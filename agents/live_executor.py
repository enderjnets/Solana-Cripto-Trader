"""Live trading router — sync wrappers over LiveDriftClient that mirror
the paper_open_position / paper_update_positions interface in executor.py.

v1 scope: SOL-PERP only. Non-SOL signals are rejected with a warning.

The router is invoked per cycle by executor.py.run(safe=False). It:
  1. Reuses paper_open_position for filters + sizing + SL/TP + dict shape.
  2. Overrides execution: opens/closes positions on Drift instead of simulating.
  3. Rolls back portfolio mutations if the Drift tx fails, so the portfolio
     never reports a position that isn't actually on-chain.

Each cycle runs in its own event loop (asyncio.run). The LiveDriftClient is
re-initialized per cycle — roughly 1s overhead, acceptable at the 120s cycle
cadence. A persistent client is a future optimization.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from agents.live_drift_client import LiveDriftClient

log = logging.getLogger("live_executor")

LIVE_SUPPORTED_SYMBOLS = {"SOL"}

# SOLAA-59: opt into limit orders to eliminate slippage. Default off for now;
# flip via env var so operators can A/B compare against market execution.
USE_LIMIT_ORDERS = os.environ.get("LIVE_USE_LIMIT_ORDERS", "false").lower() == "true"
# Offset from oracle mid — LONG buys 3 bps below mid, SHORT sells 3 bps above.
LIMIT_OFFSET_BPS = int(os.environ.get("LIVE_LIMIT_OFFSET_BPS", "3"))
LIMIT_TIMEOUT_SEC = int(os.environ.get("LIVE_LIMIT_TIMEOUT_SEC", "60"))


def _rollback(portfolio: dict, pos: dict) -> None:
    """Undo paper_open_position's portfolio mutations (remove position, refund margin)."""
    try:
        portfolio["positions"].remove(pos)
    except ValueError:
        pass
    portfolio["capital_usd"] = round(
        portfolio.get("capital_usd", 0) + pos.get("margin_usd", 0), 2
    )


async def _open_on_drift(pos: dict) -> tuple[Optional[str], float, bool]:
    """Open pos on Drift. Returns (tx_sig, oracle_entry_price, filled).

    `filled` is only meaningful when LIVE_USE_LIMIT_ORDERS=true; market orders
    return True optimistically (Drift's place_and_take either fills in the
    auction or leaves a standing order — we treat the tx landing as success).
    """
    client = LiveDriftClient()
    try:
        await client.initialize()
        snap = await client.snapshot()
        if snap.free_collateral_usd < pos["margin_usd"]:
            raise RuntimeError(
                f"insufficient Drift collateral: have ${snap.free_collateral_usd:.2f}, "
                f"need ${pos['margin_usd']:.2f}"
            )
        if USE_LIMIT_ORDERS:
            # LONG: bid slightly below mark. SHORT: ask slightly above mark.
            offset = snap.sol_perp_mark * (LIMIT_OFFSET_BPS / 10_000.0)
            limit_price = (
                snap.sol_perp_mark - offset
                if pos["direction"] == "long"
                else snap.sol_perp_mark + offset
            )
            sig, filled, _ = await client.open_sol_perp_limit(
                pos["direction"], pos["base_size"], limit_price, LIMIT_TIMEOUT_SEC
            )
            return sig, limit_price, filled
        sig = await client.open_sol_perp_market(pos["direction"], pos["base_size"])
        return sig, snap.sol_perp_mark, True
    finally:
        await client.close()


async def _close_on_drift() -> Optional[str]:
    client = LiveDriftClient()
    try:
        await client.initialize()
        return await client.close_sol_perp()
    finally:
        await client.close()


def live_open_position(signal: dict, portfolio: dict, market: dict) -> Optional[dict]:
    """Live-mode counterpart to paper_open_position. Same signature.

    Reuses paper_open_position (filters, sizing, SL/TP), then swaps the
    simulated execution for a real Drift market order. Rolls back if the
    Drift tx fails so portfolio never records a ghost position.
    """
    from agents.executor import paper_open_position  # import lazily to avoid cycles

    symbol = signal.get("symbol", "")
    if symbol not in LIVE_SUPPORTED_SYMBOLS:
        log.debug(f"live mode: skipping {symbol} (only {LIVE_SUPPORTED_SYMBOLS} supported)")
        return None

    pos = paper_open_position(signal, portfolio, market)
    if pos is None:
        return None

    try:
        sig, entry_price, filled = asyncio.run(_open_on_drift(pos))
    except Exception as e:
        log.error(f"live open failed for {symbol}: {e} — rolling back")
        _rollback(portfolio, pos)
        return None

    if not filled:
        log.warning(f"live limit {symbol} timed out unfilled — rolling back")
        _rollback(portfolio, pos)
        return None

    pos["mode"] = "live"
    pos["drift_tx_sig"] = sig
    pos["drift_market_index"] = 0
    pos["live_entry_oracle_price"] = round(entry_price, 6)
    pos["live_execution_style"] = "limit" if USE_LIMIT_ORDERS else "market"
    log.info(f"✅ LIVE {pos['direction']} {pos['base_size']} {symbol} @ ${entry_price:.2f} "
             f"({'limit' if USE_LIMIT_ORDERS else 'market'}) tx={sig[:12]}…")
    return pos


def live_update_positions(portfolio: dict, market: dict, history: list) -> list[dict]:
    """Live-mode counterpart to paper_update_positions. Same signature.

    For each open live position: read current mark from Drift, check SL/TP,
    close on-chain if triggered, mutate the position dict to reflect the close.
    Returns the list of positions closed this cycle.
    """
    live_positions = [
        p for p in portfolio.get("positions", [])
        if p.get("mode") == "live" and p.get("status") == "open"
    ]
    if not live_positions:
        return []

    async def snapshot_mark() -> float:
        client = LiveDriftClient()
        try:
            await client.initialize()
            return await client.get_mark_price()
        finally:
            await client.close()

    try:
        mark = asyncio.run(snapshot_mark())
    except Exception as e:
        log.error(f"live snapshot failed: {e} — skipping update this cycle")
        return []

    closed: list[dict] = []
    for pos in live_positions:
        entry = pos.get("entry_price", 0)
        if entry <= 0:
            continue
        sl = pos.get("sl_price", 0)
        tp = pos.get("tp_price", 0)
        is_long = pos["direction"] == "long"

        hit_sl = (is_long and sl > 0 and mark <= sl) or (not is_long and sl > 0 and mark >= sl)
        hit_tp = (is_long and tp > 0 and mark >= tp) or (not is_long and tp > 0 and mark <= tp)

        if not (hit_sl or hit_tp):
            continue

        reason = "SL" if hit_sl else "TP"
        log.info(f"🔔 LIVE {pos['symbol']} {reason} hit @ ${mark:.4f} (entry ${entry:.4f}) — closing")

        try:
            sig = asyncio.run(_close_on_drift())
        except Exception as e:
            log.error(f"live close failed for {pos['symbol']}: {e} — will retry next cycle")
            continue

        if sig is None:
            # No position on Drift — already closed or never opened. Mark as closed anyway.
            log.warning(f"no Drift position found for {pos['symbol']} — marking closed locally")

        # Mirror paper close fields
        pnl_pct = ((mark / entry) - 1) * (1 if is_long else -1)
        pnl_usd = pos.get("margin_usd", 0) * pnl_pct * pos.get("leverage", 1)
        pos["status"] = "closed"
        pos["close_time"] = datetime.now(timezone.utc).isoformat()
        pos["exit_price"] = round(mark, 8)
        pos["exit_reason"] = reason
        pos["pnl_usd"] = round(pnl_usd, 4)
        pos["pnl_pct"] = round(pnl_pct * 100, 4)
        pos["drift_close_tx_sig"] = sig
        portfolio["capital_usd"] = round(
            portfolio.get("capital_usd", 0) + pos.get("margin_usd", 0) + pnl_usd, 2
        )
        history.append(pos)
        closed.append(pos)

    return closed
