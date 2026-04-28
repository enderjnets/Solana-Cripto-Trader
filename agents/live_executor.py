"""Live trading router — sync wrappers over Jupiter Perpetuals.

Replaces the Drift Protocol integration (deprecated after April 2026 hack)
with Jupiter Perpetuals via agents.jupiter_perp_adapter.

The router is invoked per cycle by executor.py.run(safe=False). It:
  1. Reuses paper_open_position for filters + sizing + SL/TP + dict shape.
  2. Overrides execution: opens/closes positions on Jupiter Perps.
  3. Rolls back portfolio mutations if the Jupiter tx fails, so the portfolio
     never reports a position that isn't actually on-chain.

Each open/close is a blocking subprocess call to `npx @jup-ag/cli`.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from agents.jupiter_perp_adapter import (
    open_perp_position,
    close_perp_position,
    get_account_snapshot,
)

log = logging.getLogger("live_executor")

LIVE_SUPPORTED_SYMBOLS = {"SOL"}

# Jupiter Perps uses market orders; limit orders are not supported via CLI.
# Slippage is controlled via --slippage on the CLI (default 200 bps = 2%).
LIVE_SLIPPAGE_BPS = int(os.environ.get("JUP_PERP_SLIPPAGE_BPS", "200"))


def _rollback(portfolio: dict, pos: dict) -> None:
    """Undo paper_open_position's portfolio mutations (remove position, refund margin)."""
    try:
        portfolio["positions"].remove(pos)
    except ValueError:
        pass
    portfolio["capital_usd"] = round(
        portfolio.get("capital_usd", 0) + pos.get("margin_usd", 0), 2
    )


def live_open_position(signal: dict, portfolio: dict, market: dict) -> Optional[dict]:
    """Live-mode counterpart to paper_open_position. Same signature.

    Reuses paper_open_position (filters, sizing, SL/TP), then swaps the
    simulated execution for a real Jupiter Perp order. Rolls back if the
    Jupiter tx fails so portfolio never records a ghost position.
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
        result = open_perp_position(signal)
    except Exception as e:
        log.error(f"live open failed for {symbol}: {e} — rolling back")
        _rollback(portfolio, pos)
        return None

    if not result or not result.success:
        log.error(f"live open rejected for {symbol}: {result.reason if result else 'no_result'} — rolling back")
        _rollback(portfolio, pos)
        return None

    pos["mode"] = "live"
    pos["jup_tx_sig"] = result.tx_signature
    pos["jup_position_pubkey"] = result.position_pubkey
    pos["live_entry_price"] = round(result.entry_price, 6)
    pos["live_execution_style"] = "market"
    log.info(f"✅ LIVE {pos['direction']} ${result.size_usd:.2f} {symbol} @ ${result.entry_price:.2f} "
             f"tx={result.tx_signature[:12]}…")
    return pos


def live_update_positions(portfolio: dict, market: dict, history: list) -> list[dict]:
    """Live-mode counterpart to paper_update_positions. Same signature.

    For each open live position: read current mark from market data, check SL/TP,
    close on-chain if triggered, mutate the position dict to reflect the close.
    Returns the list of positions closed this cycle.
    """
    live_positions = [
        p for p in portfolio.get("positions", [])
        if p.get("mode") == "live" and p.get("status") == "open"
    ]
    if not live_positions:
        return []

    # Use market price from the market dict (same source as paper trading)
    mark = market.get("price_usd", 0) if market else 0
    if mark <= 0:
        log.warning("live_update_positions: no valid market price, skipping")
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

        pubkey = pos.get("jup_position_pubkey", "")
        if not pubkey:
            log.error(f"live close failed for {pos['symbol']}: no position pubkey stored — will retry next cycle")
            continue

        try:
            result = close_perp_position(position_pubkey=pubkey)
        except Exception as e:
            log.error(f"live close failed for {pos['symbol']}: {e} — will retry next cycle")
            continue

        if not result or not result.success:
            log.error(f"live close rejected for {pos['symbol']}: {result.reason if result else 'no_result'} — will retry next cycle")
            continue

        # Mirror paper close fields
        pnl_pct = ((mark / entry) - 1) * (1 if is_long else -1)
        pnl_usd = pos.get("margin_usd", 0) * pnl_pct * pos.get("leverage", 1)
        pos["status"] = "closed"
        pos["close_time"] = datetime.now(timezone.utc).isoformat()
        pos["exit_price"] = round(mark, 8)
        pos["exit_reason"] = reason
        pos["pnl_usd"] = round(pnl_usd, 4)
        pos["pnl_pct"] = round(pnl_pct * 100, 4)
        pos["jup_close_tx_sig"] = result.tx_signature
        portfolio["capital_usd"] = round(
            portfolio.get("capital_usd", 0) + pos.get("margin_usd", 0) + pnl_usd, 2
        )
        history.append(pos)
        closed.append(pos)

    return closed
