"""
sol_topup.py — Auto-replenish SOL fuel for transaction fees (v2.11.0-live)

When the hot wallet's SOL balance drops below SOL_MIN_BALANCE, swap a small
amount of USDC -> SOL via Jupiter so the bot keeps running without manual
intervention.

Safety gates (defense-in-depth):
  1. SOL_AUTO_TOPUP_ENABLED=true (feature flag, default false)
  2. LIVE_TRADING_ENABLED=true (paper does not burn SOL)
  3. kill switch not active
  4. SOL balance < SOL_MIN_BALANCE
  5. USDC balance - topup >= SOL_TOPUP_MIN_USDC_RESERVE
  6. spent_today + topup <= SOL_TOPUP_MAX_DAILY_USD
"""
from __future__ import annotations

import os
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

log = logging.getLogger('sol_topup')

STATE_FILE = Path(__file__).parent / "data" / "sol_topup_state.json"


@dataclass
class TopupResult:
    attempted: bool = False
    success: bool = False
    in_usd: float = 0.0
    out_sol: float = 0.0
    signature: Optional[str] = None
    price_impact_pct: float = 0.0
    reason: str = ""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"date": _today_utc(), "spent_usd_today": 0.0, "events": []}
    try:
        s = json.loads(STATE_FILE.read_text())
    except Exception:
        return {"date": _today_utc(), "spent_usd_today": 0.0, "events": []}
    if s.get("date") != _today_utc():
        s = {"date": _today_utc(), "spent_usd_today": 0.0, "events": s.get("events", [])[-50:]}
    return s


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.warning(f"sol_topup: failed to persist state: {e}")


def fuel_status(rpc, wallet) -> dict:
    """Read-only snapshot for dashboard /api/sol/fuel."""
    import solana_rpc as _srpc
    state = _load_state()
    try:
        sol_balance = rpc.get_balance_sol(wallet.pubkey)
    except Exception as e:
        sol_balance = None
        log.warning(f"fuel_status sol balance error: {e}")
    try:
        usdc_balance = rpc.get_token_balance(wallet.pubkey, _srpc.MINT_USDC)
    except Exception as e:
        usdc_balance = None
        log.warning(f"fuel_status usdc balance error: {e}")

    min_bal = _env_float("SOL_MIN_BALANCE", 0.01)
    approx_sol_per_trade = 0.00005  # conservative estimate (priority fee + sig)
    trades_remaining = int(sol_balance / approx_sol_per_trade) if sol_balance else None

    return {
        "enabled": _env_bool("SOL_AUTO_TOPUP_ENABLED", False),
        "live": _env_bool("LIVE_TRADING_ENABLED", False),
        "sol_balance": sol_balance,
        "sol_min_balance": min_bal,
        "usdc_balance": usdc_balance,
        "usdc_reserve_floor": _env_float("SOL_TOPUP_MIN_USDC_RESERVE", 2.0),
        "trades_remaining_estimate": trades_remaining,
        "topup_amount_usd": _env_float("SOL_TOPUP_AMOUNT_USD", 1.0),
        "max_daily_usd": _env_float("SOL_TOPUP_MAX_DAILY_USD", 3.0),
        "spent_today_usd": state.get("spent_usd_today", 0.0),
        "events_today": len([e for e in state.get("events", []) if e.get("ts", "").startswith(state.get("date", ""))]),
        "last_event": state.get("events", [])[-1] if state.get("events") else None,
    }


def maybe_topup(rpc, wallet, dry_run: bool = False) -> Optional[TopupResult]:
    """Main entry point. Returns None if no action taken (gate blocked),
    TopupResult otherwise. Never raises — all errors surfaced via reason field."""
    # Gate 1: feature flag
    if not _env_bool("SOL_AUTO_TOPUP_ENABLED", False):
        return None
    # Gate 2: live only
    if not _env_bool("LIVE_TRADING_ENABLED", False):
        return None

    # Gate 3: kill switch
    try:
        import safety
        ks_active, ks_reason = safety.is_kill_switch_active()
        if ks_active:
            log.info(f"sol_topup: kill switch active ({ks_reason}) — skip")
            return None
    except Exception as e:
        log.warning(f"sol_topup: kill switch check failed: {e}")
        return None

    # Read balances
    try:
        import solana_rpc
        sol_balance = rpc.get_balance_sol(wallet.pubkey)
    except Exception as e:
        log.warning(f"sol_topup: get_balance_sol failed: {e}")
        return None

    min_balance = _env_float("SOL_MIN_BALANCE", 0.01)
    # Gate 4: balance threshold
    if sol_balance >= min_balance:
        return None

    topup_usd = _env_float("SOL_TOPUP_AMOUNT_USD", 1.0)
    reserve = _env_float("SOL_TOPUP_MIN_USDC_RESERVE", 2.0)
    max_daily = _env_float("SOL_TOPUP_MAX_DAILY_USD", 3.0)

    # Read USDC
    try:
        usdc_balance = rpc.get_token_balance(wallet.pubkey, solana_rpc.MINT_USDC)
    except Exception as e:
        log.warning(f"sol_topup: get_token_balance USDC failed: {e}")
        return TopupResult(attempted=False, reason=f"usdc_balance_error:{e}")

    # Gate 5: USDC reserve floor
    if usdc_balance - topup_usd < reserve:
        msg = f"skip: USDC {usdc_balance:.2f} - {topup_usd} < reserve {reserve}"
        log.warning(f"sol_topup: {msg}")
        return TopupResult(attempted=False, reason=msg)

    # Gate 6: daily cap
    state = _load_state()
    if state.get("spent_usd_today", 0.0) + topup_usd > max_daily:
        msg = f"skip: daily cap reached (spent={state['spent_usd_today']:.2f} + {topup_usd} > {max_daily})"
        log.warning(f"sol_topup: {msg}")
        return TopupResult(attempted=False, reason=msg)

    # Execute swap
    log.info(f"🔋 SOL FUEL LOW ({sol_balance:.6f} < {min_balance}) → topup ${topup_usd} USDC → SOL")
    try:
        import jupiter_swap
        swap = jupiter_swap.JupiterSwap(wallet=wallet, rpc=rpc)
        usdc_lamports = int(topup_usd * 1_000_000)  # USDC has 6 decimals
        swap_result = swap.execute_swap(
            input_mint=solana_rpc.MINT_USDC,
            output_mint=solana_rpc.MINT_SOL,
            amount_lamports=usdc_lamports,
            slippage_bps=100,
            priority_fee_level='medium',
            dry_run=dry_run,
        )
    except Exception as e:
        log.error(f"sol_topup: execute_swap raised: {e}")
        return TopupResult(attempted=True, success=False, in_usd=topup_usd, reason=f"swap_exception:{e}")

    out_sol = swap_result.out_amount / 1e9 if swap_result.out_amount else 0.0

    if dry_run:
        log.info(f"sol_topup DRY RUN: would swap ${topup_usd} → {out_sol:.6f} SOL (impact {swap_result.price_impact_pct:.4f}%)")
        return TopupResult(attempted=True, success=False, in_usd=topup_usd,
                            out_sol=out_sol, price_impact_pct=swap_result.price_impact_pct,
                            reason="dry_run")

    if swap_result.error or not swap_result.confirmed:
        reason = swap_result.error or "not_confirmed"
        log.error(f"sol_topup: swap failed — {reason}")
        return TopupResult(attempted=True, success=False, in_usd=topup_usd, reason=reason,
                            signature=swap_result.signature)

    # Success — persist state
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "signature": swap_result.signature,
        "in_usd": topup_usd,
        "out_sol": out_sol,
        "price_impact_pct": swap_result.price_impact_pct,
        "sol_before": sol_balance,
    }
    state["spent_usd_today"] = state.get("spent_usd_today", 0.0) + topup_usd
    state.setdefault("events", []).append(event)
    state["events"] = state["events"][-100:]  # cap history
    _save_state(state)

    log.info(f"✅ SOL topup confirmed: {swap_result.signature} — +{out_sol:.6f} SOL")

    # Fire-and-forget Paperclip notification
    try:
        import paperclip_client
        if hasattr(paperclip_client, '_create_issue'):
            paperclip_client._create_issue(
                title=f"SOL Auto-Topup: ${topup_usd} USDC → {out_sol:.6f} SOL",
                description=(f"SOL balance was {sol_balance:.6f} (< {min_balance}). "
                             f"Swapped ${topup_usd} USDC for {out_sol:.6f} SOL. "
                             f"Signature: {swap_result.signature}. "
                             f"Daily spent: ${state['spent_usd_today']:.2f}/{max_daily}."),
                priority="low",
                status="done",
            )
    except Exception as e:
        log.debug(f"sol_topup: paperclip log failed (non-fatal): {e}")

    return TopupResult(attempted=True, success=True, in_usd=topup_usd,
                        out_sol=out_sol, signature=swap_result.signature,
                        price_impact_pct=swap_result.price_impact_pct,
                        reason="ok")
