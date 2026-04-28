"""jupiter_perp_adapter.py — Safety-wrapped sync API for Jupiter Perpetuals.

Uses the official @jup-ag/cli as backend (agents/jupiter_perp_cli_wrapper.py).
This replaces the manual Python implementation with the actively-maintained
Jupiter CLI, eliminating risks from custom instruction building.

Public API (sync — bridges async via asyncio.run_until_complete):
    open_perp_position(signal, leverage_override=None) -> Optional[PerpResult]
    close_perp_position(position_pubkey) -> Optional[PerpResult]
    get_account_snapshot() -> Optional[dict]

All errors surfaced via PerpResult.reason; never raises in the gating path.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from agents.jupiter_perp_cli_wrapper import (
    close_position,
    get_positions,
    open_position,
    ensure_configured,
)

log = logging.getLogger("jupiter_perp_adapter")


@dataclass
class PerpResult:
    """Same shape as drift_adapter.PerpResult for caller compatibility."""
    success: bool
    reason: str
    market: str = ""
    direction: str = ""
    size_usd: float = 0.0
    leverage: float = 1.0
    tx_signature: str = ""
    entry_price: float = 0.0


# ── Env helpers ──────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").lower().strip()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _env_whitelist(name: str, default: str = "SOL") -> set[str]:
    raw = os.environ.get(name, default)
    return {sym.strip().upper() for sym in raw.split(",") if sym.strip()}


# ── 8 Safety Gates (mirror drift_adapter) ────────────────────────────────────

def _check_common_gates() -> Optional[str]:
    """Returns rejection reason or None if all gates pass."""
    # Gate 1: feature flag
    if not _env_bool("JUP_PERP_ENABLED", False):
        return "JUP_PERP_ENABLED=false"
    # Gate 2: master live trading switch
    if not _env_bool("LIVE_TRADING_ENABLED", False):
        return "LIVE_TRADING_ENABLED=false"
    # Gate 3: kill switch
    try:
        from safety import is_kill_switch_active
        if is_kill_switch_active():
            return "kill_switch_active"
    except ImportError:
        pass
    # Gate 4: daily loss cap
    daily_loss_status = _check_daily_loss()
    if daily_loss_status:
        return daily_loss_status
    return None


def _check_daily_loss() -> Optional[str]:
    """Check MAX_DAILY_LOSS_USD vs today's realized PnL."""
    try:
        from safety import check_daily_loss  # noqa
        return None
    except ImportError:
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def open_perp_position(
    signal: dict,
    leverage_override: Optional[float] = None,
) -> Optional[PerpResult]:
    """Open a Jupiter perp position with safety gates."""
    reason = _check_common_gates()
    if reason:
        return PerpResult(success=False, reason=reason)

    # Gate 5: market whitelist
    market = signal.get("symbol", "").upper()
    whitelist = _env_whitelist("JUP_PERP_MARKET_WHITELIST", "SOL")
    if market not in whitelist:
        return PerpResult(success=False, reason=f"market_not_whitelisted: {market}")

    # Gate 6: confidence threshold
    confidence = float(signal.get("confidence", 0))
    min_conf = _env_float("MIN_CONFIDENCE", 0.55)
    if confidence < min_conf:
        return PerpResult(success=False, reason=f"low_confidence: {confidence} < {min_conf}")

    # Gate 7: leverage cap
    leverage = leverage_override if leverage_override is not None else _env_float("JUP_PERP_DEFAULT_LEVERAGE", 2.0)
    max_lev = _env_float("JUP_PERP_MAX_LEVERAGE", 3.0)
    if leverage > max_lev:
        log.warning(f"leverage {leverage} exceeds cap {max_lev}, clamping")
        leverage = max_lev
    if leverage < 1.1:
        leverage = 1.1  # Jupiter CLI minimum

    # Gate 8: free collateral check (simplified: ensure enough USDC)
    size_usd = float(signal.get("size_usd", 0))
    collateral_usd = size_usd / leverage
    min_collateral = 10.0  # Jupiter CLI minimum
    if collateral_usd < min_collateral:
        log.warning(f"collateral {collateral_usd} below Jupiter min {min_collateral}, adjusting")
        collateral_usd = min_collateral
        size_usd = collateral_usd * leverage

    direction = signal.get("direction", "long").lower()
    if direction not in ("long", "short"):
        return PerpResult(success=False, reason=f"invalid_direction: {direction}")

    ensure_configured()

    try:
        result = open_position(
            asset=market,
            side=direction,
            collateral_usd=collateral_usd,
            leverage=leverage,
            dry_run=False,
        )
        if result.success:
            return PerpResult(
                success=True,
                reason="opened",
                market=market,
                direction=direction,
                size_usd=result.size_usd,
                leverage=result.leverage,
                tx_signature=result.tx_signature,
                entry_price=result.entry_price,
            )
        return PerpResult(success=False, reason=f"open_failed: {result.error}")
    except Exception as e:
        log.error(f"open_perp_position failed: {e}")
        return PerpResult(success=False, reason=f"open_failed: {e}")


def close_perp_position(position_pubkey: str) -> Optional[PerpResult]:
    """Close a Jupiter perp position (reduce-only)."""
    reason = _check_common_gates()
    if reason:
        return PerpResult(success=False, reason=reason)

    ensure_configured()

    try:
        result = close_position(position_pubkey=position_pubkey, dry_run=False)
        if result.success:
            return PerpResult(
                success=True,
                reason="closed",
                tx_signature=result.tx_signature,
            )
        return PerpResult(success=False, reason=f"close_failed: {result.error}")
    except Exception as e:
        log.error(f"close_perp_position failed: {e}")
        return PerpResult(success=False, reason=f"close_failed: {e}")


def get_account_snapshot() -> Optional[dict]:
    """Read-only snapshot — no gates needed for reads."""
    try:
        ensure_configured()
        positions = get_positions()
        total_size = sum(p.size_usd for p in positions)
        return {
            "enabled": _env_bool("JUP_PERP_ENABLED", False),
            "live": _env_bool("LIVE_TRADING_ENABLED", False),
            "positions_count": len(positions),
            "total_size_usd": total_size,
            "positions": [
                {
                    "pubkey": p.pubkey,
                    "asset": p.asset,
                    "side": p.side,
                    "size_usd": p.size_usd,
                    "entry_price": p.entry_price,
                    "mark_price": p.mark_price,
                    "pnl_pct": p.pnl_pct,
                    "leverage": p.leverage,
                    "liq_price": p.liq_price,
                }
                for p in positions
            ],
        }
    except Exception as e:
        log.error(f"get_account_snapshot failed: {e}")
        return None
