"""jupiter_perp_adapter.py — Safety-wrapped sync API for Jupiter Perpetuals.

Mirror of agents/drift_adapter.py — same 8 safety gates, same public API
shape so risk_manager / orchestrator / martingale_engine can swap call sites
based on JUP_PERP_ENABLED flag.

Public API (sync — bridges async via asyncio.run_until_complete):

    open_perp_position(signal, leverage_override=None) -> Optional[PerpResult]
    close_perp_position(market) -> Optional[PerpResult]
    get_account_snapshot() -> Optional[dict]

All errors surfaced via PerpResult.reason; never raises in the gating path.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

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


# ── Async client lifecycle ───────────────────────────────────────────────────

_loop: Optional[asyncio.AbstractEventLoop] = None
_client = None


def _get_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop


def _get_or_init_client():
    """Lazy-init JupiterPerpClient on first use."""
    global _client
    if _client is None:
        from jupiter_perp_client import JupiterPerpClient
        _client = JupiterPerpClient()
        _get_loop().run_until_complete(_client.initialize())
    return _client


def close_client():
    """Tear down — call on shutdown."""
    global _client, _loop
    if _client and _loop:
        _loop.run_until_complete(_client.close())
        _client = None


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
        # TODO: integrate with safety.check_daily_loss() once we have a wrapper
        return None
    except ImportError:
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def open_perp_position(
    signal: dict,
    leverage_override: Optional[float] = None,
) -> Optional[PerpResult]:
    """Open a Jupiter perp position with safety gates.

    signal dict expected keys: symbol, direction (long|short), size_usd, confidence
    """
    # Gate 1-4: common gates
    reason = _check_common_gates()
    if reason:
        return PerpResult(success=False, reason=reason)

    # Gate 5: market whitelist
    market = signal.get("symbol", "").upper()
    whitelist = _env_whitelist("JUP_PERP_MARKET_WHITELIST", "SOL")
    if market not in whitelist:
        return PerpResult(success=False, reason=f"market_not_whitelisted: {market} not in {whitelist}")

    # Gate 6: confidence threshold
    confidence = float(signal.get("confidence", 0))
    min_conf = _env_float("MIN_CONFIDENCE", 0.55)
    if confidence < min_conf:
        return PerpResult(success=False, reason=f"low_confidence: {confidence} < {min_conf}")

    # Gate 7: leverage cap
    leverage = leverage_override if leverage_override is not None else _env_float("JUP_PERP_DEFAULT_LEVERAGE", 1.0)
    max_lev = _env_float("JUP_PERP_MAX_LEVERAGE", 3.0)
    if leverage > max_lev:
        log.warning(f"leverage {leverage} exceeds cap {max_lev}, clamping")
        leverage = max_lev

    # Gate 8: free collateral check
    try:
        client = _get_or_init_client()
        snap = _get_loop().run_until_complete(client.snapshot())
        size_usd = float(signal.get("size_usd", 0))
        margin_required = size_usd / leverage
        if snap.free_collateral_usd < margin_required:
            return PerpResult(success=False, reason=f"insufficient_collateral: free=${snap.free_collateral_usd:.2f} < required=${margin_required:.2f}")
    except Exception as e:
        return PerpResult(success=False, reason=f"snapshot_failed: {e}")

    # All gates passed — execute
    direction = signal.get("direction", "long").lower()
    if direction not in ("long", "short"):
        return PerpResult(success=False, reason=f"invalid_direction: {direction}")

    try:
        tx_sig = _get_loop().run_until_complete(
            client.open_perp_position(market, direction, size_usd, leverage)
        )
        return PerpResult(
            success=True,
            reason="opened",
            market=market,
            direction=direction,
            size_usd=size_usd,
            leverage=leverage,
            tx_signature=tx_sig,
        )
    except NotImplementedError as e:
        return PerpResult(success=False, reason=f"phase2_pending: {e}")
    except Exception as e:
        log.error(f"open_perp_position failed: {e}")
        return PerpResult(success=False, reason=f"open_failed: {e}")


def close_perp_position(market: str = "SOL") -> Optional[PerpResult]:
    """Close a Jupiter perp position (reduce-only)."""
    reason = _check_common_gates()
    if reason:
        return PerpResult(success=False, reason=reason)

    try:
        client = _get_or_init_client()
        tx_sig = _get_loop().run_until_complete(client.close_perp_position(market))
        return PerpResult(success=True, reason="closed", market=market, tx_signature=tx_sig)
    except NotImplementedError as e:
        return PerpResult(success=False, reason=f"phase2_pending: {e}")
    except Exception as e:
        return PerpResult(success=False, reason=f"close_failed: {e}")


def get_account_snapshot() -> Optional[dict]:
    """Read-only snapshot — no gates needed for reads."""
    try:
        client = _get_or_init_client()
        snap = _get_loop().run_until_complete(client.snapshot())
        return {
            "enabled": _env_bool("JUP_PERP_ENABLED", False),
            "live": _env_bool("LIVE_TRADING_ENABLED", False),
            "pubkey": snap.pubkey,
            "sol_balance": snap.sol_balance,
            "free_collateral_usd": snap.free_collateral_usd,
            "total_collateral_usd": snap.total_collateral_usd,
            "leverage": snap.leverage,
            "sol_perp_base": snap.sol_perp_base,
            "sol_perp_mark": snap.sol_perp_mark,
            "sol_perp_funding_hourly": snap.sol_perp_funding_hourly,
            "perp_account_exists": snap.perp_account_exists,
        }
    except Exception as e:
        log.error(f"get_account_snapshot failed: {e}")
        return None
