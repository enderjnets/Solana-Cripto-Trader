"""
drift_adapter.py — Safety-wrapped entry point for Drift Protocol perps trading.

Wraps agents/drift_client.py (LiveDriftClient) with 8 gates so the bot can
replace the Jupiter spot path with Drift perps (longs + shorts, 1-2x leverage)
without losing any existing safety layer.

Public API (sync — bridges driftpy's async API via asyncio.run_until_complete):

    open_perp_position(signal, leverage_override=None) -> Optional[PerpResult]
    close_perp_position(symbol) -> Optional[PerpResult]
    get_account_snapshot() -> Optional[dict]

All errors surfaced via PerpResult.reason; never raises in the gating path.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger('drift_adapter')


@dataclass
class PerpResult:
    success: bool = False
    symbol: str = ""
    direction: str = ""
    size_sol: float = 0.0
    leverage: float = 0.0
    signature: Optional[str] = None
    reason: str = ""
    filled: bool = False
    entry_price: float = 0.0


# ── env helpers ──
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _env_whitelist(name: str, default: str = "SOL") -> set[str]:
    raw = os.environ.get(name, default)
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


# ── event loop singleton (drift client keeps state across calls) ──
_client_cache: dict = {"loop": None, "client": None}


def _get_loop():
    if _client_cache["loop"] is None or _client_cache["loop"].is_closed():
        _client_cache["loop"] = asyncio.new_event_loop()
    return _client_cache["loop"]


def _get_or_init_client():
    """Returns an initialized LiveDriftClient instance (cached across calls)."""
    loop = _get_loop()
    client = _client_cache.get("client")
    if client is not None:
        return client
    import drift_client as _dc
    client = _dc.LiveDriftClient()
    loop.run_until_complete(client.initialize())
    _client_cache["client"] = client
    return client


def close_client():
    """Release the cached drift client (call on shutdown)."""
    client = _client_cache.get("client")
    if client is not None:
        try:
            _get_loop().run_until_complete(client.close())
        except Exception as e:
            log.warning(f"close_client error: {e}")
    _client_cache["client"] = None


# ── gate helpers ──
def _check_common_gates() -> Optional[str]:
    """Return None if all gates pass, else a reason string."""
    # Gate 1: feature flag
    if not _env_bool("DRIFT_ENABLED", False):
        return "drift_disabled"
    # Gate 2: live trading
    if not _env_bool("LIVE_TRADING_ENABLED", False):
        return "live_trading_disabled"
    # Gate 3: kill switch
    try:
        import safety
        ks, reason = safety.is_kill_switch_active()
        if ks:
            return f"kill_switch_active:{reason}"
    except Exception as e:
        log.warning(f"safety import failed: {e}")
        return f"safety_import_error:{e}"
    return None


def _check_daily_loss() -> Optional[str]:
    """Gate 4: daily loss cap."""
    try:
        import safety, json
        from pathlib import Path
        pf_file = Path(__file__).parent / "data" / "portfolio.json"
        pf_data = json.loads(pf_file.read_text()) if pf_file.exists() else None
        hit, todays_pnl = safety.check_daily_loss(pf_data)
        if hit:
            return f"daily_loss_exceeded:{todays_pnl:.2f}"
    except Exception as e:
        log.warning(f"daily loss check failed: {e}")
    return None


# ── public API ──
def open_perp_position(signal: dict, leverage_override: Optional[float] = None) -> Optional[PerpResult]:
    """Open a Drift perp position from a bot signal.

    signal: {"symbol": "SOL", "direction": "long"|"short", "confidence": 0.8, "suggested_size_usd": 2.0}
    leverage_override: if given, overrides DRIFT_DEFAULT_LEVERAGE for this trade.
    Returns None if any gate blocks, PerpResult otherwise.
    """
    # Common gates 1-3
    reason = _check_common_gates()
    if reason:
        log.info(f"drift_adapter: skip — {reason}")
        return None
    # Gate 4: daily loss
    reason = _check_daily_loss()
    if reason:
        log.warning(f"drift_adapter: {reason}")
        return PerpResult(reason=reason)

    # Gate 5: market whitelist
    symbol = str(signal.get("symbol", "")).upper()
    whitelist = _env_whitelist("DRIFT_MARKET_WHITELIST", "SOL")
    if symbol not in whitelist:
        log.info(f"drift_adapter: skip {symbol} — not in DRIFT_MARKET_WHITELIST ({whitelist})")
        return None

    # Gate 6: confidence
    confidence = float(signal.get("confidence", 0))
    min_conf = _env_float("MIN_CONFIDENCE", 0.75)
    if confidence < min_conf:
        log.info(f"drift_adapter: skip {symbol} — confidence {confidence:.2f} < {min_conf:.2f}")
        return None

    # Gate 7: leverage cap
    max_lev = _env_float("DRIFT_MAX_LEVERAGE", 2.0)
    default_lev = _env_float("DRIFT_DEFAULT_LEVERAGE", 1.0)
    leverage = leverage_override if leverage_override is not None else default_lev
    if leverage > max_lev:
        log.warning(f"drift_adapter: leverage {leverage} > cap {max_lev} — clamping")
        leverage = max_lev
    if leverage < 1.0:
        leverage = 1.0

    # Direction validation
    direction = str(signal.get("direction", "")).lower()
    if direction not in ("long", "short"):
        return PerpResult(reason=f"invalid_direction:{direction}")

    # Size: notional = suggested_size_usd; collateral needed = notional / leverage
    size_usd = float(signal.get("suggested_size_usd", 2.0))
    max_collateral = _env_float("DRIFT_MAX_COLLATERAL_USD", 3.0)

    # Initialize client (cached)
    try:
        client = _get_or_init_client()
        loop = _get_loop()
    except Exception as e:
        log.error(f"drift_adapter: client init failed: {e}")
        return PerpResult(reason=f"client_init_error:{e}")

    # Gate 8: free collateral check
    try:
        snap = loop.run_until_complete(client.snapshot())
    except Exception as e:
        log.error(f"drift_adapter: snapshot failed: {e}")
        return PerpResult(reason=f"snapshot_error:{e}")

    if not snap.drift_user_exists:
        log.error("drift_adapter: drift user not initialized — run tools/drift_setup.py first")
        return PerpResult(reason="drift_user_not_initialized")

    if snap.total_collateral_usd > max_collateral:
        log.warning(f"drift_adapter: total collateral ${snap.total_collateral_usd:.2f} > cap ${max_collateral:.2f}")
        return PerpResult(reason="collateral_over_cap")

    collateral_needed = size_usd / leverage
    if snap.free_collateral_usd < collateral_needed:
        log.warning(f"drift_adapter: insufficient free collateral — "
                    f"need ${collateral_needed:.2f} have ${snap.free_collateral_usd:.2f}")
        return PerpResult(reason=f"insufficient_collateral:{snap.free_collateral_usd:.2f}<{collateral_needed:.2f}")

    # Execute open
    size_sol = size_usd / snap.sol_perp_mark if snap.sol_perp_mark > 0 else 0
    if size_sol <= 0:
        return PerpResult(reason="invalid_size_zero")

    log.info(f"🎯 DRIFT OPEN: {symbol}-PERP {direction} size=${size_usd} "
             f"({size_sol:.4f} SOL) leverage={leverage}x @ mark ${snap.sol_perp_mark:.2f}")
    try:
        sig = loop.run_until_complete(
            client.open_sol_perp_market(direction=direction, size_sol=size_sol, slippage_bps=100)
        )
    except Exception as e:
        log.error(f"drift_adapter: open failed: {e}")
        return PerpResult(reason=f"open_error:{e}", symbol=symbol, direction=direction, size_sol=size_sol, leverage=leverage)

    return PerpResult(
        success=True, symbol=symbol, direction=direction,
        size_sol=size_sol, leverage=leverage, signature=sig,
        reason="ok", filled=True, entry_price=snap.sol_perp_mark,
    )


def close_perp_position(symbol: str = "SOL") -> Optional[PerpResult]:
    """Close current Drift perp position for symbol (reduce-only market order)."""
    reason = _check_common_gates()
    if reason:
        log.info(f"drift_adapter.close: skip — {reason}")
        return None

    try:
        client = _get_or_init_client()
        loop = _get_loop()
    except Exception as e:
        return PerpResult(reason=f"client_init_error:{e}")

    try:
        snap = loop.run_until_complete(client.snapshot())
        sig = loop.run_until_complete(client.close_sol_perp(slippage_bps=100))
    except Exception as e:
        log.error(f"drift_adapter: close failed: {e}")
        return PerpResult(reason=f"close_error:{e}")

    if sig is None:
        return PerpResult(reason="no_open_position", symbol=symbol)

    return PerpResult(
        success=True, symbol=symbol,
        direction="short" if snap.sol_perp_base > 0 else "long",  # opposite of existing
        size_sol=abs(snap.sol_perp_base),
        signature=sig, reason="ok", filled=True,
        entry_price=snap.sol_perp_mark,
    )


def get_account_snapshot() -> Optional[dict]:
    """Read-only snapshot for dashboard + reconcile. None if Drift disabled."""
    if not _env_bool("DRIFT_ENABLED", False):
        return {"enabled": False}
    if not _env_bool("LIVE_TRADING_ENABLED", False):
        return {"enabled": True, "live": False}
    try:
        client = _get_or_init_client()
        loop = _get_loop()
        snap = loop.run_until_complete(client.snapshot())
    except Exception as e:
        log.warning(f"drift_adapter.snapshot error: {e}")
        return {"enabled": True, "live": True, "error": str(e)}

    return {
        "enabled": True,
        "live": True,
        "pubkey": snap.pubkey,
        "user_exists": snap.drift_user_exists,
        "sol_native": snap.sol_balance,
        "collateral_free_usd": snap.free_collateral_usd,
        "collateral_total_usd": snap.total_collateral_usd,
        "current_leverage": snap.leverage,
        "sol_perp_base": snap.sol_perp_base,
        "sol_perp_mark": snap.sol_perp_mark,
        "funding_hourly": snap.sol_perp_funding_hourly,
        "max_leverage_cap": _env_float("DRIFT_MAX_LEVERAGE", 2.0),
        "max_collateral_cap": _env_float("DRIFT_MAX_COLLATERAL_USD", 3.0),
        "whitelist": sorted(_env_whitelist("DRIFT_MARKET_WHITELIST", "SOL")),
    }
