"""jupiter_perp_adapter.py — Safety-wrapped entry point for Jupiter Perpetuals.

Drop-in replacement for drift_adapter.py. Uses the official @jup-ag/cli via
agents/jupiter_perp_cli_wrapper.py.

Public API (sync):
    open_perp_position(signal, leverage_override=None) -> Optional[PerpResult]
    close_perp_position(symbol="SOL") -> Optional[PerpResult]
    get_account_snapshot() -> Optional[dict]

All errors surfaced via PerpResult.reason; never raises in the gating path.
"""
from __future__ import annotations

import logging
import os
import time
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


# ── gate helpers ──
def _check_common_gates() -> Optional[str]:
    """Return None if all gates pass, else a reason string."""
    # Gate 1: feature flag
    if not _env_bool("JUP_PERP_ENABLED", False):
        return "jup_perp_disabled"
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
    """Open a Jupiter perp position from a bot signal.

    signal: {"symbol": "SOL", "direction": "long"|"short", "confidence": 0.8, "suggested_size_usd": 2.0}
    leverage_override: if given, overrides JUP_PERP_DEFAULT_LEVERAGE for this trade.
    Returns None if any gate blocks, PerpResult otherwise.
    """
    # Common gates 1-3
    reason = _check_common_gates()
    if reason:
        log.info(f"jupiter_perp_adapter: skip — {reason}")
        return None
    # Gate 4: daily loss
    reason = _check_daily_loss()
    if reason:
        log.warning(f"jupiter_perp_adapter: {reason}")
        return PerpResult(reason=reason)

    # Gate 5: market whitelist
    symbol = str(signal.get("symbol", "")).upper()
    whitelist = _env_whitelist("JUP_PERP_MARKET_WHITELIST", "SOL")
    if symbol not in whitelist:
        log.info(f"jupiter_perp_adapter: skip {symbol} — not in JUP_PERP_MARKET_WHITELIST ({whitelist})")
        return None

    # Gate 6: confidence
    confidence = float(signal.get("confidence", 0))
    min_conf = _env_float("MIN_CONFIDENCE", 0.55)
    if confidence < min_conf:
        log.info(f"jupiter_perp_adapter: skip {symbol} — confidence {confidence:.2f} < {min_conf:.2f}")
        return None

    # Gate 7: leverage cap
    max_lev = _env_float("JUP_PERP_MAX_LEVERAGE", 3.0)
    default_lev = _env_float("JUP_PERP_DEFAULT_LEVERAGE", 2.0)
    leverage = leverage_override if leverage_override is not None else default_lev
    if leverage > max_lev:
        log.warning(f"jupiter_perp_adapter: leverage {leverage} > cap {max_lev} — clamping")
        leverage = max_lev
    if leverage < 1.1:
        leverage = 1.1  # Jupiter CLI minimum

    # Direction validation
    direction = str(signal.get("direction", "")).lower()
    if direction not in ("long", "short"):
        return PerpResult(reason=f"invalid_direction:{direction}")

    # Size: collateral needed = size_usd / leverage
    size_usd = float(signal.get("suggested_size_usd", 10.0))
    min_collateral = 10.0  # Jupiter CLI minimum
    collateral_usd = size_usd / leverage
    if collateral_usd < min_collateral:
        log.warning(f"jupiter_perp_adapter: collateral {collateral_usd:.2f} below Jupiter min {min_collateral}, adjusting")
        collateral_usd = min_collateral
        size_usd = collateral_usd * leverage

    # Ensure CLI is configured
    try:
        ensure_configured()
    except Exception as e:
        log.error(f"jupiter_perp_adapter: CLI config failed: {e}")
        return PerpResult(reason=f"cli_config_error:{e}")

    # Execute open
    log.info(f"🎯 JUPITER PERP OPEN: {symbol} {direction} collateral=${collateral_usd:.2f} leverage={leverage}x")
    try:
        result = open_position(
            asset=symbol,
            side=direction,
            collateral_usd=collateral_usd,
            leverage=leverage,
            dry_run=False,
        )
    except Exception as e:
        log.error(f"jupiter_perp_adapter: open failed: {e}")
        return PerpResult(reason=f"open_error:{e}", symbol=symbol, direction=direction, size_sol=0, leverage=leverage)

    if not result.success:
        return PerpResult(reason=f"open_failed:{result.error}", symbol=symbol, direction=direction, size_sol=0, leverage=leverage)

    # Derive size_sol from size_usd and entry_price for portfolio compatibility
    entry_price = result.entry_price if result.entry_price > 0 else 1.0
    size_sol = result.size_usd / entry_price if entry_price > 0 else 0

    return PerpResult(
        success=True, symbol=symbol, direction=direction,
        size_sol=size_sol, leverage=result.leverage, signature=result.tx_signature,
        reason="ok", filled=True, entry_price=entry_price,
    )


def close_perp_position(symbol: str = "SOL") -> Optional[PerpResult]:
    """Close current Jupiter perp position for symbol (reduce-only market order).
    Closes ALL open positions for the symbol (Jupiter CLI does not support partial close
    via simple command; we close each position individually)."""
    reason = _check_common_gates()
    if reason:
        log.info(f"jupiter_perp_adapter.close: skip — {reason}")
        return None

    try:
        ensure_configured()
    except Exception as e:
        return PerpResult(reason=f"cli_config_error:{e}")

    try:
        positions = get_positions()
    except Exception as e:
        log.error(f"jupiter_perp_adapter: get_positions failed: {e}")
        return PerpResult(reason=f"get_positions_error:{e}")

    symbol_upper = symbol.upper()
    target_positions = [p for p in positions if p.asset.upper() == symbol_upper]

    if not target_positions:
        return PerpResult(reason="no_open_position", symbol=symbol)

    closed_any = False
    last_sig = None
    total_size_sol = 0.0
    for p in target_positions:
        try:
            result = close_position(position_pubkey=p.pubkey, dry_run=False)
            if result.success:
                closed_any = True
                last_sig = result.tx_signature
                # size_sol approximation
                entry = p.entry_price if p.entry_price > 0 else 1.0
                total_size_sol += p.size_usd / entry
            else:
                log.warning(f"jupiter_perp_adapter: close failed for {p.pubkey}: {result.error}")
        except Exception as e:
            log.error(f"jupiter_perp_adapter: close exception for {p.pubkey}: {e}")

    if not closed_any:
        return PerpResult(reason="close_failed", symbol=symbol)

    return PerpResult(
        success=True, symbol=symbol,
        direction="short",  # placeholder — direction of the closing action
        size_sol=total_size_sol,
        signature=last_sig, reason="ok", filled=True,
        entry_price=target_positions[0].entry_price if target_positions else 0,
    )


def get_account_snapshot() -> Optional[dict]:
    """Read-only snapshot for dashboard + reconcile. None if Jupiter Perps disabled."""
    if not _env_bool("JUP_PERP_ENABLED", False):
        return {"enabled": False}
    if not _env_bool("LIVE_TRADING_ENABLED", False):
        return {"enabled": True, "live": False}
    try:
        ensure_configured()
        positions = get_positions()
    except Exception as e:
        log.warning(f"jupiter_perp_adapter.snapshot error: {e}")
        return {"enabled": True, "live": True, "error": str(e)}

    total_size = sum(p.size_usd for p in positions)
    return {
        "enabled": True,
        "live": True,
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
