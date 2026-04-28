"""
safety.py — centralized safety nets for live trading (v2.9.0-live, Sprint 1)

Exposes 5 fail-safe helpers:
  - is_kill_switch_active()   -> (bool, reason_str)
  - activate_kill_switch(reason), deactivate_kill_switch()
  - is_whitelisted(symbol)    -> bool       (TRADE_WHITELIST env, empty = pass-through)
  - check_daily_loss(portfolio) -> (hit: bool, todays_pnl_usd: float)
  - is_slippage_acceptable(expected, actual, side) -> (ok: bool, actual_bps: int)
  - validate_startup()        -> list[str] of fatal errors (empty = OK)

All env vars are optional. When not set, behavior falls back to current paper semantics
so this module is 100% safe to merge into master branch without side effects.

Env flags (default: all disabled / pass-through):
  MAX_DAILY_LOSS_USD     float  (0 = disabled)
  MAX_SLIPPAGE_BPS       int    (0 = disabled)
  TRADE_WHITELIST        comma-separated symbols (empty = allow all)
  LIVE_TRADING_ENABLED   true|false  (default false)

Kill switch file: /tmp/solana_live_killswitch (override with KILLSWITCH_FILE env)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
KILLSWITCH_FILE = Path(os.environ.get('KILLSWITCH_FILE', '/tmp/solana_live_killswitch'))

# ── Env config (read once at import; re-import/reload to pick up changes) ──
def _env_float(k: str, default: float) -> float:
    try: return float(os.environ.get(k, str(default)))
    except Exception: return default

def _env_int(k: str, default: int) -> int:
    try: return int(os.environ.get(k, str(default)))
    except Exception: return default

MAX_DAILY_LOSS_USD   = _env_float('MAX_DAILY_LOSS_USD', 0.0)
MAX_SLIPPAGE_BPS     = _env_int('MAX_SLIPPAGE_BPS', 0)
TRADE_WHITELIST      = frozenset(
    s.strip().upper()
    for s in os.environ.get('TRADE_WHITELIST', '').split(',')
    if s.strip()
)
LIVE_TRADING_ENABLED = os.environ.get('LIVE_TRADING_ENABLED', 'false').lower() == 'true'


# ══════════════════════════════════════════════════════════════════
# Kill switch — file-based, checked every cycle
# ══════════════════════════════════════════════════════════════════
def is_kill_switch_active() -> tuple[bool, str]:
    """True if killswitch file exists. Returns (active, reason)."""
    try:
        if KILLSWITCH_FILE.exists():
            reason = KILLSWITCH_FILE.read_text(errors='ignore').strip()[:200]
            return True, reason or 'no reason given'
    except Exception:
        pass
    return False, ''


def activate_kill_switch(reason: str = 'manual') -> None:
    try:
        KILLSWITCH_FILE.write_text(
            f'{datetime.now(timezone.utc).isoformat()} :: {reason}'
        )
    except Exception:
        pass  # Fail-safe; en peor caso el proximo ciclo sigue


def deactivate_kill_switch() -> None:
    try: KILLSWITCH_FILE.unlink(missing_ok=True)
    except Exception: pass


# ══════════════════════════════════════════════════════════════════
# Token whitelist — empty whitelist = allow all (paper behavior)
# ══════════════════════════════════════════════════════════════════
def is_whitelisted(symbol: str) -> bool:
    if not TRADE_WHITELIST:
        return True
    return (symbol or '').upper() in TRADE_WHITELIST


# ══════════════════════════════════════════════════════════════════
# Daily loss limit — only engages if MAX_DAILY_LOSS_USD > 0
# ══════════════════════════════════════════════════════════════════
def check_daily_loss(portfolio: dict | None = None) -> tuple[bool, float]:
    """
    Returns (limit_hit, todays_pnl_usd). Uses trade_history.json for realized
    P&L today. If MAX_DAILY_LOSS_USD <= 0, limit is disabled -> always False.
    """
    if MAX_DAILY_LOSS_USD <= 0:
        return False, 0.0
    try:
        trade_history_file = DATA / 'trade_history.json'
        if not trade_history_file.exists():
            return False, 0.0
        raw = json.loads(trade_history_file.read_text())
        trades = raw if isinstance(raw, list) else (raw.get('trades', []) or [])
        today = datetime.now(timezone.utc).date().isoformat()
        todays_pnl = sum(
            float(t.get('pnl_usd', 0) or 0)
            for t in trades
            if (t.get('close_time') or t.get('open_time') or '').startswith(today)
        )
        # Incluir P&L no realizado de posiciones abiertas (opcional, mas conservador)
        if portfolio and isinstance(portfolio, dict):
            for p in portfolio.get('positions', []):
                if p.get('status') == 'open':
                    todays_pnl += float(p.get('unrealized_pnl_usd', 0) or p.get('pnl_usd', 0) or 0)
        if todays_pnl < 0 and abs(todays_pnl) >= MAX_DAILY_LOSS_USD:
            return True, todays_pnl
        return False, todays_pnl
    except Exception:
        return False, 0.0


# ══════════════════════════════════════════════════════════════════
# Slippage check — for live swaps (disabled in paper)
# ══════════════════════════════════════════════════════════════════
def is_slippage_acceptable(expected_price: float, actual_price: float,
                           side: str = 'buy') -> tuple[bool, int]:
    """
    Returns (ok, actual_bps).
    Buy/long: adverse if actual > expected (pagaste mas).
    Sell/short: adverse if actual < expected (recibiste menos).
    """
    if MAX_SLIPPAGE_BPS <= 0 or expected_price <= 0:
        return True, 0
    if side.lower() in ('buy', 'long'):
        diff = actual_price - expected_price
    else:
        diff = expected_price - actual_price
    bps = int(max(0, diff) / expected_price * 10000)
    return bps <= MAX_SLIPPAGE_BPS, bps


# ══════════════════════════════════════════════════════════════════
# Startup validation — called once at orchestrator boot
# ══════════════════════════════════════════════════════════════════
def validate_startup() -> list:
    """
    Returns list of fatal errors. Empty list = OK to proceed.
    Si LIVE_TRADING_ENABLED=false, solo chequea que el kill switch no este activo.
    """
    errors = []
    # Kill switch no debe estar activo al arrancar
    active, reason = is_kill_switch_active()
    if active:
        errors.append(
            f'Kill switch activo ({reason}). Eliminar {KILLSWITCH_FILE} para arrancar.'
        )
    # Validacion adicional solo si live trading esta habilitado
    if LIVE_TRADING_ENABLED:
        if not os.environ.get('HOT_WALLET_PRIVATE_KEY'):
            errors.append('LIVE_TRADING_ENABLED=true pero HOT_WALLET_PRIVATE_KEY ausente')
        if not os.environ.get('SOLANA_RPC_URL'):
            errors.append('LIVE_TRADING_ENABLED=true pero SOLANA_RPC_URL ausente')
        if MAX_DAILY_LOSS_USD <= 0:
            errors.append('LIVE_TRADING_ENABLED=true requiere MAX_DAILY_LOSS_USD > 0')
        if not TRADE_WHITELIST:
            errors.append('LIVE_TRADING_ENABLED=true requiere TRADE_WHITELIST no vacia')
        if MAX_SLIPPAGE_BPS <= 0:
            errors.append('LIVE_TRADING_ENABLED=true requiere MAX_SLIPPAGE_BPS > 0 (ej: 100 = 1%%)')
    return errors


# ══════════════════════════════════════════════════════════════════
# Status snapshot — para dashboard/API
# ══════════════════════════════════════════════════════════════════
def status_snapshot() -> dict:
    active, reason = is_kill_switch_active()
    return {
        'kill_switch_active': active,
        'kill_switch_reason': reason,
        'max_daily_loss_usd': MAX_DAILY_LOSS_USD,
        'max_slippage_bps': MAX_SLIPPAGE_BPS,
        'trade_whitelist': sorted(TRADE_WHITELIST),
        'live_trading_enabled': LIVE_TRADING_ENABLED,
        'ts': datetime.now(timezone.utc).isoformat(),
    }
