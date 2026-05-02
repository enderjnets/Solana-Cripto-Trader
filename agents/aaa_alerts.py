#!/usr/bin/env python3
"""
AAA Alerts — Notificaciones Telegram para eventos criticos del sistema AAA.
Rate limiting por tipo para evitar spam.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("aaa_alerts")

# -- Paths ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
COOLDOWN_FILE = BASE_DIR / "aaa_data" / "alert_cooldowns.json"

# -- Telegram sender (lazy import to avoid circular deps) -------------------

def _send_telegram(message: str) -> bool:
    try:
        from reporter import send_telegram
        return send_telegram(message)
    except Exception as e:
        log.debug(f"Telegram send failed: {e}")
        return False


# -- Cooldown manager -------------------------------------------------------

def _load_cooldowns() -> Dict[str, str]:
    try:
        if COOLDOWN_FILE.exists():
            return json.loads(COOLDOWN_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_cooldowns(cooldowns: Dict[str, str]) -> None:
    try:
        COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        COOLDOWN_FILE.write_text(json.dumps(cooldowns, indent=2))
    except Exception as e:
        log.debug(f"Cooldown save failed: {e}")


def _is_on_cooldown(alert_key: str, minutes: int) -> bool:
    cooldowns = _load_cooldowns()
    last = cooldowns.get(alert_key)
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
        return datetime.now(timezone.utc) < last_dt + timedelta(minutes=minutes)
    except Exception:
        return False


def _record_cooldown(alert_key: str) -> None:
    cooldowns = _load_cooldowns()
    cooldowns[alert_key] = datetime.now(timezone.utc).isoformat()
    _save_cooldowns(cooldowns)


# -- Alert functions --------------------------------------------------------

def alert_first_trade(agent: str, symbol: str, direction: str, pnl: float = 0.0) -> None:
    key = f"first_trade_{agent}"
    if _is_on_cooldown(key, 999999):  # Only once per agent
        return
    msg = f"🎯 <b>Primer trade {agent}</b>\n{symbol.upper()} {direction.upper()}\nPnL: ${pnl:+.2f}"
    if _send_telegram(msg):
        _record_cooldown(key)
        log.info(f"Alert sent: first trade {agent} {symbol}")


def alert_gate_advance(old_phase: int, new_phase: int, leader: str) -> None:
    key = f"gate_{new_phase}"
    if _is_on_cooldown(key, 999999):  # Only once per phase
        return
    msg = f"🧬 <b>Gate avanza {old_phase} → {new_phase}</b>\nLider: {leader}\nFase: {new_phase}/5"
    if _send_telegram(msg):
        _record_cooldown(key)
        log.info(f"Alert sent: gate advance {old_phase}→{new_phase}")


def alert_evolution_applied(agent: str, changes: Dict, confidence: float) -> None:
    key = f"evolution_{agent}"
    if _is_on_cooldown(key, 60):  # Max 1 per hour
        return
    changes_str = ", ".join(f"{k}={v}" for k, v in changes.items())
    msg = f"🔄 <b>Self-Evolution {agent}</b>\nConfianza: {confidence:.0%}\nCambios: {changes_str}"
    if _send_telegram(msg):
        _record_cooldown(key)
        log.info(f"Alert sent: evolution applied {agent}")


def alert_rollback(agent: str, reason: str, sharpe: float) -> None:
    key = f"rollback_{agent}"
    if _is_on_cooldown(key, 60):  # Max 1 per hour
        return
    msg = f"🔄 <b>Rollback {agent}</b>\nRazon: {reason}\nSharpe: {sharpe:.2f}"
    if _send_telegram(msg):
        _record_cooldown(key)
        log.info(f"Alert sent: rollback {agent}")


def alert_ab_test_result(agent: str, variant: str, result: str, improvement: float = 0.0) -> None:
    key = f"abtest_{agent}_{variant}"
    if _is_on_cooldown(key, 999999):  # Only once per test
        return
    emoji = "✅" if result == "promoted" else "❌"
    msg = f"{emoji} <b>A/B Test {agent}</b>\nVariante: {variant}\nResultado: {result}\nMejora: {improvement:.2f}x"
    if _send_telegram(msg):
        _record_cooldown(key)
        log.info(f"Alert sent: A/B test {result} {variant}")


def alert_drawdown(agent: str, dd_pct: float) -> None:
    key = f"drawdown_{agent}"
    if _is_on_cooldown(key, 30):  # Max 1 per 30 min
        return
    msg = f"⚠️ <b>Drawdown {agent}</b>\nDD: {dd_pct:.1f}%"
    if _send_telegram(msg):
        _record_cooldown(key)
        log.info(f"Alert sent: drawdown {agent} {dd_pct:.1f}%")
