"""agents/telegram_signals.py — broadcast trade signals to Telegram channel.

v2.12.22 — Phase 2 prep. DISABLED by default via TELEGRAM_ENABLED=false.

When enabled (Phase 2):
    1. Create bot via @BotFather on Telegram → TELEGRAM_SIGNALS_BOT_TOKEN
    2. Create public channel, add bot as admin → TELEGRAM_SIGNALS_CHANNEL (e.g. @SolanaCriptoTrader_Signals)
    3. Set TELEGRAM_ENABLED=true in .env
    4. Restart watchdog — hooks activate automatically

Same hook signatures as agents/paperclip_client.py so both broadcast in parallel.
All errors silently swallowed — never raises so trading is never blocked.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("telegram_signals")

# ── Config ─────────────────────────────────────────────────────────

TELEGRAM_ENABLED = os.environ.get("TELEGRAM_ENABLED", "false").lower() == "true"
BOT_TOKEN = os.environ.get("TELEGRAM_SIGNALS_BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("TELEGRAM_SIGNALS_CHANNEL", "")
TIMEOUT = 5  # seconds — don't block trading

# Warn once on import if enabled but missing config
_WARNED = False


def _ready() -> bool:
    global _WARNED
    if not TELEGRAM_ENABLED:
        return False
    if not BOT_TOKEN or not CHANNEL_ID:
        if not _WARNED:
            log.warning(
                "TELEGRAM_ENABLED=true but TELEGRAM_SIGNALS_BOT_TOKEN or "
                "TELEGRAM_SIGNALS_CHANNEL missing — skipping sends"
            )
            _WARNED = True
        return False
    return True


def _send(text: str, parse_mode: str = "Markdown") -> bool:
    if not _ready():
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHANNEL_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=TIMEOUT,
        )
        if not r.ok:
            log.debug(f"telegram send failed: {r.status_code} {r.text[:150]}")
        return r.ok
    except Exception as e:
        log.debug(f"telegram send exception: {e}")
        return False


# ── Trade Event Hooks (same signatures as paperclip_client) ─────────

def on_trade_opened(position: dict) -> None:
    """Called post-open. Silently no-op if telegram disabled."""
    if not _ready():
        return
    symbol = position.get("symbol", "?")
    direction = str(position.get("direction", "long")).upper()
    entry = float(position.get("entry_price", 0) or 0)
    sl = float(position.get("sl_price", 0) or 0)
    tp = float(position.get("tp_price", 0) or 0)
    conf = float(position.get("confidence", 0) or 0)
    leverage = position.get("leverage", 1)
    size = float(position.get("margin_usd", position.get("size_usd", 0)) or 0)
    strategy = str(position.get("strategy", "")).replace("_", " ")
    arrow = "🟢" if direction == "LONG" else "🔴"
    text = (
        f"{arrow} *{direction} {symbol}*\n"
        f"Entry: `${entry:.4f}`  Size: `${size:.2f}` ({leverage}x)\n"
        f"SL: `${sl:.4f}`  TP: `${tp:.4f}`\n"
        f"LLM conf: `{conf:.0%}`"
    )
    if strategy:
        text += f"\n_{strategy}_"
    _send(text)


def on_trade_closed(position: dict) -> None:
    """Called post-close. Silently no-op if telegram disabled."""
    if not _ready():
        return
    symbol = position.get("symbol", "?")
    pnl_usd = float(position.get("pnl_usd", 0) or 0)
    pnl_pct = float(position.get("pnl_pct", 0) or 0)
    reason = position.get("close_reason", "?")
    if pnl_usd > 0:
        emoji = "🟢"
    elif pnl_usd < 0:
        emoji = "🔴"
    else:
        emoji = "⚪"
    text = (
        f"{emoji} *CLOSED {symbol}*\n"
        f"PnL: `${pnl_usd:+.4f}` ({pnl_pct:+.3f}%)\n"
        f"Reason: `{reason}`"
    )
    _send(text)


def on_daily_report(metrics: dict) -> None:
    """Optional daily summary. Silently no-op if telegram disabled."""
    if not _ready():
        return
    text = (
        f"📊 *Daily Report*\n"
        f"Trades: {metrics.get('total_trades', 0)}  "
        f"WR: {float(metrics.get('win_rate', 0) or 0):.1f}%\n"
        f"Capital: `${float(metrics.get('capital_usd', 0) or 0):.2f}`  "
        f"PnL: `${float(metrics.get('total_pnl', 0) or 0):+.4f}`"
    )
    _send(text)
