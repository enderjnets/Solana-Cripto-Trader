"""
Paperclip AI Integration — Solana Cripto Trader
Tracks trades, alerts, performance reports, and AI learning in Paperclip.
All calls are non-blocking (try/except) — bot continues if Paperclip is down.
"""
import json
import logging
import os
import requests
from datetime import datetime, timezone

log = logging.getLogger("paperclip")

# ── Config ──────────────────────────────────────────────────────────
PAPERCLIP_API = "http://100.88.47.99:3100"
COMPANY_ID = "782b926b-4fb7-424a-a881-f368b0f79e3c"
PAPERCLIP_API_KEY = os.environ.get("PAPERCLIP_API_KEY", "")
TIMEOUT = 5  # seconds — don't block trading
HEADERS = {"Authorization": f"Bearer {PAPERCLIP_API_KEY}", "Content-Type": "application/json"}


# ── Low-level API ───────────────────────────────────────────────────

def _create_issue(title: str, description: str, priority: str = "medium", status: str = "todo") -> str | None:
    """Create a Paperclip issue. Returns issue ID or None."""
    try:
        r = requests.post(
            f"{PAPERCLIP_API}/api/companies/{COMPANY_ID}/issues",
            json={"title": title, "description": description, "priority": priority, "status": status},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        if r.ok:
            issue = r.json()
            identifier = f"{issue.get('identifier', '?')}"
            log.info(f"📋 Paperclip: {identifier} — {title}")
            return issue.get("id")
        else:
            log.warning(f"📋 Paperclip create failed: {r.status_code}")
    except Exception as e:
        log.debug(f"📋 Paperclip unavailable: {e}")
    return None


def _update_issue(issue_id: str, status: str = None, comment: str = None):
    """Update issue status and/or add comment."""
    if not issue_id:
        return
    try:
        if status:
            requests.patch(
                f"{PAPERCLIP_API}/api/issues/{issue_id}",
                json={"status": status},
                headers=HEADERS,
                timeout=TIMEOUT,
            )
        if comment:
            requests.post(
                f"{PAPERCLIP_API}/api/issues/{issue_id}/comments",
                json={"body": comment},
                headers=HEADERS,
                timeout=TIMEOUT,
            )
    except Exception:
        pass


def _close_issue(issue_id: str, summary: str):
    """Close issue with summary comment."""
    _update_issue(issue_id, status="done", comment=summary)


# ── Trade Events ────────────────────────────────────────────────────

def on_trade_opened(position: dict) -> str | None:
    """Create issue when a new trade is opened."""
    symbol = position.get("symbol", "?")
    direction = position.get("direction", "?").upper()
    leverage = position.get("leverage", 1)
    entry = position.get("entry_price", 0)
    margin = position.get("margin_usd", 0)
    strategy = position.get("strategy", "?")
    sl = position.get("sl_price", 0)
    tp = position.get("tp_price", 0)
    exit_mode = position.get("exit_mode", "fixed")

    title = f"[OPEN] {symbol} {direction} {leverage}x @ ${entry:.6f}"
    desc = (
        f"**{symbol} {direction}** — {strategy}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Entry | ${entry:.8f} |\n"
        f"| Margin | ${margin:.2f} |\n"
        f"| Leverage | {leverage}x |\n"
        f"| SL | ${sl:.8f} |\n"
        f"| TP | ${tp:.8f} |\n"
        f"| Exit Mode | {exit_mode} |\n"
        f"| Strategy | {strategy} |\n"
        f"| Confidence | {position.get('confidence', 0):.0%} |"
    )
    return _create_issue(title, desc, priority="medium", status="in_progress")


def on_trade_closed(position: dict):
    """Close the issue when trade is closed."""
    issue_id = position.get("paperclip_issue_id")
    if not issue_id:
        return

    symbol = position.get("symbol", "?")
    direction = position.get("direction", "?").upper()
    pnl = position.get("pnl_usd", 0)
    pnl_pct = position.get("pnl_pct", 0)
    reason = position.get("close_reason", "?")
    is_win = pnl > 0

    result = "WIN" if is_win else "LOSS"
    sign = "+" if pnl > 0 else ""

    summary = (
        f"## {result}: {symbol} {direction}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| P&L | {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%) |\n"
        f"| Close Reason | {reason} |\n"
        f"| Entry | ${position.get('entry_price', 0):.8f} |\n"
        f"| Exit | ${position.get('close_price', 0):.8f} |"
    )
    _close_issue(issue_id, summary)

    # Also create a summary issue for easy filtering
    emoji = "✅" if is_win else "❌"
    title = f"[{result}] {symbol} {direction} {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%) {reason}"
    _create_issue(title, summary, priority="low" if is_win else "medium", status="done")


# ── Alerts ──────────────────────────────────────────────────────────

def on_alert(alert_type: str, details: str, priority: str = "high") -> str | None:
    """Create alert issue for errors, drawdown, etc."""
    title = f"[ALERT] {alert_type}"
    return _create_issue(title, details, priority=priority, status="todo")


def on_error(error_msg: str, context: str = "") -> str | None:
    """Create urgent issue for system errors."""
    title = f"[ERROR] {error_msg[:80]}"
    desc = f"**Error**: {error_msg}\n\n**Context**: {context}\n\n**Time**: {datetime.now(timezone.utc).isoformat()}"
    return _create_issue(title, desc, priority="urgent", status="todo")


# ── Auto-Learner ────────────────────────────────────────────────────

def on_learner_adaptation(old_params: dict, new_params: dict, analysis: str, lessons: list) -> str | None:
    """Create issue documenting parameter changes."""
    changes = []
    for key in new_params:
        old_val = old_params.get(key)
        new_val = new_params.get(key)
        if old_val != new_val:
            changes.append(f"| {key} | {old_val} | {new_val} |")

    if not changes:
        return None

    changes_table = "| Param | Before | After |\n|---|---|---|\n" + "\n".join(changes)
    lessons_text = "\n".join(f"- {l}" for l in lessons) if lessons else "No new lessons"

    title = f"[LEARN] Parameter adaptation ({len(changes)} changes)"
    desc = (
        f"## Parameter Changes\n\n{changes_table}\n\n"
        f"## Analysis\n{analysis[:500]}\n\n"
        f"## New Lessons\n{lessons_text}"
    )
    return _create_issue(title, desc, priority="medium", status="done")


# ── Daily Report ────────────────────────────────────────────────────

def on_daily_report(stats: dict) -> str | None:
    """Create daily performance report issue."""
    capital = stats.get("capital_usd", 0)
    pnl = stats.get("total_pnl", 0)
    trades = stats.get("total_trades", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    wr = stats.get("win_rate", 0)
    dd = stats.get("drawdown_pct", 0)

    sign = "+" if pnl >= 0 else ""
    emoji = "📈" if pnl >= 0 else "📉"

    title = f"[DAILY] {emoji} {sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%) {wins}W/{losses}L WR {wr:.0f}%"
    desc = (
        f"## Daily Performance Report\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Capital | ${capital:.2f} |\n"
        f"| P&L | {sign}${pnl:.2f} |\n"
        f"| Trades | {trades} |\n"
        f"| Wins/Losses | {wins}/{losses} |\n"
        f"| Win Rate | {wr:.1f}% |\n"
        f"| Drawdown | {dd:.2f}% |\n"
        f"| Time | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |"
    )
    return _create_issue(title, desc, priority="low", status="done")


# ── Reset ───────────────────────────────────────────────────────────

def on_reset(old_capital: float, new_capital: float, reason: str = "Manual reset") -> str | None:
    """Create issue when bot is reset."""
    title = f"[RESET] ${old_capital:.2f} → ${new_capital:.2f} — {reason}"
    desc = (
        f"## Bot Reset\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Previous Capital | ${old_capital:.2f} |\n"
        f"| New Capital | ${new_capital:.2f} |\n"
        f"| Reason | {reason} |\n"
        f"| Time | {datetime.now(timezone.utc).isoformat()} |"
    )
    return _create_issue(title, desc, priority="high", status="done")
