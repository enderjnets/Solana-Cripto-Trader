#!/usr/bin/env python3
"""
📊 Agente 5: Reporter
Genera reportes de portfolio, métricas de trading y alertas vía Telegram.
Reporte diario a las 8AM MT. Alertas inmediatas para eventos importantes.

Uso:
    python3 reporter.py           # Reporte del ciclo actual
    python3 reporter.py --daily   # Fuerza reporte diario completo
    python3 reporter.py --alert   # Solo alertas pendientes
"""

import os
import sys
import json
import logging
import argparse
import requests
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

# Paperclip integration
try:
    from agents.paperclip_client import on_daily_report
    _PAPERCLIP = True
except ImportError:
    _PAPERCLIP = False

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE  = DATA_DIR / "portfolio.json"
HISTORY_FILE    = DATA_DIR / "trade_history.json"
MARKET_FILE     = DATA_DIR / "market_latest.json"
REPORT_FILE     = DATA_DIR / "daily_report.json"
ALERTS_STATE    = DATA_DIR / "alerts_state.json"

# .env del proyecto
ENV_FILE = Path(__file__).parent.parent / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("reporter")

# ─── Config desde .env ───────────────────────────────────────────────────────

def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


ENV = load_env()
TELEGRAM_TOKEN = ENV.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = ENV.get("TELEGRAM_CHAT_ID", "771213858")
MINIMAX_KEY    = None  # Cargado bajo demanda

# ─── Carga de Datos ───────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"capital_usd": 1000.0, "initial_capital": 1000.0, "positions": [],
            "total_trades": 0, "wins": 0, "losses": 0, "status": "ACTIVE"}


def load_history() -> list:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def load_market() -> dict:
    if MARKET_FILE.exists():
        with open(MARKET_FILE) as f:
            return json.load(f)
    return {}


def load_alerts_state() -> dict:
    if ALERTS_STATE.exists():
        with open(ALERTS_STATE) as f:
            return json.load(f)
    return {"last_daily_report": "", "alerted_trades": []}


def save_alerts_state(state: dict):
    with open(ALERTS_STATE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Métricas de Trading ─────────────────────────────────────────────────────

def calculate_metrics(portfolio: dict, history: list) -> dict:
    """Calcula métricas completas del sistema de trading."""
    total_trades = portfolio.get("total_trades", 0)
    wins = portfolio.get("wins", 0)
    losses = portfolio.get("losses", 0)
    capital = portfolio.get("capital_usd", 1000.0)
    initial = portfolio.get("initial_capital", 1000.0)

    # Win Rate
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    # P&L total realizado — todos los trades en history son cerrados (no tienen campo "status")
    closed_trades = [t for t in history if "pnl_usd" in t]
    total_pnl = sum(t.get("pnl_usd", 0) for t in closed_trades)

    # P&L no realizado + valor de posiciones abiertas
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    unrealized_pnl = sum(p.get("pnl_usd", 0) for p in open_positions)
    # Corrección: usar margin_usd (lo que el trader aportó), NO size_usd/notional (que es 3x el margen)
    # size_usd = notional_value (margen × leverage), margin_usd = capital real inmovilizado
    invested_in_positions = sum(p.get("margin_usd", p.get("size_usd", 0)) for p in open_positions)

    # Avg win / avg loss
    winning_trades = [t for t in closed_trades if t.get("pnl_usd", 0) > 0]
    losing_trades  = [t for t in closed_trades if t.get("pnl_usd", 0) <= 0]

    avg_win  = (sum(t["pnl_usd"] for t in winning_trades) / len(winning_trades)) if winning_trades else 0
    avg_loss = (sum(abs(t["pnl_usd"]) for t in losing_trades) / len(losing_trades)) if losing_trades else 0

    # Profit Factor
    gross_profit = sum(t["pnl_usd"] for t in winning_trades)
    gross_loss   = sum(abs(t["pnl_usd"]) for t in losing_trades)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)

    # Equity total = capital libre + invertido en posiciones + P&L no realizado
    total_value = capital + invested_in_positions + unrealized_pnl
    drawdown = max(0.0, (initial - total_value) / initial * 100) if initial > 0 else 0.0

    # Sharpe básico (si hay trades suficientes)
    sharpe = 0.0
    if len(closed_trades) >= 5:
        returns = [t.get("pnl_pct", 0) / 100 for t in closed_trades]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return)**2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_return / std_return * (365**0.5)) if std_return > 0 else 0.0

    return {
        "capital_usd": round(capital, 2),
        "initial_capital": round(initial, 2),
        "total_pnl": round(total_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_value": round(total_value, 2),
        "return_pct": round((total_value - initial) / initial * 100, 2) if initial > 0 else 0,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "drawdown_pct": round(drawdown, 2),
        "open_positions": len(open_positions),
    }


# ─── Telegram ────────────────────────────────────────────────────────────────

ALERTS_FILE = DATA_DIR / "pending_alerts.json"

def queue_alert(message: str):
    """Guarda alerta en archivo para que Eko la envíe via OpenClaw."""
    alerts = []
    if ALERTS_FILE.exists():
        try:
            raw = json.loads(ALERTS_FILE.read_text())
            # BUG FIX: pending_alerts.json may be a dict {"alerts": [...]} instead of a list
            if isinstance(raw, list):
                alerts = raw
            elif isinstance(raw, dict):
                alerts = raw.get("alerts", [])
        except Exception:
            alerts = []
    alerts.append({"ts": datetime.now(ZoneInfo("America/Denver")).isoformat(), "msg": message})
    # Keep last 20 alerts
    ALERTS_FILE.write_text(json.dumps(alerts[-20:], indent=2))

def send_telegram(message: str) -> bool:
    """Envía mensaje a Telegram via OpenClaw gateway (más confiable que bot directo)."""
    try:
        # Primero intentar via OpenClaw gateway
        gw_url = "http://127.0.0.1:18789/v1/message"
        gw_token = "d2fe31da8a0060bcc7f525bed03e2d"
        resp = requests.post(gw_url, json={
            "channel": "telegram",
            "to": "771213858",
            "message": message,
        }, headers={"Authorization": f"Bearer {gw_token}"}, timeout=10)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    # Fallback: guardar en cola para que Eko envíe
    queue_alert(message)
    log.info("📥 Alerta encolada para envío via Eko")
    return True


def send_telegram_voice(text: str) -> bool:
    """
    Genera nota de voz via MiniMax TTS y la envía a Telegram.
    Usa el mismo sistema que BitTrader.
    """
    try:
        # Cargar MiniMax API key
        minimax_key_file = Path("/home/enderj/.openclaw/workspace/bittrader/keys/minimax.json")
        if not minimax_key_file.exists():
            log.warning("⚠️  minimax.json no encontrado, skip TTS")
            return False

        with open(minimax_key_file) as f:
            minimax_cfg = json.load(f)
        api_key = minimax_cfg.get("minimax_api_key", "")

        if not api_key:
            return False

        # MiniMax TTS API
        tts_url = "https://api.minimax.io/v1/t2a_v2"
        payload = {
            "model": "speech-02-hd",
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": "Wisdom_of_a_Sage",
                "speed": 1.0,
                "vol": 1.0,
                "pitch": 0
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            }
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(tts_url, json=payload, headers=headers, timeout=30)

        if resp.status_code != 200:
            log.warning(f"⚠️  MiniMax TTS error: {resp.status_code}")
            return False

        data = resp.json()
        audio_url = data.get("data", {}).get("audio", "")

        if not audio_url:
            # Puede venir como bytes
            audio_bytes = resp.content
        else:
            # Descargar audio
            audio_resp = requests.get(audio_url, timeout=15)
            audio_bytes = audio_resp.content

        # Enviar a Telegram como voice note
        if not TELEGRAM_TOKEN:
            return False

        tg_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVoice"
        files = {"voice": ("report.mp3", audio_bytes, "audio/mpeg")}
        data = {"chat_id": TELEGRAM_CHAT}
        tg_resp = requests.post(tg_url, data=data, files=files, timeout=20)
        tg_resp.raise_for_status()
        return True

    except Exception as e:
        log.error(f"❌ TTS error: {e}")
        return False


# ─── Generación de Reportes ──────────────────────────────────────────────────

def format_cycle_report(metrics: dict, portfolio: dict, market: dict) -> str:
    """Reporte corto de ciclo (cada 60s, no se envía a Telegram salvo alertas)."""
    status = portfolio.get("status", "ACTIVE")
    status_emoji = "✅" if status == "ACTIVE" else "⛔"
    open_pos = portfolio.get("positions", [])
    open_pos = [p for p in open_pos if p.get("status") == "open"]

    lines = [
        f"📊 <b>Cycle Report</b> — {datetime.now(ZoneInfo('America/Denver')).strftime('%H:%M:%S')}",
        f"{status_emoji} Estado: {status}",
        f"💰 Capital: ${metrics['capital_usd']:.2f} (inicial: ${metrics['initial_capital']:.2f})",
        f"📈 P&L realizado: ${metrics['total_pnl']:+.2f} ({metrics['return_pct']:+.2f}%)",
        f"🔮 P&L no realizado: ${metrics['unrealized_pnl']:+.2f}",
        f"📉 Drawdown: {metrics['drawdown_pct']:.2f}%",
        f"🏆 Win rate: {metrics['win_rate']:.1f}% ({metrics['wins']}W/{metrics['losses']}L)",
        f"📊 Posiciones abiertas: {metrics['open_positions']}/5",
    ]

    if open_pos:
        lines.append("\n<b>Posiciones activas:</b>")
        for p in open_pos:
            arrow = "🟢" if p["direction"] == "long" else "🔴"
            pnl = p.get("pnl_usd", 0)
            pnl_pct = p.get("pnl_pct", 0)
            lines.append(f"  {arrow} {p['symbol']}: ${p['size_usd']:.0f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

    return "\n".join(lines)


def format_daily_report(metrics: dict, portfolio: dict, history: list) -> str:
    """Reporte diario completo para 8AM MT."""
    now = datetime.now(ZoneInfo("America/Denver"))
    lines = [
        f"🌅 <b>Reporte Diario — Solana Bot</b>",
        f"📅 {now.strftime('%A %d %B %Y, %H:%M')} MT",
        "",
        f"💰 <b>Capital:</b> ${metrics['capital_usd']:.2f}",
        f"📊 <b>Valor total:</b> ${metrics['total_value']:.2f}",
        f"📈 <b>Retorno:</b> {metrics['return_pct']:+.2f}%",
        f"📉 <b>Drawdown:</b> {metrics['drawdown_pct']:.2f}%",
        "",
        f"🏆 <b>Win Rate:</b> {metrics['win_rate']:.1f}%",
        f"📊 <b>Trades totales:</b> {metrics['total_trades']} ({metrics['wins']}W / {metrics['losses']}L)",
        f"💚 <b>Avg win:</b> ${metrics['avg_win_usd']:.2f}",
        f"❤️ <b>Avg loss:</b> ${metrics['avg_loss_usd']:.2f}",
        f"⚡ <b>Profit factor:</b> {metrics['profit_factor']:.2f}",
        f"📐 <b>Sharpe ratio:</b> {metrics['sharpe_ratio']:.2f}",
        "",
        f"📊 <b>Posiciones abiertas:</b> {metrics['open_positions']}",
    ]

    # Últimos 5 trades
    recent = sorted(history, key=lambda x: x.get("close_time", ""), reverse=True)[:5]
    if recent:
        lines.append("\n<b>Últimos trades:</b>")
        for t in recent:
            arrow = "✅" if t.get("pnl_usd", 0) > 0 else "❌"
            lines.append(f"  {arrow} {t['symbol']} [{t['strategy']}] ${t.get('pnl_usd', 0):+.2f} via {t.get('close_reason','?')}")

    return "\n".join(lines)


def format_tts_daily(metrics: dict) -> str:
    """Texto para TTS del reporte diario."""
    return (
        f"Buenos días Ender. Reporte del bot de Solana. "
        f"Capital actual: {metrics['capital_usd']:.0f} dólares. "
        f"Retorno: {metrics['return_pct']:+.1f} por ciento. "
        f"Win rate: {metrics['win_rate']:.0f} por ciento con {metrics['total_trades']} trades totales. "
        f"Drawdown actual: {metrics['drawdown_pct']:.1f} por ciento. "
        f"Posiciones abiertas: {metrics['open_positions']}. "
        f"Profit factor: {metrics['profit_factor']:.2f}. "
        f"Que tengas buen trading hoy."
    )


# ─── Alertas ─────────────────────────────────────────────────────────────────

def check_and_send_alerts(portfolio: dict, history: list, metrics: dict,
                          alerts_state: dict) -> list:
    """Envía alertas inmediatas para eventos importantes."""
    alerts_sent = []
    alerted_ids = set(alerts_state.get("alerted_trades", []))

    # 1. Trades cerrados (win o loss)
    recent_closed = [t for t in history
                     if t.get("status") == "closed"
                     and t.get("id") not in alerted_ids]

    for trade in recent_closed[-3:]:  # Max 3 alertas por ciclo
        pnl = trade.get("pnl_usd", 0)
        is_win = pnl > 0
        emoji = "✅ WIN" if is_win else "❌ LOSS"
        reason = trade.get("close_reason", "?")
        msg = (
            f"{emoji} — <b>{trade['symbol']}</b>\n"
            f"Estrategia: {trade.get('strategy', '?')}\n"
            f"P&L: ${pnl:+.2f} ({trade.get('pnl_pct', 0):+.2f}%)\n"
            f"Cierre: {reason} @ ${trade.get('close_price', 0):.6f}"
        )
        if send_telegram(msg):
            alerted_ids.add(trade["id"])
            alerts_sent.append(f"{trade['symbol']} {reason}")

    # 2. Drawdown > 5%
    if metrics["drawdown_pct"] >= 5.0:
        dd_key = f"dd_{int(metrics['drawdown_pct'])}"
        if dd_key not in alerted_ids:
            msg = (
                f"⚠️ <b>ALERTA DRAWDOWN</b>\n"
                f"Drawdown actual: {metrics['drawdown_pct']:.1f}%\n"
                f"Capital: ${metrics['capital_usd']:.2f}\n"
                f"Máximo permitido: 10%"
            )
            if send_telegram(msg):
                alerted_ids.add(dd_key)
                alerts_sent.append(f"drawdown_{metrics['drawdown_pct']:.1f}%")

    # 3. Sistema PAUSED o STOPPED
    status = portfolio.get("status", "ACTIVE")
    if status in ("PAUSED", "STOPPED"):
        status_key = f"status_{status}"
        if status_key not in alerted_ids:
            msg = (
                f"🚨 <b>SISTEMA {status}</b>\n"
                f"Drawdown: {metrics['drawdown_pct']:.1f}%\n"
                f"Capital: ${metrics['capital_usd']:.2f}\n"
                f"Acción requerida manualmente."
            )
            if send_telegram(msg):
                alerted_ids.add(status_key)
                alerts_sent.append(f"sistema_{status}")

    # Actualizar estado de alertas
    alerts_state["alerted_trades"] = list(alerted_ids)[-200:]  # Max 200 IDs
    save_alerts_state(alerts_state)

    return alerts_sent


def should_send_daily_report(alerts_state: dict) -> bool:
    """Verifica si es hora del reporte diario (8AM MT)."""
    now_mt = datetime.now(ZoneInfo("America/Denver"))
    last_daily = alerts_state.get("last_daily_report", "")

    # Solo entre 7:50 y 8:10 AM
    if not (7 <= now_mt.hour <= 8 and (now_mt.hour != 8 or now_mt.minute <= 10)):
        return False

    # No enviar dos veces el mismo día
    today = now_mt.strftime("%Y-%m-%d")
    if last_daily == today:
        return False

    return True


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(daily: bool = False, alert_only: bool = False) -> dict:
    log.info("=" * 50)
    log.info("📊 REPORTER — iniciando")
    log.info("=" * 50)

    portfolio = load_portfolio()
    history   = load_history()
    market    = load_market()
    alerts_state = load_alerts_state()

    metrics = calculate_metrics(portfolio, history)

    # Log métricas al console siempre
    log.info(f"💰 Equity: ${metrics['total_value']:.2f} (libre: ${metrics['capital_usd']:.2f}) | Retorno: {metrics['return_pct']:+.2f}%")
    log.info(f"🏆 Win Rate: {metrics['win_rate']:.1f}% ({metrics['wins']}W/{metrics['losses']}L de {metrics['total_trades']} trades)")
    log.info(f"📉 Drawdown: {metrics['drawdown_pct']:.2f}% | Posiciones: {metrics['open_positions']}")
    log.info(f"⚡ Profit Factor: {metrics['profit_factor']:.2f} | Sharpe: {metrics['sharpe_ratio']:.2f}")

    alerts_sent = []
    telegram_ok = bool(TELEGRAM_TOKEN)

    if not alert_only:
        # Verificar si hay que enviar reporte diario
        if daily or should_send_daily_report(alerts_state):
            log.info("🌅 Enviando reporte diario...")
            report_text = format_daily_report(metrics, portfolio, history)
            if telegram_ok:
                send_telegram(report_text)
                # TTS de nota de voz
                tts_text = format_tts_daily(metrics)
                if send_telegram_voice(tts_text):
                    log.info("🎤 Nota de voz enviada")
                else:
                    log.warning("⚠️  TTS falló, solo texto enviado")
            alerts_state["last_daily_report"] = datetime.now(ZoneInfo("America/Denver")).strftime("%Y-%m-%d")
            save_alerts_state(alerts_state)

            # Paperclip: crear issue de reporte diario
            if _PAPERCLIP:
                try:
                    on_daily_report(metrics)
                except Exception:
                    pass

    # Alertas inmediatas
    alerts_sent = check_and_send_alerts(portfolio, history, metrics, alerts_state)
    if alerts_sent:
        log.info(f"🔔 Alertas enviadas: {', '.join(alerts_sent)}")
    else:
        log.info("🔕 Sin alertas nuevas")

    # Guardar reporte
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "alerts_sent": alerts_sent,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"💾 Guardado en {REPORT_FILE}")

    # FIX: Actualizar equity_history.json con snapshot actual (estaba sin actualizar desde Apr 3)
    EQUITY_HISTORY_FILE = DATA_DIR / "equity_history.json"
    try:
        eq_data = {"equity": [], "dates": []}
        if EQUITY_HISTORY_FILE.exists():
            with open(EQUITY_HISTORY_FILE) as f:
                eq_data = json.load(f)
        eq_data["equity"].append(round(metrics.get("total_value", metrics.get("capital_usd", 0)), 2))
        eq_data["dates"].append(datetime.now(timezone.utc).isoformat())
        # Retener solo últimos 500 puntos
        if len(eq_data["equity"]) > 500:
            eq_data["equity"] = eq_data["equity"][-500:]
            eq_data["dates"] = eq_data["dates"][-500:]
        with open(EQUITY_HISTORY_FILE, "w") as f:
            json.dump(eq_data, f, indent=2)
    except Exception as e:
        log.warning(f"equity_history update failed: {e}")

    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reporter Agent")
    parser.add_argument("--daily", action="store_true", help="Forzar reporte diario")
    parser.add_argument("--alert", action="store_true", help="Solo verificar alertas")
    args = parser.parse_args()

    run(daily=args.daily, alert_only=args.alert)
    sys.exit(0)
