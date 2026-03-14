#!/usr/bin/env python3
"""
📊 DAILY REPORTER - Envía resumen diario del trading al final del día

ARCHIVO: agents/daily_reporter.py
"""

from pathlib import Path
import json
from datetime import datetime, timezone
import subprocess
import sys

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent))

from daily_profit_target import DailyProfitState, PortfolioEvaluator

STATE_FILE = Path.home() / ".config" / "solana-jupiter-bot" / "master_state.json"


def send_telegram_message(message: str):
    """Envía un mensaje por Telegram usando la CLI de OpenClaw"""
    try:
        result = subprocess.run(
            ["openclaw", "message", "send", "telegram", "--to", "771213858", "--message", message],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error enviando a Telegram: {e}")
        return False


def check_and_send_daily_report():
    """Verifica si es final del día y envía el reporte si es necesario"""
    state = DailyProfitState()
    evaluator = PortfolioEvaluator()

    # Cargar portfolio
    if not STATE_FILE.exists():
        print(f"No se encontró master_state.json")
        return False

    with open(STATE_FILE) as f:
        portfolio = json.load(f)

    # Verificar si hay trades hoy
    trades_today = state.trades_today
    positions_closed = state.positions_closed_today

    # Solo enviar reporte si hubo actividad
    if trades_today == 0 and positions_closed == 0:
        print(f"No hubo actividad hoy. No se envía reporte.")
        return False

    # Verificar si ya se envió el reporte de hoy
    last_report = state.last_decision.get("timestamp", "")
    if last_report and last_report.startswith(datetime.now(timezone.utc).date().isoformat()):
        if "DAILY_REPORT_SENT" in str(state.last_decision):
            print(f"Reporte ya enviado hoy. No se reenvía.")
            return False

    # Generar resumen
    summary = state.get_daily_summary(portfolio)

    # Marcar reporte enviado
    state.record_decision(
        "DAILY_REPORT_SENT",
        f"Reporte diario enviado con {summary['trades_ejecutados']} trades (Profit NETO: ${summary['profit_net_usd']:.2f})"
    )

    # Generar texto del reporte
    report_text = state.send_daily_report(summary)

    # Enviar a Telegram
    print(f"Enviando reporte diario a Telegram...")
    print(f"\n{report_text}\n")

    success = send_telegram_message(report_text)

    if success:
        print(f"✅ Reporte enviado exitosamente")
        return True
    else:
        print(f"❌ Error enviando reporte")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("📊 DAILY REPORTER - Verificando si enviar reporte diario")
    print("=" * 60)

    success = check_and_send_daily_report()

    if success:
        print("\n✅ Reporte enviado exitosamente")
    else:
        print("\nℹ️ No se envió reporte (sin actividad o ya enviado)")

    print("=" * 60)
