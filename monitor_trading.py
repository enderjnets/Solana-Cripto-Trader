#!/usr/bin/env python3
"""
Trading Bot Monitor - Genera resúmenes de la operativa
Uso: python3 monitor_trading.py [--notify]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path("/home/enderj/.openclaw/workspace/solana-jupiter-bot")

def load_state():
    """Cargar estado actual del trading"""
    try:
        with open(PROJECT_DIR / "data" / "paper_trading_state.json") as f:
            state = json.load(f)
        return state
    except:
        return None

def load_trades():
    """Cargar historial de trades"""
    # Los trades están en paper_trading_state.json
    try:
        with open(PROJECT_DIR / "data" / "paper_trading_state.json") as f:
            state = json.load(f)
        return state.get("trades", [])
    except:
        return []

def calculate_portfolio_value(state):
    """Calcular valor actual del portfolio"""
    if not state:
        return 0

    balance = state.get("balance_usd", 0)
    open_trades = state.get("trades", [])

    # Valor actual de posiciones abiertas
    open_value = sum(t.get("size", 0) for t in open_trades)

    total_value = balance + open_value
    return total_value

def generate_summary():
    """Generar resumen completo"""
    state = load_state()

    if not state:
        return "❌ No se pudo cargar el estado del trading"

    # Métricas básicas
    balance = state.get("balance_usd", 0)
    stats = state.get("stats", {})
    open_trades = state.get("trades", [])
    total_trades = stats.get("total_trades", 0)
    win_rate = stats.get("win_rate", 0)
    total_pnl = stats.get("total_pnl", 0)

    # Calcular portfolio actual
    portfolio_value = calculate_portfolio_value(state)
    initial_balance = state.get("initial_balance", 500)  # Usar initial_balance del estado
    overall_pnl = portfolio_value - initial_balance
    overall_pnl_pct = (overall_pnl / initial_balance) * 100 if initial_balance > 0 else 0

    # Métricas avanzadas
    winning_trades = stats.get("winning_trades", 0)
    losing_trades = stats.get("losing_trades", 0)
    max_streak = state.get("best_streak", 0)
    worst_streak = state.get("worst_streak", 0)

    # Generar reporte
    report = f"""
{'='*60}
🦞 SOLANA TRADING BOT - RESUMEN OPERATIVA
{'='*60}

⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} MST

💰 VALOR DEL PORTFOLIO
{'─'*60}
Balance: ${balance:,.2f}
Posiciones: ${sum(t.get('size', 0) for t in open_trades):,.2f}
Total: ${portfolio_value:,.2f}
P&L: ${overall_pnl:+,.2f} ({overall_pnl_pct:+.1f}%)

📊 ESTADÍSTICAS DE TRADING
{'─'*60}
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%  ({winning_trades}G / {losing_trades}P)
P&L Acumulado: ${total_pnl:+,.2f}

🎯 RACHAS
{'─'*60}
Mejor racha: {max_streak} ganadores 📈
Peor racha: {worst_streak} perdedores 📉

📌 POSICIONES ABIERTAS ({len(open_trades)})
{'─'*60}"""

    for i, trade in enumerate(open_trades, 1):
        symbol = trade.get("symbol", "UNKNOWN")
        direction = trade.get("direction", "neutral")
        size = trade.get("size", 0)
        entry = trade.get("entry_price", 0)
        pnl_pct = trade.get("pnl_pct", 0)

        direction_emoji = "📈" if direction == "bullish" else "📉"
        pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"

        report += f"""
{i}. {symbol} {direction_emoji}
   Dirección: {direction.upper()}
   Tamaño: ${size:,.2f}
   Entry: ${entry}
   P&L: {pnl_emoji} {pnl_pct:+.1f}%"""

    report += f"""

{'='*60}
📈 SISTEMA: Auto-Improver ACTIVO
   Parámetros: 15% / 3% / 6%
   Reentrenamiento: Cada ~20 ciclos
{'='*60}
"""

    return report

def send_notification(message, priority="normal"):
    """Enviar notificación vía archivo de logs"""
    try:
        notification_log = PROJECT_DIR / "data" / "notifications.log"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(notification_log, 'a') as f:
            f.write(f"[{timestamp}] [{priority.upper()}] {message}\n")

        print(f"✅ Notificación enviada")
    except Exception as e:
        print(f"❌ Error enviando notificación: {e}")

if __name__ == "__main__":
    # Generar reporte
    report = generate_summary()

    # Mostrar reporte
    print(report)

    # Enviar notificación si se especifica
    if "--notify" in sys.argv:
        # Versión corta para notificación
        state = load_state()
        short_report = f"""
📊 RESUMEN TRADING - {datetime.now().strftime('%H:%M')}
💰 Portfolio: ${calculate_portfolio_value(state):,.2f}
📈 Trades: {state.get('stats', {}).get('total_trades', 0)}
🎯 Win Rate: {state.get('stats', {}).get('win_rate', 0):.1f}%
📌 Open: {len(state.get('trades', []))} posiciones
"""
        send_notification(short_report.strip(), "normal")
