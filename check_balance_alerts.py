#!/usr/bin/env python3
"""
Alerta de Trading - Monitorea cambios de capital >10%
Envía alertas a Telegram cuando hay cambios significativos
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_DIR = Path("/home/enderj/.openclaw/workspace/solana-jupiter-bot")
STATE_FILE = PROJECT_DIR / "data" / "paper_trading_state.json"
BALANCE_FILE = PROJECT_DIR / "data" / "last_balance.json"
ALERT_THRESHOLD = 10  # 10% change

def load_state():
    """Cargar estado actual"""
    with open(STATE_FILE) as f:
        return json.load(f)

def load_last_balance():
    """Cargar último balance conocido"""
    try:
        with open(BALANCE_FILE) as f:
            return json.load(f)
    except:
        return None

def save_last_balance(balance):
    """Guardar último balance"""
    with open(BALANCE_FILE, 'w') as f:
        json.dump({
            "balance": balance,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

def calculate_portfolio_value(state):
    """Calcular valor del portfolio"""
    balance = state.get("balance_usd", 0)
    trades = state.get("trades", [])
    total_size = sum(t.get("size", 0) for t in trades)
    return balance + total_size

def send_alert(message, priority="high"):
    """Enviar alerta"""
    notification_log = PROJECT_DIR / "data" / "notifications.log"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(notification_log, 'a') as f:
        f.write(f"[{timestamp}] [{priority.upper()}] {message}\n")

    print(f"🚨 ALERTA ENVIADA: {message}")

def check_alerts():
    """Verificar alertas de cambio de capital"""
    state = load_state()
    last_balance = load_last_balance()

    if not last_balance:
        # Primera ejecución, guardar balance actual
        save_last_balance(state.get("balance_usd", 0))
        print("✅ Primer balance registrado")
        return

    # Calcular valores
    current_balance = state.get("balance_usd", 0)
    last_balance_value = last_balance["balance"]
    change = current_balance - last_balance_value
    change_pct = (change / last_balance_value) * 100 if last_balance_value > 0 else 0

    # Verificar umbral
    if abs(change_pct) >= ALERT_THRESHOLD:
        direction = "📈 PERDIDA" if change < 0 else "📉 GANANCIA"
        alert_msg = f"""
🚨 ALERTA DE CAPITAL >{ALERT_THRESHOLD}%

{direction} DETECTADA

Balance anterior: ${last_balance_value:,.2f}
Balance actual: ${current_balance:,.2f}
Cambio: ${change:+,.2f} ({change_pct:+.1f}%)

Portfolio total: ${calculate_portfolio_value(state):,.2f}
Posiciones abiertas: {len(state.get('trades', []))}
"""

        send_alert(alert_msg.strip(), "high")

        # Actualizar último balance
        save_last_balance(current_balance)
    else:
        print(f"✅ Cambio normal: {change:+.1f}% (umbral: {ALERT_THRESHOLD}%)")

if __name__ == "__main__":
    check_alerts()
