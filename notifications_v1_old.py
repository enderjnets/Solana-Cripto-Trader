#!/usr/bin/env python3
"""
Notification System for Trading Bot
Sends alerts to Telegram via OpenClaw message system
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("notifications")


class TelegramNotifier:
    """Sends trading alerts to Telegram via OpenClaw"""

    def __init__(self):
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "771213858")
        self.enabled = True
        logger.info(f"📱 TelegramNotifier initialized for chat: {self.chat_id}")

    def _send(self, message: str, priority: str = "normal"):
        """Send message via OpenClaw"""
        try:
            # Import OpenClaw message module
            from openclaw_tools import send_telegram_message

            send_telegram_message(
                chat_id=self.chat_id,
                message=message,
                priority=priority
            )
            logger.debug(f"✅ Notification sent: {message[:50]}...")
            return True
        except ImportError:
            # Fallback: Write to notification log
            log_file = PROJECT_ROOT / "data" / "notifications.log"
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"{datetime.now()} | {message}\n")
            logger.info(f"📝 Notification logged (OpenClaw not available): {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")
            return False

    # ========== TRADE NOTIFICATIONS ==========

    def trade_opened(self, symbol: str, direction: str, entry: float, size: float, reason: str = ""):
        """Alert when a trade is opened"""
        direction_emoji = "📈" if direction == "bullish" else "📉"
        message = f"""
{direction_emoji} **NUEVA POSICIÓN ABIERTA**

💎 Token: {symbol}
🎯 Dirección: {direction.upper()}
💵 Entry: ${entry:,.6f}
📊 Size: ${size:,.2f}
📝 Razón: {reason}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="normal")

    def trade_closed(self, symbol: str, entry: float, exit: float, pnl: float, pnl_percent: float, reason: str = ""):
        """Alert when a trade is closed"""
        pnl_emoji = "✅" if pnl > 0 else "❌"
        message = f"""
{pnl_emoji} **POSICIÓN CERRADA**

💎 Token: {symbol}
💵 Entry: ${entry:,.6f}
🎯 Exit: ${exit:,.6f}
📊 P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)
📝 Razón: {reason}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        # High priority for significant P&L
        priority = "high" if abs(pnl) >= 50 else "normal"
        return self._send(message, priority=priority)

    def stop_loss_hit(self, symbol: str, entry: float, sl: float, pnl: float):
        """Critical alert when stop loss is hit"""
        message = f"""
🛑 **STOP LOSS EJECUTADO** 🛑

💎 Token: {symbol}
💵 Entry: ${entry:,.6f}
🎯 SL: ${sl:,.6f}
📊 P&L: ${pnl:,.2f}

⚠️ TRADE CERRADO POR STOP LOSS
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="critical")

    def take_profit_hit(self, symbol: str, entry: float, tp: float, pnl: float):
        """Alert when take profit is hit"""
        message = f"""
🎯 **TAKE PROFIT EJECUTADO** 🎯

💎 Token: {symbol}
💵 Entry: ${entry:,.6f}
🎯 TP: ${tp:,.6f}
📊 P&L: ${pnl:,.2f}

✅ OBJETIVO ALCANZADO
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="high")

    # ========== PORTFOLIO NOTIFICATIONS ==========

    def balance_change(self, old_balance: float, new_balance: float, change_percent: float):
        """Alert when balance changes significantly (>5%)"""
        change_emoji = "📈" if new_balance > old_balance else "📉"
        message = f"""
{change_emoji} **CAMBIO SIGNIFICATIVO EN BALANCE**

💰 Balance anterior: ${old_balance:,.2f}
💰 Nuevo balance: ${new_balance:,.2f}
📊 Cambio: {change_percent:+.2f}%

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="high")

    def daily_summary(self, balance: float, total_trades: int, win_rate: float, daily_pnl: float, open_positions: int):
        """Daily trading summary"""
        win_emoji = "✅" if win_rate >= 50 else "⚠️"
        pnl_emoji = "📈" if daily_pnl > 0 else "📉"
        message = f"""
📊 **RESUMEN DIARIO DE TRADING**

💰 Balance actual: ${balance:,.2f}
{pnl_emoji} P&L del día: ${daily_pnl:,.2f}
📈 Total trades: {total_trades}
{win_emoji} Win rate: {win_rate:.1f}
📊 Posiciones abiertas: {open_positions}

📅 {datetime.now().strftime('%Y-%m-%d')}
⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="normal")

    # ========== SYSTEM NOTIFICATIONS ==========

    def system_error(self, error: str, context: str = ""):
        """Critical system error"""
        message = f"""
🚨 **ERROR DEL SISTEMA** 🚨

❌ Error: {error}
📝 Contexto: {context}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="critical")

    def system_warning(self, warning: str, context: str = ""):
        """System warning"""
        message = f"""
⚠️ **ADVERTENCIA DEL SISTEMA**

📝 Warning: {warning}
📝 Contexto: {context}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="high")

    def market_alert(self, alert: str):
        """Market-related alert (volatility, etc.)"""
        message = f"""
📢 **ALERTA DE MERCADO**

{alert}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="normal")

    # ========== RISK NOTIFICATIONS ==========

    def risk_limit_warning(self, limit_type: str, current: float, max_limit: float):
        """Alert when approaching risk limits"""
        message = f"""
⚠️ **LÍMITE DE RIESGO CERCANO**

📊 Tipo: {limit_type}
📈 Actual: {current:,.2f}
🎯 Límite: {max_limit:,.2f}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="high")

    def dynamic_risk_changed(self, mode: str, reason: str):
        """Alert when dynamic risk mode changes"""
        message = f"""
🧠 **CAMBIO EN MODO DE RIESGO**

📊 Nuevo modo: {mode}
📝 Razón: {reason}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self._send(message, priority="normal")


# Global notifier instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


if __name__ == "__main__":
    # Test the notifier
    notifier = get_notifier()

    print("Testing Telegram notifier...")
    notifier.trade_opened("WIF", "bearish", 0.206, 100.0, "RSI oversold")
    notifier.trade_closed("SOL", 80.0, 85.0, 50.0, 0.625, "Take profit")
    notifier.system_error("Connection timeout", "Kraken API")
    notifier.daily_summary(1000.0, 20, 55.0, 50.0, 2)

    print("Test complete!")
