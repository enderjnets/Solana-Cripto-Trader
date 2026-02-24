#!/usr/bin/env python3
"""
Notification System v2 - File-based logging
Writes notifications to a log file that OpenClaw can read and send
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
logger = logging.getLogger("notifications")


class NotificationLogger:
    """Logs notifications to a file for OpenClaw to read and send"""

    def __init__(self):
        self.notification_log = PROJECT_ROOT / "data" / "notifications.log"
        self.notification_log.parent.mkdir(exist_ok=True)
        self.enabled = True
        logger.info(f"📱 NotificationLogger initialized: {self.notification_log}")

    def _log(self, message: str, priority: str = "normal"):
        """Log notification to file"""
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{priority.upper()}] {message}\n"

        try:
            with open(self.notification_log, "a", encoding="utf-8") as f:
                f.write(log_entry)
            logger.debug(f"✅ Notification logged: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to log notification: {e}")
            return False

    # ========== TRADE NOTIFICATIONS ==========

    def trade_opened(self, symbol: str, direction: str, entry: float, size: float, reason: str = ""):
        """Alert when a trade is opened"""
        direction_emoji = "📈" if direction == "bullish" else "📉"
        message = f"""{direction_emoji} NUEVA POSICIÓN ABIERTA
💎 Token: {symbol}
🎯 Dirección: {direction.upper()}
💵 Entry: ${entry:,.6f}
📊 Size: ${size:,.2f}
📝 Razón: {reason}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="normal")

    def trade_closed(self, symbol: str, entry: float, exit: float, pnl: float, pnl_percent: float, reason: str = ""):
        """Alert when a trade is closed"""
        pnl_emoji = "✅" if pnl > 0 else "❌"
        message = f"""{pnl_emoji} POSICIÓN CERRADA
💎 Token: {symbol}
💵 Entry: ${entry:,.6f}
🎯 Exit: ${exit:,.6f}
📊 P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)
📝 Razón: {reason}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        # High priority for significant P&L
        priority = "high" if abs(pnl) >= 50 else "normal"
        return self._log(message.strip(), priority=priority)

    def stop_loss_hit(self, symbol: str, entry: float, sl: float, pnl: float):
        """Critical alert when stop loss is hit"""
        message = f"""🛑 STOP LOSS EJECUTADO 🛑
💎 Token: {symbol}
💵 Entry: ${entry:,.6f}
🎯 SL: ${sl:,.6f}
📊 P&L: ${pnl:,.2f}
⚠️ TRADE CERRADO POR STOP LOSS
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="critical")

    def take_profit_hit(self, symbol: str, entry: float, tp: float, pnl: float):
        """Alert when take profit is hit"""
        message = f"""🎯 TAKE PROFIT EJECUTADO 🎯
💎 Token: {symbol}
💵 Entry: ${entry:,.6f}
🎯 TP: ${tp:,.6f}
📊 P&L: ${pnl:,.2f}
✅ OBJETIVO ALCANZADO
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="high")

    # ========== PORTFOLIO NOTIFICATIONS ==========

    def balance_change(self, old_balance: float, new_balance: float, change_percent: float):
        """Alert when balance changes significantly (>5%)"""
        change_emoji = "📈" if new_balance > old_balance else "📉"
        message = f"""{change_emoji} CAMBIO SIGNIFICATIVO EN BALANCE
💰 Balance anterior: ${old_balance:,.2f}
💰 Nuevo balance: ${new_balance:,.2f}
📊 Cambio: {change_percent:+.2f}%
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="high")

    def daily_summary(self, balance: float, total_trades: int, win_rate: float, daily_pnl: float, open_positions: int):
        """Daily trading summary"""
        win_emoji = "✅" if win_rate >= 50 else "⚠️"
        pnl_emoji = "📈" if daily_pnl > 0 else "📉"
        message = f"""📊 RESUMEN DIARIO DE TRADING
💰 Balance actual: ${balance:,.2f}
{pnl_emoji} P&L del día: ${daily_pnl:,.2f}
📈 Total trades: {total_trades}
{win_emoji} Win rate: {win_rate:.1f}%
📊 Posiciones abiertas: {open_positions}
📅 {datetime.now().strftime('%Y-%m-%d')}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="normal")

    # ========== SYSTEM NOTIFICATIONS ==========

    def system_error(self, error: str, context: str = ""):
        """Critical system error"""
        message = f"""🚨 ERROR DEL SISTEMA 🚨
❌ Error: {error}
📝 Contexto: {context}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="critical")

    def system_warning(self, warning: str, context: str = ""):
        """System warning"""
        message = f"""⚠️ ADVERTENCIA DEL SISTEMA
📝 Warning: {warning}
📝 Contexto: {context}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="high")

    def market_alert(self, alert: str):
        """Market-related alert (volatility, etc.)"""
        message = f"""📢 ALERTA DE MERCADO
{alert}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="normal")

    # ========== RISK NOTIFICATIONS ==========

    def risk_limit_warning(self, limit_type: str, current: float, max_limit: float):
        """Alert when approaching risk limits"""
        message = f"""⚠️ LÍMITE DE RIESGO CERCANO
📊 Tipo: {limit_type}
📈 Actual: {current:,.2f}
🎯 Límite: {max_limit:,.2f}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="high")

    def dynamic_risk_changed(self, mode: str, reason: str):
        """Alert when dynamic risk mode changes"""
        message = f"""🧠 CAMBIO EN MODO DE RIESGO
📊 Nuevo modo: {mode}
📝 Razón: {reason}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="normal")

    def system_started(self, mode: str = "DAY TRADING"):
        """Alert when trading system starts"""
        message = f"""🚀 SISTEMA DE TRADING INICIADO
📊 Modo: {mode}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        return self._log(message.strip(), priority="normal")

    def system_stopped(self, reason: str = "Manual shutdown"):
        """Alert when trading system stops"""
        message = f"""🛑 SISTEMA DE TRADING DETENIDO
📝 Razón: {reason}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        return self._log(message.strip(), priority="high")

    def system_error(self, error: str, context: str = ""):
        """Critical alert when system encounters an error"""
        message = f"""❌ ERROR DEL SISTEMA
📝 Error: {error}
📝 Contexto: {context}
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        return self._log(message.strip(), priority="high")


# Global notifier instance
_notifier: Optional[NotificationLogger] = None


def get_notifier() -> NotificationLogger:
    """Get or create the global notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = NotificationLogger()
    return _notifier


if __name__ == "__main__":
    # Test the notifier
    notifier = get_notifier()

    print("Testing notification logger...")
    notifier.trade_opened("WIF", "bearish", 0.206, 100.0, "RSI oversold")
    notifier.trade_closed("SOL", 80.0, 85.0, 50.0, 0.625, "Take profit")
    notifier.system_error("Connection timeout", "Kraken API")
    notifier.daily_summary(1000.0, 20, 55.0, 50.0, 2)

    print(f"Test complete! Check: {notifier.notification_log}")
