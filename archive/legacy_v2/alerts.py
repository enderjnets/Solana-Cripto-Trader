#!/usr/bin/env python3
"""
Alerts Module for Solana Trading Bot
====================================
Send Telegram notifications for trading events.
"""

import os
import asyncio
from typing import Optional

# Try to import Telegram
try:
    import telegram
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "771213858")


class AlertManager:
    """Simple alert manager for Telegram notifications."""
    
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_TOKEN
        self.chat_id = chat_id or CHAT_ID
        self.enabled = bool(self.token and self.chat_id and HAS_TELEGRAM)
        
    async def send(self, message: str, parse_mode: str = None):
        """Send alert to Telegram."""
        if not self.enabled:
            print(f"[ALERT] {message}")
            return
            
        try:
            bot = telegram.Bot(token=self.token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            print(f"[ALERT ERROR] {e}")
    
    # === Trading Alerts ===
    
    async def position_opened(self, symbol: str, direction: str, leverage: float, entry_price: float):
        """Alert when position is opened."""
        emoji = "📈" if direction == "long" else "📉"
        msg = f"""🆕 *Position Opened*

{emoji} *{symbol}* {direction.upper()}
💪 Leverage: {leverage:.2f}x
💵 Entry: ${entry_price:.2f}"""
        await self.send(msg, parse_mode="Markdown")
    
    async def position_closed(self, symbol: str, direction: str, pnl: float, reason: str):
        """Alert when position is closed."""
        emoji = "✅" if pnl > 0 else "❌"
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        msg = f"""🔴 *Position Closed*

{emoji} *{symbol}* {direction.upper()}
💰 PnL: {pnl_str}
📋 Reason: {reason}"""
        await self.send(msg, parse_mode="Markdown")
    
    # === System Alerts ===
    
    async def disconnected(self, reason: str = "Unknown"):
        """Alert when bot disconnects."""
        msg = f"""🔴 *Bot Disconnected*

Reason: {reason}
⏰ Time: Now"""
        await self.send(msg, parse_mode="Markdown")
    
    async def connected(self):
        """Alert when bot connects/reconnects."""
        msg = """🟢 *Bot Connected*

Trading resumed."""
        await self.send(msg, parse_mode="Markdown")
    
    async def error(self, error: str):
        """Alert on error."""
        msg = f"""⚠️ *Error*

{error}"""
        await self.send(msg, parse_mode="Markdown")


# Singleton instance
alerts = AlertManager()
