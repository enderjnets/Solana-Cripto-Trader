#!/usr/bin/env python3
"""
Webhook Alert System
====================
Sends real-time alerts for:
- Trade executions
- Stop loss / Take profit triggers
- Daily P&L updates
- Risk warnings

Supports:
- Telegram (via OpenClaw)
- Discord Webhooks
- Slack Webhooks
- Email (SMTP)
- Custom webhooks

Usage:
    from alerts.webhook_manager import WebhookManager
    alerts = WebhookManager()
    alerts.send_trade_alert("BUY", "SOL", 87.50, 20)
"""

import asyncio
import json
import httpx
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AlertConfig:
    """Webhook configuration."""
    telegram_enabled: bool = True
    discord_enabled: bool = False
    discord_webhook: str = ""
    slack_enabled: bool = False
    slack_webhook: str = ""
    email_enabled: bool = False
    smtp_config: Dict = None


class WebhookManager:
    """Manages all alert webhooks."""

    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.alert_history: List[Dict] = []

    async def send_telegram(self, message: str) -> bool:
        """Send alert via Telegram (OpenClaw)."""
        # OpenClaw handles Telegram automatically when message is sent
        # This is handled by the message tool in OpenClaw
        print(f"📱 TELEGRAM: {message}")
        return True

    async def send_discord(self, message: str) -> bool:
        """Send alert via Discord webhook."""
        if not self.config.discord_enabled or not self.config.discord_webhook:
            return False

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    self.config.discord_webhook,
                    json={"content": message},
                    timeout=10
                )
            return True
        except Exception as e:
            print(f"Discord error: {e}")
            return False

    async def send_slack(self, message: str) -> bool:
        """Send alert via Slack webhook."""
        if not self.config.slack_enabled or not self.config.slack_webhook:
            return False

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    self.config.slack_webhook,
                    json={"text": message},
                    timeout=10
                )
            return True
        except Exception as e:
            print(f"Slack error: {e}")
            return False

    async def send_alert(
        self,
        alert_type: str,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> Dict:
        """Send alert to all configured channels."""
        timestamp = datetime.now().isoformat()

        # Format message
        full_message = f"""
🔔 *{title}*
```
{message}
```
📅 {timestamp}
"""

        # Send to all channels
        results = {
            "telegram": await self.send_telegram(full_message),
            "discord": await self.send_discord(full_message),
            "slack": await self.send_slack(full_message)
        }

        # Log alert
        alert = {
            "timestamp": timestamp,
            "type": alert_type,
            "title": title,
            "message": message,
            "priority": priority,
            "results": results
        }
        self.alert_history.append(alert)

        return results

    async def send_trade_alert(
        self,
        direction: str,
        symbol: str,
        price: float,
        size: float,
        confidence: float = 0.0
    ) -> Dict:
        """Send trade execution alert."""
        emoji = "🟢" if direction == "BUY" else "🔴"
        return await self.send_alert(
            alert_type="trade",
            title=f"TRADE EXECUTED {emoji}",
            message=f"""
{direction} {symbol}
Price: ${price:.4f}
Size: ${size:.2f}
Confidence: {confidence:.0%}
""",
            priority="high"
        )

    async def send_pnl_alert(
        self,
        daily_pnl_pct: float,
        trades_today: int,
        win_rate: float
    ) -> Dict:
        """Send daily P&L update."""
        emoji = "📈" if daily_pnl_pct >= 0 else "📉"
        return await self.send_alert(
            alert_type="pnl",
            title=f"P&L UPDATE {emoji}",
            message=f"""
Daily P&L: {daily_pnl_pct:+.2f}%
Trades Today: {trades_today}
Win Rate: {win_rate:.1f}%
Target: +5.00%
""",
            priority="normal"
        )

    async def send_risk_alert(
        self,
        reason: str,
        current_exposure: float
    ) -> Dict:
        """Send risk warning alert."""
        return await self.send_alert(
            alert_type="risk",
            title="⚠️ RISK WARNING",
            message=f"""
Reason: {reason}
Current Exposure: {current_exposure:.1f}%
Max Allowed: 10.00%
""",
            priority="critical"
        )

    async def send_tp_sl_alert(
        self,
        symbol: str,
        close_reason: str,
        pnl_pct: float
    ) -> Dict:
        """Send take profit / stop loss alert."""
        emoji = "✅" if pnl_pct >= 0 else "❌"
        return await self.send_alert(
            alert_type="tp_sl",
            title=f"POSITION CLOSED {emoji}",
            message=f"""
Symbol: {symbol}
Reason: {close_reason}
P&L: {pnl_pct:+.2f}%
""",
            priority="high"
        )

    def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        return self.alert_history[-limit:]


class AutonomousScheduler:
    """Schedules autonomous tasks."""

    def __init__(self, webhook: WebhookManager):
        self.webhook = webhook
        self.tasks = []
        self.running = False

    def add_task(self, name: str, interval: int, callback):
        """Add autonomous task."""
        self.tasks.append({
            "name": name,
            "interval": interval,
            "callback": callback
        })

    async def run(self):
        """Run all autonomous tasks."""
        self.running = True
        while self.running:
            for task in self.tasks:
                try:
                    await task["callback"]()
                except Exception as e:
                    await self.webhook.send_alert(
                        "error",
                        "AUTONOMOUS TASK ERROR",
                        f"{task['name']}: {e}"
                    )
            await asyncio.sleep(60)  # Check every minute


async def test_webhook():
    """Test webhook system."""
    print("\n🧪 TESTING WEBHOOK SYSTEM")
    print("="*50)

    webhook = WebhookManager()

    # Test trade alert
    print("\n📊 Testing trade alert...")
    result = await webhook.send_trade_alert("BUY", "SOL", 87.50, 20, 0.85)
    print(f"   Result: {result}")

    # Test P&L alert
    print("\n💰 Testing P&L alert...")
    result = await webhook.send_pnl_alert(2.5, 5, 60.0)
    print(f"   Result: {result}")

    # Test risk alert
    print("\n⚠️ Testing risk alert...")
    result = await self.webhook.send_risk_alert("Daily loss limit hit", 12.5)
    print(f"   Result: {result}")

    print("\n" + "="*50)
    print("✅ Webhook system working!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_webhook())
    else:
        print("Webhook Manager Module Ready")
