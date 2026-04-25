#!/usr/bin/env python3
"""
Auto-Learning Notification System
=================================
Sistema de notificaciones automáticas para eventos importantes.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from auto_learning_wrapper import get_wrapper

class NotificationSystem:
    """Sistema de notificaciones automáticas"""
    
    def __init__(self):
        self.wrapper = get_wrapper()
        self.last_trades = 0
        self.last_generation = 0
        self.last_daily_pnl = 0.0
        self.log_file = Path("/tmp/auto_learning_notifications.log")
        self.notification_count = 0
        
        # Load last state
        self._load_state()
        
        # Setup log file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_state(self):
        """Load last known state"""
        state_file = Path("/tmp/auto_learning_notification_state.json")
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.last_trades = data.get("last_trades", 0)
                self.last_generation = data.get("last_generation", 0)
                self.last_daily_pnl = data.get("last_daily_pnl", 0.0)
            except:
                pass
    
    def _save_state(self):
        """Save current state"""
        state_file = Path("/tmp/auto_learning_notification_state.json")
        data = {
            "last_trades": self.last_trades,
            "last_generation": self.last_generation,
            "last_daily_pnl": self.last_daily_pnl,
            "timestamp": datetime.now().isoformat()
        }
        state_file.write_text(json.dumps(data, indent=2))
    
    def _log(self, message: str, priority: str = "medium"):
        """Log notification"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{priority.upper()}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        print(log_line.strip())
        self.notification_count += 1
    
    def check_important_events(self):
        """Check for important events and send notifications"""
        status = self.wrapper.get_status()
        
        current_trades = status['total_trades']
        current_generation = status['current_generation']
        current_daily_pnl = status['daily_pnl']
        
        # Event 1: First trade executed
        if current_trades == 1 and self.last_trades == 0:
            self._log("🎯 **FIRST TRADE EXECUTED**\n\nAuto-learning system has executed its first trade!\n\nThe system is now learning from real trades.", "high")
        
        # Event 2: New trades executed
        if current_trades > self.last_trades and current_trades > 1:
            new_trades = current_trades - self.last_trades
            self._log(f"📊 **NEW TRADES EXECUTED**\n\n{new_trades} new trades completed.\n\nTotal trades: {current_trades}", "medium")
        
        # Event 3: Daily target reached (5%)
        if current_daily_pnl >= 0.05 and self.last_daily_pnl < 0.05:
            pnl_pct = current_daily_pnl * 100
            self._log(f"🎉 **DAILY TARGET REACHED!**\n\nDaily PnL: +{pnl_pct:.2f}%\n\nTarget of +5% achieved!", "high")
        
        # Event 4: Significant profit (3%+)
        if current_daily_pnl >= 0.03 and self.last_daily_pnl < 0.03:
            pnl_pct = current_daily_pnl * 100
            self._log(f"📈 **SIGNIFICANT PROFIT**\n\nDaily PnL: +{pnl_pct:.2f}%\n\nGreat progress towards daily target!", "medium")
        
        # Event 5: Max drawdown warning
        if current_daily_pnl <= -0.10 and self.last_daily_pnl > -0.10:
            pnl_pct = current_daily_pnl * 100
            self._log(f"⚠️ **MAX DRAWDOWN WARNING**\n\nDaily PnL: {pnl_pct:.2f}%\n\nSystem is adjusting risk parameters.", "high")
        
        # Event 6: Strategy evolution completed
        if current_generation > self.last_generation and current_generation > 0:
            self._log(f"🧬 **STRATEGY EVOLUTION COMPLETE**\n\nNew generation: {current_generation}\n\nStrategies have evolved based on recent performance.", "high")
        
        # Event 7: Evolution threshold reached
        if current_trades >= 50 and self.last_trades < 50:
            self._log(f"🔬 **EVOLUTION THRESHOLD REACHED**\n\n50+ trades completed.\n\nNext strategy evolution in 1 hour.", "medium")
        
        # Event 8: 100 trades milestone
        if current_trades >= 100 and self.last_trades < 100:
            self._log(f"🏆 **100 TRADES MILESTONE**\n\nTotal trades: {current_trades}\n\nSystem has accumulated significant learning data!", "medium")
        
        # Update state
        self.last_trades = current_trades
        self.last_generation = current_generation
        self.last_daily_pnl = current_daily_pnl
        self._save_state()
        
        return status
    
    def run(self, interval: int = 300):
        """Run notification system"""
        self._log("="*60)
        self._log("🚀 NOTIFICATION SYSTEM STARTED")
        self._log("="*60)
        self._log(f"Check interval: {interval} seconds")
        self._log("Monitoring for important events...")
        self._log("="*60)
        
        while True:
            try:
                self.check_important_events()
                time.sleep(interval)
            except KeyboardInterrupt:
                self._log("🛑 Notification system stopped")
                break
            except Exception as e:
                self._log(f"❌ Error: {str(e)}", "high")
                time.sleep(60)

if __name__ == "__main__":
    system = NotificationSystem()
    system.run(interval=300)  # Check every 5 minutes
