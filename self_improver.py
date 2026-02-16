"""
Self-Improving Trading System
Tracks performance and adjusts parameters automatically
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class SelfImprover:
    def __init__(self, state_file: str = "data/self_improvement.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {
            "trades_by_signal": {},  # signal_type -> {"wins": 0, "losses": 0}
            "total_trades": 0,
            "last_update": None
        }
    
    def _save_state(self):
        self.state["last_update"] = datetime.now().isoformat()
        self.state_file.write_text(json.dumps(self.state, indent=2))
    
    def record_trade(self, symbol: str, direction: str, pnl_pct: float):
        """Record a completed trade"""
        signal_key = f"{symbol}_{direction}"
        
        if signal_key not in self.state["trades_by_signal"]:
            self.state["trades_by_signal"][signal_key] = {"wins": 0, "losses": 0}
        
        if pnl_pct > 0:
            self.state["trades_by_signal"][signal_key]["wins"] += 1
        else:
            self.state["trades_by_signal"][signal_key]["losses"] += 1
        
        self.state["total_trades"] += 1
        self._save_state()
    
    def get_win_rate(self, symbol: str = None, direction: str = None) -> float:
        """Get win rate for a signal type"""
        if symbol and direction:
            signal_key = f"{symbol}_{direction}"
            if signal_key in self.state["trades_by_signal"]:
                data = self.state["trades_by_signal"][signal_key]
                total = data["wins"] + data["losses"]
                if total > 0:
                    return data["wins"] / total
        
        # Overall win rate
        total_wins = sum(d["wins"] for d in self.state["trades_by_signal"].values())
        total_losses = sum(d["losses"] for d in self.state["trades_by_signal"].values())
        total = total_wins + total_losses
        
        if total > 0:
            return total_wins / total
        return 0.5  # Default 50%
    
    def get_adjusted_confidence_threshold(self, base_threshold: float = 10) -> float:
        """Adjust confidence threshold based on recent performance"""
        win_rate = self.get_win_rate()
        
        # If win rate > 60%, lower threshold to take more trades
        # If win rate < 40%, raise threshold to be more selective
        if win_rate > 0.6:
            return max(5, base_threshold - 5)
        elif win_rate > 0.7:
            return max(3, base_threshold - 8)
        elif win_rate < 0.4:
            return min(20, base_threshold + 5)
        elif win_rate < 0.3:
            return min(30, base_threshold + 10)
        
        return base_threshold
    
    def get_stats(self) -> Dict:
        """Get improvement stats"""
        return {
            "total_trades": self.state["total_trades"],
            "overall_win_rate": self.get_win_rate(),
            "signals": self.state["trades_by_signal"]
        }

# Test
if __name__ == "__main__":
    improver = SelfImprover()
    improver.record_trade("SOL", "bullish", 2.5)
    improver.record_trade("SOL", "bullish", -1.0)
    print(improver.get_stats())
