#!/usr/bin/env python3
"""
Auto-Tuner Module
=================
Automatically adjusts trading parameters to achieve daily profit targets.

Target: 5% daily profit
Logic:
- If profit < 5% and WR > 50%: Lower confidence threshold (more trades)
- If profit < 5% and WR < 40%: Increase risk per trade
- If profit > 5%: Lower risk (preserve gains)
- If profit < 0%: Pause mode - only high confidence trades
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

logger = logging.getLogger("auto_tuner")

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "auto_tuner_state.json"


class AutoTuner:
    """
    Auto-tuning system for achieving 5% daily profit target.
    
    Adjusts:
    - confidence_threshold: Minimum confidence to take a trade
    - risk_per_trade: % of balance risked per trade
    - max_trades_per_day: Maximum trades per day
    """
    
    # Limits
    MIN_CONFIDENCE = 5
    MAX_CONFIDENCE = 50
    MIN_RISK = 0.05  # 5%
    MAX_RISK = 0.20  # 20%
    MIN_TRADES = 5
    MAX_TRADES = 30
    
    # Targets
    DAILY_PROFIT_TARGET = 0.05  # 5%
    WR_GOOD = 0.50  # 50%
    WR_BAD = 0.40  # 40%
    
    def __init__(self):
        self.state = self._load_state()
        self.last_adjustment_time = None
    
    def _load_state(self) -> Dict:
        """Load saved state"""
        try:
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text())
        except Exception as e:
            logger.warning(f"Could not load auto-tuner state: {e}")
        
        return {
            "confidence_threshold": 10,
            "risk_per_trade": 0.10,
            "max_trades_per_day": 20,
            "daily_profit_target": 0.05,
            "adjustments_today": 0,
            "last_adjustment_date": None,
            "total_adjustments": 0,
            "current_mode": "normal"
        }
    
    def _save_state(self):
        """Save current state"""
        try:
            STATE_FILE.write_text(json.dumps(self.state, indent=2))
        except Exception as e:
            logger.warning(f"Could not save auto-tuner state: {e}")
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current parameters"""
        return {
            "confidence_threshold": self.state["confidence_threshold"],
            "risk_per_trade": self.state["risk_per_trade"],
            "max_trades_per_day": self.state["max_trades_per_day"]
        }
    
    def reset_daily(self):
        """Reset daily counters if new day"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.get("last_adjustment_date") != today:
            self.state["adjustments_today"] = 0
            self.state["last_adjustment_date"] = today
            self._save_state()
    
    def analyze_and_adjust(
        self, 
        daily_pnl_pct: float, 
        win_rate: float,
        trades_today: int,
        recent_pnl: float = 0
    ) -> Dict[str, Any]:
        """
        Main auto-tuning logic.
        
        Args:
            daily_pnl_pct: Daily profit/loss as percentage (e.g., 0.02 = 2%)
            win_rate: Win rate from recent trades (0.0-1.0)
            trades_today: Number of trades executed today
            recent_pnl: Recent P/L for direction analysis
            
        Returns:
            Dict with adjustment info and current parameters
        """
        # Handle None values
        if daily_pnl_pct is None:
            daily_pnl_pct = 0.0
        if win_rate is None:
            win_rate = 0.5
        
        self.reset_daily()
        
        # Max 3 adjustments per day
        if self.state["adjustments_today"] >= 3:
            logger.info("🎚️ Auto-Tuner: Max daily adjustments reached")
            return {
                "adjusted": False,
                "reason": "max_adjustments_reached",
                "parameters": self.get_parameters()
            }
        
        # Handle None values safely
        if daily_pnl_pct is None:
            daily_pnl_pct = 0.0
        if win_rate is None:
            win_rate = 0.5
        if target is None:
            target = 0.05
        if trades_today is None:
            trades_today = 0
        
        target = self.state.get("daily_profit_target", 0.05)
        adjustment_made = False
        reason = "no_change"
        
        # === MODE 1: PROFIT > 5% - PRESERVE GAINS ===
        if daily_pnl_pct > target:
            new_risk = max(self.state["risk_per_trade"] * 0.8, self.MIN_RISK)
            self.state["risk_per_trade"] = round(new_risk, 2)
            self.state["current_mode"] = "preserve"
            adjustment_made = True
            reason = f"profit_above_target_{daily_pnl_pct:.1%}_lowering_risk"
            logger.info(f"🎚️ Auto-Tuner: Profit {daily_pnl_pct:.1%} > 5% → Lowering risk to {new_risk:.0%}")
        
        # === MODE 2: PROFIT < 0% - PAUSE MODE ===
        elif daily_pnl_pct < 0:
            # Only take high confidence trades
            new_conf = min(self.state["confidence_threshold"] + 10, self.MAX_CONFIDENCE)
            self.state["confidence_threshold"] = new_conf
            self.state["current_mode"] = "pause"
            adjustment_made = True
            reason = f"negative_profit_{daily_pnl_pct:.1%}_pausing"
            logger.info(f"🎚️ Auto-Tuner: Loss {daily_pnl_pct:.1%} → Raising confidence to {new_conf}%")
        
        # === MODE 3: 0% < PROFIT < 5% - ADJUST ===
        else:
            if win_rate > self.WR_GOOD:
                # Good win rate but not enough profit → more trades
                if trades_today < self.state["max_trades_per_day"]:
                    new_conf = max(self.state["confidence_threshold"] - 5, self.MIN_CONFIDENCE)
                    self.state["confidence_threshold"] = new_conf
                    adjustment_made = True
                    reason = f"good_wr_{win_rate:.0%}_more_trades"
                    logger.info(f"🎚️ Auto-Tuner: WR {win_rate:.0%} good → Lowering confidence to {new_conf}% for more trades")
            
            elif win_rate < self.WR_BAD:
                # Bad win rate → less trades, higher risk per trade
                new_risk = min(self.state["risk_per_trade"] * 1.2, self.MAX_RISK)
                new_conf = min(self.state["confidence_threshold"] + 5, self.MAX_CONFIDENCE)
                self.state["risk_per_trade"] = round(new_risk, 2)
                self.state["confidence_threshold"] = new_conf
                adjustment_made = True
                reason = f"low_wr_{win_rate:.0%}_higher_risk"
                logger.info(f"🎚️ Auto-Tuner: WR {win_rate:.0%} low → Risk {new_risk:.0%}, Conf {new_conf}%")
            
            else:
                # 40-50% win rate → balanced adjustment
                new_risk = min(self.state["risk_per_trade"] * 1.1, self.MAX_RISK)
                self.state["risk_per_trade"] = round(new_risk, 2)
                adjustment_made = True
                reason = f"moderate_wr_{win_rate:.0%}_slight_risk_increase"
                logger.info(f"🎚️ Auto-Tuner: WR {win_rate:.0%} moderate → Slight risk increase to {new_risk:.0%}")
        
        if adjustment_made:
            self.state["adjustments_today"] += 1
            self.state["total_adjustments"] += 1
            self._save_state()
        
        return {
            "adjusted": adjustment_made,
            "reason": reason,
            "parameters": self.get_parameters(),
            "mode": self.state["current_mode"],
            "daily_pnl_pct": daily_pnl_pct,
            "win_rate": win_rate,
            "trades_today": trades_today
        }
    
    def get_status(self) -> Dict:
        """Get current tuner status"""
        return {
            **self.get_parameters(),
            "mode": self.state["current_mode"],
            "adjustments_today": self.state["adjustments_today"],
            "total_adjustments": self.state["total_adjustments"],
            "daily_profit_target": self.state["daily_profit_target"]
        }


# Test
if __name__ == "__main__":
    tuner = AutoTuner()
    
    # Test scenarios
    print("=== Auto-Tuner Test ===\n")
    
    # Scenario 1: Profit > 5%
    result = tuner.analyze_and_adjust(0.06, 0.60, 15)
    print(f"Scenario 1: {result}")
    
    # Scenario 2: Loss
    result = tuner.analyze_and_adjust(-0.02, 0.35, 10)
    print(f"Scenario 2: {result}")
    
    # Scenario 3: Good WR, not enough profit
    result = tuner.analyze_and_adjust(0.02, 0.55, 12)
    print(f"Scenario 3: {result}")
    
    # Scenario 4: Low WR
    result = tuner.analyze_and_adjust(0.01, 0.30, 8)
    print(f"Scenario 4: {result}")
    
    print("\n=== Status ===")
    print(tuner.get_status())
