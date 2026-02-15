#!/usr/bin/env python3
"""
Risk Agent
==========
Specialized agent for risk assessment and limit checking.

Features:
- Position size validation
- Drawdown monitoring
- Daily loss limits
- Portfolio risk analysis
- Trade approval/rejection

Usage:
    agent = RiskAgent()
    result = agent.validate_trade(trade_signal)
    result = agent.check_portfolio_risk()
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config

logger = logging.getLogger("risk_agent")

# ============================================================================
# RISK LIMITS
# ============================================================================
@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_pct: float = 0.15  # 15% max per trade
    max_daily_loss_pct: float = 0.10  # 10% daily loss limit
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    max_concurrent_trades: int = 5  # Max open trades
    min_confidence: float = 0.30  # Min signal confidence (30%)
    risk_per_trade_pct: float = 0.10  # 10% risk per trade


@dataclass
class TradeRisk:
    """Risk assessment for a trade"""
    approved: bool
    risk_score: float  # 0-1 (1 = high risk)
    reasons: List[str]
    suggestions: List[str]


# ============================================================================
# RISK AGENT
# ============================================================================
class RiskAgent:
    """
    Risk assessment agent for trading operations.
    
    Validates trades before execution and monitors portfolio risk.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.limits = RiskLimits()
        self.state_file = PROJECT_ROOT / "data" / "risk_state.json"
        self._load_state()
        
    def _load_state(self):
        """Load risk state from file"""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.daily_pnl = data.get("daily_pnl", 0.0)
                self.daily_trades = data.get("daily_trades", 0)
                self.daily_start_balance = data.get("daily_start_balance", 500.0)
                self.open_trades = data.get("open_trades", [])
            except:
                self._reset_state()
        else:
            self._reset_state()
    
    def _reset_state(self):
        """Reset daily state"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_balance = 500.0  # Default paper balance
        self.open_trades = []
        self._save_state()
    
    def _save_state(self):
        """Save risk state to file"""
        data = {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_start_balance": self.daily_start_balance,
            "open_trades": self.open_trades,
            "last_update": datetime.now().isoformat()
        }
        self.state_file.parent.mkdir(exist_ok=True)
        self.state_file.write_text(json.dumps(data, indent=2))
    
    # =========================================================================
    # TRADE VALIDATION
    # =========================================================================
    def validate_trade(self, trade_signal: Dict) -> TradeRisk:
        """
        Validate a trade signal against risk limits.
        
        Args:
            trade_signal: Dict with 'symbol', 'direction', 'size', 'stop_loss', 'take_profit', 'confidence'
            
        Returns:
            TradeRisk with approval status and risk assessment
        """
        reasons = []
        suggestions = []
        risk_score = 0.0
        
        # Check confidence
        confidence = trade_signal.get("confidence", 0.0)
        if confidence < self.limits.min_confidence:
            reasons.append(f"Low confidence: {confidence:.0%} < {self.limits.min_confidence:.0%}")
            risk_score += 0.3
            suggestions.append("Wait for higher confidence signal")
        
        # Check position size
        size_pct = trade_signal.get("size_pct", 0.0)
        if size_pct > self.limits.max_position_pct:
            reasons.append(f"Position too large: {size_pct:.1%} > {self.limits.max_position_pct:.1%}")
            risk_score += 0.2
            suggestions.append(f"Reduce position to {self.limits.max_position_pct:.1%} of balance")
        
        # Check concurrent trades
        if len(self.open_trades) >= self.limits.max_concurrent_trades:
            reasons.append(f"Max concurrent trades reached: {len(self.open_trades)}")
            risk_score += 0.2
            suggestions.append("Close or wait for existing trades")
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance if self.daily_start_balance > 0 else 0
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            reasons.append(f"Daily loss limit reached: {daily_loss_pct:.1%}")
            risk_score += 0.3
            suggestions.append("Stop trading for today")
        
        # Calculate risk/reward ratio
        sl_pct = abs(trade_signal.get("stop_loss_pct", 0.03))
        tp_pct = abs(trade_signal.get("take_profit_pct", 0.06))
        
        if sl_pct > 0:
            rr_ratio = tp_pct / sl_pct
            if rr_ratio < 1.5:
                reasons.append(f"Low R/R ratio: {rr_ratio:.1f}:1 < 1.5:1")
                risk_score += 0.1
                suggestions.append("Wait for better R/R setup")
        
        # Determine approval
        approved = risk_score < 0.5
        
        logger.info(f"📊 Risk Assessment: {'✅ APPROVED' if approved else '❌ REJECTED'} (risk: {risk_score:.2f})")
        
        return TradeRisk(
            approved=approved,
            risk_score=risk_score,
            reasons=reasons,
            suggestions=suggestions
        )
    
    # =========================================================================
    # PORTFOLIO RISK
    # =========================================================================
    def check_portfolio_risk(self) -> Dict:
        """Check overall portfolio risk"""
        open_count = len(self.open_trades)
        total_exposure = sum(t.get("size", 0) for t in self.open_trades)
        
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance if self.daily_start_balance > 0 else 0
        
        risk_level = "LOW"
        if daily_loss_pct > 0.05 or open_count > 3:
            risk_level = "MEDIUM"
        if daily_loss_pct > 0.08 or open_count >= self.limits.max_concurrent_trades:
            risk_level = "HIGH"
        
        return {
            "risk_level": risk_level,
            "open_trades": open_count,
            "total_exposure": total_exposure,
            "daily_pnl": self.daily_pnl,
            "daily_loss_pct": daily_loss_pct,
            "recommendations": self._get_risk_recommendations(risk_level)
        }
    
    def _get_risk_recommendations(self, risk_level: str) -> List[str]:
        """Get recommendations based on risk level"""
        if risk_level == "LOW":
            return ["Trading normally", "Monitor for changes"]
        elif risk_level == "MEDIUM":
            return ["Reduce position sizes", "Be selective with signals"]
        else:  # HIGH
            return ["Consider stopping", "Review open positions", "Wait for clearer signals"]
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    def open_trade(self, trade: Dict):
        """Register an open trade"""
        self.open_trades.append(trade)
        self.daily_trades += 1
        self._save_state()
    
    def close_trade(self, trade_id: str, pnl: float):
        """Register a closed trade"""
        self.open_trades = [t for t in self.open_trades if t.get("id") != trade_id]
        self.daily_pnl += pnl
        self._save_state()
    
    def reset_daily(self):
        """Reset daily counters (call at start of each day)"""
        self._reset_state()
        logger.info("📅 Daily risk state reset")
    
    # =========================================================================
    # STATUS
    # =========================================================================
    def status(self) -> Dict:
        """Get agent status"""
        portfolio_risk = self.check_portfolio_risk()
        
        return {
            "agent": "risk_agent",
            "status": "active",
            "limits": {
                "max_position_pct": self.limits.max_position_pct,
                "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                "max_concurrent_trades": self.limits.max_concurrent_trades,
                "min_confidence": self.limits.min_confidence
            },
            "portfolio": portfolio_risk,
            "daily": {
                "trades": self.daily_trades,
                "pnl": self.daily_pnl,
                "start_balance": self.daily_start_balance
            }
        }


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Risk Agent CLI")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--check", action="store_true", help="Check portfolio risk")
    parser.add_argument("--reset", action="store_true", help="Reset daily state")
    
    args = parser.parse_args()
    
    agent = RiskAgent()
    
    if args.status:
        print(json.dumps(agent.status(), indent=2))
    elif args.check:
        print(json.dumps(agent.check_portfolio_risk(), indent=2))
    elif args.reset:
        agent.reset_daily()
    else:
        print("Risk Agent - Trading Risk Management")
        print("Usage:")
        print("  python3 risk_agent.py --status   # Show status")
        print("  python3 risk_agent.py --check    # Check portfolio risk")
        print("  python3 risk_agent.py --reset    # Reset daily state")
