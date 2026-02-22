#!/usr/bin/env python3
"""
Auto-Improvement Module for Trading System
==========================================
Automatically improves trading strategy based on performance metrics.

Features:
- Tracks performance metrics (win rate, P&L, trades)
- Retrains models periodically
- Adjusts parameters based on results
- Saves best configurations

Usage:
    from auto_improver import AutoImprover
    improver = AutoImprover()
    improver.record_trade(result)
    improver.should_retrain()  # Check if should retrain
    new_params = improver.get_best_params()
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Record of a single trade for analysis."""
    symbol: str
    direction: str  # bullish/bearish
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    pnl_percent: float
    duration_seconds: int
    confidence: float
    timestamp: str

@dataclass
class PerformanceMetrics:
    """Current performance metrics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_win: float = 0.0
    worst_loss: float = 0.0
    avg_confidence: float = 0.0
    avg_duration: float = 0.0
    win_rate: float = 0.0
    avg_pnl_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceMetrics':
        return cls(**data)

class AutoImprover:
    """
    Auto-improvement system for trading strategies.
    
    Tracks performance and automatically adjusts:
    - Position sizing based on win rate
    - Stop loss / take profit levels
    - Confidence thresholds
    - Risk parameters
    """
    
    def __init__(self, state_file: str = "auto_improve_state.json"):
        self.state_file = state_file
        self.trades: List[TradeRecord] = []
        self.metrics = PerformanceMetrics()
        self.best_metrics = PerformanceMetrics()
        self.cycles_since_retrain = 0
        self.retrain_interval = 20  # Retrain every 20 cycles
        self.best_params = self._load_best_params()
        self.load_state()
        
    def _load_best_params(self) -> Dict:
        """Load best known parameters."""
        default = {
            "min_confidence": 0.25,
            "position_size_pct": 0.10,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "max_positions": 5,
            "min_win_rate": 0.50,
        }
        
        if os.path.exists("best_params.json"):
            try:
                with open("best_params.json") as f:
                    return json.load(f)
            except:
                pass
        return default
    
    def _save_best_params(self):
        """Save best parameters."""
        with open("best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        logger.info(f"💾 Saved best params: {self.best_params}")
    
    def load_state(self):
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data.get('trades', [])]
                    self.metrics = PerformanceMetrics.from_dict(data.get('metrics', {}))
                    self.cycles_since_retrain = data.get('cycles_since_retrain', 0)
                    self.best_metrics = PerformanceMetrics.from_dict(data.get('best_metrics', {}))
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def save_state(self):
        """Save state to file."""
        data = {
            'trades': [asdict(t) for t in self.trades[-100:]],  # Keep last 100
            'metrics': self.metrics.to_dict(),
            'cycles_since_retrain': self.cycles_since_retrain,
            'best_metrics': self.best_metrics.to_dict(),
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def record_trade(self, trade_data: Dict):
        """Record a completed trade."""
        trade = TradeRecord(
            symbol=trade_data.get('symbol', 'UNKNOWN'),
            direction=trade_data.get('direction', 'bullish'),
            entry_price=trade_data.get('entry_price', 0),
            exit_price=trade_data.get('exit_price', 0),
            size_usd=trade_data.get('size_usd', 0),
            pnl=trade_data.get('pnl', 0),
            pnl_percent=trade_data.get('pnl_percent', 0),
            duration_seconds=trade_data.get('duration_seconds', 0),
            confidence=trade_data.get('confidence', 0),
            timestamp=datetime.now().isoformat()
        )
        
        self.trades.append(trade)
        self._update_metrics()
        self.save_state()
        
        logger.info(f"📊 Recorded trade: {trade.symbol} {trade.pnl:+.2f} ({trade.pnl_percent:+.2f}%)")
        
        # Check if this is a new best
        if self.metrics.win_rate > self.best_metrics.win_rate:
            self.best_metrics = PerformanceMetrics(
                total_trades=self.metrics.total_trades,
                wins=self.metrics.wins,
                losses=self.metrics.losses,
                total_pnl=self.metrics.total_pnl,
                avg_win=self.metrics.avg_win,
                avg_loss=self.metrics.avg_loss,
                best_win=self.metrics.best_win,
                worst_loss=self.metrics.worst_loss,
                avg_confidence=self.metrics.avg_confidence,
                avg_duration=self.metrics.avg_duration,
                win_rate=self.metrics.win_rate,
                avg_pnl_percent=self.metrics.avg_pnl_percent
            )
            self._adjust_params()
            self._save_best_params()
    
    def _update_metrics(self):
        """Update performance metrics from trades."""
        if not self.trades:
            return
            
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        self.metrics.total_trades = len(self.trades)
        self.metrics.wins = len(wins)
        self.metrics.losses = len(losses)
        self.metrics.win_rate = len(wins) / len(self.trades) if self.trades else 0
        self.metrics.total_pnl = sum(t.pnl for t in self.trades)
        
        if wins:
            self.metrics.avg_win = sum(t.pnl for t in wins) / len(wins)
            self.metrics.best_win = max(t.pnl for t in wins)
        
        if losses:
            self.metrics.avg_loss = sum(t.pnl for t in losses) / len(losses)
            self.metrics.worst_loss = min(t.pnl for t in losses)
        
        if self.trades:
            self.metrics.avg_confidence = sum(t.confidence for t in self.trades) / len(self.trades)
            self.metrics.avg_duration = sum(t.duration_seconds for t in self.trades) / len(self.trades)
            self.metrics.avg_pnl_percent = sum(t.pnl_percent for t in self.trades) / len(self.trades)
    
    def _adjust_params(self):
        """Adjust parameters based on best performance."""
        wr = self.best_metrics.win_rate
        
        # Adjust confidence threshold based on win rate
        if wr >= 0.70:
            self.best_params['min_confidence'] = 0.20  # More aggressive
            self.best_params['position_size_pct'] = 0.15
        elif wr >= 0.60:
            self.best_params['min_confidence'] = 0.25
            self.best_params['position_size_pct'] = 0.12
        elif wr >= 0.50:
            self.best_params['min_confidence'] = 0.30
            self.best_params['position_size_pct'] = 0.10
        else:
            self.best_params['min_confidence'] = 0.35  # More conservative
            self.best_params['position_size_pct'] = 0.08
        
        # Adjust risk based on avg P&L
        if self.best_metrics.avg_pnl_percent > 2:
            self.best_params['take_profit_pct'] = 0.08
            self.best_params['stop_loss_pct'] = 0.025
        elif self.best_metrics.avg_pnl_percent > 0:
            self.best_params['take_profit_pct'] = 0.06
            self.best_params['stop_loss_pct'] = 0.03
        else:
            self.best_params['take_profit_pct'] = 0.05
            self.best_params['stop_loss_pct'] = 0.02
        
        logger.info(f"🔧 Adjusted params: win_rate={wr:.1%}, params={self.best_params}")
    
    def should_retrain(self, cycle_count: int) -> bool:
        """Check if should retrain model."""
        self.cycles_since_retrain = cycle_count
        return cycle_count >= self.retrain_interval
    
    def get_best_params(self) -> Dict:
        """Get current best parameters."""
        return self.best_params.copy()
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics."""
        return self.metrics
    
    def get_summary(self) -> str:
        """Get summary string."""
        return (
            f"📊 Auto-Improve Summary:\n"
            f"   Trades: {self.metrics.total_trades}\n"
            f"   Win Rate: {self.metrics.win_rate:.1%}\n"
            f"   Total P&L: ${self.metrics.total_pnl:.2f}\n"
            f"   Avg P&L: {self.metrics.avg_pnl_percent:+.2f}%\n"
            f"   Cycles: {self.cycles_since_retrain}/{self.retrain_interval}\n"
            f"   Best WR: {self.best_metrics.win_rate:.1%}"
        )


# Test
if __name__ == "__main__":
    improver = AutoImprover()
    
    # Simulate some trades
    for i in range(10):
        improver.record_trade({
            'symbol': 'SOL',
            'direction': 'bullish',
            'entry_price': 100,
            'exit_price': 105 if i < 6 else 98,
            'size_usd': 100,
            'pnl': 5 if i < 6 else -2,
            'pnl_percent': 5 if i < 6 else -2,
            'duration_seconds': 300,
            'confidence': 0.7
        })
    
    print(improver.get_summary())
    print(f"\nBest params: {improver.get_best_params()}")
