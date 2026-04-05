#!/usr/bin/env python3
"""
Auto-Learning Wrapper
=====================
Wrapper simple para conectar auto-learner con el master orchestrator.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from auto_learner import AutoLearningOrchestrator, MarketState

class AutoLearningWrapper:
    """Wrapper para integrar auto-learner con cualquier sistema de trading"""
    
    def __init__(self):
        self.learner = AutoLearningOrchestrator()
        self.running = False
        
    def process_trade(self, trade_data: Dict) -> Dict:
        """
        Process a completed trade and feed to auto-learner.
        
        Args:
            trade_data: Dict with trade information
                - symbol: str
                - direction: 'long' or 'short'
                - entry_price: float
                - exit_price: float
                - pnl: float (absolute)
                - pnl_pct: float (percentage)
                - win: bool
                - holding_time: float (seconds)
                - rsi: float (optional)
                - ema_diff: float (optional)
                - volatility: float (optional)
                
        Returns:
            Updated parameters if changed
        """
        # Create market state
        market_state = {
            "rsi": trade_data.get('rsi', 50),
            "ema_diff": trade_data.get('ema_diff', 0),
            "volatility": trade_data.get('volatility', 0.02),
            "volume_ratio": trade_data.get('volume_ratio', 1.0),
            "price_momentum": trade_data.get('price_momentum', 0),
            "bb_position": trade_data.get('bb_position', 0.5),
            "trend_strength": trade_data.get('trend_strength', 0),
            "recent_pnl": self.learner.daily_pnl,
            "trade_count": self.learner.total_trades,
            "time_of_day": datetime.now().hour / 24
        }
        
        # Record to learner
        self.learner.record_trade_result({
            "trade_id": trade_data.get('trade_id', f"trade_{datetime.now().timestamp()}"),
            "strategy_id": trade_data.get('strategy_id', 'default'),
            "pnl": trade_data['pnl'],
            "pnl_pct": trade_data['pnl_pct'],
            "win": trade_data['win'],
            "holding_time": trade_data.get('holding_time', 0),
            "market_state": market_state,
            "confidence": trade_data.get('confidence', 0.5),
            "metadata": {
                "symbol": trade_data.get('symbol', 'unknown'),
                "direction": trade_data.get('direction', 'long'),
                "entry_price": trade_data.get('entry_price'),
                "exit_price": trade_data.get('exit_price')
            }
        })
        
        # Get updated parameters
        params = self.learner.param_optimizer.get_current_parameters()
        
        return {
            "status": "recorded",
            "total_trades": self.learner.total_trades,
            "daily_pnl": self.learner.daily_pnl,
            "current_params": params
        }
    
    def should_trade(self, market_data: Dict) -> Tuple[bool, float, str]:
        """
        Determine if we should trade based on RL.
        
        Args:
            market_data: Dict with market indicators
                - rsi: float
                - ema_fast: float
                - ema_slow: float
                - close: float
                - volume: float
                - avg_volume: float
                
        Returns:
            (should_trade, confidence, direction)
        """
        # Create market state
        state = MarketState(
            rsi=market_data.get('rsi', 50),
            ema_diff=(market_data.get('ema_fast', 0) - market_data.get('ema_slow', 0)) / market_data.get('close', 1),
            volatility=market_data.get('volatility', 0.02),
            volume_ratio=market_data.get('volume', 1) / market_data.get('avg_volume', 1),
            price_momentum=market_data.get('momentum', 0),
            bb_position=market_data.get('bb_position', 0.5),
            trend_strength=abs(market_data.get('ema_diff', 0)) * 100,
            recent_pnl=self.learner.daily_pnl,
            trade_count=self.learner.total_trades,
            time_of_day=datetime.now().hour / 24
        )
        
        # Get RL decision
        should_trade, confidence, direction = self.learner.should_trade(state)
        
        return should_trade, confidence, direction
    
    def get_current_parameters(self) -> Dict:
        """Get current adaptive parameters"""
        return self.learner.param_optimizer.get_current_parameters()
    
    def get_status(self) -> Dict:
        """Get full learning status"""
        return self.learner.get_status()
    
    def start(self):
        """Start the auto-learning system"""
        self.learner.start()
        self.running = True
        
    def stop(self):
        """Stop the auto-learning system"""
        self.learner.stop()
        self.running = False


# Singleton instance
_wrapper_instance = None

def get_wrapper() -> AutoLearningWrapper:
    """Get singleton wrapper instance"""
    global _wrapper_instance
    if _wrapper_instance is None:
        _wrapper_instance = AutoLearningWrapper()
    return _wrapper_instance


if __name__ == "__main__":
    print("="*80)
    print("AUTO-LEARNING WRAPPER")
    print("="*80)
    
    # Test wrapper
    wrapper = AutoLearningWrapper()
    
    print("\n📊 Estado inicial:")
    status = wrapper.get_status()
    print(f"  Generación: {status['current_generation']}")
    print(f"  Trades: {status['total_trades']}")
    
    # Simulate trade
    print("\n🧪 Simulando trade de prueba...")
    result = wrapper.process_trade({
        "symbol": "SOL",
        "direction": "long",
        "entry_price": 100.0,
        "exit_price": 102.5,
        "pnl": 25.0,
        "pnl_pct": 0.025,
        "win": True,
        "holding_time": 3600,
        "rsi": 45,
        "ema_diff": 0.01
    })
    
    print(f"  Status: {result['status']}")
    print(f"  Total trades: {result['total_trades']}")
    print(f"  Daily PnL: {result['daily_pnl']:.2%}")
    
    # Test decision
    print("\n🎯 Probando decisión de trading...")
    should, conf, direction = wrapper.should_trade({
        "rsi": 35,
        "ema_fast": 102,
        "ema_slow": 100,
        "close": 101,
        "volume": 1000,
        "avg_volume": 900
    })
    
    print(f"  Should trade: {should}")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Direction: {direction}")
    
    print("\n✅ Wrapper funcionando correctamente")
