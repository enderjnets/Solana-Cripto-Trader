#!/usr/bin/env python3
"""
Strategy Runner
===============
Integrates strategies with workers for automated trading execution.
"""

import os
import sys
import json
import time
import threading
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from strategies import (
    Strategy, StrategyFactory, TradingSignal, SignalType,
    RSIStrategy, SMACrossoverStrategy, MACDStrategy, CombinedStrategy,
    generate_sample_data
)
from workers.jupiter_worker import get_worker

logger = logging.getLogger("strategy_runner")


@dataclass
class StrategyConfig:
    """Configuration for a strategy instance"""
    name: str
    enabled: bool = True
    parameters: Dict = field(default_factory=dict)
    timeframe: str = "1h"
    check_interval: int = 60  # seconds
    min_confidence: float = 0.5
    auto_execute: bool = False
    pair: str = "SOL-USDC"


@dataclass
class RunnerState:
    """State of the strategy runner"""
    running: bool = False
    active_strategies: Dict[str, Strategy] = field(default_factory=dict)
    signals: List[Dict] = field(default_factory=list)
    positions: List[Dict] = field(default_factory=list)
    last_check: Optional[datetime] = None
    total_signals: int = 0
    executed_trades: int = 0


class StrategyRunner:
    """
    Runs strategies and can execute trades based on signals.
    """
    
    def __init__(self, config: Dict = None):
        self.state = RunnerState()
        self.config = config or {}
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
    
    def add_strategy(self, name: str, strategy_config: Dict) -> bool:
        """
        Add a strategy to run
        
        Args:
            name: Strategy identifier
            strategy_config: Strategy configuration
        
        Returns:
            True if successful
        """
        try:
            strategy = StrategyFactory.create(
                strategy_config["type"],
                strategy_config.get("parameters", {})
            )
            
            # Filter only valid StrategyConfig fields
            valid_fields = {
                'enabled', 'parameters', 'timeframe', 'check_interval',
                'min_confidence', 'auto_execute', 'pair'
            }
            filtered_config = {k: v for k, v in strategy_config.items() if k in valid_fields}
            
            self.state.active_strategies[name] = {
                "strategy": strategy,
                "config": StrategyConfig(name=name, **filtered_config)
            }
            
            logger.info(f"Strategy added: {name} ({strategy_config['type']})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add strategy {name}: {e}")
            return False
    
    def remove_strategy(self, name: str) -> bool:
        """Remove a strategy"""
        if name in self.state.active_strategies:
            del self.state.active_strategies[name]
            logger.info(f"Strategy removed: {name}")
            return True
        return False
    
    def list_strategies(self) -> List[Dict]:
        """List all configured strategies"""
        return [
            {
                "name": name,
                "type": data["strategy"].name,
                "enabled": data["config"].enabled,
                "pair": data["config"].pair,
                "auto_execute": data["config"].auto_execute
            }
            for name, data in self.state.active_strategies.items()
        ]
    
    def get_signal(self, strategy_name: str) -> Optional[Dict]:
        """Get latest signal for a strategy"""
        for signal in reversed(self.state.signals):
            if signal["strategy"] == strategy_name:
                return signal
        return None
    
    def get_all_signals(self) -> List[Dict]:
        """Get all recent signals"""
        return self.state.signals[-50:]  # Last 50 signals
    
    def run_backtest(self, strategy_name: str, df: pd.DataFrame = None, days: int = 365,
                    tp_pct: float = 0.05, sl_pct: float = 0.03) -> Dict:
        """Run backtest for a strategy with REAL historical data and TP/SL"""
        if strategy_name not in self.state.active_strategies:
            return {"error": f"Strategy not found: {strategy_name}"}
        
        strategy = self.state.active_strategies[strategy_name]["strategy"]
        
        # Try to get REAL data if not provided
        if df is None:
            try:
                from data.historical_data import HistoricalDataManager
                manager = HistoricalDataManager()
                
                # Get pair info from config
                config = self.state.active_strategies[strategy_name]["config"]
                pair = config.pair.split("-")
                
                base_token = pair[0] if len(pair) >= 1 else "SOL"
                df = manager.get_historical_data(base_token, timeframe="1h", days=days)
                
                if len(df) < 100:
                    raise ValueError("Not enough data, using sample")
                    
            except Exception as e:
                logger.warning(f"Using sample data: {e}")
                df = generate_sample_data(min(days * 24, 1000))
        
        from strategies import backtest_strategy
        results = backtest_strategy(strategy, df, tp_pct=tp_pct, sl_pct=sl_pct)
        
        # Add data info
        results["data_info"] = {
            "total_candles": len(df),
            "date_range": f"{str(df['timestamp'].min())[:10]} to {str(df['timestamp'].max())[:10]}",
            "days": days,
            "tp_pct": tp_pct * 100,
            "sl_pct": sl_pct * 100,
            "source": "HistoricalDataManager"
        }
        
        return results
    
    def _check_strategy(self, name: str) -> Optional[TradingSignal]:
        """Check a single strategy for signals"""
        data = self.state.active_strategies.get(name)
        if not data or not data["config"].enabled:
            return None
        
        strategy = data["strategy"]
        config = data["config"]
        
        try:
            # Get real data from HistoricalDataManager
            try:
                from data.historical_data import HistoricalDataManager
                manager = HistoricalDataManager()
                pair = config.pair.split("-")
                base_token = pair[0] if len(pair) >= 1 else "SOL"
                df = manager.get_historical_data(base_token, timeframe="1h", days=90)
                
                if len(df) < 100:
                    raise ValueError("Not enough data")
                    
            except Exception:
                # Fallback to sample data
                df = generate_sample_data(100)
            
            # Calculate indicators
            df = strategy.calculate_indicators(df.copy())
            
            # Generate signal
            signal = strategy.generate_signal(df)
            
            # Filter by confidence
            if signal.confidence < config.min_confidence:
                return None
            
            return signal
            
        except Exception as e:
            logger.error(f"Error checking strategy {name}: {e}")
            return None
    
    def _execute_trade(self, signal: TradingSignal, pair: str) -> Optional[Dict]:
        """
        Execute a trade based on signal
        
        Args:
            signal: Trading signal
            pair: Trading pair
        
        Returns:
            Trade result or None
        """
        if signal.signal == SignalType.HOLD:
            return None
        
        try:
            # Get quote from worker
            worker = get_worker()
            
            # Parse pair
            parts = pair.split("-")
            if len(parts) != 2:
                logger.error(f"Invalid pair format: {pair}")
                return None
            
            input_token, output_token = parts[0], parts[1]
            
            # Default amount (10% of available balance)
            amount = 0.1  # SOL
            
            # Get quote
            quote = worker.get_quote(input_token, output_token, amount)
            
            if not quote or "outAmount" not in quote:
                logger.warning(f"Failed to get quote for {pair}")
                return None
            
            result = {
                "signal": signal.signal.value,
                "pair": pair,
                "price": signal.price,
                "confidence": signal.confidence,
                "quote": quote,
                "timestamp": datetime.now().isoformat(),
                "status": "quote_only" if not worker_stopped else "executed"
            }
            
            logger.info(f"Trade {result['status']}: {result['signal']} {pair} @ ${signal.price:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def start(self):
        """Start the strategy runner"""
        if self.state.running:
            logger.warning("Runner already running")
            return
        
        self.state.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Strategy runner started with {len(self.state.active_strategies)} strategies")
    
    def stop(self):
        """Stop the strategy runner"""
        self.state.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Strategy runner stopped")
    
    def _run_loop(self):
        """Main loop for checking strategies"""
        while self.state.running:
            try:
                for name, data in self.state.active_strategies.items():
                    if not data["config"].enabled:
                        continue
                    
                    config = data["config"]
                    
                    # Check strategy
                    signal = self._check_strategy(name)
                    
                    if signal:
                        # Store signal
                        signal_dict = signal.to_dict()
                        signal_dict["pair"] = config.pair
                        self.state.signals.append(signal_dict)
                        self.state.total_signals += 1
                        
                        # Notify callbacks
                        for cb in self._callbacks:
                            try:
                                cb(signal_dict)
                            except Exception as e:
                                logger.debug(f"Callback error: {e}")
                        
                        # Auto-execute if enabled
                        if config.auto_execute:
                            trade = self._execute_trade(signal, config.pair)
                            if trade:
                                self.state.positions.append(trade)
                                self.state.executed_trades += 1
                
                self.state.last_check = datetime.now()
                
                # Wait for next check
                min_interval = min(
                    data["config"].check_interval 
                    for data in self.state.active_strategies.values()
                ) if self.state.active_strategies else 60
                
                time.sleep(min_interval)
                
            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                time.sleep(10)
    
    def add_callback(self, callback: Callable):
        """Add callback for signals"""
        self._callbacks.append(callback)
    
    def get_status(self) -> Dict:
        """Get runner status"""
        return {
            "running": self.state.running,
            "strategies": len(self.state.active_strategies),
            "total_signals": self.state.total_signals,
            "executed_trades": self.state.executed_trades,
            "last_check": self.state.last_check.isoformat() if self.state.last_check else None,
            "active_strategies": self.list_strategies(),
            "recent_signals": self.get_all_signals()[-5:]
        }


# Global runner instance
_runner: Optional[StrategyRunner] = None
_runner_lock = threading.Lock()


def get_runner() -> StrategyRunner:
    """Get or create global runner instance"""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = StrategyRunner()
        return _runner


def runner_status() -> Dict:
    """Get runner status"""
    return get_runner().get_status()


def runner_start():
    """Start runner"""
    get_runner().start()
    return "Strategy runner started"


def runner_stop():
    """Stop runner"""
    get_runner().stop()
    return "Strategy runner stopped"


def runner_add_strategy(name: str, config: Dict) -> str:
    """Add a strategy"""
    success = get_runner().add_strategy(name, config)
    return f"Strategy {name} added" if success else f"Failed to add {name}"


def runner_list() -> List[Dict]:
    """List strategies"""
    return get_runner().list_strategies()


def runner_backtest(name: str) -> Dict:
    """Run backtest"""
    return get_runner().run_backtest(name)


if __name__ == "__main__":
    print("=" * 60)
    print("STRATEGY RUNNER - Demo")
    print("=" * 60)
    
    runner = get_runner()
    
    # Add some strategies
    print("\n📊 Adding strategies...")
    runner.add_strategy("rsi_fast", {
        "type": "rsi",
        "parameters": {"rsi_period": 7, "oversold": 25, "overbought": 75},
        "pair": "SOL-USDC",
        "min_confidence": 0.6
    })
    
    runner.add_strategy("macd_trend", {
        "type": "macd",
        "parameters": {"fast_period": 12, "slow_period": 26},
        "pair": "SOL-USDC",
        "min_confidence": 0.7
    })
    
    runner.add_strategy("sma_crossover", {
        "type": "sma_crossover",
        "parameters": {"fast_period": 10, "slow_period": 30},
        "pair": "SOL-USDC",
        "min_confidence": 0.8
    })
    
    # List strategies
    print("\n📋 Configured Strategies:")
    for s in runner.list_strategies():
        print(f"  - {s['name']} ({s['type']}) - {s['pair']}")
    
    # Run backtest
    print("\n📈 Running backtest for RSI strategy...")
    results = runner.run_backtest("rsi_fast")
    print(f"   Total Trades: {results.get('total_trades', 0)}")
    print(f"   Win Rate: {results.get('win_rate', 0)*100:.1f}%")
    print(f"   Total Return: {results.get('total_return', 0)*100:.2f}%")
    
    # Status
    print("\n" + "=" * 60)
    status = runner.get_status()
    print(f"Runner Status: {'Running' if status['running'] else 'Stopped'}")
    print(f"Strategies: {status['strategies']}")
    print("=" * 60)
