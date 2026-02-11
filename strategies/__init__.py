#!/usr/bin/env python3
"""
Trading Strategies Module
=========================
Strategies for automated trading on Solana/Jupiter.

Features:
- RSI-based strategies
- Moving Average Crossovers
- MACD Strategy
- Bollinger Bands
- Support/Resistance
- Custom multi-indicator strategies
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from enum import Enum

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("strategies")


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TradingSignal:
    """Trading signal output"""
    signal: SignalType
    strategy: str
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "signal": self.signal.value,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Position:
    """Open position"""
    position_type: PositionType
    entry_price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class Strategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.required_indicators = []
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators"""
        pass
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        """Generate trading signal"""
        pass
    
    def before_run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-processing before signal generation"""
        return df
    
    def after_run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processing"""
        return df
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate strategy conditions"""
        if len(df) < 50:
            return False, f"Insufficient data: {len(df)} candles"
        return True, "OK"


class RSIStrategy(Strategy):
    """
    RSI-based strategy
    Buy when oversold, sell when overbought
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "exit_oversold": 40,
            "exit_overbought": 60,
        }
        config = {**default_config, **(config or {})}
        super().__init__("RSI Strategy", config)
        self.required_indicators = ["rsi"]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.config["rsi_period"]
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        rsi = df['rsi'].iloc[-1]
        price = df['close'].iloc[-1]
        
        if rsi <= self.config["oversold"]:
            return TradingSignal(
                signal=SignalType.BUY,
                strategy=self.name,
                confidence=min(1.0, (30 - rsi) / 30),
                price=price,
                timestamp=datetime.now(),
                metadata={"rsi": rsi, "threshold": self.config["oversold"]}
            )
        elif rsi >= self.config["overbought"]:
            return TradingSignal(
                signal=SignalType.SELL,
                strategy=self.name,
                confidence=min(1.0, (rsi - 70) / 30),
                price=price,
                timestamp=datetime.now(),
                metadata={"rsi": rsi, "threshold": self.config["overbought"]}
            )
        
        return TradingSignal(
            signal=SignalType.HOLD,
            strategy=self.name,
            confidence=0.0,
            price=price,
            timestamp=datetime.now(),
            metadata={"rsi": rsi}
        )


class SMACrossoverStrategy(Strategy):
    """
    Simple Moving Average Crossover
    Buy when short SMA crosses above long SMA
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            "fast_period": 10,
            "slow_period": 30,
            "confirmation_bars": 2,
        }
        config = {**default_config, **(config or {})}
        super().__init__("SMA Crossover", config)
        self.required_indicators = ["sma_fast", "sma_slow"]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.config["fast_period"]
        slow = self.config["slow_period"]
        
        df['sma_fast'] = df['close'].rolling(window=fast).mean()
        df['sma_slow'] = df['close'].rolling(window=slow).mean()
        
        # Trend direction
        df['sma_trend'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        price = df['close'].iloc[-1]
        current_trend = df['sma_trend'].iloc[-1]
        prev_trend = df['sma_trend'].iloc[-2] if len(df) > 1 else current_trend
        
        if prev_trend == -1 and current_trend == 1:
            return TradingSignal(
                signal=SignalType.BUY,
                strategy=self.name,
                confidence=0.8,
                price=price,
                timestamp=datetime.now(),
                metadata={
                    "sma_fast": df['sma_fast'].iloc[-1],
                    "sma_slow": df['sma_slow'].iloc[-1]
                }
            )
        elif prev_trend == 1 and current_trend == -1:
            return TradingSignal(
                signal=SignalType.SELL,
                strategy=self.name,
                confidence=0.8,
                price=price,
                timestamp=datetime.now(),
                metadata={
                    "sma_fast": df['sma_fast'].iloc[-1],
                    "sma_slow": df['sma_slow'].iloc[-1]
                }
            )
        
        return TradingSignal(
            signal=SignalType.HOLD,
            strategy=self.name,
            confidence=0.0,
            price=price,
            timestamp=datetime.now(),
            metadata={"trend": "bullish" if current_trend == 1 else "bearish"}
        )


class EMACrossoverStrategy(Strategy):
    """
    Exponential Moving Average Crossover
    Faster response than SMA
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        }
        config = {**default_config, **(config or {})}
        super().__init__("EMA Crossover", config)
        self.required_indicators = ["ema_fast", "ema_slow"]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.config["fast_period"]
        slow = self.config["slow_period"]
        
        df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        
        # Distance from EMA
        df['ema_distance'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow'] * 100
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        price = df['close'].iloc[-1]
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        distance = df['ema_distance'].iloc[-1]
        
        if ema_fast > ema_slow and distance > 0.5:
            return TradingSignal(
                signal=SignalType.BUY,
                strategy=self.name,
                confidence=min(0.9, abs(distance) / 2),
                price=price,
                timestamp=datetime.now(),
                metadata={"ema_fast": ema_fast, "ema_slow": ema_slow}
            )
        elif ema_fast < ema_slow and distance < -0.5:
            return TradingSignal(
                signal=SignalType.SELL,
                strategy=self.name,
                confidence=min(0.9, abs(distance) / 2),
                price=price,
                timestamp=datetime.now(),
                metadata={"ema_fast": ema_fast, "ema_slow": ema_slow}
            )
        
        return TradingSignal(
            signal=SignalType.HOLD,
            strategy=self.name,
            confidence=0.0,
            price=price,
            timestamp=datetime.now(),
            metadata={"ema_distance": distance}
        )


class MACDStrategy(Strategy):
    """
    MACD Strategy (Moving Average Convergence Divergence)
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        }
        config = {**default_config, **(config or {})}
        super().__init__("MACD Strategy", config)
        self.required_indicators = ["macd", "macd_signal", "macd_hist"]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = self.config["fast_period"]
        slow = self.config["slow_period"]
        signal = self.config["signal_period"]
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        price = df['close'].iloc[-1]
        macd = df['macd'].iloc[-1]
        signal = df['macd_signal'].iloc[-1]
        hist = df['macd_hist'].iloc[-1]
        prev_hist = df['macd_hist'].iloc[-2] if len(df) > 1 else hist
        
        # Golden Cross (MACD crosses above signal)
        if prev_hist < 0 and hist > 0 and macd > signal:
            return TradingSignal(
                signal=SignalType.BUY,
                strategy=self.name,
                confidence=0.85,
                price=price,
                timestamp=datetime.now(),
                metadata={"macd": macd, "signal": signal, "hist": hist}
            )
        
        # Death Cross (MACD crosses below signal)
        if prev_hist > 0 and hist < 0 and macd < signal:
            return TradingSignal(
                signal=SignalType.SELL,
                strategy=self.name,
                confidence=0.85,
                price=price,
                timestamp=datetime.now(),
                metadata={"macd": macd, "signal": signal, "hist": hist}
            )
        
        return TradingSignal(
            signal=SignalType.HOLD,
            strategy=self.name,
            confidence=0.0,
            price=price,
            timestamp=datetime.now(),
            metadata={"macd_hist": hist}
        )


class BollingerBandStrategy(Strategy):
    """
    Bollinger Bands Strategy
    Buy at lower band, sell at upper band
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            "period": 20,
            "std_dev": 2.0,
            "exit_multiplier": 1.5,
        }
        config = {**default_config, **(config or {})}
        super().__init__("Bollinger Bands", config)
        self.required_indicators = ["bb_upper", "bb_middle", "bb_lower", "bb_width"]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.config["period"]
        std = self.config["std_dev"]
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (std * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        price = df['close'].iloc[-1]
        bb_pos = df['bb_position'].iloc[-1]
        
        # Buy at lower band with bounce
        if bb_pos < 0.1:
            return TradingSignal(
                signal=SignalType.BUY,
                strategy=self.name,
                confidence=0.8,
                price=price,
                timestamp=datetime.now(),
                metadata={"bb_position": bb_pos, "lower": df['bb_lower'].iloc[-1]}
            )
        
        # Sell at upper band
        if bb_pos > 0.9:
            return TradingSignal(
                signal=SignalType.SELL,
                strategy=self.name,
                confidence=0.8,
                price=price,
                timestamp=datetime.now(),
                metadata={"bb_position": bb_pos, "upper": df['bb_upper'].iloc[-1]}
            )
        
        # Middle band bounce
        if bb_pos < 0.3:
            return TradingSignal(
                signal=SignalType.BUY,
                strategy=self.name,
                confidence=0.5,
                price=price,
                timestamp=datetime.now(),
                metadata={"bb_position": bb_pos, "note": "bounce"}
            )
        elif bb_pos > 0.7:
            return TradingSignal(
                signal=SignalType.SELL,
                strategy=self.name,
                confidence=0.5,
                price=price,
                timestamp=datetime.now(),
                metadata={"bb_position": bb_pos, "note": "reversal"}
            )
        
        return TradingSignal(
            signal=SignalType.HOLD,
            strategy=self.name,
            confidence=0.0,
            price=price,
            timestamp=datetime.now(),
            metadata={"bb_position": bb_pos}
        )


class CombinedStrategy(Strategy):
    """
    Multi-indicator combined strategy
    Requires multiple confirmations for higher confidence
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            "require_rsi": True,
            "require_sma": True,
            "require_macd": False,
            "min_confidence": 0.6,
        }
        config = {**default_config, **(config or {})}
        super().__init__("Combined Strategy", config)
        self.required_indicators = ["rsi", "sma_fast", "sma_slow"]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI
        period = 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # SMA
        df['sma_fast'] = df['close'].rolling(window=10).mean()
        df['sma_slow'] = df['close'].rolling(window=30).mean()
        df['sma_trend'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> TradingSignal:
        price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        sma_trend = df['sma_trend'].iloc[-1]
        
        buy_score = 0.0
        sell_score = 0.0
        reasons = []
        
        # RSI analysis
        if self.config["require_rsi"]:
            if rsi < 30:
                buy_score += 0.4
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                sell_score += 0.4
                reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # SMA analysis
        if self.config["require_sma"]:
            if sma_trend == 1:
                buy_score += 0.3
                reasons.append("SMA bullish")
            else:
                sell_score += 0.3
                reasons.append("SMA bearish")
        
        # Determine signal
        if buy_score >= self.config["min_confidence"]:
            return TradingSignal(
                signal=SignalType.BUY,
                strategy=self.name,
                confidence=buy_score,
                price=price,
                timestamp=datetime.now(),
                metadata={"buy_score": buy_score, "reasons": reasons}
            )
        elif sell_score >= self.config["min_confidence"]:
            return TradingSignal(
                signal=SignalType.SELL,
                strategy=self.name,
                confidence=sell_score,
                price=price,
                timestamp=datetime.now(),
                metadata={"sell_score": sell_score, "reasons": reasons}
            )
        
        return TradingSignal(
            signal=SignalType.HOLD,
            strategy=self.name,
            confidence=0.0,
            price=price,
            timestamp=datetime.now(),
            metadata={"buy_score": buy_score, "sell_score": sell_score}
        )


# Strategy Factory
class StrategyFactory:
    """Factory for creating strategies"""
    
    _strategies = {
        "rsi": RSIStrategy,
        "sma_crossover": SMACrossoverStrategy,
        "ema_crossover": EMACrossoverStrategy,
        "macd": MACDStrategy,
        "bollinger": BollingerBandStrategy,
        "combined": CombinedStrategy,
    }
    
    @classmethod
    def create(cls, strategy_name: str, config: Dict = None) -> Strategy:
        """Create a strategy by name"""
        if strategy_name.lower() not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return cls._strategies[strategy_name.lower()](config)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategies"""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls, name: str) -> Dict:
        """Get strategy information"""
        strategies = {
            "rsi": {
                "name": "RSI Strategy",
                "description": "Buy oversold, sell overbought",
                "parameters": ["rsi_period", "oversold", "overbought"],
                "timeframe": "1h or 4h recommended"
            },
            "sma_crossover": {
                "name": "SMA Crossover",
                "description": "Golden/death cross of moving averages",
                "parameters": ["fast_period", "slow_period"],
                "timeframe": "4h or 1d recommended"
            },
            "ema_crossover": {
                "name": "EMA Crossover",
                "description": "Faster moving average crossovers",
                "parameters": ["fast_period", "slow_period", "signal_period"],
                "timeframe": "1h or 4h recommended"
            },
            "macd": {
                "name": "MACD Strategy",
                "description": "MACD crossover with histogram",
                "parameters": ["fast_period", "slow_period", "signal_period"],
                "timeframe": "4h or 1d recommended"
            },
            "bollinger": {
                "name": "Bollinger Bands",
                "description": "Mean reversion at bands",
                "parameters": ["period", "std_dev"],
                "timeframe": "1h recommended"
            },
            "combined": {
                "name": "Combined Strategy",
                "description": "Multi-indicator with confirmations",
                "parameters": ["require_rsi", "require_sma", "min_confidence"],
                "timeframe": "4h recommended"
            }
        }
        return strategies.get(name, {})


def generate_sample_data(n_candles: int = 1000) -> pd.DataFrame:
    """Generate realistic sample OHLCV data for testing"""
    import numpy as np
    
    np.random.seed(42)
    
    # Generate price with trends
    t = np.arange(n_candles)
    trend = np.sin(t / 50) * 0.02 + t / 500 * 0.001
    noise = np.random.randn(n_candles) * 0.02
    
    close = 100 * np.exp(trend + noise)
    
    # OHLC
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_candles, freq='1h'),
        'open': close * (1 + np.random.randn(n_candles) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n_candles)) * 0.01),
        'low': close * (1 - np.abs(np.random.randn(n_candles)) * 0.01),
        'close': close,
        'volume': np.abs(np.random.randn(n_candles) * 1000) + 500
    })
    
    # Ensure high > low
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + np.random.rand(n_candles) * 0.01
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - np.random.rand(n_candles) * 0.01
    
    return df


def backtest_strategy(strategy: Strategy, df: pd.DataFrame, 
                                   tp_pct: float = 0.05, sl_pct: float = 0.03) -> Dict:
    """
    Backtest with Take Profit and Stop Loss
    
    Args:
        strategy: Strategy to test
        df: Price data  
        tp_pct: Take profit percentage (default 5%)
        sl_pct: Stop loss percentage (default 3%)
    
    Returns:
        Dictionary with backtest results
    """
    # Calculate indicators
    df = strategy.calculate_indicators(df.copy())
    
    # Generate signals
    signals = []
    positions = []
    position = None
    
    for i in range(len(df)):
        price = df.iloc[i]['close']
        signal = strategy.generate_signal(df.iloc[:i+1])
        signals.append(signal)
        
        # Check TP/SL first if in position
        if position is not None:
            entry_price = position["entry_price"]
            pnl_pct = (price - entry_price) / entry_price
            
            # Take Profit
            if pnl_pct >= tp_pct:
                positions.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": tp_pct,
                    "exit_reason": "TP",
                })
                position = None
                continue
            
            # Stop Loss
            if pnl_pct <= -sl_pct:
                positions.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": -sl_pct,
                    "exit_reason": "SL",
                })
                position = None
                continue
        
        # Signal-based entries/exits
        if signal.signal == SignalType.BUY and position is None:
            position = {"entry_price": signal.price, "entry_time": signal.timestamp}
        elif signal.signal == SignalType.SELL and position is not None:
            pnl = (signal.price - position["entry_price"]) / position["entry_price"]
            positions.append({
                "entry_price": position["entry_price"],
                "exit_price": signal.price,
                "pnl": pnl,
                "exit_reason": "SIGNAL",
            })
            position = None
    
    # Close any open position at end
    if position is not None:
        last_price = df.iloc[-1]['close']
        pnl = (last_price - position["entry_price"]) / position["entry_price"]
        positions.append({
            "entry_price": position["entry_price"],
            "exit_price": last_price,
            "pnl": pnl,
            "exit_reason": "EOD",
        })
    
    # Calculate statistics
    if positions:
        pnls = [p["pnl"] for p in positions]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        tp_count = sum(1 for p in positions if p.get("exit_reason") == "TP")
        sl_count = sum(1 for p in positions if p.get("exit_reason") == "SL")
        
        return {
            "strategy": strategy.name,
            "total_trades": len(positions),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(positions) if positions else 0,
            "avg_pnl": np.mean(pnls) if pnls else 0,
            "max_pnl": max(pnls) if pnls else 0,
            "min_pnl": min(pnls) if pnls else 0,
            "total_return": sum(pnls),
            "tp_count": tp_count,
            "sl_count": sl_count,
            "signals": [s.to_dict() for s in signals[-10:]]
        }
    
    return {
        "strategy": strategy.name,
        "total_trades": 0,
        "win_rate": 0,
        "avg_pnl": 0,
        "max_pnl": 0,
        "min_pnl": 0,
        "total_return": 0,
        "tp_count": 0,
        "sl_count": 0,
        "signals": []
    }

if __name__ == "__main__":
    # Demo strategies
    print("=" * 60)
    print("STRATEGIES MODULE - Demo")
    print("=" * 60)
    
    # Generate sample data
    df = generate_sample_data(500)
    
    # Test RSI Strategy
    print("\n📊 Testing RSI Strategy...")
    rsi_strategy = RSIStrategy()
    df = rsi_strategy.calculate_indicators(df.copy())
    signal = rsi_strategy.generate_signal(df)
    print(f"   Signal: {signal.signal.value} (confidence: {signal.confidence:.2f})")
    
    # Test MACD Strategy
    print("\n📊 Testing MACD Strategy...")
    macd_strategy = MACDStrategy()
    df = macd_strategy.calculate_indicators(df.copy())
    signal = macd_strategy.generate_signal(df)
    print(f"   Signal: {signal.signal.value} (confidence: {signal.confidence:.2f})")
    
    # Test Combined Strategy
    print("\n📊 Testing Combined Strategy...")
    combined = CombinedStrategy()
    df = combined.calculate_indicators(df.copy())
    signal = combined.generate_signal(df)
    print(f"   Signal: {signal.signal.value} (confidence: {signal.confidence:.2f})")
    
    print("\n" + "=" * 60)
    print("Available Strategies:")
    for name in StrategyFactory.list_strategies():
        info = StrategyFactory.get_strategy_info(name)
        print(f"  - {info.get('name', name)}")
    
    print("\n" + "=" * 60)
