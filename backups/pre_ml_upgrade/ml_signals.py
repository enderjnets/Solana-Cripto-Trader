#!/usr/bin/env python3
"""
ML Signal Generator v2 - AGGRESSIVE MODE
======================================
Optimized for more trades and better signals.

Changes from v1:
- Lower signal threshold (0.3 → 0.15)
- Higher weights for strong signals
- Added volume-based signals
- More aggressive momentum detection

Usage:
    from ml.ml_signals import MLSignalGenerator
    signals = generator.generate(data)
"""

import json
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MarketData:
    """Market data for ML analysis."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    timestamp: str


@dataclass
class Signal:
    """ML-generated signal."""
    symbol: str
    direction: str
    confidence: float
    reason: str
    indicators: Dict[str, float]


class TechnicalIndicators:
    """Calculate technical indicators."""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 7) -> float:
        """Calculate RSI (faster period for more signals)."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 65.0  # Slightly bullish on neutral

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1] if prices else 0

        alpha = 2 / (period + 1)
        ema = prices[-1]
        for price in reversed(prices[:-1]):
            ema = alpha * price + (1 - alpha) * ema
        return ema

    @staticmethod
    def calculate_momentum(prices: List[float], period: int = 5) -> float:
        """Calculate momentum (faster period)."""
        if len(prices) < period:
            return 100.0

        current = prices[-1]
        past = prices[-period]
        return (current / past) * 100

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 7) -> float:
        """Calculate volatility."""
        if len(prices) < period:
            return 0.02

        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)


class MLSignalGenerator:
    """
    AGGRESSIVE ML-based signal generator.
    Optimized for more trades and higher returns.
    """

    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        self.signals_generated = 0
        self.trades_executed = 0

    def add_price(self, symbol: str, price: float):
        """Add price to history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

        # Keep last 30 prices (reduced for faster signals)
        if len(self.price_history[symbol]) > 30:
            self.price_history[symbol] = self.price_history[symbol][-30:]

    def generate_signal(self, data: Dict) -> Optional[Signal]:
        """Generate AGGRESSIVE ML signal."""
        symbol = data["symbol"]
        price = data["price"]
        change = data["change"]
        volume = data.get("volume", 0)

        # Add price to history
        self.add_price(symbol, price)
        history = self.price_history.get(symbol, [price])

        # Calculate indicators (faster periods)
        rsi = self.calculate_rsi(history, 7)
        ema_5 = self.calculate_ema(history, 5)
        ema_15 = self.calculate_ema(history, 15)
        momentum = self.calculate_momentum(history, 5)
        volatility = self.calculate_volatility(history, 7)

        # AGGRESSIVE scoring
        scores = {"rsi": 0.0, "ema": 0.0, "momentum": 0.0, "trend": 0.0, "volume": 0.0}
        reasons = []

        # RSI component (more sensitive)
        if rsi < 35:  # Lower threshold
            scores["rsi"] = 0.9
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65:  # Lower threshold
            scores["rsi"] = -0.9
            reasons.append(f"RSI overbought ({rsi:.1f})")
        else:
            scores["rsi"] = (rsi - 50) / 15  # More sensitive

        # EMA crossover (faster)
        if ema_5 > ema_15:
            scores["ema"] = 0.6
            reasons.append("EMA 5>15 (bullish)")
        else:
            scores["ema"] = -0.6
            reasons.append("EMA 5<15 (bearish)")

        # Momentum (stronger signals)
        if momentum > 102:
            scores["momentum"] = 0.7
            reasons.append(f"Strong momentum ({momentum:.1f}%)")
        elif momentum < 98:
            scores["momentum"] = -0.7
            reasons.append(f"Weak momentum ({momentum:.1f}%)")
        else:
            scores["momentum"] = (momentum - 100) / 5

        # 24h change (larger threshold)
        if change > 3:
            scores["trend"] = 0.5
            reasons.append(f"Strong 24h ({change:+.1f}%)")
        elif change < -3:
            scores["trend"] = -0.5
            reasons.append(f"Weak 24h ({change:+.1f}%)")
        else:
            scores["trend"] = change / 10

        # Volume bonus
        if volume > 1000000:
            scores["volume"] = 0.3
            reasons.append(f"High volume ${volume/1e6:.1f}M")

        # Weighted AGGRESSIVE ensemble
        weights = {
            "rsi": 0.25,
            "ema": 0.20,
            "momentum": 0.25,
            "trend": 0.20,
            "volume": 0.10
        }

        total_score = sum(scores[k] * weights[k] for k in scores)

        # LOWER volatility penalty (more trades)
        vol_penalty = min(volatility * 3, 0.2)
        confidence = abs(total_score) - vol_penalty
        confidence = max(0.15, min(0.90, confidence))  # LOWER minimum

        # LOWER threshold (more signals)
        threshold = 0.15

        if total_score > threshold:
            direction = "BUY"
            self.signals_generated += 1
        elif total_score < -threshold:
            direction = "SELL"
            self.signals_generated += 1
        else:
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reason=" | ".join(reasons[:3]),
            indicators={
                "rsi": rsi,
                "ema_5": ema_5,
                "ema_15": ema_15,
                "momentum": momentum,
                "volatility": volatility
            }
        )


async def test_aggressive_ml():
    """Test aggressive ML generator."""
    print("\n🧪 TESTING AGGRESSIVE ML SIGNAL GENERATOR")
    print("="*60)

    # Create generator
    ml = MLSignalGenerator()

    # Add fake history
    base_prices = {"SOL": 87.0, "BONK": 0.000006, "WIF": 2.5}
    for symbol, base in base_prices.items():
        for i in range(15):
            variation = np.random.normal(0, 0.03)
            ml.add_price(symbol, base * (1 + variation))

    # Test data
    test_data = [
        {"symbol": "SOL", "price": 87.5, "change": 2.5, "volume": 50000000},
        {"symbol": "BONK", "price": 0.0000065, "change": 8.0, "volume": 2000000},
        {"symbol": "WIF", "price": 2.6, "change": -1.5, "volume": 800000},
    ]

    signals = []
    for data in test_data:
        signal = ml.generate_signal(data)
        if signal:
            signals.append(signal)
            emoji = "🟢" if signal.direction == "BUY" else "🔴"
            print(f"\n{emoji} {signal.symbol}: {signal.direction}")
            print(f"   Confidence: {signal.confidence:.0%}")
            print(f"   {signal.reason}")
            print(f"   RSI: {signal.indicators['rsi']:.1f}")
        else:
            print(f"\n⚪ {data['symbol']}: No signal")

    print(f"\n📊 Signals generated: {ml.signals_generated}")
    print(f"🎯 Threshold: 0.15 (aggressive)")
    print("="*60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_aggressive_ml())
    else:
        print("ML Signal Generator v2 (AGGRESSIVE MODE) Ready")
