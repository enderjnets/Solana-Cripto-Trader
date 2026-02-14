#!/usr/bin/env python3
"""
Unified Brain v2 - ML-Powered Trading System
============================================
Single brain that combines:
- Token Scout (from agent_brain.py)
- Strategy Optimizer (from agent_brain.py)
- Trading Team (from trading_team.py)
- WebSocket real-time data
- Jito bundles
- Database persistence
- ML Signal Generator (NEW)

Eliminates duplicate processes and consolidates all functionality.

Usage:
    python3 unified_brain.py --start     # Start unified system
    python3 unified_brain.py --status   # Check status
"""

import json
import asyncio
import httpx
import logging
import signal
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Import modules
from api.websocket_client import WebSocketSimulator
from api.jito_client import JitoClient, JitoConfig
from db.database import SQLiteDatabase, Trade as DBTrade
from ml.ml_signals import MLSignalGenerator, MarketData
from cache.redis_manager import RedisSimulator, PriceCache, TradeStateManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("unified_brain")

# Paths
PROJECT_ROOT = Path(__file__).parent
STATE_FILE = PROJECT_ROOT / "unified_brain_state.json"
DB_FILE = PROJECT_ROOT / "db" / "unified_trading.db"

# Settings
DAILY_TARGET_PCT = 0.05  # 5%
INITIAL_CAPITAL = 500.0
TRADE_SIZE = 20
CYCLE_INTERVAL = 60  # 1 minute


@dataclass
class Trade:
    """Trade record."""
    id: str
    time: str
    symbol: str
    direction: str
    entry_price: float
    size: float
    current_price: float
    pnl_pct: float
    pnl_value: float
    status: str
    strategy: str


class TokenScout:
    """Scans tokens for opportunities (from agent_brain.py)."""

    CORE_TOKENS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
        "cbBTC": "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
        "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "JTO": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
    }

    async def scan(self) -> List[Dict]:
        """Scan for trading opportunities."""
        opportunities = []

        async with httpx.AsyncClient() as client:
            for symbol, address in self.CORE_TOKENS.items():
                try:
                    resp = await client.get(
                        f"https://api.dexscreener.com/latest/dex/tokens/{address}",
                        timeout=10
                    )
                    data = resp.json()
                    pairs = data.get("pairs", [])

                    if pairs:
                        pair = pairs[0]
                        price = float(pair.get("priceUsd", 0))
                        change_24h = float(pair.get("priceChange", {}).get("h24", 0))
                        volume_24h = float(pair.get("volume", {}).get("h24", 0))

                        # Calculate score
                        score = 0
                        reasons = []

                        if change_24h > 5:
                            score += change_24h
                            reasons.append(f"Momentum +{change_24h:.1f}%")
                        elif change_24h < -5:
                            score += abs(change_24h) * 0.8
                            reasons.append(f"Dip {change_24h:.1f}%")

                        if volume_24h > 500000:
                            score += 1

                        if score > 2:
                            opportunities.append({
                                "symbol": symbol,
                                "price": price,
                                "change": change_24h,
                                "volume": volume_24h,
                                "score": score,
                                "reasons": reasons
                            })

                except Exception:
                    continue

        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities[:8]


class MLSignalGenerator:
    """
    ML-based signal generator for intelligent trading decisions.
    Uses RSI, EMA crossovers, and momentum analysis.
    """

    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}

    def add_price(self, symbol: str, price: float):
        """Add price to history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > 50:
            self.price_history[symbol] = self.price_history[symbol][-50:]

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 70.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1] if prices else 0

        alpha = 2 / (period + 1)
        ema = prices[-1]
        for price in reversed(prices[:-1]):
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def generate_signal(self, data: Dict) -> Optional[Dict]:
        """Generate ML signal for a token."""
        symbol = data["symbol"]
        price = data["price"]
        change = data["change"]

        # Add price to history
        self.add_price(symbol, price)
        history = self.price_history.get(symbol, [price])

        # Calculate indicators
        rsi = self.calculate_rsi(history)
        ema_9 = self.calculate_ema(history, 9)
        ema_21 = self.calculate_ema(history, 21)

        # Ensemble scoring
        score = 0
        reasons = []

        # RSI component
        if rsi < 30:
            score += 0.8
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.8
            reasons.append(f"RSI overbought ({rsi:.1f})")
        else:
            score += (rsi - 50) / 50

        # EMA crossover
        if ema_9 > ema_21:
            score += 0.5
            reasons.append("EMA 9>21 (bullish)")
        else:
            score -= 0.5
            reasons.append("EMA 9<21 (bearish)")

        # 24h change
        if change > 5:
            score += 0.5
            reasons.append(f"Strong 24h ({change:+.1f}%)")
        elif change < -5:
            score -= 0.5
            reasons.append(f"Weak 24h ({change:+.1f}%)")

        # Determine signal
        if score > 0.3:
            return {
                "symbol": symbol,
                "direction": "BUY",
                "confidence": min(score, 0.95),
                "reason": " | ".join(reasons[:2]),
                "indicators": {"rsi": rsi, "ema_9": ema_9, "ema_21": ema_21}
            }
        elif score < -0.3:
            return {
                "symbol": symbol,
                "direction": "SELL",
                "confidence": min(abs(score), 0.95),
                "reason": " | ".join(reasons[:2]),
                "indicators": {"rsi": rsi, "ema_9": ema_9, "ema_21": ema_21}
            }

        return None


class StrategyOptimizer:
    """Optimizes trading strategies."""

    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.iterations = 0
        self.best_params = {"threshold": 0.3, "risk_pct": 0.05}

    def analyze(self, trades: List[Trade]) -> Dict:
        """Analyze performance and adjust strategy."""
        self.iterations += 1

        closed = [t for t in trades if t.status == "closed"]
        self.wins = sum(1 for t in closed if t.pnl_pct > 0)
        self.losses = len(closed) - self.wins
        self.total_pnl = sum(t.pnl_value for t in closed)

        win_rate = self.wins / len(closed) * 100 if closed else 0

        # Adjust parameters
        if win_rate < 40:
            self.best_params["threshold"] += 0.1
        elif win_rate > 60:
            self.best_params["threshold"] -= 0.05

        return {
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "iterations": self.iterations,
            "params": self.best_params
        }


class Trader:
    """Executes trades with ML signals."""

    def __init__(self, jito: JitoClient, ml: MLSignalGenerator):
        self.trades: List[Trade] = []
        self.daily_pnl_pct = 0.0
        self.trades_today = 0
        self.jito = jito
        self.ml = ml

    def execute(self, data: Dict, ml_signal: Dict) -> Optional[Trade]:
        """Execute a trade based on ML signal."""
        if len(self.trades) >= 10:
            return None

        direction = ml_signal["direction"]
        trade = Trade(
            id=f"trade_{datetime.now().strftime('%H%M%S')}",
            time=datetime.now().strftime("%H:%M:%S"),
            symbol=data["symbol"],
            direction=direction,
            entry_price=data["price"],
            size=TRADE_SIZE,
            current_price=data["price"],
            pnl_pct=0.0,
            pnl_value=0.0,
            status="open",
            strategy="ml_ensemble"
        )

        self.trades.append(trade)
        self.trades_today += 1
        return trade

    def update_prices(self, prices: Dict[str, float]):
        """Update prices and P&L."""
        for trade in self.trades:
            if trade.status == "open" and trade.symbol in prices:
                trade.current_price = prices[trade.symbol]

                if trade.direction == "BUY":
                    trade.pnl_pct = (trade.current_price - trade.entry_price) / trade.entry_price * 100
                else:
                    trade.pnl_pct = (trade.entry_price - trade.current_price) / trade.entry_price * 100

                trade.pnl_value = TRADE_SIZE * trade.pnl_pct / 100

                if trade.pnl_pct >= 10:
                    trade.status = "closed"
                elif trade.pnl_pct <= -5:
                    trade.status = "closed"

        closed = [t for t in self.trades if t.status == "closed"]
        self.daily_pnl_pct = sum(t.pnl_pct for t in closed)


class RiskManager:
    """Controls risk and exposure."""

    def __init__(self):
        self.max_daily_loss = 0.10
        self.max_positions = 10

    def can_trade(self, trades: List[Trade], daily_pnl_pct: float) -> bool:
        """Check if trading is allowed."""
        if daily_pnl_pct <= -self.max_daily_loss * 100:
            return False
        if len([t for t in trades if t.status == "open"]) >= self.max_positions:
            return False
        return True


class UnifiedBrain:
    """Unified brain combining all components."""

    def __init__(self):
        # Initialize all modules
        self.ws = WebSocketSimulator()
        self.jito = JitoClient(JitoConfig(enabled=True))
        self.db = SQLiteDatabase(str(DB_FILE))
        self.ml = MLSignalGenerator()

        # Initialize Redis cache
        self.redis = RedisSimulator()
        self.price_cache = PriceCache(self.redis)
        self.trade_state = TradeStateManager(self.redis)

        # Initialize agents
        self.scout = TokenScout()
        self.optimizer = StrategyOptimizer()
        self.trader = Trader(self.jito, self.ml)
        self.risk = RiskManager()

        self.running = False
        self.cycle_count = 0

    async def run_cycle(self):
        """Run one complete cycle."""
        self.cycle_count += 1

        print(f"\n{'='*60}")
        print(f"🧠 UNIFIED BRAIN v2 (ML) - Cycle {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # 1. Scout scans
        print("\n🧭 SCOUT: Scanning tokens...")
        opportunities = await self.scout.scan()
        print(f"   Found {len(opportunities)} opportunities")

        # 2. ML generates signals
        print("\n🧠 ML SIGNAL GENERATOR:")
        ml_signals = []
        for opp in opportunities[:5]:
            signal = self.ml.generate_signal(opp)
            if signal:
                ml_signals.append(signal)
                emoji = "🟢" if signal["direction"] == "BUY" else "🔴"
                print(f"   {emoji} {signal['symbol']}: {signal['direction']} ({signal['confidence']:.0%})")
                print(f"      {signal['reason']}")
                print(f"      RSI: {signal['indicators']['rsi']:.1f}")

        # 3. Trader executes
        print("\n💰 TRADER: Executing ML signals...")
        if not self.risk.can_trade(self.trader.trades, self.trader.daily_pnl_pct):
            print("   🛑 Risk limit hit")
        else:
            for signal in ml_signals[:3]:
                # Find opportunity data
                opp = next((o for o in opportunities if o["symbol"] == signal["symbol"]), None)
                if opp:
                    trade = self.trader.execute(opp, signal)
                    if trade:
                        print(f"   ✅ {trade.direction} {trade.symbol} @ ${trade.entry_price:.4f}")
                        print(f"      Confidence: {signal['confidence']:.0%} | {signal['reason']}")

                        # Save to database
                        db_trade = DBTrade(
                            id=trade.id,
                            symbol=trade.symbol,
                            direction=trade.direction,
                            entry_price=trade.entry_price,
                            exit_price=None,
                            size=trade.size,
                            pnl=None,
                            pnl_pct=None,
                            status="open",
                            timestamp=trade.time,
                            strategy="ml_ensemble"
                        )
                        self.db.add_trade(db_trade)

        # 4. Update prices
        prices = {opp["symbol"]: opp["price"] for opp in opportunities}
        self.trader.update_prices(prices)

        # 5. Optimizer analyzes
        print("\n🧠 OPTIMIZER: Analyzing...")
        analysis = self.optimizer.analyze(self.trader.trades)
        print(f"   Win Rate: {analysis['win_rate']:.1f}%")
        print(f"   P&L: ${analysis['total_pnl']:.2f}")
        print(f"   Threshold: {analysis['params']['threshold']:.2f}")

        # 6. Progress
        closed = [t for t in self.trader.trades if t.status == "closed"]
        daily_pnl = sum(t.pnl_pct for t in closed)
        target = DAILY_TARGET_PCT * 100

        print(f"\n📊 PROGRESS: {daily_pnl:+.2f}% / +{target}% target")
        print(f"   Trades: {self.trader.trades_today}")
        print(f"   ML Signals: {len(ml_signals)}")

        # 7. Open positions
        print(f"\n📋 OPEN POSITIONS:")
        for trade in self.trader.trades:
            if trade.status == "open":
                emoji = "🟢" if trade.pnl_pct >= 0 else "🔴"
                print(f"   {emoji} {trade.symbol}: {trade.pnl_pct:+.2f}%")

        return {
            "opportunities": len(opportunities),
            "ml_signals": len(ml_signals),
            "trades": self.trader.trades_today,
            "pnl_pct": daily_pnl,
            "win_rate": analysis["win_rate"]
        }

    def save_state(self):
        """Save brain state."""
        state = {
            "brain": "unified_v3",
            "version": "3.0",
            "modules": {
                "websocket": True,
                "jito": True,
                "database": True,
                "ml_signals": True,
                "redis_cache": True,
                "scout": True,
                "optimizer": True
            },
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "cycles": self.cycle_count,
                "trades_today": self.trader.trades_today,
                "daily_pnl_pct": self.trader.daily_pnl_pct,
                "total_pnl": self.optimizer.total_pnl,
                "iterations": self.optimizer.iterations
            }
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    async def run(self):
        """Main loop."""
        logger.info("\n" + "="*60)
        logger.info("🧠 UNIFIED BRAIN v3 - ML + REDIS POWERED")
        logger.info("="*60)
        logger.info(f"   Initial Capital: ${INITIAL_CAPITAL}")
        logger.info(f"   Daily Target: +{DAILY_TARGET_PCT*100}%")
        logger.info(f"   Trade Size: ${TRADE_SIZE}")
        logger.info(f"   ML Signals: ✅ Enabled")
        logger.info(f"   Redis Cache: ✅ Enabled")
        logger.info("="*60)

        self.running = True
        while self.running:
            try:
                await self.run_cycle()
                self.save_state()
                await asyncio.sleep(CYCLE_INTERVAL)
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(10)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Brain v3")
    parser.add_argument("--fast", action="store_true", help="Fast mode (30s cycles)")
    args = parser.parse_args()

    global CYCLE_INTERVAL
    if args.fast:
        CYCLE_INTERVAL = 30

    brain = UnifiedBrain()
    asyncio.run(brain.run())


if __name__ == "__main__":
    main()
