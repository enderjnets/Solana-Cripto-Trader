#!/usr/bin/env python3
"""
Unified Brain - Consolidated Trading System
==========================================
Single brain that combines:
- Token Scout (from agent_brain.py)
- Strategy Optimizer (from agent_brain.py)
- Trading Team (from trading_team.py)
- WebSocket real-time data
- Jito bundles
- Database persistence

Eliminates duplicate processes and consolidates all functionality.

Usage:
    python3 unified_brain.py --start     # Start unified system
    python3 unified_brain.py --status    # Check status
"""

import json
import asyncio
import httpx
import logging
import signal
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Import modules
from api.websocket_client import WebSocketSimulator
from api.jito_client import JitoClient, JitoConfig
from db.database import SQLiteDatabase, Trade as DBTrade

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


class StrategyOptimizer:
    """Optimizes trading strategies (from agent_brain.py)."""

    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.iterations = 0
        self.best_params = {"threshold": 5.0, "risk_pct": 0.05}

    def analyze(self, trades: List[Trade]) -> Dict:
        """Analyze performance and adjust strategy."""
        self.iterations += 1

        closed = [t for t in trades if t.status == "closed"]
        self.wins = sum(1 for t in closed if t.pnl_pct > 0)
        self.losses = len(closed) - self.wins
        self.total_pnl = sum(t.pnl_value for t in closed)

        win_rate = self.wins / len(closed) * 100 if closed else 0

        # Adjust parameters based on performance
        if win_rate < 40:
            self.best_params["threshold"] += 1.0  # stricter signals
        elif win_rate > 60:
            self.best_params["threshold"] -= 0.5  # more signals

        return {
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "iterations": self.iterations,
            "params": self.best_params
        }


class Trader:
    """Executes trades with Jito support."""

    def __init__(self, jito: JitoClient):
        self.trades: List[Trade] = []
        self.daily_pnl_pct = 0.0
        self.trades_today = 0
        self.jito = jito

    def execute(self, opportunity: Dict) -> Optional[Trade]:
        """Execute a trade."""
        if len(self.trades) >= 10:
            return None

        direction = "BUY" if opportunity["change"] > 0 else "SELL"

        trade = Trade(
            id=f"trade_{datetime.now().strftime('%H%M%S')}",
            time=datetime.now().strftime("%H:%M:%S"),
            symbol=opportunity["symbol"],
            direction=direction,
            entry_price=opportunity["price"],
            size=TRADE_SIZE,
            current_price=opportunity["price"],
            pnl_pct=0.0,
            pnl_value=0.0,
            status="open",
            strategy="momentum"
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

                # Auto-close at 10% profit or 5% loss
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

        # Initialize agents
        self.scout = TokenScout()
        self.optimizer = StrategyOptimizer()
        self.trader = Trader(self.jito)
        self.risk = RiskManager()

        self.running = False
        self.cycle_count = 0

    async def run_cycle(self):
        """Run one complete cycle."""
        self.cycle_count += 1

        print(f"\n{'='*60}")
        print(f"🧠 UNIFIED BRAIN CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # 1. Scout scans
        print("\n🧭 SCOUT: Scanning tokens...")
        opportunities = await self.scout.scan()
        print(f"   Found {len(opportunities)} opportunities")

        if opportunities:
            for i, opp in enumerate(opportunities[:5], 1):
                print(f"   {i}. {opp['symbol']}: Score {opp['score']:.1f} ({opp['change']:+.1f}%)")

        # 2. Trader executes
        print("\n💰 TRADER: Executing trades...")
        for opp in opportunities[:5]:
            if not self.risk.can_trade(self.trader.trades, self.trader.daily_pnl_pct):
                print("   🛑 Risk limit hit")
                break

            trade = self.trader.execute(opp)
            if trade:
                print(f"   ✅ {trade.direction} {trade.symbol} @ ${trade.entry_price:.4f}")

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
                    strategy=trade.strategy
                )
                self.db.add_trade(db_trade)

        # 3. Update prices
        prices = {opp["symbol"]: opp["price"] for opp in opportunities}
        self.trader.update_prices(prices)

        # 4. Optimizer analyzes
        print("\n🧠 OPTIMIZER: Analyzing...")
        analysis = self.optimizer.analyze(self.trader.trades)
        print(f"   Win Rate: {analysis['win_rate']:.1f}%")
        print(f"   P&L: ${analysis['total_pnl']:.2f}")
        print(f"   Threshold: {analysis['params']['threshold']:.1f}")

        # 5. Progress
        closed = [t for t in self.trader.trades if t.status == "closed"]
        daily_pnl = sum(t.pnl_pct for t in closed)
        target = DAILY_TARGET_PCT * 100

        print(f"\n📊 PROGRESS: {daily_pnl:+.2f}% / +{target}% target")
        print(f"   Trades: {self.trader.trades_today}")
        print(f"   Open: {len([t for t in self.trader.trades if t.status == 'open'])}")

        # 6. Open positions
        print(f"\n📋 OPEN POSITIONS:")
        for trade in self.trader.trades:
            if trade.status == "open":
                emoji = "🟢" if trade.pnl_pct >= 0 else "🔴"
                print(f"   {emoji} {trade.symbol}: {trade.pnl_pct:+.2f}%")

        return {
            "opportunities": len(opportunities),
            "trades": self.trader.trades_today,
            "pnl_pct": daily_pnl,
            "win_rate": analysis["win_rate"]
        }

    def save_state(self):
        """Save brain state."""
        state = {
            "brain": "unified",
            "version": "1.0",
            "modules": {
                "websocket": True,
                "jito": True,
                "database": True,
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
        logger.info("🧠 UNIFIED BRAIN - CONSOLIDATED SYSTEM")
        logger.info("="*60)
        logger.info(f"   Initial Capital: ${INITIAL_CAPITAL}")
        logger.info(f"   Daily Target: +{DAILY_TARGET_PCT*100}%")
        logger.info(f"   Trade Size: ${TRADE_SIZE}")
        logger.info(f"   Cycle Interval: {CYCLE_INTERVAL}s")
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
    parser = argparse.ArgumentParser(description="Unified Brain")
    parser.add_argument("--fast", action="store_true", help="Fast mode (30s cycles)")
    args = parser.parse_args()

    global CYCLE_INTERVAL
    if args.fast:
        CYCLE_INTERVAL = 30

    brain = UnifiedBrain()
    asyncio.run(brain.run())


if __name__ == "__main__":
    main()
