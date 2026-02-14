#!/usr/bin/env python3
"""
Trading Team Coordinator
=====================
Unified team working together to achieve 5% daily target.

Team Members:
- Scout (Market Scanner) - Finds opportunities
- Trader (Paper) - Executes trades
- Optimizer - Improves strategies
- Risk Manager - Controls exposure

Goal: 5% daily ($25 on $500)

Usage:
    python3 trading_team.py --start     # Start the team
    python3 trading_team.py --status    # Check team status
"""

import json
import asyncio
import httpx
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Paths
PROJECT_ROOT = Path(__file__).parent
PAPER_STATE_FILE = PROJECT_ROOT / "paper_trading_state.json"

# Settings
DAILY_TARGET_PCT = 0.05  # 5%
INITIAL_CAPITAL = 500.0
TRADE_SIZE = 20
MAX_RISK_PCT = 0.10  # Max 10% risk per day


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
    status: str  # open, closed


class ScoutAgent:
    """Scouts market for opportunities."""

    def __init__(self):
        self.tokens = [
            ("SOL", "So11111111111111111111111111111111111111112"),
            ("JUP", "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2"),
            ("WIF", "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP"),
            ("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"),
            ("POPCAT", "7GCihgDB8Fe6JKr2mG9VDLxrGkZaGtD1W89VjMW9w8s"),
        ]

    async def scan(self) -> List[Dict]:
        """Scan for opportunities."""
        opportunities = []

        async with httpx.AsyncClient() as client:
            for symbol, address in self.tokens:
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
                            reasons.append(f"Strong momentum (+{change_24h:.1f}%)")
                        elif change_24h < -5:
                            score += abs(change_24h) * 0.8
                            reasons.append(f"Dip opportunity ({change_24h:.1f}%)")

                        if volume_24h > 1000000:
                            score += 2
                            reasons.append(f"High volume ${volume_24h/1e6:.1f}M")

                        if score > 3:
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

        # Sort by score
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities[:5]


class TraderAgent:
    """Executes trades."""

    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_pnl_pct = 0.0
        self.trades_today = 0

    def execute(self, opportunity: Dict) -> Optional[Trade]:
        """Execute a trade."""
        # Check risk limits
        if self.daily_pnl_pct <= -MAX_RISK_PCT * 100:
            return None  # Stop trading - daily loss limit hit

        if len(self.trades) >= 10:
            return None  # Max trades per day

        trade = Trade(
            id=f"trade_{datetime.now().strftime('%H%M%S')}",
            time=datetime.now().strftime("%H:%M:%S"),
            symbol=opportunity["symbol"],
            direction="BUY",
            entry_price=opportunity["price"],
            size=TRADE_SIZE,
            current_price=opportunity["price"],
            pnl_pct=0.0,
            pnl_value=0.0,
            status="open"
        )

        self.trades.append(trade)
        self.trades_today += 1
        return trade

    def update_prices(self, prices: Dict[str, float]):
        """Update trade prices and P&L."""
        for trade in self.trades:
            if trade.status == "open" and trade.symbol in prices:
                trade.current_price = prices[trade.symbol]
                trade.pnl_pct = (trade.current_price - trade.entry_price) / trade.entry_price * 100
                trade.pnl_value = TRADE_SIZE * trade.pnl_pct / 100

                # Close at 10% profit or 5% loss
                if trade.pnl_pct >= 10:
                    trade.status = "closed"
                elif trade.pnl_pct <= -5:
                    trade.status = "closed"

        # Calculate daily P&L
        closed = [t for t in self.trades if t.status == "closed"]
        self.daily_pnl_pct = sum(t.pnl_pct for t in closed)


class OptimizerAgent:
    """Improves strategy based on results."""

    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.iterations = 0

    def analyze(self, trades: List[Trade]) -> Dict:
        """Analyze performance and return recommendations."""
        self.iterations += 1

        closed = [t for t in trades if t.status == "closed"]
        self.wins = sum(1 for t in closed if t.pnl_pct > 0)
        self.losses = len(closed) - self.wins
        self.total_pnl = sum(t.pnl_value for t in closed)

        win_rate = self.wins / len(closed) * 100 if closed else 0

        recommendations = []

        # Adjust strategy based on results
        if win_rate < 40:
            recommendations.append("Win rate too low - increase entry threshold")
            recommendations.append("Consider waiting for stronger signals")
        elif win_rate > 60:
            recommendations.append("Strategy working - can increase trade frequency")

        if self.total_pnl < 0:
            recommendations.append("In negative - review risk management")

        return {
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "recommendations": recommendations,
            "iterations": self.iterations
        }


class RiskManager:
    """Controls risk and exposure."""

    def __init__(self):
        self.max_daily_loss = 0.10  # 10%
        self.max_position_size = 0.20  # 20% per trade
        self.current_exposure = 0.0

    def check_limits(self, trades: List[Trade], daily_pnl_pct: float) -> Dict:
        """Check risk limits."""
        status = {
            "can_trade": True,
            "reasons": []
        }

        # Check daily loss limit
        if daily_pnl_pct <= -self.max_daily_loss * 100:
            status["can_trade"] = False
            status["reasons"].append("Daily loss limit hit (10%)")

        # Check exposure
        self.current_exposure = sum(abs(t.pnl_pct) for t in trades if t.status == "open") / 100

        return status


class TradingTeam:
    """Coordinates all agents toward 5% daily goal."""

    def __init__(self):
        self.scout = ScoutAgent()
        self.trader = TraderAgent()
        self.optimizer = OptimizerAgent()
        self.risk = RiskManager()

    async def run_cycle(self):
        """Run one team cycle."""
        print(f"\n{'='*60}")
        print(f"🔄 TRADING TEAM CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # 1. Scout scans market
        print("\n🧭 SCOUT: Scanning market...")
        opportunities = await self.scout.scan()
        print(f"   Found {len(opportunities)} opportunities")

        if opportunities:
            for i, opp in enumerate(opportunities[:3], 1):
                print(f"   {i}. {opp['symbol']}: Score {opp['score']:.1f}")
                print(f"      Price: ${opp['price']:.4f} | Change: {opp['change']:+.1f}%")

        # 2. Trader executes
        print("\n💰 TRADER: Executing trades...")
        for opp in opportunities[:3]:  # Take top 3
            risk_status = self.risk.check_limits(
                self.trader.trades,
                self.trader.daily_pnl_pct
            )
            if not risk_status["can_trade"]:
                print(f"   🛑 Cannot trade: {risk_status['reasons']}")
                break

            trade = self.trader.execute(opp)
            if trade:
                print(f"   ✅ Executed: {trade.direction} {trade.symbol} @ ${trade.entry_price:.4f}")

        # 3. Update prices
        prices = {opp["symbol"]: opp["price"] for opp in opportunities}
        self.trader.update_prices(prices)

        # 4. Optimizer analyzes
        print("\n🧠 OPTIMIZER: Analyzing performance...")
        analysis = self.optimizer.analyze(self.trader.trades)
        print(f"   Win Rate: {analysis['win_rate']:.1f}%")
        print(f"   P&L: ${analysis['total_pnl']:.2f}")

        # 5. Risk check
        print("\n🛡️ RISK MANAGER: Checking limits...")
        risk_status = self.risk.check_limits(
            self.trader.trades,
            self.trader.daily_pnl_pct
        )
        print(f"   Can Trade: {'✅' if risk_status['can_trade'] else '❌'}")

        # 6. Progress toward 5% goal
        closed = [t for t in self.trader.trades if t.status == "closed"]
        daily_pnl_pct = sum(t.pnl_pct for t in closed)
        target_pct = DAILY_TARGET_PCT * 100

        print(f"\n📊 PROGRESS TOWARD +{target_pct}% DAILY GOAL")
        print(f"   Current: {daily_pnl_pct:+.2f}%")
        print(f"   Target: +{target_pct}%")
        progress = (daily_pnl_pct / target_pct * 100) if target_pct > 0 else 0
        bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
        print(f"   [{bar}] {progress:.1f}%")

        # 7. Open positions
        print(f"\n📋 OPEN POSITIONS ({len([t for t in self.trader.trades if t.status == 'open'])})")
        for trade in self.trader.trades:
            if trade.status == "open":
                emoji = "🟢" if trade.pnl_pct >= 0 else "🔴"
                print(f"   {emoji} {trade.symbol}: ${trade.entry_price:.4f} → ${trade.current_price:.4f} ({trade.pnl_pct:+.2f}%)")

        return {
            "opportunities": len(opportunities),
            "trades_executed": self.trader.trades_today,
            "daily_pnl_pct": daily_pnl_pct,
            "target_pct": target_pct,
            "win_rate": analysis["win_rate"]
        }

    def save_state(self):
        """Save team state."""
        state = {
            "team_status": "active",
            "timestamp": datetime.now().isoformat(),
            "trader": {
                "trades_today": self.trader.trades_today,
                "daily_pnl_pct": self.trader.daily_pnl_pct,
                "open_positions": len([t for t in self.trader.trades if t.status == "open"])
            },
            "optimizer": {
                "iterations": self.optimizer.iterations,
                "total_pnl": self.optimizer.total_pnl
            }
        }
        PAPER_STATE_FILE.write_text(json.dumps(state, indent=2))


async def main():
    team = TradingTeam()

    print("\n" + "="*60)
    print("🚀 TRADING TEAM - 5% DAILY GOAL")
    print("="*60)
    print(f"Initial Capital: ${INITIAL_CAPITAL}")
    print(f"Daily Target: +{DAILY_TARGET_PCT*100}% (${INITIAL_CAPITAL * DAILY_TARGET_PCT})")
    print(f"Trade Size: ${TRADE_SIZE}")
    print("="*60 + "\n")

    cycle = 0
    while True:
        cycle += 1
        result = await team.run_cycle()
        team.save_state()

        print(f"\n📈 SUMMARY - Cycle {cycle}")
        print(f"   Opportunities: {result['opportunities']}")
        print(f"   Trades: {result['trades_executed']}")
        print(f"   P&L: {result['daily_pnl_pct']:+.2f}% / +{result['target_pct']}% target")
        print(f"   Win Rate: {result['win_rate']:.1f}%")

        await asyncio.sleep(30)  # Every 30 seconds


if __name__ == "__main__":
    asyncio.run(main())
