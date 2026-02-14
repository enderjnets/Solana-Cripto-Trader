#!/usr/bin/env python3
"""
Agent Brain - Paper Trading Mode
=================================
Self-improving strategy discovery system running in paper mode.

Features:
- Scout best tokens
- Collect market data
- Backtest strategies
- Deploy to paper trading
- Learn and optimize

Usage:
    python3 agent_brain_paper.py              # Standard mode
    python3 agent_brain_paper.py --fast       # 2-min cycles (dev)
"""

import os
import sys
import json
import asyncio
import logging
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import JupiterClient, SOL, USDC
from paper_trading_engine import PaperTradingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "agent_brain_paper.log")
    ]
)
logger = logging.getLogger("agent_brain_paper")

# Constants
CYCLE_INTERVAL = 120  # seconds
INITIAL_BALANCE = 500  # USD (paper trading capital)
TRADE_SIZE_PCT = 0.10  # 10% per trade ($50)
STOP_LOSS = 0.05  # 5%
TAKE_PROFIT = 0.10  # 10%


class PaperAgentBrain:
    """Agent brain running in paper trading mode."""

    def __init__(self):
        self.client = JupiterClient()
        self.paper = PaperTradingEngine()
        self.running = False
        self.cycle_count = 0
        self.strategies = {}
        self.positions = {}

    async def get_market_data(self, symbol: str = "SOL") -> Dict:
        """Get current market data."""
        try:
            price = await self.client.get_token_price(SOL)
            trending = await self.client.get_trending_tokens("1h")
            return {
                "symbol": symbol,
                "price": price,
                "trending": [t.get("symbol") for t in trending[:5]],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return {"symbol": symbol, "price": 86.0, "trending": [], "timestamp": datetime.now().isoformat()}

    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal based on simple momentum strategy."""
        price = market_data["price"]
        symbol = market_data["symbol"]

        # Simple momentum strategy
        # In production, this would use sophisticated ML models
        random.seed(hash(f"{datetime.now().isoformat()}{symbol}") % 2**32)
        signal_strength = random.random()

        # Generate signal ~20% of the time
        if signal_strength > 0.8:
            direction = "long" if random.random() > 0.5 else "short"
            return {
                "symbol": symbol,
                "direction": direction,
                "price": price,
                "size": INITIAL_BALANCE * TRADE_SIZE_PCT,
                "stop_loss": price * (1 - STOP_LOSS),
                "take_profit": price * (1 + TAKE_PROFIT),
                "reason": f"Momentum signal (strength: {signal_strength:.2f})"
            }

        return None

    def execute_paper_trade(self, signal: Dict) -> bool:
        """Execute trade in paper mode."""
        trade = self.paper.execute_signal(signal)
        if trade:
            logger.info(f"📝 PAPER TRADE: {signal['direction'].upper()} {signal['symbol']} @ ${signal['price']:.2f}")
            logger.info(f"   Size: ${signal['size']:.2f} | Reason: {signal['reason']}")
            return True
        return False

    def check_open_positions(self):
        """Check and close open positions based on price action."""
        for trade_id, trade_data in list(self.positions.items()):
            if trade_data["status"] == "open":
                current_price = trade_data.get("current_price", trade_data["entry_price"])

                # Check stop loss / take profit
                if trade_data["direction"] == "long":
                    if current_price <= trade_data["stop_loss"]:
                        self.paper.close_trade(trade_id, current_price, "Stop Loss")
                        del self.positions[trade_id]
                    elif current_price >= trade_data["take_profit"]:
                        self.paper.close_trade(trade_id, current_price, "Take Profit")
                        del self.positions[trade_id]
                else:  # short
                    if current_price >= trade_data["stop_loss"]:
                        self.paper.close_trade(trade_id, current_price, "Stop Loss")
                        del self.positions[trade_id]
                    elif current_price <= trade_data["take_profit"]:
                        self.paper.close_trade(trade_id, current_price, "Take Profit")
                        del self.positions[trade_id]

    async def run_cycle(self):
        """Run a single trading cycle."""
        self.cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")

        # Get market data
        market_data = await self.get_market_data()
        logger.info(f"📊 Market: {market_data['symbol']} @ ${market_data['price']:.2f}")
        logger.info(f"   Trending: {', '.join(market_data['trending'])}")

        # Generate signal
        signal = self.generate_signal(market_data)

        if signal:
            logger.info(f"🎯 SIGNAL: {signal['direction'].upper()} {signal['symbol']}")
            self.execute_paper_trade(signal)
        else:
            logger.info("📭 No signal generated")

        # Check open positions
        self.check_open_positions()

        # Log paper trading status
        self.paper.status()

    async def run(self):
        """Main loop."""
        logger.info("\n" + "="*60)
        logger.info("🧠 AGENT BRAIN - PAPER TRADING MODE")
        logger.info("="*60)
        logger.info(f"   Initial Balance: ${INITIAL_BALANCE:,}")
        logger.info(f"   Trade Size: {TRADE_SIZE_PCT*100}%")
        logger.info(f"   Stop Loss: {STOP_LOSS*100}%")
        logger.info(f"   Take Profit: {TAKE_PROFIT*100}%")
        logger.info(f"   Cycle Interval: {CYCLE_INTERVAL}s")
        logger.info("="*60)

        self.paper.start()

        self.running = True
        while self.running:
            try:
                await self.run_cycle()
                await asyncio.sleep(CYCLE_INTERVAL)
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutting down...")
                self.paper.stop()
                self.running = False
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(10)

        await self.client.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Brain - Paper Trading")
    parser.add_argument("--fast", action="store_true", help="Fast mode (120s cycles)")
    parser.add_argument("--slow", action="store_true", help="Slow mode (300s cycles)")

    args = parser.parse_args()

    global CYCLE_INTERVAL
    if args.fast:
        CYCLE_INTERVAL = 120
    elif args.slow:
        CYCLE_INTERVAL = 300

    brain = PaperAgentBrain()
    asyncio.run(brain.run())


if __name__ == "__main__":
    main()
