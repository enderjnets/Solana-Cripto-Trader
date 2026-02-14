#!/usr/bin/env python3
"""
Aggressive Signal Generator for Lateral Markets
==============================================
Enhanced strategy that generates more signals in ranging conditions.

Usage:
    python3 aggressive_signals.py --start     # Start aggressive mode
    python3 aggressive_signals.py --stop      # Stop
    python3 aggressive_signals.py --status   # Check status
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent
PAPER_STATE_FILE = PROJECT_ROOT / "paper_trading_state.json"


@dataclass
class Signal:
    """Trading signal."""
    time: str
    symbol: str
    direction: str  # long, short
    price: float
    strength: float
    strategy: str
    reason: str


class AggressiveSignalGenerator:
    """Generates more signals for lateral markets."""

    def __init__(self):
        self.state = self._load_state()
        self.signals: List[Signal] = []

    def _load_state(self):
        """Load paper trading state."""
        if PAPER_STATE_FILE.exists():
            return json.loads(PAPER_STATE_FILE.read_text())
        return {
            "balance_usd": 500.0,
            "trades": [],
            "stats": {"total_trades": 0, "winning_trades": 0}
        }

    def _save_state(self):
        """Save state."""
        self.state["signals"] = [
            {
                "time": s.time,
                "symbol": s.symbol,
                "direction": s.direction,
                "price": s.price,
                "strength": s.strength,
                "strategy": s.strategy,
                "reason": s.reason
            }
            for s in self.signals[-50:]
        ]
        PAPER_STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate recent volatility."""
        if len(prices) < 5:
            return 0.02
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)

    def detect_range(self, prices: List[float], window: int = 20) -> Optional[Dict]:
        """Detect if price is in a range."""
        if len(prices) < window:
            return None

        recent = prices[-window:]
        high = max(recent)
        low = min(recent)
        mid = (high + low) / 2
        current = prices[-1]

        range_pct = (high - low) / mid
        position = (current - low) / (high - low) if high > low else 0.5

        return {
            "high": high,
            "low": low,
            "mid": mid,
            "range_pct": range_pct,
            "position": position,  # 0 = bottom, 1 = top
            "is_range": range_pct < 0.03  # 3% range = lateral
        }

    def generate_aggressive_signal(
        self,
        symbol: str,
        price: float,
        prices: List[float],
        price_change: float
    ) -> Optional[Signal]:
        """Generate aggressive signal for lateral market."""

        # Check for range
        range_info = self.detect_range(prices)
        if not range_info:
            return None

        volatility = self.calculate_volatility(prices)

        # AGGRESSIVE STRATEGY 1: Range bounce
        if range_info["is_range"]:
            position = range_info["position"]
            reason = f"Range bounce (price {position*100:.0f}% from bottom)"

            # Near bottom → LONG
            if position < 0.35:
                return Signal(
                    time=datetime.now().isoformat(),
                    symbol=symbol,
                    direction="long",
                    price=price,
                    strength=0.7 + (0.3 * (1 - position)),
                    strategy="range_bounce_long",
                    reason=f"{reason} - Oversold"
                )

            # Near top → SHORT
            elif position > 0.65:
                return Signal(
                    time=datetime.now().isoformat(),
                    symbol=symbol,
                    direction="short",
                    price=price,
                    strength=0.7 + (0.3 * position),
                    strategy="range_bounce_short",
                    reason=f"{reason} - Overbought"
                )

        # AGGRESSIVE STRATEGY 2: Micro trend continuation
        if abs(price_change) > volatility * 0.5:
            direction = "long" if price_change > 0 else "short"
            return Signal(
                time=datetime.now().isoformat(),
                symbol=symbol,
                direction=direction,
                price=price,
                strength=0.6,
                strategy="micro_trend",
                reason=f"Momentum: {price_change*100:.2f}% in period"
            )

        # AGGRESSIVE STRATEGY 3: Mean reversion
        if len(prices) >= 10:
            ma10 = np.mean(prices[-10:])
            deviation = (price - ma10) / ma10

            if deviation < -0.015:  # -1.5% from MA
                return Signal(
                    time=datetime.now().isoformat(),
                    symbol=symbol,
                    direction="long",
                    price=price,
                    strength=0.65,
                    strategy="mean_reversion_long",
                    reason=f"Below MA10 by {abs(deviation)*100:.2f}%"
                )
            elif deviation > 0.015:  # +1.5% from MA
                return Signal(
                    time=datetime.now().isoformat(),
                    symbol=symbol,
                    direction="short",
                    price=price,
                    strength=0.65,
                    strategy="mean_reversion_short",
                    reason=f"Above MA10 by {deviation*100:.2f}%"
                )

        # AGGRESSIVE STRATEGY 4: VWAP bounce
        if len(prices) >= 20:
            vwap = np.mean(prices[-20:])
            vwap_dev = (price - vwap) / vwap

            if -0.01 < vwap_dev < 0:
                return Signal(
                    time=datetime.now().isoformat(),
                    symbol=symbol,
                    direction="long",
                    price=price,
                    strength=0.55,
                    strategy="vwap_bounce_long",
                    reason=f"Near VWAP ({(vwap_dev)*100:.2f}%)"
                )
            elif 0 < vwap_dev < 0.01:
                return Signal(
                    time=datetime.now().isoformat(),
                    symbol=symbol,
                    direction="short",
                    price=price,
                    strength=0.55,
                    strategy="vwap_bounce_short",
                    reason=f"Near VWAP ({vwap_dev*100:.2f}%)"
                )

        return None

    def execute_signal(self, signal: Signal) -> bool:
        """Execute signal in paper trading."""
        if signal.strength < 0.5:
            return False

        # Add to state
        self.signals.append(signal)
        self._save_state()

        print(f"\n🎯 AGGRESSIVE SIGNAL GENERATED")
        print(f"   Symbol:   {signal.symbol}")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Price:    ${signal.price:.2f}")
        print(f"   Strength: {signal.strength:.2f}")
        print(f"   Strategy: {signal.strategy}")
        print(f"   Reason:   {signal.reason}")
        print(f"   ─────────────────────────────────")

        return True


class AggressivePaperTrader:
    """Paper trader with aggressive signal generation."""

    def __init__(self):
        self.generator = AggressiveSignalGenerator()
        self.running = False
        self.prices: List[float] = []

    async def get_price(self) -> float:
        """Get current SOL price."""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112")
            data = resp.json()
            return float(data["priceUsd"])

    async def run(self):
        """Main loop."""
        print("\n" + "="*60)
        print("🚀 AGGRESSIVE PAPER TRADING MODE")
        print("="*60)
        print("Strategies enabled:")
        print("  • Range bounce (lateral markets)")
        print("  • Micro trend continuation")
        print("  • Mean reversion from MA10")
        print("  • VWAP bounce")
        print("="*60)

        self.running = True
        cycle = 0

        while self.running:
            cycle += 1
            print(f"\n🔄 CYCLE {cycle} - {datetime.now().strftime('%H:%M:%S')}")

            try:
                # Get current price
                price = await self.get_price()
                self.prices.append(price)

                # Keep last 50 prices
                if len(self.prices) > 50:
                    self.prices = self.prices[-50:]

                # Calculate price change
                if len(self.prices) >= 2:
                    price_change = (self.prices[-1] - self.prices[-2]) / self.prices[-2]
                else:
                    price_change = 0

                print(f"   Price: ${price:.2f} ({price_change*100:+.2f}%)")
                print(f"   Prices tracked: {len(self.prices)}")

                # Generate signal
                if len(self.prices) >= 10:
                    signal = self.generator.generate_aggressive_signal(
                        symbol="SOL",
                        price=price,
                        prices=self.prices,
                        price_change=price_change
                    )

                    if signal:
                        self.generator.execute_signal(signal)
                    else:
                        print(f"   📭 No signal generated")
                else:
                    print(f"   📊 Collecting price data...")

            except Exception as e:
                print(f"   ❌ Error: {e}")

            await asyncio.sleep(60)  # Check every minute

    def stop(self):
        """Stop trading."""
        self.running = False


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggressive Paper Trading")
    parser.add_argument("--start", action="store_true", help="Start aggressive trading")
    parser.add_argument("--stop", action="store_true", help="Stop trading")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    trader = AggressivePaperTrader()

    if args.start:
        await trader.run()
    elif args.stop:
        trader.stop()
        print("\n🛑 Trading stopped")
    elif args.status:
        state = trader.generator._load_state()
        print(f"\n📊 STATUS")
        print(f"   Signals generated: {len(trader.generator.signals)}")
        print(f"   Balance: ${state.get('balance_usd', 500):.2f}")
    else:
        await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
