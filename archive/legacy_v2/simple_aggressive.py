#!/usr/bin/env python3
"""
Simple Aggressive Paper Trading
===============================
Generates signals every 30 seconds for lateral markets.

Usage:
    python3 simple_aggressive.py
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
import httpx
import numpy as np

PAPER_STATE_FILE = Path(__file__).parent / "paper_trading_state.json"


def get_sol_price() -> float:
    """Get current SOL price."""
    try:
        resp = httpx.get("https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112", timeout=5)
        data = resp.json()
        return float(data["priceUsd"])
    except:
        return 86.50  # Fallback


def analyze_market(prices: list) -> dict:
    """Analyze market and return signals."""
    if len(prices) < 5:
        return {"signal": None, "reason": "Collecting data..."}

    prices_arr = np.array(prices)
    current = prices_arr[-1]
    ma5 = np.mean(prices_arr[-5:])
    ma10 = np.mean(prices_arr[-10:])
    volatility = np.std(prices_arr[-10:]) / np.mean(prices_arr[-10:])

    # Range detection
    high = max(prices_arr[-20:])
    low = min(prices_arr[-20:])
    position = (current - low) / (high - low) if high > low else 0.5

    # Signal generation
    if position < 0.3:
        return {
            "signal": "LONG",
            "reason": f"Price at {position*100:.0f}% of range (bottom)",
            "strength": 0.8
        }
    elif position > 0.7:
        return {
            "signal": "SHORT",
            "reason": f"Price at {position*100:.0f}% of range (top)",
            "strength": 0.8
        }
    elif current < ma5 * 0.995:
        return {
            "signal": "LONG",
            "reason": f"Below MA5 by {(1-current/ma5)*100:.2f}%",
            "strength": 0.6
        }
    elif current > ma5 * 1.005:
        return {
            "signal": "SHORT",
            "reason": f"Above MA5 by {(current/ma5-1)*100:.2f}%",
            "strength": 0.6
        }
    elif volatility < 0.005:
        return {
            "signal": "WAIT",
            "reason": f"Low volatility ({volatility*100:.2f}%)",
            "strength": 0
        }

    return {"signal": None, "reason": "No clear signal", "strength": 0}


def main():
    print("\n" + "="*60)
    print("🚀 SIMPLE AGGRESSIVE PAPER TRADING")
    print("="*60)
    print("Checking every 30 seconds for trading signals...")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    prices = []
    cycle = 0

    while True:
        cycle += 1
        current_time = datetime.now().strftime("%H:%M:%S")

        try:
            price = get_sol_price()
            prices.append(price)
            if len(prices) > 50:
                prices = prices[-50:]

            analysis = analyze_market(prices)

            print(f"[{current_time}] Cycle {cycle}")
            print(f"  SOL: ${price:.2f} | Prices: {len(prices)}")

            if analysis["signal"]:
                print(f"  🎯 SIGNAL: {analysis['signal']}")
                print(f"     Reason: {analysis['reason']}")
                print(f"     Strength: {analysis['strength']:.1f}")
            else:
                print(f"  📭 {analysis['reason']}")

            print()

        except KeyboardInterrupt:
            print("\n👋 Stopped")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
