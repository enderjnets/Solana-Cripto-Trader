#!/usr/bin/env python3
"""
Simple Paper Trader - Executes Aggressive Signals
=============================================
Generates and executes trades every 30 seconds.

Usage:
    python3 simple_paper_trader.py
"""

import json
import time
import asyncio
import httpx
import numpy as np
from datetime import datetime
from pathlib import Path

PAPER_STATE_FILE = Path(__file__).parent / "paper_trading_state.json"

# Settings
TRADE_SIZE = 50  # $50 per trade
STOP_LOSS = 0.05  # 5%
TAKE_PROFIT = 0.10  # 10%


def get_sol_price() -> float:
    """Get current SOL price from Raydium."""
    try:
        resp = httpx.get(
            "https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112",
            timeout=5
        )
        data = resp.json()
        # Get first SOL pair
        pairs = data.get("pairs", [])
        for pair in pairs:
            if "solana" in pair.get("chainId", "").lower():
                return float(pair["priceUsd"])
        return 86.50
    except:
        return 86.50


def analyze_and_trade(prices: list, current_price: float) -> dict:
    """Analyze market and return trade decision."""
    if len(prices) < 5:
        return {"action": "WAIT", "reason": "Collecting data"}

    prices_arr = np.array(prices)
    current = current_price
    ma5 = np.mean(prices_arr[-5:])
    ma10 = np.mean(prices_arr[-10:])

    # Range detection
    high = max(prices_arr[-20:]) if len(prices_arr) >= 20 else max(prices_arr)
    low = min(prices_arr[-20:]) if len(prices_arr) >= 20 else min(prices_arr)
    position = (current - low) / (high - low) if high > low else 0.5

    # AGGRESSIVE STRATEGIES

    # 1. Range extremes
    if position < 0.25:
        return {
            "action": "BUY",
            "reason": f"Price at {position*100:.0f}% of range (bottom)",
            "stop_loss": current * (1 - STOP_LOSS),
            "take_profit": current * (1 + TAKE_PROFIT)
        }
    elif position > 0.75:
        return {
            "action": "SELL",
            "reason": f"Price at {position*100:.0f}% of range (top)",
            "stop_loss": current * (1 + STOP_LOSS),
            "take_profit": current * (1 - TAKE_PROFIT)
        }

    # 2. MA cross
    if current < ma5 * 0.998:
        return {
            "action": "BUY",
            "reason": f"Below MA5 by {(1-current/ma5)*100:.2f}%",
            "stop_loss": current * (1 - STOP_LOSS),
            "take_profit": current * (1 + TAKE_PROFIT)
        }
    elif current > ma5 * 1.002:
        return {
            "action": "SELL",
            "reason": f"Above MA5 by {(current/ma5-1)*100:.2f}%",
            "stop_loss": current * (1 + STOP_LOSS),
            "take_profit": current * (1 - TAKE_PROFIT)
        }

    # 3. MA10 bounce
    if current < ma10 * 0.995:
        return {
            "action": "BUY",
            "reason": f"Below MA10 by {(1-current/ma10)*100:.2f}%",
            "stop_loss": current * (1 - STOP_LOSS),
            "take_profit": current * (1 + TAKE_PROFIT)
        }

    return {"action": "WAIT", "reason": "No clear signal"}


def save_trade(trade: dict):
    """Save trade to state file."""
    state = {
        "enabled": True,
        "start_time": "2026-02-14T07:13:34.723492",
        "balance_usd": 500.0,
        "initial_balance": 500.0,
        "trades": [],
        "stats": {
            "total_trades": 1,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "current_streak": 0,
            "best_streak": 0,
            "worst_streak": 0
        },
        "signals": [trade]
    }
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    print("\n" + "="*60)
    print("🚀 SIMPLE PAPER TRADER - AGGRESSIVE MODE")
    print("="*60)
    print(f"Trade Size: ${TRADE_SIZE}")
    print(f"Stop Loss: {STOP_LOSS*100}%")
    print(f"Take Profit: {TAKE_PROFIT*100}%")
    print(f"Cycle: 30 seconds")
    print("="*60 + "\n")

    prices = []
    cycle = 0

    while True:
        cycle += 1
        current_time = datetime.now().strftime("%H:%M:%S")

        try:
            price = get_sol_price()
            prices.append(price)
            if len(prices) > 100:
                prices = prices[-100:]

            decision = analyze_and_trade(prices, price)

            print(f"[{current_time}] Cycle {cycle} | SOL: ${price:.2f}")
            print(f"   Prices tracked: {len(prices)}")

            if decision["action"] in ["BUY", "SELL"]:
                print(f"   🎯 SIGNAL: {decision['action']}")
                print(f"   Reason: {decision['reason']}")
                print(f"   SL: ${decision['stop_loss']:.2f} | TP: ${decision['take_profit']:.2f}")

                # Save trade
                trade = {
                    "time": current_time,
                    "symbol": "SOL",
                    "direction": decision["action"],
                    "price": price,
                    "size": TRADE_SIZE,
                    "stop_loss": decision["stop_loss"],
                    "take_profit": decision["take_profit"],
                    "reason": decision["reason"]
                }
                save_trade(trade)
                print(f"   ✅ Trade saved to state")
            else:
                print(f"   📭 {decision['reason']}")

            print()

        except KeyboardInterrupt:
            print("\n👋 Stopped")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
