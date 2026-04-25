#!/usr/bin/env python3
"""
Multi-Token Trading Agent v2
============================
Scans entire market and executes trades across ALL profitable tokens.

Usage:
    python3 multi_token_agent.py
"""

import json
import asyncio
import httpx
from datetime import datetime
from pathlib import Path

PAPER_STATE_FILE = Path(__file__).parent / "paper_trading_state.json"

TRADE_SIZE = 20  # $20 per token
MAX_TOKENS = 5


async def scan_market():
    """Scan for trading opportunities."""
    tokens = [
        ("SOL", "So11111111111111111111111111111111111111112"),
        ("JUP", "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2"),
        ("WIF", "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP"),
        ("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"),
        ("POPCAT", "7GCihgDB8Fe6JKr2mG9VDLxrGkZaGtD1W89VjMW9w8s"),
    ]

    signals = []
    async with httpx.AsyncClient() as client:
        for symbol, address in tokens:
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
                    change = float(pair.get("priceChange", {}).get("h24", 0))
                    volume = float(pair.get("volume", {}).get("h24", 0))

                    # Generate signal
                    if change > 5:
                        signals.append({
                            "symbol": symbol,
                            "price": price,
                            "change": change,
                            "signal": "BUY",
                            "reason": f"Momentum +{change:.1f}%"
                        })
                    elif change < -5:
                        signals.append({
                            "symbol": symbol,
                            "price": price,
                            "change": change,
                            "signal": "BUY",
                            "reason": f"Dip opportunity {change:.1f}%"
                        })
                    elif volume > 500000 and abs(change) > 2:
                        signals.append({
                            "symbol": symbol,
                            "price": price,
                            "change": change,
                            "signal": "SELL" if change > 0 else "BUY",
                            "reason": f"High volume ${volume/1e6:.1f}M"
                        })

            except Exception as e:
                continue

    return signals


async def main():
    print("\n" + "="*60)
    print("🚀 MULTI-TOKEN TRADING AGENT v2")
    print("="*60)

    state = {
        "balance_usd": 500.0,
        "trades": [],
        "signals": []
    }

    while True:
        print(f"\n🔄 {datetime.now().strftime('%H:%M:%S')} - Scanning market...")

        signals = await scan_market()

        if signals:
            print(f"\n🎯 Found {len(signals)} signals:")
            for s in signals:
                emoji = "🟢" if s["signal"] == "BUY" else "🔴"
                print(f"   {emoji} {s['symbol']}: {s['signal']} @ ${s['price']:.6f} ({s['change']:+.1f}%)")
                print(f"      Reason: {s['reason']}")

                # Execute trade
                trade = {
                    "time": datetime.now().isoformat(),
                    "symbol": s["symbol"],
                    "direction": s["signal"],
                    "price": s["price"],
                    "size": TRADE_SIZE,
                    "reason": s["reason"],
                    "status": "open",
                    "pnl": 0
                }
                state["trades"].append(trade)
        else:
            print("   No signals found")

        # Calculate performance
        closed = [t for t in state["trades"] if t["status"] == "closed"]
        total_pnl = sum(t.get("pnl", 0) for t in closed)
        win_rate = len([t for t in closed if t.get("pnl", 0) > 0]) / len(closed) * 100 if closed else 0

        print(f"\n📊 Performance:")
        print(f"   Total Trades: {len(state['trades'])}")
        print(f"   Closed: {len(closed)}")
        print(f"   P&L: ${total_pnl:.2f}")
        print(f"   Win Rate: {win_rate:.1f}%")

        # Save state
        state["signals"] = signals[:5]
        PAPER_STATE_FILE.write_text(json.dumps(state, indent=2))

        await asyncio.sleep(30)  # Every 30 seconds


if __name__ == "__main__":
    asyncio.run(main())
