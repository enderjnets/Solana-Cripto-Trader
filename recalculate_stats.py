#!/usr/bin/env python3
"""
Recalculate Trading Stats from Scratch
====================================
Rebuilds all stats based on actual trade history, eliminating corruption.
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
STATE_FILE = PROJECT_DIR / "data" / "paper_trading_state.json"

def recalculate_all_stats():
    """Recalculate all statistics from trades"""
    with open(STATE_FILE) as f:
        state = json.load(f)

    trades = state.get("trades", [])

    # Initialize new stats
    new_stats = {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "total_fees": 0.0,
        "liquidations": 0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "current_streak": 0,
        "best_streak": 0,
        "worst_streak": 0,
    }

    # Recalculate from closed trades
    current_streak = 0
    best_streak = 0
    worst_streak = 0

    for trade in trades:
        if trade.get("status") == "closed":
            new_stats["total_trades"] += 1
            pnl = trade.get("pnl", 0)
            new_stats["total_pnl"] += pnl

            # Count wins/losses
            if pnl > 0:
                new_stats["winning_trades"] += 1
                current_streak += 1 if current_streak >= 0 else 1
                if current_streak > best_streak:
                    best_streak = current_streak
            else:
                new_stats["losing_trades"] += 1
                current_streak -= 1 if current_streak <= 0 else -1
                if abs(current_streak) > worst_streak:
                    worst_streak = abs(current_streak)

            # Count liquidations
            if trade.get("reason") == "LIQUIDATION":
                new_stats["liquidations"] += 1

            # Add fees
            entry_fee = trade.get("entry_fee", 0)
            exit_fee = trade.get("exit_fee", 0)
            new_stats["total_fees"] += entry_fee + exit_fee

    # Calculate win rate
    if new_stats["total_trades"] > 0:
        new_stats["win_rate"] = (new_stats["winning_trades"] / new_stats["total_trades"]) * 100

    new_stats["best_streak"] = best_streak
    new_stats["worst_streak"] = worst_streak
    new_stats["current_streak"] = current_streak

    # Replace old stats
    old_stats = state["stats"].copy()
    state["stats"] = new_stats

    # Save state
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    # Print comparison
    print("\n" + "="*60)
    print("📊 STATS RECALCULATION COMPLETE")
    print("="*60)
    print(f"\n{'Metric':<20} {'Old':>15} {'New':>15} {'Change':>15}")
    print("-"*60)
    print(f"{'Total Trades':<20} {old_stats['total_trades']:>15} {new_stats['total_trades']:>15} {new_stats['total_trades'] - old_stats['total_trades']:>+15}")
    print(f"{'Winning Trades':<20} {old_stats['winning_trades']:>15} {new_stats['winning_trades']:>15} {new_stats['winning_trades'] - old_stats['winning_trades']:>+15}")
    print(f"{'Losing Trades':<20} {old_stats['losing_trades']:>15} {new_stats['losing_trades']:>15} {new_stats['losing_trades'] - old_stats['losing_trades']:>+15}")
    print(f"{'Total P&L':<20} ${old_stats['total_pnl']:>13.2f} ${new_stats['total_pnl']:>13.2f} ${new_stats['total_pnl'] - old_stats['total_pnl']:>+13.2f}")
    print(f"{'Total Fees':<20} ${old_stats['total_fees']:>13.2f} ${new_stats['total_fees']:>13.2f} ${new_stats['total_fees'] - old_stats['total_fees']:>+13.2f}")
    print(f"{'Win Rate':<20} {old_stats['win_rate']:>14.1f}% {new_stats['win_rate']:>14.1f}% {new_stats['win_rate'] - old_stats['win_rate']:>+14.1f}%")
    print(f"{'Liquidations':<20} {old_stats['liquidations']:>15} {new_stats['liquidations']:>15} {new_stats['liquidations'] - old_stats['liquidations']:>+15}")
    print(f"{'Best Streak':<20} {old_stats['best_streak']:>15} {new_stats['best_streak']:>15} {new_stats['best_streak'] - old_stats['best_streak']:>+15}")
    print(f"{'Worst Streak':<20} {old_stats['worst_streak']:>15} {new_stats['worst_streak']:>15} {new_stats['worst_streak'] - old_stats['worst_streak']:>+15}")
    print("="*60)

    # Verify P&L consistency
    balance = state.get("balance_usd", 0)
    initial_balance = state.get("initial_balance", 500)
    expected_pnl = balance - initial_balance

    print(f"\n📈 Balance vs Stats P&L:")
    print(f"   Current Balance: ${balance:,.2f}")
    print(f"   Initial Balance: ${initial_balance:,.2f}")
    print(f"   Expected P&L:    ${expected_pnl:,.2f}")
    print(f"   Stats P&L:       ${new_stats['total_pnl']:,.2f}")
    print(f"   Difference:      ${new_stats['total_pnl'] - expected_pnl:,.2f} (fees + open positions)")

    print("\n✅ Stats recalculated from trade history!")

if __name__ == "__main__":
    recalculate_all_stats()
