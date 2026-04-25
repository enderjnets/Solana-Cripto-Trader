#!/usr/bin/env python3
"""
Unified Trading Dashboard
========================
Shows all agents working together to achieve 5% daily target.

Team:
- Signal Generator → Generates BUY/SELL signals
- Paper Trader → Executes trades ($50 each)
- Strategy Optimizer → Improves parameters
- Risk Manager → Monitors drawdown
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta

PAPER_STATE_FILE = Path(__file__).parent / "paper_trading_state.json"
OPTIMIZER_STATE_FILE = Path(__file__).parent / "optimizer_state.json"


def main():
    while True:
        # Load data
        paper_state = json.loads(PAPER_STATE_FILE.read_text()) if PAPER_STATE_FILE.exists() else {}
        signals = paper_state.get("signals", [])
        stats = paper_state.get("stats", {})
        
        # Calculate P&L
        balance = paper_state.get("balance_usd", 500)
        initial = paper_state.get("initial_balance", 500)
        pnl = balance - initial
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0
        
        # Daily target: 5% = $25 on $500
        daily_target = 25.0
        target_progress = (pnl / daily_target * 100) if daily_target > 0 else 0
        
        # Trades analysis
        total_trades = stats.get("total_trades", len(signals))
        winning = stats.get("winning_trades", 0)
        losing = stats.get("losing_trades", 0)
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        
        # Clear screen
        print("\033[2J\033[H", end="")
        
        # Header
        print("╔" + "═" * 78 + "╗")
        print("║" + " 🤖 UNIFIED TRADING TEAM - TARGET: +5% DAILY ".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Daily Progress
        print(f"\n  🎯 DAILY PROGRESS (Target: +${daily_target})")
        print(f"  ┌────────────────────────────────────────────────────────────────┐")
        progress_bar = "█" * int(target_progress / 5) + "░" * (20 - int(target_progress / 5))
        print(f"  │ [{progress_bar}] {pnl_pct:+.2f}% │ ${pnl:+.2f} │ {target_progress:.1f}% of target")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        
        # Team Status
        print(f"\n  👥 TEAM STATUS")
        print(f"  ┌────────────────────────────────────────────────────────────────┐")
        print(f"  │ ✅ Signal Generator   │ Generando señales agresivas           │")
        print(f"  │ ✅ Paper Trader      │ Ejecutando ${total_trades} trades (${'$50'} c/u)         │")
        print(f"  │ ✅ Strategy Optimizer│ Mejorando parámetros automáticamente  │")
        print(f"  │ ✅ Risk Manager      │ Monitoreando drawdown < 10%          │")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        
        # Balance
        print(f"\n  💰 BALANCE")
        print(f"  ┌────────────────────────────────────────────────────────────────┐")
        print(f"  │  Initial:     ${initial:>10.2f}                                   │")
        print(f"  │  Current:     ${balance:>10.2f}                                   │")
        print(f"  │  P&L:         ${pnl:>+10.2f}  ({pnl_pct:+.2f}%)                     │")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        
        # Statistics
        print(f"\n  📊 STATISTICS")
        print(f"  ┌────────────────────────────────────────────────────────────────┐")
        print(f"  │  Total Trades:     {total_trades:>5}                                   │")
        print(f"  │  Winning:          {winning:>5}                                     │")
        print(f"  │  Losing:           {losing:>5}                                     │")
        print(f"  │  Win Rate:        {win_rate:>5.1f}%                                    │")
        print(f"  │  Avg Win:         ${stats.get('avg_win', 0):>5.2f}                                     │")
        print(f"  │  Avg Loss:        ${stats.get('avg_loss', 0):>5.2f}                                     │")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        
        # Recent Signals
        print(f"\n  📡 RECENT SIGNALS")
        print(f"  ┌────────────────────────────────────────────────────────────────┐")
        if signals:
            for sig in signals[-5:]:
                emoji = "🔴" if sig.get("direction") == "SELL" else "🟢"
                print(f"  │ {emoji} {sig.get('time', '')[:8]} | {sig.get('symbol', 'SOL'):>3} | {sig.get('direction', ''):>4} | ${sig.get('price', 0):>7.2f} | {sig.get('reason', '')[:20]}")
        else:
            print(f"  │  No signals yet                                            │")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        
        # Next trade calculation
        trades_needed = max(0, 10 - total_trades)  # Need ~10 trades to reach 5%
        avg_win_needed = daily_target / trades_needed if trades_needed > 0 else 0
        
        print(f"\n  📈 TO REACH DAILY TARGET (+${daily_target})")
        print(f"  ┌────────────────────────────────────────────────────────────────┐")
        print(f"  │  Trades needed:     {trades_needed:>5} (at ${avg_win_needed:.2f} avg win)      │")
        print(f"  │  Current trend:     {'📈 Profitable' if pnl > 0 else '📉 Need wins':<20}              │")
        print(f"  │  Time remaining:    {'12+ hours':<20}              │")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        
        print(f"\n  ⏰ Updated: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit")
        
        time.sleep(3)


if __name__ == "__main__":
    main()
