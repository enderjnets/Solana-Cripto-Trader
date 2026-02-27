#!/usr/bin/env python3
"""Monitor trading system for successful trades"""

import json
import time
from datetime import datetime

def monitor_trades():
    state_file = '/home/enderj/.openclaw/workspace/solana-jupiter-bot/data/paper_trading_state.json'
    log_file = '/home/enderj/.openclaw/workspace/solana-jupiter-bot/unified_trading_system.log'

    print("📊 MONITORING TRADING SYSTEM")
    print("="*60)

    # Read current state
    with open(state_file) as f:
        state = json.load(f)

    balance = state['balance_usd']
    total_trades = state['stats']['total_trades']
    winning_trades = state['stats']['winning_trades']
    losing_trades = state['stats']['losing_trades']
    win_rate = state['stats']['win_rate']

    print(f"\n📈 Current Balance: ${balance:.2f}")
    print(f"📊 Total Trades: {total_trades}")
    print(f"✅ Winning: {winning_trades} | ❌ Losing: {losing_trades}")
    print(f"🎯 Win Rate: {win_rate:.1f}%")

    # Check for recent trades
    if total_trades > 0:
        print(f"\n📜 Recent Trades:")
        for trade in state['trades'][-5:]:  # Last 5 trades
            if trade['status'] == 'closed':
                pnl = trade['pnl']
                pnl_pct = trade['pnl_pct']
                emoji = "✅" if pnl > 0 else "❌"
                print(f"   {emoji} {trade['symbol']} {trade['direction']}: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            else:
                print(f"   🟢 {trade['symbol']} {trade['direction']}: ${trade['size']:.2f} OPEN")

    # Check for errors in log
    print(f"\n🔍 Checking for errors...")
    with open(log_file, 'r') as f:
        lines = f.readlines()
        error_count = sum(1 for line in lines[-100:] if 'ERROR' in line and 'open_position' in line)
        if error_count > 0:
            print(f"   ⚠️ Found {error_count} 'open_position' errors in last 100 log lines")
        else:
            print(f"   ✅ No 'open_position' API errors found")

    # Check for successful trade execution
    print(f"\n🔍 Checking for successful trades...")
    with open(log_file, 'r') as f:
        lines = f.readlines()
        trade_opened_count = sum(1 for line in lines[-100:] if 'Trade opened' in line)
        if trade_opened_count > 0:
            print(f"   ✅ Found {trade_opened_count} successful trade executions")
        else:
            print(f"   ℹ️ No trades opened in last 100 log lines (normal if no signals)")

    print("\n" + "="*60)

if __name__ == '__main__':
    monitor_trades()
