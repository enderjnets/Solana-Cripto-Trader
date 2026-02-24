#!/usr/bin/env python3
"""
COMPLETE PAPER TRADING RESET AND FIX
Resetea y recalcula todo desde cero
"""

import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

PAPER_STATE_FILE = Path(__file__).parent / "data" / "paper_trading_state.json"
BACKUP_FILE = Path(__file__).parent / "data" / "paper_trading_state_final_backup.json"

def main():
    print("=" * 60)
    print("🔧 COMPLETE PAPER TRADING RESET & FIX")
    print("=" * 60)
    
    # Load current state
    with open(PAPER_STATE_FILE, 'r') as f:
        state = json.load(f)
    
    # Backup
    with open(BACKUP_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"\n✅ Backup saved: {BACKUP_FILE}")
    
    # Constants
    INITIAL_BALANCE = 700.0  # Fixed starting balance
    TAKER_FEE = 0.0005  # 0.05%
    
    trades = state.get('trades', [])
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    open_trades = [t for t in trades if t.get('status') == 'open']
    
    print(f"\n📊 Found {len(closed_trades)} closed trades, {len(open_trades)} open trades")
    
    # Start fresh from initial balance
    balance = INITIAL_BALANCE
    total_pnl = 0.0
    total_fees = 0.0
    winning = 0
    losing = 0
    margin_locked = 0.0
    
    print("\n📋 **RECALCULATING ALL TRADES:**\n")
    
    # Process closed trades
    for i, t in enumerate(closed_trades, 1):
        symbol = t.get('symbol', 'UNKNOWN')
        direction = t.get('direction', 'unknown')
        entry = t.get('entry_price', 0)
        exit_p = t.get('exit_price', 0)
        size = t.get('size', 0)  # Notional
        margin = t.get('margin', size)
        
        # Simulate: Open trade
        balance -= margin  # Deduct margin
        entry_fee = size * TAKER_FEE
        balance -= entry_fee
        total_fees += entry_fee
        
        # Calculate P&L
        if direction in ['bullish', 'long']:
            pnl_raw = size * (exit_p - entry) / entry if entry > 0 else 0
        else:
            pnl_raw = size * (entry - exit_p) / entry if entry > 0 else 0
        
        # Exit fee
        exit_fee = size * TAKER_FEE
        total_fees += exit_fee
        
        # Net P&L
        pnl_net = pnl_raw - exit_fee
        total_pnl += pnl_net
        
        # Close trade: return margin + pnl
        balance += margin + pnl_net
        
        # Update trade record
        t['pnl'] = round(pnl_net, 2)
        t['pnl_pct'] = round((pnl_raw / margin * 100) if margin > 0 else 0, 2)
        t['entry_fee'] = round(entry_fee, 4)
        t['exit_fee'] = round(exit_fee, 4)
        
        if pnl_net > 0:
            winning += 1
        elif pnl_net < 0:
            losing += 1
        
        emoji = '✅' if pnl_net > 0 else '❌'
        print(f"  {i}. {symbol} {direction[:4]}: Entry ${entry:.6f} → Exit ${exit_p:.6f} | {emoji} ${pnl_net:.2f} ({t['pnl_pct']:+.2f}%)")
    
    # Process open trades
    print(f"\n🔓 **OPEN TRADES:**\n")
    for t in open_trades:
        symbol = t.get('symbol', 'UNKNOWN')
        direction = t.get('direction', 'unknown')
        entry = t.get('entry_price', 0)
        size = t.get('size', 0)
        margin = t.get('margin', size)
        
        # Deduct margin and entry fee
        balance -= margin
        margin_locked += margin
        entry_fee = size * TAKER_FEE
        balance -= entry_fee
        total_fees += entry_fee
        
        # Update trade
        t['entry_fee'] = round(entry_fee, 4)
        
        print(f"  {symbol} {direction[:4]} @ ${entry:.6f} | Margin: ${margin:.2f} | Fee: ${entry_fee:.4f}")
    
    # Calculate stats
    total_trades = len(closed_trades)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0
    
    # Update state
    state['balance_usd'] = round(balance, 2)
    state['initial_balance'] = INITIAL_BALANCE
    state['margin_used'] = round(margin_locked, 2)
    
    if 'stats' not in state:
        state['stats'] = {}
    
    state['stats']['total_trades'] = total_trades
    state['stats']['winning_trades'] = winning
    state['stats']['losing_trades'] = losing
    state['stats']['win_rate'] = round(win_rate, 1)
    state['stats']['total_pnl'] = round(total_pnl, 2)
    state['stats']['total_fees'] = round(total_fees, 2)
    
    # Save
    with open(PAPER_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    # Report
    print("\n" + "=" * 60)
    print("✅ **STATE COMPLETELY RECALCULATED**")
    print("=" * 60)
    print(f"\n💰 Balance: ${balance:.2f}")
    print(f"💵 Initial: ${INITIAL_BALANCE:.2f}")
    print(f"📈 Growth: ${balance - INITIAL_BALANCE:.2f} ({((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.2f}%)")
    print(f"🔓 Margin Locked: ${margin_locked:.2f}")
    print(f"📊 Total P&L: ${total_pnl:.2f}")
    print(f"💸 Total Fees: ${total_fees:.2f}")
    print(f"✅ Winning: {winning}")
    print(f"❌ Losing: {losing}")
    print(f"📈 Win Rate: {win_rate:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
