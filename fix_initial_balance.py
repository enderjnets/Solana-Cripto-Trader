#!/usr/bin/env python3
"""
Fix Paper Trading - Restore correct initial_balance and margin_used
"""

import json
from pathlib import Path

PAPER_STATE_FILE = Path(__file__).parent / "data" / "paper_trading_state.json"

def main():
    print("🔧 Arreglando paper trading state...\n")

    with open(PAPER_STATE_FILE, 'r') as f:
        state = json.load(f)

    # Set correct initial_balance (fixed value)
    state['initial_balance'] = 700.0

    # Calculate margin_used from open trades
    open_trades = [t for t in state.get('trades', []) if t.get('status') == 'open']
    margin_used = sum(t.get('margin', t.get('size', 0)) for t in open_trades)
    state['margin_used'] = round(margin_used, 2)

    balance = state.get('balance_usd', 0)
    equity = balance + margin_used
    initial = state['initial_balance']
    growth = equity - initial

    print(f"✅ Initial Balance: ${initial:.2f}")
    print(f"💰 Balance: ${balance:.2f}")
    print(f"🔓 Margin Used: ${margin_used:.2f}")
    print(f"💎 Equity: ${equity:.2f}")
    print(f"📈 Growth: ${growth:.2f} ({(growth/initial*100):+.2f}%)")

    # Save
    with open(PAPER_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ Estado arreglado y guardado")

if __name__ == "__main__":
    main()
