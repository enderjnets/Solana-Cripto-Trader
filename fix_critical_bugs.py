#!/usr/bin/env python3
"""
Critical Bug Fixes - Trading System
===================================
Fixes:
1. Balance negative protection (already in paper_trading_engine.py, line 387-394)
2. Portfolio value calculation (fixed in monitor_trading.py)
3. Hardcoded initial_balance (fixed in monitor_trading.py)
4. Add comprehensive validation and logging
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent
STATE_FILE = PROJECT_DIR / "data" / "paper_trading_state.json"
BACKUP_DIR = PROJECT_DIR / "data" / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

def backup_state():
    """Backup current state before fixes"""
    if STATE_FILE.exists():
        backup_file = BACKUP_DIR / f"paper_trading_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_file.write_text(STATE_FILE.read_text())
        print(f"✅ Backup created: {backup_file.name}")

def validate_state():
    """Validate state and report issues"""
    with open(STATE_FILE) as f:
        state = json.load(f)

    issues = []

    # Check 1: Balance negative
    balance = state.get("balance_usd", 0)
    if balance < 0:
        issues.append(f"❌ CRITICAL: Balance is negative (${balance:,.2f})")

    # Check 2: initial_balance consistency
    initial_balance = state.get("initial_balance", 500)
    if initial_balance != 500:
        issues.append(f"⚠️ WARNING: initial_balance is ${initial_balance}, should be $500")

    # Check 3: Margin consistency
    open_trades = [t for t in state.get("trades", []) if t.get("status") == "open"]
    calculated_margin = sum(t.get("margin", t.get("size", 0)) for t in open_trades)
    saved_margin = state.get("margin_used", 0)
    if abs(calculated_margin - saved_margin) > 0.01:
        issues.append(f"⚠️ WARNING: margin_used inconsistency (saved: ${saved_margin:.2f}, calculated: ${calculated_margin:.2f})")

    # Check 4: Portfolio value sanity
    portfolio_value = balance + sum(t.get("size", 0) for t in open_trades)
    if portfolio_value < 0:
        issues.append(f"❌ CRITICAL: Portfolio value is negative (${portfolio_value:,.2f})")

    # Check 5: P&L calculation
    total_pnl = state.get("stats", {}).get("total_pnl", 0)
    expected_pnl = balance - initial_balance
    if abs(total_pnl - expected_pnl) > 100:  # Allow small differences from fees
        issues.append(f"⚠️ WARNING: P&L inconsistency (stats: ${total_pnl:.2f}, expected: ${expected_pnl:.2f})")

    return issues

def fix_state():
    """Apply fixes to state"""
    with open(STATE_FILE) as f:
        state = json.load(f)

    fixes_applied = []

    # Fix 1: Reset negative balance to 0 (minimum)
    if state.get("balance_usd", 0) < 0:
        state["balance_usd"] = 0
        fixes_applied.append("✅ Fixed negative balance (set to $0)")

    # Fix 2: Correct initial_balance
    if state.get("initial_balance") != 500:
        state["initial_balance"] = 500
        fixes_applied.append("✅ Fixed initial_balance (set to $500)")

    # Fix 3: Recalculate margin_used from open trades
    open_trades = [t for t in state.get("trades", []) if t.get("status") == "open"]
    calculated_margin = sum(t.get("margin", t.get("size", 0)) for t in open_trades)
    state["margin_used"] = calculated_margin
    fixes_applied.append(f"✅ Recalculated margin_used (${calculated_margin:.2f})")

    # Fix 4: Validate and fix trade P&L values
    for trade in state.get("trades", []):
        if trade.get("status") == "closed":
            size = trade.get("size", 0)
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            direction = trade.get("direction", "")

            # Recalculate P&L
            if entry_price > 0 and exit_price > 0:
                if direction in ["bullish", "long"]:
                    expected_pct = (exit_price - entry_price) / entry_price
                else:
                    expected_pct = (entry_price - exit_price) / entry_price

                expected_pnl = size * expected_pct
                saved_pnl = trade.get("pnl", 0)

                # If difference is huge (> 50%), fix it
                if abs(expected_pnl - saved_pnl) > abs(saved_pnl * 0.5):
                    trade["pnl"] = expected_pnl
                    trade["pnl_pct"] = expected_pct * 100

    # Save fixed state
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    return fixes_applied

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🔧 CRITICAL BUG FIXES - Trading System")
    print("="*60)

    if not STATE_FILE.exists():
        print(f"❌ ERROR: State file not found: {STATE_FILE}")
        return

    # Backup before fixing
    print("\n📦 Creating backup...")
    backup_state()

    # Validate current state
    print("\n🔍 Validating current state...")
    issues = validate_state()

    if not issues:
        print("✅ No issues found - state is healthy!")
        return

    print(f"\n⚠️  Found {len(issues)} issues:")
    for issue in issues:
        print(f"   {issue}")

    # Apply fixes
    print("\n🔧 Applying fixes...")
    fixes = fix_state()

    print(f"\n✅ Applied {len(fixes)} fixes:")
    for fix in fixes:
        print(f"   {fix}")

    # Re-validate
    print("\n🔍 Re-validating state...")
    remaining_issues = validate_state()

    if remaining_issues:
        print(f"⚠️  {len(remaining_issues)} issues remain:")
        for issue in remaining_issues:
            print(f"   {issue}")
    else:
        print("✅ All issues fixed! State is now healthy.")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
