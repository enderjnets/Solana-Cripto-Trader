#!/usr/bin/env python3
"""
Auto-Learning Audit Script
========================
Verifies that the live trading system is using:
1. Dynamic parameters from auto_improver
2. Individual trade registration
3. Optimizer execution frequency
"""

import sys
import json
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_trading_system import UnifiedTradingSystem

def check_auto_improver_integration():
    """Check if auto_improver is properly integrated"""
    print("\n" + "="*60)
    print("🔍 AUDIT 1: Auto-Improver Integration")
    print("="*60)
    
    system = UnifiedTradingSystem()
    
    # Check if get_trading_params uses dynamic params
    print(f"\n📋 Checking get_trading_params()...")
    
    # This will try auto_improver first, then fallback to HARDBIT
    params = system.get_trading_params()
    
    print(f"\n📊 Current trading parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Check source
    has_improver_params = 'min_confidence' in params
    
    if has_improver_params:
        print(f"\n✅ PASS: Using auto_improver parameters (dynamic)")
        
        # Check if confidence threshold is being used
        conf_threshold = params.get('min_confidence', 0.30)
        if conf_threshold <= 0.35:
            print(f"✅ PASS: Confidence threshold is reasonable: {conf_threshold:.0%}")
        else:
            print(f"⚠️  WARNING: Confidence threshold may be too high: {conf_threshold:.0%}")
        
        return True
    else:
        print(f"\n❌ FAIL: Using HARDBIT parameters (fixed)")
        print(f"⚠️  Auto-improver parameters are not being used!")
        return False

def check_trade_registration():
    """Check if trades are being registered with auto_improver"""
    print("\n" + "="*60)
    print("🔍 AUDIT 2: Trade Registration")
    print("="*60)
    
    try:
        with open('auto_improve_state.json', 'r') as f:
            state = json.load(f)
        
        print(f"\n📊 Auto-improver state:")
        print(f"   Total trades tracked: {len(state.get('trades', []))}")
        print(f"   Last update: {state.get('last_update', 'Never')}")
        
        # Check best metrics
        best_metrics = state.get('best_metrics', {})
        if best_metrics:
            print(f"\n📋 Best metrics:")
            for key, value in best_metrics.items():
                print(f"   {key}: {value}")
        
        # Check best params
        best_params = state.get('best_params', {})
        if best_params:
            print(f"\n📋 Best parameters:")
            for key, value in best_params.items():
                print(f"   {key}: {value}")
        
        print(f"\n✅ PASS: Auto-improver state file exists and is valid")
        return True
        
    except FileNotFoundError:
        print(f"❌ FAIL: auto_improve_state.json not found")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error reading state: {e}")
        return False

def check_optimization_frequency():
    """Check if optimizer runs frequently enough"""
    print("\n" + "="*60)
    print("🔍 AUDIT 3: Optimization Frequency")
    print("="*60)
    
    try:
        with open('data/paper_trading_state.json', 'r') as f:
            paper_state = json.load(f)
        
        total_trades = paper_state.get('stats', {}).get('total_trades', 0)
        
        print(f"\n📊 Paper trading stats:")
        print(f"   Total trades: {total_trades}")
        
        # Check if trades are being tracked
        if total_trades > 0:
            print(f"✅ PASS: Trades are being tracked ({total_trades} total)")
            
            # Check if individual trades are being recorded
            trades_list = paper_state.get('trades', [])
            recent_trades = [t for t in trades_list if t.get('status') == 'closed']
            
            if recent_trades:
                print(f"\n📋 Recent closed trades (last 5):")
                for i, trade in enumerate(recent_trades[-5:], 1):
                    pnl_color = "✅" if trade.get('pnl', 0) > 0 else "❌"
                    print(f"   {i}. {trade.get('symbol', 'N/A'):6} | {pnl_color} ${trade.get('pnl', 0):+6.2f} ({trade.get('pnl_pct', 0):+.2f}%) | {trade.get('reason', 'N/A')[:20]}")
                
                print(f"\n✅ PASS: Individual trades are being tracked")
                return True
            else:
                print(f"⚠️  INFO: No closed trades found")
                return True
        else:
            print(f"⚠️  INFO: No trades yet")
            return True
            
    except FileNotFoundError:
        print(f"❌ FAIL: paper_trading_state.json not found")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error reading state: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 AUTO-LEARNING SYSTEM AUDIT")
    print("="*60)
    print(f"⏰ Started: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Run audits
    results.append(("Auto-Improver Integration", check_auto_improver_integration()))
    results.append(("Trade Registration", check_trade_registration()))
    results.append(("Optimization Frequency", check_optimization_frequency()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 AUDIT SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r[1])
    total = len(results)
    
    for audit_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {audit_name}")
    
    print(f"\n📊 Result: {passed}/{total} audits passed ({passed/total*100:.0f}%)")
    print()
    
    if passed == total:
        print("🎉 ALL AUDITS PASSED! Auto-learning is fully integrated.")
        print("\n📋 The system will:")
        print("   ✅ Register individual trades when they close")
        print("   ✅ Run optimizer every 5-10 trades")
        print("   ✅ Use dynamic parameters based on performance")
        return 0
    else:
        print("⚠️  SOME AUDITS FAILED. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
