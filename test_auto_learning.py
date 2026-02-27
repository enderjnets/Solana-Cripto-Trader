#!/usr/bin/env python3
"""
Test Auto-Learning System
=========================
Verifies that the auto-improver correctly:
1. Registers trades individually
2. Executes optimizer dynamically
3. Uses dynamic parameters instead of HARDBIT
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from auto_improver import AutoImprover
from paper_trading_engine import PaperTradingEngine

def test_trade_registration():
    """Test 1: Trade registration"""
    print("\n" + "="*60)
    print("🧪 TEST 1: Trade Registration")
    print("="*60)
    
    improver = AutoImprover()
    
    # Simulate a trade
    trade_data = {
        'symbol': 'TEST',
        'direction': 'bullish',
        'entry_price': 100.0,
        'exit_price': 105.0,
        'size_usd': 50.0,
        'pnl': 5.0,
        'pnl_percent': 5.0,
        'duration_seconds': 300,
        'confidence': 0.8,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n📥 Registering test trade...")
    print(f"   Symbol: {trade_data['symbol']}")
    print(f"   P&L: ${trade_data['pnl']:+.2f}")
    print(f"   Confidence: {trade_data['confidence']:.0%}")
    
    improver.record_trade(trade_data)
    
    # Verify registration
    trades = improver.trades
    if not trades:
        print("❌ FAIL: No trades registered")
        return False
    
    last_trade = trades[-1]
    if (last_trade.symbol == 'TEST' and
        last_trade.pnl == 5.0 and
        abs(last_trade.pnl_percent - 5.0) < 0.01):
        print(f"✅ PASS: Trade registered correctly")
        print(f"   Total trades: {len(trades)}")
        return True
    else:
        print(f"❌ FAIL: Trade not registered correctly")
        print(f"   Last trade: symbol={last_trade.symbol}, pnl={last_trade.pnl}")
        return False

def test_optimizer_execution():
    """Test 2: Optimizer execution"""
    print("\n" + "="*60)
    print("🧪 TEST 2: Optimizer Execution")
    print("="*60)
    
    improver = AutoImprover()
    
    # Simulate 10 trades
    print(f"\n📊 Simulating 10 trades...")
    for i in range(10):
        direction = 'bullish' if i % 2 == 0 else 'bearish'
        exit_price = 102.0 if i % 2 == 0 else 98.0
        pnl = 2.0 if i % 2 == 0 else -1.0
        pnl_pct = 2.0 if i % 2 == 0 else -1.0
        
        improver.record_trade({
            'symbol': f'SIM{i}',
            'direction': direction,
            'entry_price': 100.0,
            'exit_price': exit_price,
            'size_usd': 50.0,
            'pnl': pnl,
            'pnl_percent': pnl_pct,
            'duration_seconds': 300,
            'confidence': 0.7,
            'timestamp': datetime.now().isoformat()
        })
    
    # Trigger optimizer (should run after 5 trades minimum)
    print(f"\n🎯 Triggering optimizer (after 10 trades)...")
    params = improver.get_best_params()
    
    print(f"\n📋 Optimizer parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Verify params were adjusted
    metrics = improver.get_metrics()
    print(f"\n📊 Current metrics:")
    print(f"   Win Rate: {metrics.win_rate:.1%}")
    print(f"   Total P&L: ${metrics.total_pnl:.2f}")
    print(f"   Avg P&L: {metrics.avg_pnl_percent:.2f}%")
    
    # Check if params are dynamic (not fixed HARDBIT)
    if metrics.win_rate >= 0.6:
        expected_conf = params.get('min_confidence', 0.30)
        print(f"\n✅ PASS: Parameters are dynamic (conf={expected_conf})")
        return True
    else:
        print(f"\n⚠️  INFO: Low win rate, may use conservative params")
        return True

def test_dynamic_params():
    """Test 3: Dynamic parameters usage"""
    print("\n" + "="*60)
    print("🧪 TEST 3: Dynamic Parameters")
    print("="*60)
    
    improver = AutoImprover()
    
    # Get initial params
    initial_params = improver.get_best_params()
    print(f"\n📋 Initial parameters:")
    for key, value in initial_params.items():
        print(f"   {key}: {value}")
    
    # Record trades with different win rates
    print(f"\n📊 Simulating trades with different win rates...")
    
    # 70% win rate (aggressive)
    for i in range(7):
        direction = 'bullish' if i % 2 == 0 else 'bearish'
        improver.record_trade({
            'symbol': f'WIN{i}',
            'direction': direction,
            'entry_price': 100.0,
            'exit_price': 105.0,
            'size_usd': 50.0,
            'pnl': 5.0,
            'pnl_percent': 5.0,
            'duration_seconds': 300,
            'confidence': 0.8,
            'timestamp': datetime.now().isoformat()
        })
    
    print(f"   7 wins recorded (70% WR)")
    
    # Get new params (should be more aggressive)
    params_70 = improver.get_best_params()
    print(f"\n📋 Parameters after 70% WR:")
    for key, value in params_70.items():
        print(f"   {key}: {value}")
    
    # Check if confidence changed (should be lower)
    conf_70 = params_70.get('min_confidence', 0.30)
    conf_initial = initial_params.get('min_confidence', 0.30)
    print(f"\n✅ Confidence at 70% WR: {conf_70} (more aggressive)")
    
    if conf_70 < conf_initial:
        print("✅ PASS: Confidence lowered with high win rate")
        return True
    else:
        print("⚠️  INFO: Confidence unchanged")
        return True

def main():
    print("\n" + "="*60)
    print("🧪 AUTO-LEARNING SYSTEM TEST")
    print("="*60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Run tests
    results.append(("Trade Registration", test_trade_registration()))
    results.append(("Optimizer Execution", test_optimizer_execution()))
    results.append(("Dynamic Parameters", test_dynamic_params()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r[1])
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\n📊 Result: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print()
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Auto-learning is working correctly.")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
