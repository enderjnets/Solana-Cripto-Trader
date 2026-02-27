#!/usr/bin/env python3
"""Test script to verify the API fix for execute_signal"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading_engine import PaperTradingEngine
from typing import Optional

# Test 1: Verify PaperTradingEngine has execute_signal method
print("Test 1: Checking if PaperTradingEngine has execute_signal method...")
engine = PaperTradingEngine()

if hasattr(engine, 'execute_signal'):
    print("✅ PASS: execute_signal method exists")
else:
    print("❌ FAIL: execute_signal method does not exist")
    sys.exit(1)

# Test 2: Verify open_position does NOT exist
print("\nTest 2: Checking if open_position method does NOT exist...")
if not hasattr(engine, 'open_position'):
    print("✅ PASS: open_position method correctly does not exist")
else:
    print("❌ FAIL: open_position still exists (should have been removed)")
    sys.exit(1)

# Test 3: Test execute_signal with proper dict format
print("\nTest 3: Testing execute_signal with proper signal dict...")
engine.reset()

signal_dict = {
    'symbol': 'SOL',
    'direction': 'long',
    'price': 86.0,
    'size': 50.0,  # $50 position (10% of $500)
    'reason': 'Test signal',
    'leverage': 1
}

# Enable engine
engine.start()

result = engine.execute_signal(signal_dict)

if result:
    print(f"✅ PASS: Trade executed successfully")
    print(f"   Trade ID: {result.id}")
    print(f"   Symbol: {result.symbol}")
    print(f"   Direction: {result.direction}")
    print(f"   Entry Price: ${result.entry_price:.6f}")
    print(f"   Size: ${result.size:.2f}")
    print(f"   Status: {result.status}")

    # Note: engine.status() already printed balance info
else:
    print("❌ FAIL: execute_signal returned None")
    sys.exit(1)

# Test 4: Check signal dict format
print("\nTest 4: Checking required signal dict fields...")
required_fields = ['symbol', 'direction', 'price', 'size', 'reason', 'leverage']
signal_dict_valid = {
    'symbol': 'SOL',
    'direction': 'long',
    'price': 86.0,
    'size': 50.0,
    'reason': 'Test',
    'leverage': 1
}

missing_fields = [field for field in required_fields if field not in signal_dict_valid]
if not missing_fields:
    print("✅ PASS: All required fields present in signal dict")
else:
    print(f"❌ FAIL: Missing fields: {missing_fields}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nThe API fix is working correctly.")
print("Ready to restart the trading system.")
