#!/usr/bin/env python3
"""
Diagnose why the trading system is not generating signals.
"""

import sys
from unified_trading_system import UnifiedTradingSystem, MLSignalGenerator
from typing import List, Dict

def main():
    print("="*60)
    print("🔍 TRADING SIGNAL DIAGNOSTIC")
    print("="*60)
    
    # Initialize ML signal generator
    ml_gen = MLSignalGenerator()
    
    # Simulate some price history
    symbols = ['SOL', 'BTC', 'ETH', 'WIF', 'JUP']
    
    print("\n📊 Simulating price history...")
    for symbol in symbols:
        # Add some fake prices to simulate history
        base_price = 100.0
        for i in range(10):
            price = base_price + (i * 0.5) + (i % 2)
            ml_gen.update_price(symbol, price)
        print(f"   {symbol}: {len(ml_gen.price_history.get(symbol, []))} prices")
    
    print("\n🎯 Testing signal generation...")
    for symbol in symbols:
        signal = ml_gen.generate_signal(symbol)
        print(f"\n   {symbol}:")
        print(f"      Direction: {signal['direction']}")
        print(f"      Confidence: {signal['confidence']:.1f}%")
        print(f"      Ensemble Score: {signal.get('ensemble_score', 0):.3f}")
        
        if signal['confidence'] > 0:
            components = signal.get('components', {})
            print(f"      RSI: {components.get('rsi', {}).get('value', 0):.1f}")
            print(f"      EMA Short: {components.get('ema_crossover', {}).get('short', 0):.2f}")
            print(f"      EMA Long: {components.get('ema_crossover', {}).get('long', 0):.2f}")
    
    print("\n" + "="*60)
    print("🎯 Threshold Check:")
    print("="*60)
    
    for symbol in symbols:
        signal = ml_gen.generate_signal(symbol)
        
        # Check thresholds
        ensemble_ok = abs(signal.get('ensemble_score', 0)) > 0.10
        confidence_ok = signal['confidence'] >= 25
        
        print(f"\n   {symbol}:")
        print(f"      Ensemble > 0.10: {'✅' if ensemble_ok else '❌'} ({signal.get('ensemble_score', 0):.3f})")
        print(f"      Confidence >= 25%: {'✅' if confidence_ok else '❌'} ({signal['confidence']:.1f}%)")
        print(f"      Would Trade: {'✅' if ensemble_ok and confidence_ok else '❌'}")
    
    print("\n" + "="*60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
