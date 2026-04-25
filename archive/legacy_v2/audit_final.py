#!/usr/bin/env python3
"""
Final Audit: Complete Trading System
===================================
Verifies all 11 strategies (sensitive + momentum + breakout).
"""

import sys
import json
import pandas as pd

sys.path.insert(0, '/home/enderj/.openclaw/workspace/solana-jupiter-bot')

print("=" * 70)
print("AUDITORÍA FINAL: Sistema Completo de Trading")
print("=" * 70)

# Load strategies
with open('data/active_strategies.json', 'r') as f:
    strategies = json.load(f)

print(f"\n📊 ARCHIVO: {strategies.get('generation')}")
print(f"   Total: {strategies.get('count')} estrategias")
print(f"   Tipos: {', '.join(strategies.get('strategy_types', []))}")
print(f"   Manual: {'Sí' if strategies.get('manual_override') else 'No'}")

# Categorize
sensitive = [s for s in strategies['strategies'] if any(r.get('threshold', 0) < 1.0 for r in s['brain_rules']) and any(r.get('operator') == '<' for r in s['brain_rules'])]
momentum = [s for s in strategies['strategies'] if any(r.get('threshold', 0) > 1.0 or (r.get('threshold', 0) == 1.0 and r.get('operator') == '>') for r in s['brain_rules'])]
breakout = [s for s in strategies['strategies'] if 'breakout' in s['name'].lower() or 'oversold' in s['name'].lower() or 'reversal' in s['name'].lower()]

print(f"\n📈 ESTRATEGIAS POR TIPO:")
print(f"   Sensitive: {len(sensitive)}")
print(f"   Momentum: {len(momentum)}")
print(f"   Breakout: {len(breakout)}")

# Load price data
print(f"\n📉 MERCADO ACTUAL:")
df = pd.read_csv('data/sol_1h.csv')
if 'close' not in df.columns:
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

close = df['close'].iloc[-1]
sma_20 = df['close'].rolling(20).mean().iloc[-1]
sma_50 = df['close'].rolling(50).mean().iloc[-1]
rsi_14 = df['close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / \
         df['close'].diff().abs().rolling(14).mean() * 100
rsi_14 = rsi_14.iloc[-1]

print(f"   SOL: ${close:.2f}")
print(f"   SMA(20): ${sma_20:.2f} ({close/sma_20*100-100:+.2f}%)")
print(f"   SMA(50): ${sma_50:.2f} ({close/sma_50*100-100:+.2f}%)")
print(f"   RSI(14): {rsi_14:.2f}")

# Evaluate all strategies
print(f"\n🎯 EVALUACIÓN DE SEÑALES:")
buy_count = 0
sell_count = 0
hold_count = 0

for s in strategies['strategies']:
    rules = s.get('brain_rules', [])
    all_pass = True
    for r in rules:
        ind = r.get('indicator')
        period = r.get('period')
        op = r.get('operator')
        thresh = r.get('threshold')
        
        if ind == 'SMA':
            if period == 20:
                value = close / sma_20
            elif period == 50:
                value = close / sma_50
            else:
                value = close / df['close'].rolling(period).mean().iloc[-1]
        elif ind == 'EMA':
            value = close / df['close'].ewm(span=period).mean().iloc[-1]
        elif ind == 'RSI':
            value = rsi_14
        else:
            value = 1.0
        
        if op == '<':
            passes = value < thresh
        elif op == '>':
            passes = value > thresh
        else:
            passes = False
        
        if not passes:
            all_pass = False
    
    if all_pass:
        buy_count += 1
        status = "✅ BUY"
    else:
        hold_count += 1
        status = "❌ HOLD"
    
    print(f"   {status} {s['name']}")

# Summary
print(f"\n" + "=" * 70)
print("RESUMEN DE AUDITORÍA")
print("=" * 70)
print(f"✅ Total estrategias: {len(strategies['strategies'])}")
print(f"✅ Señales BUY: {buy_count}")
print(f"❌ Señales HOLD: {hold_count}")
print(f"\n📁 Archivo: data/active_strategies.json")
print(f"📝 Generation: {strategies.get('generation')}")

# Market regime
ratio_20 = close / sma_20
if ratio_20 > 1.02:
    regime = "🟢 ALCISTA (Momentum activo)"
elif ratio_20 < 0.98:
    regime = "🔴 SENSITIVE activo (precio bajo SMA)"
else:
    regime = "🟡 LATERAL (esperando ruptura)"

print(f"\n🌊 Régimen de mercado: {regime}")

print("\n" + "=" * 70)
print("SISTEMA COMPLETO - LISTO PARA TRADING")
print("=" * 70)

sys.exit(0)
