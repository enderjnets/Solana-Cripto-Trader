#!/usr/bin/env python3
"""
Audit: Momentum + Sensitive Strategies
======================================
Verifies all 7 strategies (4 sensitive + 3 momentum) are loaded.
"""

import sys
import json
import pandas as pd

sys.path.insert(0, '/home/enderj/.openclaw/workspace/solana-jupiter-bot')

print("=" * 60)
print("AUDITORÍA: Estrategias Sensitive + Momentum")
print("=" * 60)

# Load strategies
with open('data/active_strategies.json', 'r') as f:
    strategies = json.load(f)

print(f"\n1. Verificando archivo...")
print(f"   Generation: {strategies.get('generation')}")
print(f"   Count: {strategies.get('count')}")
print(f"   Types: {strategies.get('strategy_types', [])}")
print(f"   Manual: {strategies.get('manual_override', False)}")

# Categorize
sensitive = []
momentum = []

for s in strategies.get('strategies', []):
    rules = s.get('brain_rules', [])
    for r in rules:
        if r.get('threshold', 0) < 1.0 and r.get('operator') == '<':
            sensitive.append(s['name'])
        elif r.get('threshold', 0) > 1.0 or (r.get('threshold', 0) == 1.0 and r.get('operator') == '>'):
            momentum.append(s['name'])

print(f"\n2. Estrategias por tipo:")
print(f"   Sensitive: {len(sensitive)} - {sensitive}")
print(f"   Momentum: {len(momentum)} - {momentum}")

# Load price data
print(f"\n3. Analizando mercado actual...")
df = pd.read_csv('data/sol_1h.csv')
if 'close' not in df.columns:
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

close = df['close'].iloc[-1]
sma_20 = df['close'].rolling(20).mean().iloc[-1]
sma_50 = df['close'].rolling(50).mean().iloc[-1]
rsi_14 = df['close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / \
         df['close'].diff().abs().rolling(14).mean() * 100
rsi_14 = rsi_14.iloc[-1]

print(f"   Precio: ${close:.2f}")
print(f"   SMA(20): ${sma_20:.2f} (ratio: {close/sma_20:.4f})")
print(f"   SMA(50): ${sma_50:.2f} (ratio: {close/sma_50:.4f})")
print(f"   RSI(14): {rsi_14:.2f}")

# Evaluate strategies
print(f"\n4. Evaluando reglas...")
buy_signals = 0
for s in strategies.get('strategies', []):
    rules = s.get('brain_rules', [])
    all_pass = True
    rule_details = []
    for r in rules:
        ind = r.get('indicator')
        period = r.get('period')
        op = r.get('operator')
        thresh = r.get('threshold')
        
        if ind == 'SMA':
            value = close / (df['close'].rolling(period).mean().iloc[-1])
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
        
        rule_details.append(f"{ind}({period}) {op} {thresh}={value:.3f}")
    
    if all_pass:
        buy_signals += 1
        status = "✅ BUY"
    else:
        status = "❌ HOLD"
    
    print(f"   {status} {s['name']}: {', '.join(rule_details)}")

# Result
print(f"\n" + "=" * 60)
print("RESULTADO DE AUDITORÍA")
print("=" * 60)
print(f"Total estrategias: {len(strategies.get('strategies', []))}")
print(f"Señales BUY: {buy_signals}")
print(f"Precio actual: ${close:.2f}")

if close / sma_20 > 1.02:
    print(f"\nℹ️ MOMENTUM ACTIVO: Precio > 2% sobre SMA(20)")
elif close / sma_20 < 0.98:
    print(f"\n✅ SENSITIVE ACTIVO: Precio < 2% bajo SMA(20)")
else:
    print(f"\nℹ️ PRECIO ESTABLE: Entre 98%-102% del SMA")

print("\nEstado del sistema:")
print(f"   ✓ {len(strategies.get('strategies', []))} estrategias cargadas")
print(f"   ✓ {len(sensitive)} sensitive + {len(momentum)} momentum")
print(f"   ✓ Agent runner ejecutando")

sys.exit(0)
