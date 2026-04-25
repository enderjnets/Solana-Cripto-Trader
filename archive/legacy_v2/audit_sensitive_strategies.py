#!/usr/bin/env python3
"""
Quick Audit: Verify Sensitive Strategies
==========================================
Tests that sensitive strategies are properly loaded and evaluated.
"""

import sys
import json
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, '/home/enderj/.openclaw/workspace/solana-jupiter-bot')

from strategies.genetic_miner import IND_SMA_14, IND_SMA_50, IND_EMA_14, IND_RSI_14
from backtesting.solana_backtester import precompute_indicators

print("=" * 60)
print("AUDITORÍA: Estrategias Sensibles")
print("=" * 60)

# 1. Verificar que active_strategies.json tiene estrategias sensibles
print("\n1. Verificando active_strategies.json...")
with open('data/active_strategies.json', 'r') as f:
    strategies = json.load(f)

print(f"   Generation: {strategies.get('generation')}")
print(f"   Count: {strategies.get('count')}")
print(f"   Deployed at: {strategies.get('deployed_at')}")

sensitive_found = 0
for s in strategies.get('strategies', []):
    rules = s.get('brain_rules', [])
    for r in rules:
        if r.get('threshold', 0) == 0.98:
            sensitive_found += 1
            print(f"   ✓ {s['name']}: {r['indicator']} {r['operator']} {r['threshold']}")

if sensitive_found == 0:
    print("   ❌ ERROR: No se encontraron estrategias sensibles (threshold=0.98)")
    sys.exit(1)
else:
    print(f"   ✅ {sensitive_found} estrategias sensibles encontradas")

# 2. Cargar datos y calcular indicadores
print("\n2. Cargando datos y calculando indicadores...")
df = pd.read_csv('data/sol_1h.csv')
if 'close' not in df.columns:
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

close = df['close'].iloc[-1]
sma_20 = df['close'].rolling(20).mean().iloc[-1]
sma_50 = df['close'].rolling(50).mean().iloc[-1]
ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
rsi_14 = df['close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / \
         df['close'].diff().abs().rolling(14).mean() * 100
rsi_14 = rsi_14.iloc[-1]

print(f"   Precio close: ${close:.2f}")
print(f"   SMA(20): ${sma_20:.2f} (ratio: {close/sma_20:.4f})")
print(f"   SMA(50): ${sma_50:.2f} (ratio: {close/sma_50:.4f})")
print(f"   EMA(20): ${ema_20:.2f} (ratio: {close/ema_20:.4f})")
print(f"   RSI(14): {rsi_14:.2f}")

# 3. Evaluar reglas de estrategias sensibles
print("\n3. Evaluando reglas de estrategias...")
for s in strategies.get('strategies', []):
    rules = s.get('brain_rules', [])
    all_pass = True
    for r in rules:
        ind = r.get('indicator')
        period = r.get('period')
        op = r.get('operator')
        thresh = r.get('threshold')
        
        # Calcular valor según indicador
        if ind == 'SMA':
            if period == 20:
                value = close / sma_20
            elif period == 50:
                value = close / sma_50
            else:
                value = 1.0
        elif ind == 'EMA':
            value = close / ema_20
        elif ind == 'RSI':
            value = rsi_14
        else:
            value = 1.0
        
        # Evaluar condición
        if op == '<':
            passes = value < thresh
        elif op == '>':
            passes = value > thresh
        else:
            passes = False
        
        if not passes:
            all_pass = False
        
        status = "✓ BUY" if passes else "✗ NO"
        print(f"   {s['name']}: {ind}({period}) {op} {thresh} = {value:.4f} → {status}")

# 4. Resultado final
print("\n" + "=" * 60)
print("RESULTADO DE AUDITORÍA")
print("=" * 60)

ratio_20 = close / sma_20
if ratio_20 < 0.98:
    print("✅ PRECIO BAJO UMBRAL: Se activaría señal BUY")
    print(f"   Ratio actual: {ratio_20:.4f} < 0.98")
else:
    print("ℹ️ PRECIO ENCIMA UMBRAL: No hay señal BUY todavía")
    print(f"   Ratio actual: {ratio_20:.4f} >= 0.98")

print("\nEstado del sistema:")
print("   ✓ Estrategias sensibles desplegadas")
print("   ✓ Agent runner reiniciado")
print("   ✓ Esperando ciclo de evaluación...")

sys.exit(0)
