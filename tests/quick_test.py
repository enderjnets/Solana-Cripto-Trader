#!/usr/bin/env python3
"""
Quick Test Suite - Solana Jupiter Bot
====================================
Tests críticos del sistema.

Usage: python3 tests/quick_test.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

print('=' * 70)
print('🧪 QUICK TEST SUITE')
print('=' * 70)

tests_passed = 0
tests_failed = 0

# Test 1: Backtester imports
print('\n1. Import Backtester... ')
try:
    from backtesting.solana_backtester import NUM_INDICATORS, IND_CLOSE, IND_RSI_BASE
    print(f'✅ NUM_INDICATORS={NUM_INDICATORS}')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 2: Genetic Miner imports
print('2. Import Genetic Miner... ')
try:
    from strategies.genetic_miner import GenomeEncoder, Genome, INDICATOR_MAP
    print(f'✅ INDICATOR_MAP size={len(INDICATOR_MAP)}')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 3: Strategy Agent imports
print('3. Import Strategy Agent... ')
try:
    from agents.strategy_agent import StrategyAgent
    print(f'✅ StrategyAgent imported')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 4: Trading Agent imports
print('4. Import Trading Agent... ')
try:
    from agents.trading_agent import TradingAgentWithAPI
    print(f'✅ TradingAgent imported')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 5: Index compatibility
print('5. Check index compatibility... ')
try:
    from backtesting.solana_backtester import IND_CLOSE as BT_CLOSE
    from strategies.genetic_miner import IND_CLOSE as MINER_CLOSE
    if BT_CLOSE == MINER_CLOSE:
        print(f'✅ IND_CLOSE match: {BT_CLOSE}')
        tests_passed += 1
    else:
        print(f'❌ IND_CLOSE mismatch: BT={BT_CLOSE}, MINER={MINER_CLOSE}')
        tests_failed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 6: Genome encoding
print('6. Test genome encoding... ')
try:
    from strategies.genetic_miner import Genome, GenomeEncoder
    genome = Genome(
        entry_rules=[{'indicator': 'RSI_14', 'period': 14, 'operator': '<', 'threshold': 30}],
        exit_rules=[],
        params={'sl_pct': 0.03, 'tp_pct': 0.06}
    )
    encoded = GenomeEncoder.encode(genome)
    if len(encoded) == GenomeEncoder.get_genome_size():
        print(f'✅ Genome encoded: {len(encoded)} floats')
        tests_passed += 1
    else:
        print(f'❌ Wrong genome size: {len(encoded)}')
        tests_failed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 7: Backtest execution
print('7. Execute backtest... ')
try:
    from backtesting.solana_backtester import run_backtest
    
    np.random.seed(42)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=300, freq='1h'),
        'open': 100 + np.cumsum(np.random.randn(300)),
        'high': 100 + np.cumsum(np.random.randn(300)),
        'low': 100 + np.cumsum(np.random.randn(300)),
        'close': 100 + np.cumsum(np.random.randn(300)),
        'volume': np.random.uniform(1e6, 1e8, 300)
    })
    
    genome = np.array([0.03, 0.06, 1, 6, 30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    result = run_backtest(df, genome)
    
    print(f'✅ Backtest: trades={result.trades}, pnl={result.pnl:.4f}')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 8: Strategy Agent initialization
print('8. Initialize Strategy Agent... ')
try:
    from agents.strategy_agent import StrategyAgent
    agent = StrategyAgent()
    status = agent.get_status()
    print(f'✅ Agent: {status["agent"]}, indicators={status["available_indicators"]}')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 9: Trading Agent initialization
print('9. Initialize Trading Agent... ')
try:
    from agents.trading_agent import TradingAgentWithAPI
    agent = TradingAgentWithAPI()
    status = agent.get_status()
    print(f'✅ Trading Agent: {status["name"]}, network={status["network"]}')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

# Test 10: Genome to backtester compatibility
print('10. Test miner → backtester... ')
try:
    from strategies.genetic_miner import Genome, GenomeEncoder
    from backtesting.solana_backtester import run_backtest
    
    genome = Genome(
        entry_rules=[{'indicator': 'SMA_14', 'period': 14, 'operator': '>', 'threshold': 100}],
        exit_rules=[],
        params={'sl_pct': 0.03, 'tp_pct': 0.06}
    )
    encoded = GenomeEncoder.encode(genome)
    
    np.random.seed(123)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='1h'),
        'open': 100 + np.cumsum(np.random.randn(200)),
        'high': 100 + np.cumsum(np.random.randn(200)),
        'low': 100 + np.cumsum(np.random.randn(200)),
        'close': 100 + np.cumsum(np.random.randn(200)),
        'volume': np.random.uniform(1e6, 1e8, 200)
    })
    
    result = run_backtest(df, encoded)
    print(f'✅ Integration: trades={result.trades}')
    tests_passed += 1
except Exception as e:
    print(f'❌ {e}')
    tests_failed += 1

print()
print('=' * 70)
print(f'📊 RESULTADO: {tests_passed} passed, {tests_failed} failed')
print('=' * 70)

if tests_failed == 0:
    print('✅ TODOS LOS TESTS PASARON')
    sys.exit(0)
else:
    print('❌ ALGUNOS TESTS FALLARON')
    sys.exit(1)
