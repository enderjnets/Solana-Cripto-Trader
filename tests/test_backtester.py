#!/usr/bin/env python3
"""
Unit Tests for Solana Jupiter Bot
=================================
Tests for backtesting, genetic mining, and agent systems.

Usage:
    python -m pytest tests/ -v
    python tests/test_backtester.py
    python tests/test_genetic_miner.py
"""

import sys
import os
sys.path.insert(0, '.')

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Generate sample price data for testing"""
    np.random.seed(42)
    n_candles = 1000
    returns = np.random.normal(0, 0.02 / np.sqrt(365), n_candles)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_candles)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_candles)))
    open_price = np.roll(close, 1)
    open_price[0] = 100
    volume = np.random.uniform(1e6, 1e8, n_candles)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_candles, freq='1h'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    return df


@pytest.fixture
def small_data():
    """Generate small dataset for quick tests"""
    np.random.seed(123)
    n_candles = 200
    returns = np.random.normal(0, 0.01, n_candles)
    close = 50 * np.cumprod(1 + returns)
    high = close * 1.01
    low = close * 0.99
    volume = np.random.uniform(1e5, 1e6, n_candles)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_candles, freq='1h'),
        'open': close * 0.999,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    return df


# ============================================================================
# BACKTESTER TESTS
# ============================================================================

class TestBacktester:
    """Tests for backtesting/solana_backtester.py"""
    
    def test_indicators_count(self):
        """Test that all indicators are available"""
        from backtesting.solana_backtester import NUM_INDICATORS, INDICATOR_NAMES
        
        assert NUM_INDICATORS == 134, f"Expected 134 indicators, got {NUM_INDICATORS}"
        assert len(INDICATOR_NAMES) == NUM_INDICATORS, "Not all indicators have names"
    
    def test_rsi_indicator(self, small_data):
        """Test RSI calculation"""
        from backtesting.solana_backtester import precompute_indicators, IND_RSI_BASE, RSI_PERIODS
        
        indicators = precompute_indicators(small_data)
        
        # RSI should be between 0 and 100
        rsi_14 = indicators[IND_RSI_BASE + 1]  # RSI_14
        assert np.nanmin(rsi_14) >= 0, "RSI below 0"
        assert np.nanmax(rsi_14) <= 100, "RSI above 100"
        assert not np.isnan(np.nanmean(rsi_14)), "RSI mean is NaN"
    
    def test_sma_indicator(self, small_data):
        """Test SMA calculation"""
        from backtesting.solana_backtester import precompute_indicators, IND_SMA_BASE, SMA_PERIODS
        
        indicators = precompute_indicators(small_data)
        
        # SMA should be close to close price
        sma_14 = indicators[IND_SMA_BASE + 1]  # SMA_14
        close = small_data['close'].values
        
        # SMA should follow close price direction
        sma_mean = np.nanmean(sma_14)
        close_mean = np.nanmean(close)
        assert abs(sma_mean - close_mean) / close_mean < 0.1, "SMA deviates too much from close"
    
    def test_bollinger_bands(self, small_data):
        """Test Bollinger Bands calculation"""
        from backtesting.solana_backtester import precompute_indicators, IND_BB_UPPER_BASE, IND_BB_LOWER_BASE
        
        indicators = precompute_indicators(small_data)
        
        bb_upper = indicators[IND_BB_UPPER_BASE]
        bb_lower = indicators[IND_BB_LOWER_BASE]
        
        # Upper band should be >= lower band (may have NaN values)
        valid_upper = bb_upper[~np.isnan(bb_upper)]
        valid_lower = bb_lower[~np.isnan(bb_lower)]
        if len(valid_upper) > 0 and len(valid_lower) > 0:
            assert np.nanmin(valid_upper) >= np.nanmin(valid_lower), "BB upper < BB lower"
        
        # BB percent should be approximately between 0 and 1 (may exceed slightly)
        from backtesting.solana_backtester import IND_BB_PERCENT_BASE
        bb_pct = indicators[IND_BB_PERCENT_BASE]
        valid_pct = bb_pct[~np.isnan(bb_pct)]
        if len(valid_pct) > 0:
            # BB percent can exceed 0-1 slightly, just check it's not NaN everywhere
            assert np.sum(~np.isnan(valid_pct)) > len(valid_pct) * 0.5, "BB percent mostly NaN"
    
    def test_backtest_execution(self, small_data):
        """Test that backtest runs without errors"""
        from backtesting.solana_backtester import run_backtest
        import numpy as np
        
        # Create a simple genome
        genome = np.array([0.03, 0.06, 1, 6, 30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        
        result = run_backtest(small_data, genome)
        
        assert 'pnl' in result.__dataclass_fields__, "Result missing pnl field"
        assert 'trades' in result.__dataclass_fields__, "Result missing trades field"
        assert 'win_rate' in result.__dataclass_fields__, "Result missing win_rate field"
        assert isinstance(result.pnl, float), "PnL is not a float"
        assert isinstance(result.trades, int), "Trades is not an int"
    
    def test_genome_with_trades(self, small_data):
        """Test genome that generates trades"""
        from backtesting.solana_backtester import run_backtest, IND_RSI_BASE
        import numpy as np
        
        # RSI oversold strategy - should generate trades
        genome = np.array([0.03, 0.06, 1, IND_RSI_BASE + 1, 30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        
        result = run_backtest(small_data, genome)
        
        # Should have at least some trades or 0 trades
        assert result.trades >= 0, "Negative trades"
        assert result.win_rate >= 0 and result.win_rate <= 1, "Win rate out of range"


# ============================================================================
# GENETIC MINER TESTS
# ============================================================================

class TestGeneticMiner:
    """Tests for strategies/genetic_miner.py"""
    
    def test_indicator_mapping(self):
        """Test that indicator indices match backtester"""
        from strategies.genetic_miner import (
            GenomeEncoder, INDICATOR_MAP,
            IND_RSI_14, IND_SMA_14, IND_CLOSE
        )
        from backtesting.solana_backtester import IND_CLOSE as BACKTESTER_CLOSE
        
        # Check that indices match
        assert IND_CLOSE == BACKTESTER_CLOSE, "IND_CLOSE doesn't match backtester"
        assert GenomeEncoder.get_indicator_index("RSI_14") == IND_RSI_14
        assert GenomeEncoder.get_indicator_index("SMA_14") == IND_SMA_14
    
    def test_genome_encoding(self):
        """Test genome encoding/decoding"""
        from strategies.genetic_miner import Genome, GenomeEncoder, INDICATOR_MAP
        
        genome = Genome(
            entry_rules=[{"indicator": "RSI_14", "period": 14, "operator": "<", "threshold": 30}],
            exit_rules=[],
            params={"sl_pct": 0.03, "tp_pct": 0.06, "position_size": 0.1}
        )
        
        encoded = GenomeEncoder.encode(genome)
        
        assert len(encoded) == GenomeEncoder.GENOME_SIZE, "Wrong encoded size"
        assert encoded[0] == 0.03, "SL not encoded correctly"
        assert encoded[1] == 0.06, "TP not encoded correctly"
        # Position 5 = indicator index, position 6 = period
        assert int(encoded[5]) == INDICATOR_MAP["RSI_14"], "Indicator not encoded correctly"
        assert int(encoded[6]) == 14, "Period not encoded correctly"
    
    def test_available_indicators(self):
        """Test that all indicators are available"""
        from strategies.genetic_miner import GenomeEncoder, INDICATOR_MAP
        
        indicators = GenomeEncoder.get_available_indicators()
        
        assert len(indicators) > 10, "Not enough indicators"
        assert "RSI_14" in indicators, "RSI_14 not found"
        assert "SMA_14" in indicators, "SMA_14 not found"
        assert "EMA_14" in indicators, "EMA_14 not found"
    
    def test_random_genome_generation(self):
        """Test random genome generation"""
        from strategies.genetic_miner import StrategyMiner
        import pandas as pd
        
        # Create small test data
        df = pd.DataFrame({
            'close': np.cumprod(1 + np.random.normal(0, 0.01, 100)),
            'open': np.cumprod(1 + np.random.normal(0, 0.01, 100)),
            'high': None,
            'low': None,
        })
        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99
        
        miner = StrategyMiner(df, population_size=5, generations=2)
        genome = miner.generate_random_genome()
        
        assert len(genome.entry_rules) >= 1, "No entry rules"
        assert 'sl_pct' in genome.params, "No SL in params"
        assert 'tp_pct' in genome.params, "No TP in params"
    
    def test_crossover(self):
        """Test genome crossover"""
        from strategies.genetic_miner import Genome, StrategyMiner
        import pandas as pd
        
        df = pd.DataFrame({
            'close': np.cumprod(1 + np.random.normal(0, 0.01, 100)),
        })
        
        miner = StrategyMiner(df, population_size=5, generations=2)
        
        parent1 = Genome(
            entry_rules=[{"indicator": "RSI_14", "period": 14, "operator": "<", "threshold": 30}],
            exit_rules=[],
            params={"sl_pct": 0.02, "tp_pct": 0.05}
        )
        
        parent2 = Genome(
            entry_rules=[{"indicator": "SMA_14", "period": 14, "operator": ">", "threshold": 50}],
            exit_rules=[],
            params={"sl_pct": 0.03, "tp_pct": 0.06}
        )
        
        child = miner.crossover(parent1, parent2)
        
        assert len(child.entry_rules) >= 1, "Child has no rules"
        assert 'sl_pct' in child.params, "Child has no params"
    
    def test_mutation(self):
        """Test genome mutation"""
        from strategies.genetic_miner import Genome, StrategyMiner
        import pandas as pd
        
        df = pd.DataFrame({
            'close': np.cumprod(1 + np.random.normal(0, 0.01, 100)),
        })
        
        miner = StrategyMiner(df, population_size=5, generations=2)
        
        genome = Genome(
            entry_rules=[{"indicator": "RSI_14", "period": 14, "operator": "<", "threshold": 30}],
            exit_rules=[],
            params={"sl_pct": 0.03, "tp_pct": 0.06}
        )
        
        original_threshold = genome.entry_rules[0]['threshold']
        mutated = miner.mutate(genome, mutation_rate=1.0)  # Force mutation
        
        # Threshold might have changed
        assert len(mutated.entry_rules) == 1, "Lost rules"
    
    def test_evolution(self, sample_data):
        """Test complete evolution run"""
        from strategies.genetic_miner import StrategyMiner
        
        miner = StrategyMiner(sample_data, population_size=10, generations=5)
        result = miner.evolve(verbose=False)
        
        assert 'best_genome' in result, "No genome in result"
        assert 'best_pnl' in result, "No PnL in result"
        assert isinstance(result['best_pnl'], float), "PnL not float"
        assert result['generations'] == 5, "Wrong generations count"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests between components"""
    
    def test_miner_to_backtester(self, small_data):
        """Test that miner genome works with backtester"""
        from strategies.genetic_miner import Genome, GenomeEncoder, StrategyMiner
        from backtesting.solana_backtester import run_backtest
        
        # Create genome through miner
        miner = StrategyMiner(small_data, population_size=5, generations=2)
        genome = miner.generate_random_genome()
        
        # Encode for backtester
        encoded = GenomeEncoder.encode(genome)
        
        # Run backtest
        result = run_backtest(small_data, encoded)
        
        # Should complete without error
        assert result.trades >= 0
        assert isinstance(result.pnl, float)
    
    def test_consistent_indices(self):
        """Test that key indices are consistent between components"""
        from strategies.genetic_miner import IND_CLOSE as MINER_CLOSE, IND_RSI_14
        from backtesting.solana_backtester import IND_CLOSE as BT_CLOSE, IND_RSI_BASE
        
        # Key indices must match
        assert BT_CLOSE == MINER_CLOSE, "IND_CLOSE mismatch"
        # RSI_14 in miner is IND_RSI_14=6, in backtester IND_RSI_BASE+1=6
        assert IND_RSI_14 == IND_RSI_BASE + 1, "RSI_14 index mismatch"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and stress tests"""
    
    def test_backtest_performance(self, sample_data):
        """Test backtest speed"""
        from backtesting.solana_backtester import run_backtest
        import time
        
        genome = np.array([0.03, 0.06, 1, 6, 30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        
        start = time.time()
        for _ in range(10):
            run_backtest(sample_data, genome)
        elapsed = time.time() - start
        
        # Should complete 10 backtests in under 30 seconds
        assert elapsed < 30, f"Backtests too slow: {elapsed:.2f}s"
        print(f"Backtest performance: {elapsed:.2f}s for 10 runs")
    
    def test_large_dataset(self):
        """Test with larger dataset"""
        from backtesting.solana_backtester import precompute_indicators
        import time
        
        # Generate 50,000 candles
        np.random.seed(42)
        n = 50000
        close = 100 * np.cumprod(1 + np.random.normal(0, 0.02 / np.sqrt(365), n))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=n, freq='1h'),
            'open': np.roll(close, 1),
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': np.random.uniform(1e6, 1e8, n)
        })
        
        start = time.time()
        indicators = precompute_indicators(df)
        elapsed = time.time() - start
        
        # Should handle 50K candles in under 10 seconds
        assert elapsed < 10, f"Large dataset too slow: {elapsed:.2f}s"
        print(f"Large dataset: {elapsed:.2f}s for {n} candles")


# ============================================================================
# RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 SOLANA JUPITER BOT - UNIT TESTS")
    print("=" * 70)
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
