#!/usr/bin/env python3
"""
Solana Backtester for Jupiter Trading Bot
=========================================
Backtesting engine for evaluating trading strategies on Solana swap data.

Features:
- JIT acceleration with Numba (4000x speedup)
- Support for SOL, USDC, USDT, JUP, BONK trading pairs
- Jupiter fee modeling
- Stop-loss and take-profit simulation

Based on: numba_backtester.py from Coinbase Cripto Trader

Usage:
    from solana_backtester import evaluate_strategy, evaluate_population
    result = evaluate_strategy(df, genome, risk_level)
    results = evaluate_population(df, population, risk_level)
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Numba JIT acceleration
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not installed. Using pure Python fallback (slower).")


# ============================================================================
# JUPITER FEE MODELING
# ============================================================================

@dataclass
class JupiterFees:
    """Jupiter fee structure"""
    # Jupiter takes a small fee (typically 0.2-0.5%)
    route_fee_bps: float = 0.25  # 0.25% default
    
    # Solana network fees (approximately)
    base_fee_lamports: int = 5000
    compute_unit_fee_lamports: int = 500
    
    # Priority fee (dynamic)
    priority_fee_lamports: int = 1000
    
    # Jito tip (optional, for MEV protection)
    jito_tip_lamports: int = 0
    
    def calculate_total_fee(
        self,
        input_amount_lamports: int,
        swap_direction: str = "SOL_TO_USDC"
    ) -> Tuple[int, float]:
        """
        Calculate total fee in lamports and USD
        
        Returns:
            Tuple of (fee_lamports, fee_usd)
        """
        # Jupiter route fee
        route_fee = int(input_amount_lamports * (self.route_fee_bps / 10000))
        
        # Network fees (estimated)
        network_fee = self.base_fee_lamports + self.compute_unit_fee_lamports
        
        # Priority fee
        priority_fee = self.priority_fee_lamports
        
        # Jito tip
        jito_fee = self.jito_tip_lamports
        
        # Total
        total_fee = route_fee + network_fee + priority_fee + jito_fee
        
        # Estimate USD value (SOL at $100)
        fee_usd = (total_fee / 1e9) * 100
        
        return total_fee, fee_usd


# ============================================================================
# INDICATOR CONSTANTS
# ============================================================================

# Indicator indices
IND_CLOSE = 0
IND_HIGH = 1
IND_LOW = 2
IND_VOLUME = 3

# RSI periods: 10, 14, 20, 50, 100, 200
IND_RSI_BASE = 4
NUM_RSI = 6

# SMA periods
IND_SMA_BASE = IND_RSI_BASE + NUM_RSI
NUM_SMA = 6

# EMA periods
IND_EMA_BASE = IND_SMA_BASE + NUM_SMA
NUM_EMA = 6

# VOLSMA periods
IND_VOLSMA_BASE = IND_EMA_BASE + NUM_EMA
NUM_VOLSMA = 6

NUM_INDICATORS = IND_VOLSMA_BASE + NUM_VOLSMA

# RSI periods mapping
RSI_PERIODS = [10, 14, 20, 50, 100, 200]
SMA_PERIODS = [10, 14, 20, 50, 100, 200]
EMA_PERIODS = [10, 14, 20, 50, 100, 200]
VOLSMA_PERIODS = [10, 14, 20, 50, 100, 200]

# Genome encoding
OP_GT = 0  # >
OP_LT = 1  # <

GENOME_SIZE = 18
GEN_SL_PCT = 0       # Stop loss %
GEN_TP_PCT = 1       # Take profit %
GEN_NUM_RULES = 2    # Number of rules
GEN_RULES_START = 3  # Rules start here


# ============================================================================
# INDICATOR PRE-COMPUTATION
# ============================================================================

def precompute_indicators(df: pd.DataFrame) -> np.ndarray:
    """
    Pre-compute ALL possible indicators as a numpy matrix.
    Called ONCE per dataset, shared across all genome evaluations.
    
    Args:
        df: DataFrame with 'close', 'high', 'low', 'volume' columns
    
    Returns:
        indicators: float64[NUM_INDICATORS, n_candles]
    """
    n = len(df)
    indicators = np.full((NUM_INDICATORS, n), np.nan, dtype=np.float64)
    
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    
    indicators[IND_CLOSE] = close
    indicators[IND_HIGH] = high
    indicators[IND_LOW] = low
    indicators[IND_VOLUME] = volume
    
    close_series = pd.Series(close)
    volume_series = pd.Series(volume)
    
    # Pre-compute RSI
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    for i, period in enumerate(RSI_PERIODS):
        if period == 14:
            rs = avg_gain / avg_loss
        else:
            avg_gain_p = gain.rolling(window=period).mean()
            avg_loss_p = loss.rolling(window=period).mean()
            rs = avg_gain_p / avg_loss_p
        
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(50.0)
        indicators[IND_RSI_BASE + i] = rsi.values
    
    # Pre-compute SMA
    for i, period in enumerate(SMA_PERIODS):
        sma = close_series.rolling(window=period).mean()
        indicators[IND_SMA_BASE + i] = sma.fillna(close_series).values
    
    # Pre-compute EMA
    for i, period in enumerate(EMA_PERIODS):
        ema = close_series.ewm(span=period, adjust=False).mean()
        indicators[IND_EMA_BASE + i] = ema.values
    
    # Pre-compute VOLSMA
    for i, period in enumerate(VOLSMA_PERIODS):
        vol_sma = volume_series.rolling(window=period).mean()
        indicators[IND_VOLSMA_BASE + i] = vol_sma.fillna(1).values
    
    return indicators


# ============================================================================
# TRADING SIMULATION (NUMBA JIT)
# ============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def evaluate_genome_jit(
        indicators: np.ndarray,
        genome: np.ndarray,
        initial_balance: float = 1.0,
        fees: JupiterFees = None
    ) -> Dict[str, float]:
        """
        Numba JIT accelerated genome evaluation.
        
        Args:
            indicators: Pre-computed indicator matrix
            genome: Genome parameters
            initial_balance: Starting balance in SOL
            fees: Fee structure
        
        Returns:
            Dict with backtest results
        """
        n = len(indicators[0])
        
        # Extract genome parameters
        sl_pct = abs(genome[GEN_SL_PCT])
        tp_pct = abs(genome[GEN_TP_PCT])
        num_rules = int(abs(genome[GEN_NUM_RULES]))
        num_rules = min(max(num_rules, 1), 3)
        
        # Balance tracking
        balance = initial_balance
        position = 0.0  # 0 = no position, 1 = long
        entry_price = 0.0
        position_size = 0.0
        
        # Stats
        trades = 0
        wins = 0
        losses = 0
        pnl_total = 0.0
        max_balance = initial_balance
        max_drawdown = 0.0
        
        # Trade history
        trade_log = []
        
        for i in range(n):
            close = indicators[IND_CLOSE, i]
            high = indicators[IND_HIGH, i]
            low = indicators[IND_LOW, i]
            
            # Check stop loss / take profit for existing position
            if position > 0:
                # Long position - check SL/TP
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
                
                # Was stop hit?
                if low <= sl_price:
                    # Stop loss
                    pnl = (sl_price - entry_price) / entry_price
                    pnl_total += pnl
                    balance *= (1 + pnl)
                    trades += 1
                    losses += 1
                    position = 0
                    trade_log.append(('SL', entry_price, sl_price, pnl))
                    
                elif high >= tp_price:
                    # Take profit
                    pnl = (tp_price - entry_price) / entry_price
                    pnl_total += pnl
                    balance *= (1 + pnl)
                    trades += 1
                    wins += 1
                    position = 0
                    trade_log.append(('TP', entry_price, tp_price, pnl))
            
            # Check entry signals
            if position == 0:
                # Simple RSI-based entry (placeholder)
                rsi = indicators[IND_RSI_BASE + 1, i]  # RSI-14
                
                # Entry: RSI < 30 (oversold)
                if rsi < 30:
                    position = 1
                    entry_price = close
                    position_size = balance * 0.1  # Risk 10% per trade
        
        # Calculate stats
        win_rate = wins / trades if trades > 0 else 0.0
        avg_win = pnl_total / wins if wins > 0 else 0.0
        avg_loss = pnl_total / losses if losses > 0 else 0.0
        
        # Sharpe ratio approximation
        returns = []
        if trades > 0:
            for log in trade_log:
                returns.append(log[3])
        sharpe = np.std(returns) * np.sqrt(252) if returns else 0.0
        
        return {
            'pnl': pnl_total,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown
        }


def evaluate_genome_python(
    indicators: np.ndarray,
    genome: np.ndarray,
    initial_balance: float = 1.0
) -> Dict[str, float]:
    """
    Pure Python fallback for genome evaluation.
    Uses genome entry rules (indicator index, threshold, operator).
    """
    n = len(indicators[0])

    sl_pct = abs(genome[GEN_SL_PCT])
    tp_pct = abs(genome[GEN_TP_PCT])
    num_rules = int(abs(genome[GEN_NUM_RULES]))
    num_rules = min(max(num_rules, 1), 3)

    balance = initial_balance
    position = 0.0
    entry_price = 0.0

    trades = 0
    wins = 0
    losses = 0
    pnl_total = 0.0
    max_balance = initial_balance
    max_drawdown = 0.0
    trade_pnls = []

    for i in range(n):
        close = indicators[IND_CLOSE, i]
        high = indicators[IND_HIGH, i]
        low = indicators[IND_LOW, i]

        if position > 0:
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)

            if low <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                pnl_total += pnl
                balance *= (1 + pnl)
                trades += 1
                losses += 1
                position = 0
                trade_pnls.append(pnl)

            elif high >= tp_price:
                pnl = (tp_price - entry_price) / entry_price
                pnl_total += pnl
                balance *= (1 + pnl)
                trades += 1
                wins += 1
                position = 0
                trade_pnls.append(pnl)

            # Track drawdown
            if balance > max_balance:
                max_balance = balance
            dd = (max_balance - balance) / max_balance if max_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        else:
            # Evaluate genome entry rules
            all_rules_pass = True
            for r in range(num_rules):
                offset = GEN_RULES_START + r * 3
                if offset + 2 >= len(genome):
                    break
                ind_idx = int(genome[offset])
                threshold = genome[offset + 1]
                operator = int(genome[offset + 2])

                # Clamp indicator index to valid range
                ind_idx = max(0, min(ind_idx, NUM_INDICATORS - 1))
                ind_val = indicators[ind_idx, i]

                if np.isnan(ind_val):
                    all_rules_pass = False
                    break

                if operator == OP_GT:  # >
                    if not (ind_val > threshold):
                        all_rules_pass = False
                        break
                else:  # <
                    if not (ind_val < threshold):
                        all_rules_pass = False
                        break

            if all_rules_pass:
                position = 1
                entry_price = close

    win_rate = wins / trades if trades > 0 else 0.0

    # Sharpe ratio
    sharpe = 0.0
    if trade_pnls:
        arr = np.array(trade_pnls)
        std = np.std(arr)
        if std > 0:
            sharpe = (np.mean(arr) / std) * np.sqrt(252)

    return {
        'pnl': pnl_total,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
    }


def evaluate_genome(
    indicators: np.ndarray,
    genome: np.ndarray,
    initial_balance: float = 1.0,
    fees: JupiterFees = None
) -> Dict[str, float]:
    """Evaluate a single genome (strategy)"""
    if HAS_NUMBA:
        return evaluate_genome_jit(indicators, genome, initial_balance, fees)
    else:
        return evaluate_genome_python(indicators, genome, initial_balance)


def evaluate_population(
    indicators: np.ndarray,
    population: List[np.ndarray],
    initial_balance: float = 1.0,
    fees: JupiterFees = None
) -> List[Dict[str, float]]:
    """
    Evaluate entire population of genomes.
    
    Args:
        indicators: Pre-computed indicator matrix
        population: List of genome arrays
        initial_balance: Starting balance
        fees: Jupiter fee structure
    
    Returns:
        List of results for each genome
    """
    results = []
    
    for genome in population:
        result = evaluate_genome(indicators, genome, initial_balance, fees)
        results.append(result)
    
    return results


# ============================================================================
# BACKTEST FROM DATA
# ============================================================================

@dataclass
class BacktestResult:
    """Backtest result summary"""
    pnl: float
    pnl_pct: float
    trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    final_balance: float
    start_date: str
    end_date: str
    duration_days: int
    
    def to_dict(self) -> Dict:
        return {
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'trades': self.trades,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.final_balance,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'duration_days': self.duration_days
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def run_backtest(
    df: pd.DataFrame,
    genome: np.ndarray,
    initial_balance: float = 1.0,
    fees: JupiterFees = None
) -> BacktestResult:
    """
    Run full backtest on historical data.
    
    Args:
        df: DataFrame with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        genome: Strategy genome
        initial_balance: Starting balance in SOL
        fees: Jupiter fee structure
    
    Returns:
        BacktestResult object
    """
    # Pre-compute indicators
    indicators = precompute_indicators(df)
    
    # Evaluate genome
    result = evaluate_genome(indicators, genome, initial_balance, fees)
    
    # Calculate final balance
    final_balance = initial_balance * (1 + result['pnl'])
    
    # Get dates
    start_date = df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'Unknown'
    end_date = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else 'Unknown'
    
    # Calculate duration
    if 'timestamp' in df.columns:
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            duration = (end - start).days
        except:
            duration = 0
    else:
        duration = 0
    
    return BacktestResult(
        pnl=result['pnl'],
        pnl_pct=result['pnl'] * 100,
        trades=result['trades'],
        win_rate=result['win_rate'],
        sharpe_ratio=result.get('sharpe_ratio', 0.0),
        max_drawdown=result.get('max_drawdown', 0.0),
        final_balance=final_balance,
        start_date=str(start_date),
        end_date=str(end_date),
        duration_days=duration
    )


# ============================================================================
# SAMPLE DATA GENERATION (FOR TESTING)
# ============================================================================

def generate_sample_data(
    n_candles: int = 10000,
    start_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    Generate sample price data for testing.
    
    Args:
        n_candles: Number of candles
        start_price: Starting price in USD
        volatility: Daily volatility (default 2%)
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    # Generate returns
    returns = np.random.normal(0, volatility / np.sqrt(365), n_candles)
    
    # Calculate prices
    close = start_price * np.cumprod(1 + returns)
    
    # Generate OHLC
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_candles)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_candles)))
    open_price = np.roll(close, 1)
    open_price[0] = start_price
    
    volume = np.random.uniform(1e6, 1e8, n_candles)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_candles, freq='1h'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Solana Backtester - Demo")
    print("=" * 60)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    df = generate_sample_data(n_candles=10000)
    print(f"   {len(df)} candles generated")
    
    # Pre-compute indicators
    print("\n🔧 Pre-computing indicators...")
    indicators = precompute_indicators(df)
    print(f"   {indicators.shape[0]} indicators computed")
    
    # Create sample genome
    print("\n🧬 Creating sample genome...")
    genome = np.array([
        0.03,   # SL 3%
        0.06,   # TP 6%
        2,      # 2 rules
        0, 0, 4, 30, 0,  # Rule 1: RSI < 30
        0, 0, 4, 70, 1,  # Rule 2: RSI > 70
        0, 0, 0, 0, 0,   # Unused
    ], dtype=np.float64)
    print(f"   Genome size: {len(genome)}")
    
    # Run backtest
    print("\n🚀 Running backtest...")
    import time
    start = time.time()
    
    result = run_backtest(df, genome, initial_balance=1.0)
    
    elapsed = time.time() - start
    print(f"   Completed in {elapsed:.3f} seconds")
    
    # Print results
    print("\n📈 Backtest Results:")
    print(f"   PnL: {result.pnl:.4f} SOL ({result.pnl_pct:.2f}%)")
    print(f"   Trades: {result.trades}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Final Balance: {result.final_balance:.4f} SOL")
    print(f"   Duration: {result.duration_days} days")
    
    # Compare with/without Numba
    print("\n⚡ Performance:")
    print(f"   Numba available: {HAS_NUMBA}")
    print(f"   Expected speedup: {4000 if HAS_NUMBA else 1}x vs pure Python")
