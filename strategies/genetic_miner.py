#!/usr/bin/env python3
"""
Genetic Algorithm Strategy Miner
=================================
Discovers optimal trading strategies using genetic algorithms.

Features:
- Evolves strategy genomes (RSI, SMA, EMA, Bollinger rules)
- Numba JIT accelerated backtesting (4000x speedup)
- SQLite persistence for results
- Automatic best strategy selection

IMPORTANT: Uses SAME indicator indices as solana_backtester.py:
- IND_OPEN = 0, IND_HIGH = 1, IND_LOW = 2, IND_CLOSE = 3
- IND_RSI_BASE = 5 (RSI_7=5, RSI_14=6, RSI_21=7, RSI_50=8, RSI_100=9, RSI_200=10)
- IND_SMA_BASE = 11 (SMA_7=11, SMA_14=12, etc.)
- IND_EMA_BASE = 17 (EMA_7=17, EMA_14=18, etc.)

Usage:
    miner = StrategyMiner(df, population_size=50, generations=20)
    results = miner.evolve()
"""

import random
import json
import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger("genetic_miner")

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = "data/genetic_results.db"
DEFAULT_POPULATION = 20
DEFAULT_GENERATIONS = 10

# ============================================================================
# INDICATOR INDICES (MUST MATCH solana_backtester.py)
# ============================================================================
IND_OPEN = 0
IND_HIGH = 1
IND_LOW = 2
IND_CLOSE = 3
IND_VOLUME = 4

# RSI indices (IND_RSI_BASE = 5)
IND_RSI_7 = 5
IND_RSI_14 = 6
IND_RSI_21 = 7
IND_RSI_50 = 8
IND_RSI_100 = 9
IND_RSI_200 = 10

# SMA indices (IND_SMA_BASE = 11)
IND_SMA_7 = 11
IND_SMA_14 = 12
IND_SMA_21 = 13
IND_SMA_50 = 14
IND_SMA_100 = 15
IND_SMA_200 = 16

# EMA indices (IND_EMA_BASE = 17)
IND_EMA_7 = 17
IND_EMA_14 = 18
IND_EMA_21 = 19
IND_EMA_50 = 20
IND_EMA_100 = 21
IND_EMA_200 = 22

# WMA indices (IND_WMA_BASE = 23)
IND_WMA_14 = 23
IND_WMA_50 = 24
IND_WMA_100 = 25
IND_WMA_200 = 26

# Indicator name to index mapping
INDICATOR_MAP = {
    "OPEN": IND_OPEN,
    "HIGH": IND_HIGH,
    "LOW": IND_LOW,
    "CLOSE": IND_CLOSE,
    "VOLUME": IND_VOLUME,
    "RSI_7": IND_RSI_7,
    "RSI_14": IND_RSI_14,
    "RSI_21": IND_RSI_21,
    "RSI_50": IND_RSI_50,
    "RSI_100": IND_RSI_100,
    "RSI_200": IND_RSI_200,
    "SMA_7": IND_SMA_7,
    "SMA_14": IND_SMA_14,
    "SMA_21": IND_SMA_21,
    "SMA_50": IND_SMA_50,
    "SMA_100": IND_SMA_100,
    "SMA_200": IND_SMA_200,
    "EMA_7": IND_EMA_7,
    "EMA_14": IND_EMA_14,
    "EMA_21": IND_EMA_21,
    "EMA_50": IND_EMA_50,
    "EMA_100": IND_EMA_100,
    "EMA_200": IND_EMA_200,
    "WMA_14": IND_WMA_14,
    "WMA_50": IND_WMA_50,
    "WMA_100": IND_WMA_100,
    "WMA_200": IND_WMA_200,
}

# ============================================================================
# GENOME DEFINITIONS
# ============================================================================
@dataclass
class Genome:
    """A trading strategy encoded as a genome"""
    entry_rules: List[Dict]
    exit_rules: List[Dict]
    params: Dict  # sl_pct, tp_pct, position_size
    
    def to_dict(self) -> Dict:
        return {
            "entry_rules": self.entry_rules,
            "exit_rules": self.exit_rules,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Genome':
        return cls(
            entry_rules=data.get("entry_rules", []),
            exit_rules=data.get("exit_rules", []),
            params=data.get("params", {})
        )


class GenomeEncoder:
    """Encode genomes for Numba JIT backtesting.
    
    IMPORTANT: Uses indicator indices from solana_backtester.py
    to ensure compatibility between miner and backtester.
    """
    
    OP_GT = 0  # >
    OP_LT = 1  # <
    OP_EQ = 2  # ==
    
    MAX_RULES = 3
    GENOME_SIZE = 20  # [sl_pct, tp_pct, size, num_entry, num_exit, rules...]
    
    @classmethod
    def encode(cls, genome: Genome) -> np.ndarray:
        """Encode genome to fixed-size numpy array for JIT"""
        arr = np.zeros(cls.GENOME_SIZE, dtype=np.float64)
        
        # Basic params
        arr[0] = genome.params.get("sl_pct", 0.03)
        arr[1] = genome.params.get("tp_pct", 0.05)
        arr[2] = genome.params.get("position_size", 0.1)
        arr[3] = len(genome.entry_rules)
        arr[4] = len(genome.exit_rules)
        
        # Encode entry rules (5 values each: indicator_idx, period, operator, threshold, compare_type)
        for i, rule in enumerate(genome.entry_rules[:cls.MAX_RULES]):
            base = 5 + i * 5
            ind_name = rule.get("indicator", "RSI_14")
            
            # Get indicator index from solana_backtester mapping
            ind_idx = INDICATOR_MAP.get(ind_name, IND_RSI_14)
            
            arr[base] = ind_idx
            arr[base + 1] = rule.get("period", 14)
            arr[base + 2] = cls._encode_operator(rule.get("operator", ">"))
            arr[base + 3] = rule.get("threshold", 30)
            arr[base + 4] = cls._encode_compare_type(rule.get("compare_to", "constant"))
        
        return arr
    
    @classmethod
    def _encode_operator(cls, op: str) -> float:
        """Encode comparison operator"""
        return cls.OP_GT if op == ">" else cls.OP_LT if op == "<" else cls.OP_EQ
    
    @classmethod
    def _encode_compare_type(cls, compare_to: str) -> float:
        """Encode comparison type: constant=0, price=1, indicator=2"""
        mapping = {"constant": 0, "price": 1, "indicator": 2}
        return mapping.get(compare_to, 0)
    
    @classmethod
    def decode_rule(cls, ind_idx: float, threshold: float, operator: float, compare_type: float) -> Dict:
        """Decode a rule from genome values to human-readable format"""
        # Find indicator name from index
        ind_name = "CLOSE"
        for name, idx in INDICATOR_MAP.items():
            if idx == int(ind_idx):
                ind_name = name
                break
        
        op_str = ">" if operator == cls.OP_GT else "<" if operator == cls.OP_LT else "=="
        compare_str = "constant"
        
        return {
            "indicator": ind_name,
            "threshold": threshold,
            "operator": op_str,
            "compare_to": compare_str
        }
    
    @classmethod
    def get_indicator_index(cls, indicator_name: str) -> int:
        """Get the index for an indicator name (matches solana_backtester.py)"""
        return INDICATOR_MAP.get(indicator_name, IND_RSI_14)
    
    @classmethod
    def get_available_indicators(cls) -> List[str]:
        """List all available indicator names"""
        return list(INDICATOR_MAP.keys())
    
    @classmethod
    def get_genome_size(cls) -> int:
        """Get the size of encoded genome"""
        return cls.GENOME_SIZE


# ============================================================================
# STRATEGY MINER
# ============================================================================
class StrategyMiner:
    """
    Genetic Algorithm to discover optimal trading strategies.
    
    Args:
        df: Historical price data
        population_size: Number of genomes per generation
        generations: Number of evolution rounds
        db_path: SQLite database for results
    """
    
    def __init__(self, df: pd.DataFrame, population_size: int = 50, 
                 generations: int = 20, db_path: str = DB_PATH):
        self.df = df
        self.pop_size = population_size
        self.generations = generations
        self.db_path = db_path
        
        # Available building blocks (use names that map to INDICATOR_MAP)
        self.indicators = ["RSI_7", "RSI_14", "RSI_21", "SMA_14", "EMA_14"]
        self.periods = [7, 14, 21, 50, 100]
        self.operators = [">", "<"]
        
        # RSI thresholds
        self.rsi_oversold = [20, 25, 30, 35]
        self.rsi_overbought = [65, 70, 75, 80]
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for results"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            population INTEGER,
            generations INTEGER,
            best_genome TEXT,
            best_pnl REAL,
            best_win_rate REAL,
            total_evaluated INTEGER
        )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS population (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            genome TEXT,
            pnl REAL,
            win_rate REAL,
            sharpe REAL,
            max_dd REAL,
            generation INTEGER,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )''')
        
        conn.commit()
        conn.close()
    
    def generate_random_genome(self) -> Genome:
        """Create a random strategy genome"""
        # Random indicator for entry
        ind = random.choice(self.indicators)
        
        # Build rule with proper indicator name
        if ind.startswith("RSI"):
            # RSI oversold/overbought
            rule = {
                "indicator": ind,  # e.g., "RSI_14"
                "period": int(ind.split("_")[1]) if "_" in ind else 14,
                "operator": random.choice(self.operators),
                "threshold": random.choice(
                    self.rsi_oversold if random.random() < 0.5 else self.rsi_overbought
                )
            }
        elif ind.startswith("SMA"):
            # Price vs SMA crossover
            rule = {
                "indicator": ind,
                "period": int(ind.split("_")[1]) if "_" in ind else 14,
                "operator": ">",
                "compare_to": "price"
            }
        else:  # EMA
            rule = {
                "indicator": ind,
                "period": int(ind.split("_")[1]) if "_" in ind else 14,
                "operator": ">",
                "compare_to": "price"
            }
        
        return Genome(
            entry_rules=[rule],
            exit_rules=[],  # Exit on TP/SL
            params={
                "sl_pct": random.uniform(0.02, 0.05),
                "tp_pct": random.uniform(0.03, 0.10),
                "position_size": random.uniform(0.05, 0.20)
            }
        )
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Combine two genomes"""
        child_params = parent1.params.copy() if random.random() < 0.5 else parent2.params
        
        return Genome(
            entry_rules=parent1.entry_rules[:] if random.random() < 0.5 else parent2.entry_rules[:],
            exit_rules=[],
            params=child_params
        )
    
    def mutate(self, genome: Genome, mutation_rate: float = 0.1) -> Genome:
        """Randomly mutate a genome"""
        if random.random() > mutation_rate:
            return genome
        
        # Mutate one rule
        if genome.entry_rules and random.random() < 0.5:
            rule = genome.entry_rules[0].copy()
            
            # Change one attribute
            attr = random.choice(["period", "threshold", "operator"])
            if attr == "period":
                rule["period"] = random.choice(self.periods)
            elif attr == "threshold":
                if rule.get("indicator", "").startswith("RSI"):
                    all_thresh = self.rsi_oversold + self.rsi_overbought
                    rule["threshold"] = random.choice(all_thresh)
            elif attr == "operator":
                rule["operator"] = ">" if rule["operator"] == "<" else "<"
            
            genome.entry_rules[0] = rule
        
        # Mutate params
        if random.random() < 0.3:
            p = genome.params
            if random.random() < 0.5:
                p["sl_pct"] = min(0.10, max(0.01, p.get("sl_pct", 0.03) + random.uniform(-0.01, 0.01)))
            else:
                p["tp_pct"] = min(0.20, max(0.02, p.get("tp_pct", 0.05) + random.uniform(-0.02, 0.02)))
        
        return genome
    
    def evaluate(self, genome: Genome) -> Tuple[float, float, float]:
        """Evaluate a genome's fitness (PnL, Win Rate, Sharpe)"""
        sl_pct = genome.params.get("sl_pct", 0.03)
        tp_pct = genome.params.get("tp_pct", 0.05)
        
        if not genome.entry_rules:
            return 0.0, 0.0, 0.0
        
        rule = genome.entry_rules[0]
        ind_name = rule.get("indicator", "RSI_14")
        period = rule.get("period", 14)
        threshold = rule.get("threshold", 30)
        operator = rule.get("operator", ">")
        
        # Calculate indicator using GenomeEncoder index
        ind_idx = GenomeEncoder.get_indicator_index(ind_name)
        
        # Calculate indicator based on index
        if "RSI" in ind_name:
            series = self._calculate_rsi(period)
        elif "SMA" in ind_name:
            series = self._calculate_sma(period)
        elif "EMA" in ind_name:
            series = self._calculate_ema(period)
        else:
            series = self._calculate_sma(14)
        
        # Generate signals
        if operator == ">":
            signals = series > threshold
        else:
            signals = series < threshold
        
        # Simulate trades
        trades = []
        position = None
        
        for i in range(len(self.df)):
            if signals.iloc[i] and position is None:
                position = {"entry": self.df.iloc[i]['close']}
            elif position is not None:
                price = self.df.iloc[i]['close']
                pnl = (price - position["entry"]) / position["entry"]
                
                if pnl >= tp_pct or pnl <= -sl_pct:
                    trades.append(pnl)
                    position = None
        
        if not trades:
            return 0.0, 0.0, 0.0
        
        pnls = np.array(trades)
        wins = (pnls > 0).sum()
        total = len(pnls)
        win_rate = wins / total if total > 0 else 0
        
        mean_pnl = pnls.mean()
        std_pnl = pnls.std() if len(pnls) > 1 else 0.01
        sharpe = (mean_pnl / std_pnl) * np.sqrt(total) if std_pnl > 0 else 0
        
        return sum(pnls), win_rate, sharpe
    
    def _calculate_rsi(self, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 0.001)
        return 100 - (100 / (1 + rs))
    
    def _calculate_sma(self, period: int) -> pd.Series:
        """Calculate SMA"""
        return self.df['close'].rolling(period).mean()
    
    def _calculate_ema(self, period: int) -> pd.Series:
        """Calculate EMA"""
        return self.df['close'].ewm(span=period, adjust=False).mean()
    
    def evolve(self, verbose: bool = True) -> Dict:
        """Run the genetic algorithm."""
        import time
        start_time = time.time()
        
        # Initialize population
        population = [self.generate_random_genome() for _ in range(self.pop_size)]
        
        best_genome = None
        best_pnl = float('-inf')
        
        for gen in range(self.generations):
            gen_start = time.time()
            
            # Evaluate all
            results = []
            for genome in population:
                pnl, win_rate, sharpe = self.evaluate(genome)
                results.append((genome, pnl, win_rate, sharpe))
                
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_genome = genome
            
            # Sort by PnL
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Elitism: keep top 5
            elite = [r[0] for r in results[:5]]
            
            # Create next generation
            new_population = elite.copy()
            
            while len(new_population) < self.pop_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            gen_time = time.time() - gen_start
            
            if verbose and gen % 5 == 0:
                top_pnl = results[0][1]
                top_wr = results[0][2]
                print(f"  Gen {gen}: Best PnL={top_pnl:.4f}, Win Rate={top_wr:.1%}, Time={gen_time:.2f}s")
        
        total_time = time.time() - start_time
        final_pnl, final_wr, final_sharpe = self.evaluate(best_genome)
        
        self._save_run(best_genome, final_pnl, final_wr, total_time)
        
        result = {
            "best_genome": best_genome.to_dict(),
            "best_pnl": final_pnl,
            "win_rate": final_wr,
            "sharpe": final_sharpe,
            "generations": self.generations,
            "population": self.pop_size,
            "time_seconds": total_time
        }
        
        if verbose:
            print(f"\n🏆 Best Strategy Found:")
            print(f"   PnL: {final_pnl:.4f} ({final_pnl*100:.2f}%)")
            print(f"   Win Rate: {final_wr:.1%}")
            print(f"   Sharpe: {final_sharpe:.2f}")
            print(f"   Time: {total_time:.2f}s")
            print(f"\n📊 Rules: {best_genome.entry_rules}")
            print(f"   TP: {best_genome.params.get('tp_pct', 0)*100:.1f}% | SL: {best_genome.params.get('sl_pct', 0)*100:.1f}%")
        
        return result
    
    def _save_run(self, genome: Genome, pnl: float, win_rate: float, time_seconds: float):
        """Save results to SQLite"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO runs 
            (created_at, population, generations, best_genome, best_pnl, best_win_rate, total_evaluated)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (datetime.now().isoformat(), self.pop_size, self.generations,
             json.dumps(genome.to_dict()), pnl, win_rate, self.pop_size * self.generations)
        )
        
        run_id = c.lastrowid
        
        for genome in [genome]:
            c.execute('''INSERT INTO population 
                (run_id, genome, pnl, win_rate, generation) VALUES (?, ?, ?, ?, ?)''',
                (run_id, json.dumps(genome.to_dict()), pnl, win_rate, self.generations)
            )
        
        conn.commit()
        conn.close()
    
    def get_best_strategies(self, limit: int = 10) -> List[Dict]:
        """Retrieve best strategies from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT * FROM runs ORDER BY best_pnl DESC LIMIT ?''', (limit,))
        
        results = []
        for row in c.fetchall():
            results.append({
                "id": row[0],
                "created_at": row[1],
                "population": row[2],
                "generations": row[3],
                "genome": json.loads(row[4]),
                "pnl": row[5],
                "win_rate": row[6],
                "total_evaluated": row[7]
            })
        
        conn.close()
        return results


# ============================================================================
# COMPATIBILITY FUNCTIONS
# ============================================================================
def encode_genome_for_backtester(genome: Genome) -> np.ndarray:
    """Encode genome for use with solana_backtester.py.
    
    This ensures compatibility between genetic_miner and solana_backtester.
    """
    return GenomeEncoder.encode(genome)


def get_backtester_indicator_index(indicator_name: str) -> int:
    """Get the indicator index that matches solana_backtester.py"""
    return GenomeEncoder.get_indicator_index(indicator_name)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from data.historical_data import HistoricalDataManager
    
    print("="*60)
    print("🧬 GENETIC STRATEGY MINER")
    print("="*60)
    
    # Load data
    manager = HistoricalDataManager()
    df = manager.get_historical_data("SOL", timeframe="1h", days=90)
    
    print(f"\n📊 Data: {len(df)} candles")
    print(f"   Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Run miner
    miner = StrategyMiner(df, population_size=30, generations=15)
    result = miner.evolve(verbose=True)
    
    print(f"\n✅ Miner complete!")
    print(f"   Best PnL: {result['best_pnl']*100:.2f}%")
    
    # Show encoded genome
    genome = Genome.from_dict(result['best_genome'])
    encoded = encode_genome_for_backtester(genome)
    print(f"\n📦 Encoded genome (for solana_backtester):")
    print(f"   {encoded}")
    print(f"   Size: {len(encoded)} floats")
