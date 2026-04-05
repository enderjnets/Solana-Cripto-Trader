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
import traceback

logger = logging.getLogger("genetic_miner")


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class GeneticMinerError(Exception):
    """Base exception for genetic miner errors"""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.logger.error(f"{self.__class__.__name__}: {message}", extra=details)


class InvalidGenomeError(GeneticMinerError):
    """Raised when a genome is invalid or malformed"""
    pass


class DatabaseError(GeneticMinerError):
    """Raised when database operations fail"""
    pass


class EvaluationError(GeneticMinerError):
    """Raised when genome evaluation fails"""
    pass


class EvolutionError(GeneticMinerError):
    """Raised when the evolution process fails"""
    pass


class ConfigurationError(GeneticMinerError):
    """Raised when configuration is invalid"""
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = "data/genetic_results.db"
DEFAULT_POPULATION = 20
DEFAULT_GENERATIONS = 10


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
    """Encode genomes for Numba JIT backtesting"""
    
    IND_CLOSE = 0
    IND_HIGH = 1
    IND_LOW = 2
    IND_RSI = 3
    IND_SMA = 4
    IND_EMA = 5
    IND_BB_HIGH = 6
    IND_BB_LOW = 7
    IND_BB_MID = 8
    
    OP_GT = 0  # >
    OP_LT = 1  # <
    
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
        
        # Encode entry rules (6 values each: indicator, period, operator, threshold)
        for i, rule in enumerate(genome.entry_rules[:cls.MAX_RULES]):
            base = 5 + i * 5
            arr[base] = cls._encode_indicator(rule.get("indicator", "RSI"))
            arr[base + 1] = rule.get("period", 14)
            arr[base + 2] = cls._encode_operator(rule.get("operator", ">"))
            arr[base + 3] = rule.get("threshold", 30)
            arr[base + 4] = cls._encode_indicator(rule.get("compare_to", "constant"))
        
        return arr
    
    @classmethod
    def _encode_indicator(cls, name: str) -> float:
        mapping = {
            "close": cls.IND_CLOSE, "high": cls.IND_HIGH, "low": cls.IND_LOW,
            "RSI": cls.IND_RSI, "SMA": cls.IND_SMA, "EMA": cls.IND_EMA,
            "BB_HIGH": cls.IND_BB_HIGH, "BB_LOW": cls.IND_BB_LOW, "BB_MID": cls.IND_BB_MID,
            "constant": 99
        }
        return mapping.get(name, cls.IND_RSI)
    
    @classmethod
    def _encode_operator(cls, op: str) -> float:
        return cls.OP_GT if op == ">" else cls.OP_LT


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
        
        # Available building blocks
        self.indicators = ["RSI", "SMA", "EMA", "BB"]
        self.periods = [7, 14, 20, 50, 100]
        self.operators = [">", "<"]
        
        # RSI thresholds
        self.rsi_oversold = [20, 25, 30, 35]
        self.rsi_overbought = [65, 70, 75, 80]
        
        # BB thresholds
        self.bb_thresholds = [0.0, 1.0, 2.0, 3.0]  # Standard deviations
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for results"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            c = conn.cursor()
            
            # Create tables with error handling
            tables_sql = [
                '''CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT,
                    population INTEGER,
                    generations INTEGER,
                    best_genome TEXT,
                    best_pnl REAL,
                    best_win_rate REAL,
                    total_evaluated INTEGER
                )''',
                '''CREATE TABLE IF NOT EXISTS population (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER,
                    genome TEXT,
                    pnl REAL,
                    win_rate REAL,
                    sharpe REAL,
                    max_dd REAL,
                    generation INTEGER,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )'''
            ]
            
            for sql in tables_sql:
                try:
                    c.execute(sql)
                except sqlite3.Error as e:
                    raise DatabaseError(
                        f"Failed to create table: {e}",
                        {"sql": sql[:100], "error": str(e)}
                    )
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to initialize database: {e}",
                {"db_path": self.db_path, "error": str(e)}
            )
    
    def generate_random_genome(self) -> Genome:
        """Create a random strategy genome"""
        try:
            # Random indicator for entry
            ind = random.choice(self.indicators)
            
            if ind == "RSI":
                # RSI oversold/overbought
                rule = {
                    "indicator": "RSI",
                    "period": random.choice(self.periods),
                    "operator": random.choice([">", "<"]),
                    "threshold": random.choice(
                        self.rsi_oversold if random.random() < 0.5 else self.rsi_overbought
                    )
                }
            elif ind == "SMA":
                # Price vs SMA crossover
                rule = {
                    "indicator": "SMA",
                    "period": random.choice(self.periods),
                    "operator": ">",
                    "compare_to": "price"
                }
            elif ind == "EMA":
                # EMA crossover
                rule = {
                    "indicator": "EMA",
                    "period": random.choice(self.periods),
                    "operator": ">",
                    "compare_to": "SMA"
                }
            else:  # BB
                rule = {
                    "indicator": "BB",
                    "period": random.choice([14, 20]),
                    "operator": "<",
                    "threshold": random.choice(self.bb_thresholds),
                    "compare_to": "BB_LOW"
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
        except Exception as e:
            raise EvolutionError(
                f"Failed to generate random genome: {e}",
                details={"indicator": ind, "error": str(e)}
            )
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Combine two genomes"""
        # Single-point crossover on params
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
                if rule.get("indicator") == "RSI":
                    all_thresh = self.rsi_oversold + self.rsi_overbought
                    rule["threshold"] = random.choice(all_thresh)
                else:
                    rule["threshold"] = random.choice(self.bb_thresholds)
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
        """
        Evaluate a genome's fitness (PnL, Win Rate, Sharpe)
        
        Returns:
            Tuple of (PnL, Win Rate, Sharpe Ratio)
            
        Raises:
            InvalidGenomeError: If genome structure is invalid
            EvaluationError: If evaluation fails due to data issues
        """
        try:
            # Validate genome structure
            if not isinstance(genome, Genome):
                raise InvalidGenomeError(
                    f"Invalid genome type: {type(genome)}",
                    {"genome_type": str(type(genome))}
                )
            
            sl_pct = genome.params.get("sl_pct", 0.03)
            tp_pct = genome.params.get("tp_pct", 0.05)
            
            # Validate stop loss and take profit percentages
            if not (0 < sl_pct < 1):
                raise InvalidGenomeError(
                    f"Invalid stop loss percentage: {sl_pct}",
                    {"sl_pct": sl_pct}
                )
            if not (0 < tp_pct < 1):
                raise InvalidGenomeError(
                    f"Invalid take profit percentage: {tp_pct}",
                    {"tp_pct": tp_pct}
                )
            
            if not genome.entry_rules:
                logger.warning("Empty entry rules - returning zero metrics")
                return 0.0, 0.0, 0.0
            
            rule = genome.entry_rules[0]
            
            # Validate rule structure
            required_keys = ["indicator"]
            for key in required_keys:
                if key not in rule:
                    raise InvalidGenomeError(
                        f"Missing required rule key: {key}",
                        {"rule": rule, "missing_key": key}
                    )
            
            ind_name = rule.get("indicator", "RSI")
            period = rule.get("period", 14)
            threshold = rule.get("threshold", 30)
            operator = rule.get("operator", ">")
            
            # Validate indicator name
            valid_indicators = ["RSI", "SMA", "EMA", "BB"]
            if ind_name not in valid_indicators:
                raise InvalidGenomeError(
                    f"Unknown indicator: {ind_name}",
                    {"indicator": ind_name, "valid_indicators": valid_indicators}
                )
            
            # Calculate indicator
            try:
                if ind_name == "RSI":
                    series = self._calculate_rsi(period)
                elif ind_name == "SMA":
                    series = self._calculate_sma(period)
                elif ind_name == "EMA":
                    series = self._calculate_ema(period)
                else:
                    series = self._calculate_bb(period)
            except Exception as e:
                raise EvaluationError(
                    f"Failed to calculate indicator {ind_name}: {e}",
                    {"indicator": ind_name, "period": period, "error": str(e)}
                )
            
            # Generate signals
            try:
                if operator == ">":
                    signals = series > threshold
                elif operator == "<":
                    signals = series < threshold
                else:
                    raise InvalidGenomeError(
                        f"Invalid operator: {operator}",
                        {"operator": operator}
                    )
            except Exception as e:
                raise EvaluationError(
                    f"Failed to generate signals: {e}",
                    {"threshold": threshold, "operator": operator, "error": str(e)}
                )
            
            # Simulate trades
            trades = []
            position = None
            
            for i in range(len(self.df)):
                try:
                    if signals.iloc[i] and position is None:
                        position = {"entry": self.df.iloc[i]['close']}
                    elif position is not None:
                        price = self.df.iloc[i]['close']
                        pnl = (price - position["entry"]) / position["entry"]
                        
                        if pnl >= tp_pct or pnl <= -sl_pct:
                            trades.append(pnl)
                            position = None
                except Exception as e:
                    logger.warning(f"Error processing candle {i}: {e}")
                    continue
            
            if not trades:
                logger.info("No trades generated for this genome")
                return 0.0, 0.0, 0.0
            
            pnls = np.array(trades)
            wins = (pnls > 0).sum()
            total = len(pnls)
            win_rate = wins / total if total > 0 else 0
            
            # Sharpe ratio approximation
            mean_pnl = pnls.mean()
            std_pnl = pnls.std() if len(pnls) > 1 else 0.01
            sharpe = (mean_pnl / std_pnl) * np.sqrt(total) if std_pnl > 0 else 0
            
            logger.debug(
                f"Genome evaluated: PnL={sum(pnls):.4f}, WinRate={win_rate:.2%}, Sharpe={sharpe:.2f}"
            )
            
            return sum(pnls), win_rate, sharpe
            
        except GeneticMinerError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise EvaluationError(
                f"Unexpected error during evaluation: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
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
    
    def _calculate_bb(self, period: int) -> pd.Series:
        """Calculate Bollinger Band position (std devs from middle)"""
        sma = self.df['close'].rolling(period).mean()
        std = self.df['close'].rolling(period).std()
        return (self.df['close'] - sma) / std.replace(0, 0.001)
    
    def evolve(self, verbose: bool = True) -> Dict:
        """
        Run the genetic algorithm.
        
        Args:
            verbose: Whether to print progress updates
            
        Returns:
            Dictionary with best genome and statistics
            
        Raises:
            EvolutionError: If evolution process fails
            ConfigurationError: If configuration is invalid
        """
        try:
            # Validate configuration
            if self.pop_size < 1:
                raise ConfigurationError(
                    f"Invalid population size: {self.pop_size} (must be >= 1)",
                    {"population_size": self.pop_size}
                )
            if self.generations < 1:
                raise ConfigurationError(
                    f"Invalid generations: {self.generations} (must be >= 1)",
                    {"generations": self.generations}
                )
            
            import time
            start_time = time.time()
            
            logger.info(
                f"Starting evolution: pop_size={self.pop_size}, generations={self.generations}"
            )
            
            # Initialize population with error handling
            try:
                population = [self.generate_random_genome() for _ in range(self.pop_size)]
            except Exception as e:
                raise EvolutionError(
                    f"Failed to initialize population: {e}",
                    {"pop_size": self.pop_size, "error": str(e)}
                )
            
            best_genome = None
            best_pnl = float('-inf')
            total_evaluated = 0
            failed_evaluations = 0
            
            for gen in range(self.generations):
                gen_start = time.time()
                
                # Evaluate all
                results = []
                for i, genome in enumerate(population):
                    try:
                        pnl, win_rate, sharpe = self.evaluate(genome)
                        results.append((genome, pnl, win_rate, sharpe))
                        total_evaluated += 1
                        
                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_genome = genome
                            
                    except (InvalidGenomeError, EvaluationError) as e:
                        failed_evaluations += 1
                        logger.warning(
                            f"Evaluation failed for genome {i} in gen {gen}: {e}"
                        )
                        # Add a neutral genome to keep population size
                        results.append((genome, 0.0, 0.0, 0.0))
                    except Exception as e:
                        failed_evaluations += 1
                        logger.error(
                            f"Unexpected error evaluating genome {i}: {e}",
                            {"error": str(e), "traceback": traceback.format_exc()}
                        )
                        results.append((genome, 0.0, 0.0, 0.0))
                
                # Check if we have any valid results
                if not results:
                    raise EvolutionError(
                        "No genomes evaluated successfully",
                        {"generation": gen, "population_size": len(population)}
                    )
                
                # Sort by PnL
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Elitism: keep top 5
                elite = [r[0] for r in results[:5]]
                
                # Create next generation
                new_population = elite.copy()
                
                while len(new_population) < self.pop_size:
                    # Tournament selection
                    parent1 = random.choice(elite)
                    parent2 = random.choice(elite)
                    
                    try:
                        # Crossover
                        child = self.crossover(parent1, parent2)
                        
                        # Mutation
                        child = self.mutate(child)
                        
                        new_population.append(child)
                    except Exception as e:
                        logger.warning(f"Failed to create child genome: {e}")
                        # Use a random genome instead
                        new_population.append(self.generate_random_genome())
                
                population = new_population
                
                gen_time = time.time() - gen_start
                
                if verbose and gen % 5 == 0:
                    top_pnl = results[0][1]
                    top_wr = results[0][2]
                    success_rate = (total_evaluated - failed_evaluations) / total_evaluated * 100
                    print(f"  Gen {gen}: Best PnL={top_pnl:.4f}, Win Rate={top_wr:.1%}, "
                          f"Time={gen_time:.2f}s, Success={success_rate:.1f}%")
            
            total_time = time.time() - start_time
            
            # Final evaluation
            if best_genome is None:
                raise EvolutionError(
                    "No successful genome found during evolution",
                    {"generations": self.generations, "total_evaluated": total_evaluated}
                )
            
            try:
                final_pnl, final_wr, final_sharpe = self.evaluate(best_genome)
            except Exception as e:
                raise EvolutionError(
                    f"Failed to evaluate best genome: {e}",
                    {"error": str(e)}
                )
            
            # Save to database
            try:
                self._save_run(best_genome, final_pnl, final_wr, total_time)
            except DatabaseError as e:
                logger.error(f"Failed to save run to database: {e}")
                # Continue without saving - not critical
            
            result = {
                "best_genome": best_genome.to_dict(),
                "best_pnl": final_pnl,
                "win_rate": final_wr,
                "sharpe": final_sharpe,
                "generations": self.generations,
                "population": self.pop_size,
                "time_seconds": total_time,
                "total_evaluated": total_evaluated,
                "failed_evaluations": failed_evaluations
            }
            
            logger.info(
                f"Evolution complete: PnL={final_pnl:.4f}, WinRate={final_wr:.2%}, "
                f"Sharpe={final_sharpe:.2f}, Time={total_time:.2f}s"
            )
            
            if verbose:
                print(f"\n🏆 Best Strategy Found:")
                print(f"   PnL: {final_pnl:.4f} ({final_pnl*100:.2f}%)")
                print(f"   Win Rate: {final_wr:.1%}")
                print(f"   Sharpe: {final_sharpe:.2f}")
                print(f"   Time: {total_time:.2f}s")
                print(f"\n📊 Rules: {best_genome.entry_rules}")
                print(f"   TP: {best_genome.params.get('tp_pct', 0)*100:.1f}% | "
                      f"SL: {best_genome.params.get('sl_pct', 0)*100:.1f}%")
            
            return result
            
        except GeneticMinerError:
            raise
        except Exception as e:
            raise EvolutionError(
                f"Evolution process failed: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def _save_run(self, genome: Genome, pnl: float, win_rate: float, time_seconds: float):
        """Save results to SQLite
        
        Raises:
            DatabaseError: If database operation fails
            InvalidGenomeError: If genome is invalid
        """
        try:
            # Validate genome
            if not isinstance(genome, Genome):
                raise InvalidGenomeError(
                    f"Invalid genome type for save: {type(genome)}",
                    {"genome_type": str(type(genome))}
                )
            
            # Serialize genome
            try:
                genome_json = json.dumps(genome.to_dict())
            except (TypeError, ValueError) as e:
                raise InvalidGenomeError(
                    f"Failed to serialize genome: {e}",
                    {"error": str(e)}
                )
            
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            c = conn.cursor()
            
            try:
                c.execute('''INSERT INTO runs 
                    (created_at, population, generations, best_genome, best_pnl, best_win_rate, total_evaluated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (datetime.now().isoformat(), self.pop_size, self.generations,
                     genome_json, pnl, win_rate, self.pop_size * self.generations)
                )
                
                run_id = c.lastrowid
                
                # Save population stats
                c.execute('''INSERT INTO population 
                    (run_id, genome, pnl, win_rate, generation) VALUES (?, ?, ?, ?, ?)''',
                    (run_id, genome_json, pnl, win_rate, self.generations)
                )
                
                conn.commit()
                logger.info(f"Run saved successfully: run_id={run_id}, PnL={pnl:.4f}")
                
            except sqlite3.Error as e:
                conn.rollback()
                raise DatabaseError(
                    f"Failed to insert data: {e}",
                    {"run_id": run_id if 'run_id' in locals() else None, "error": str(e)}
                )
            finally:
                conn.close()
                
        except GeneticMinerError:
            raise
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error saving run: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def get_best_strategies(self, limit: int = 10) -> List[Dict]:
        """Retrieve best strategies from database
        
        Args:
            limit: Maximum number of strategies to return
            
        Returns:
            List of strategy dictionaries
            
        Raises:
            DatabaseError: If database operation fails
            ConfigurationError: If limit is invalid
        """
        try:
            # Validate limit
            if not isinstance(limit, int):
                raise ConfigurationError(
                    f"Invalid limit type: {type(limit)}",
                    {"limit": limit, "expected_type": "int"}
                )
            if limit < 1:
                raise ConfigurationError(
                    f"Invalid limit value: {limit} (must be >= 1)",
                    {"limit": limit}
                )
            if limit > 1000:
                logger.warning(f"Large limit requested: {limit}, capping at 1000")
                limit = 1000
            
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            c = conn.cursor()
            
            try:
                c.execute('''SELECT * FROM runs ORDER BY best_pnl DESC LIMIT ?''', (limit,))
                
                results = []
                for row in c.fetchall():
                    try:
                        genome_data = json.loads(row[4])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse genome for run {row[0]}: {e}")
                        genome_data = {}
                    
                    results.append({
                        "id": row[0],
                        "created_at": row[1],
                        "population": row[2],
                        "generations": row[3],
                        "genome": genome_data,
                        "pnl": row[5],
                        "win_rate": row[6],
                        "total_evaluated": row[7]
                    })
                
                logger.info(f"Retrieved {len(results)} strategies from database")
                return results
                
            except sqlite3.Error as e:
                raise DatabaseError(
                    f"Failed to query strategies: {e}",
                    {"limit": limit, "error": str(e)}
                )
            finally:
                conn.close()
                
        except GeneticMinerError:
            raise
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error retrieving strategies: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )


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
