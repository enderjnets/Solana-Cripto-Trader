#!/usr/bin/env python3
"""
Strategy Agent
==============
Specialized agent for managing trading strategies.

Features:
- Genome creation and evolution
- Backtesting integration
- Strategy persistence
- Performance tracking

Usage:
    agent = StrategyAgent()
    result = agent.optimize_strategy(df, population=50, generations=20)
    genome = agent.get_best_strategy()
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.genetic_miner import (
    StrategyMiner, Genome, GenomeEncoder, INDICATOR_MAP,
    DB_PATH as GENETIC_DB_PATH
)
from backtesting.solana_backtester import (
    precompute_indicators, run_backtest, NUM_INDICATORS, INDICATOR_NAMES
)

import logging
logger = logging.getLogger("strategy_agent")


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class StrategyConfig:
    """Strategy agent configuration"""
    population_size: int = 50
    generations: int = 20
    min_pnl: float = 0.01  # 1% minimum PnL
    min_win_rate: float = 0.40  # 40% minimum win rate
    max_drawdown: float = 0.15  # 15% max drawdown
    strategy_db_path: str = "data/strategies.json"
    cache_size: int = 100  # Number of strategies to cache


# ============================================================================
# STRATEGY AGENT
# ============================================================================
class StrategyAgent:
    """
    Specialized agent for managing trading strategies.
    
    Responsibilities:
    - Create and evolve trading strategies
    - Backtest strategies against historical data
    - Track strategy performance
    - Deploy winning strategies to live trading
    
    Attributes:
        config: Agent configuration
        miner: Genetic algorithm miner
        strategy_cache: Cached strategies
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.miner = None
        self.strategy_cache = self._load_strategy_cache()
        self.current_best = None
        self.performance_history = []
        
        logger.info(f"🚀 StrategyAgent initialized")
        logger.info(f"   Population: {self.config.population_size}")
        logger.info(f"   Generations: {self.config.generations}")
    
    def _load_strategy_cache(self) -> Dict:
        """Load cached strategies from disk"""
        if os.path.exists(self.config.strategy_db_path):
            try:
                with open(self.config.strategy_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load strategy cache: {e}")
        return {"strategies": [], "deployments": []}
    
    def _save_strategy_cache(self):
        """Save strategy cache to disk"""
        try:
            with open(self.config.strategy_db_path, 'w') as f:
                json.dump(self.strategy_cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save strategy cache: {e}")
    
    def optimize_strategy(
        self,
        df: pd.DataFrame,
        population_size: int = None,
        generations: int = None,
        verbose: bool = True
    ) -> Dict:
        """
        Optimize a trading strategy using genetic algorithm.
        
        Args:
            df: Historical price data
            population_size: Number of genomes per generation
            generations: Number of evolution rounds
            verbose: Print progress
            
        Returns:
            Dict with best genome, performance metrics, and evolution stats
        """
        pop_size = population_size or self.config.population_size
        gens = generations or self.config.generations
        
        if verbose:
            print(f"\n🧬 Starting strategy optimization...")
            print(f"   Population: {pop_size}")
            print(f"   Generations: {gens}")
            print(f"   Data: {len(df)} candles")
        
        start_time = time.time()
        
        # Initialize miner
        self.miner = StrategyMiner(
            df=df,
            population_size=pop_size,
            generations=gens
        )
        
        # Run evolution
        result = self.miner.evolve(verbose=verbose)
        
        elapsed = time.time() - start_time
        
        # Validate result
        if result['best_pnl'] >= self.config.min_pnl:
            result['meets_min_pnl'] = True
        else:
            result['meets_min_pnl'] = False
            logger.warning(f"Best strategy PnL below minimum: {result['best_pnl']:.4f}")
        
        # Store in cache
        self._cache_strategy(result)
        
        if verbose:
            print(f"\n✅ Optimization complete in {elapsed:.2f}s")
            print(f"   Best PnL: {result['best_pnl']:.4f} ({result['best_pnl']*100:.2f}%)")
            print(f"   Win Rate: {result['win_rate']:.1%}")
            print(f"   Sharpe: {result['sharpe']:.2f}")
        
        return result
    
    def _cache_strategy(self, result: Dict):
        """Cache a successful strategy"""
        genome_dict = result.get('best_genome', {})
        
        strategy = {
            "id": f"strat_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "genome": genome_dict,
            "pnl": result['best_pnl'],
            "win_rate": result['win_rate'],
            "sharpe": result['sharpe'],
            "generations": result['generations'],
            "trades_estimate": int(result['best_pnl'] * 100 / 0.03),  # Rough estimate
            "status": "pending",  # pending, active, archived
            "performance": []
        }
        
        self.strategy_cache["strategies"].insert(0, strategy)
        
        # Keep only top N strategies
        self.strategy_cache["strategies"] = self.strategy_cache["strategies"][:self.config.cache_size]
        
        self._save_strategy_cache()
        
        self.current_best = strategy
        logger.info(f"Cached strategy: PnL={result['best_pnl']:.4f}")
    
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        genome: Dict,
        verbose: bool = True
    ) -> Dict:
        """
        Backtest a strategy genome against historical data.
        
        Args:
            df: Historical price data
            genome: Strategy genome dictionary
            verbose: Print progress
            
        Returns:
            BacktestResult with performance metrics
        """
        # Convert genome dict to object
        genome_obj = Genome.from_dict(genome)
        
        # Encode for backtester
        encoded = GenomeEncoder.encode(genome_obj)
        
        if verbose:
            print(f"\n🔍 Backtesting strategy...")
            print(f"   Rules: {genome.get('entry_rules', [])}")
            print(f"   SL: {genome.get('params', {}).get('sl_pct', 0)*100:.1f}%")
            print(f"   TP: {genome.get('params', {}).get('tp_pct', 0)*100:.1f}%")
        
        # Pre-compute indicators
        indicators = precompute_indicators(df)
        
        # Run backtest
        from backtesting.solana_backtester import evaluate_genome
        result = evaluate_genome(indicators, encoded)
        
        # Format result
        backtest_result = {
            "pnl": result['pnl'],
            "pnl_pct": result['pnl'] * 100,
            "trades": result['trades'],
            "wins": result['wins'],
            "losses": result['losses'],
            "win_rate": result['win_rate'],
            "sharpe_ratio": result['sharpe_ratio'],
            "max_drawdown": result['max_drawdown'],
            "status": "pass" if result['pnl'] >= self.config.min_pnl else "fail",
            "meets_criteria": (
                result['pnl'] >= self.config.min_pnl and
                result['win_rate'] >= self.config.min_win_rate and
                result['max_drawdown'] <= self.config.max_drawdown
            )
        }
        
        if verbose:
            print(f"\n📊 Backtest Results:")
            print(f"   PnL: {backtest_result['pnl_pct']:.2f}%")
            print(f"   Trades: {backtest_result['trades']} ({backtest_result['wins']}W / {backtest_result['losses']}L)")
            print(f"   Win Rate: {backtest_result['win_rate']:.1%}")
            print(f"   Sharpe: {backtest_result['sharpe_ratio']:.2f}")
            print(f"   Max DD: {backtest_result['max_drawdown']:.1%}")
            print(f"   Status: {backtest_result['status']}")
        
        return backtest_result
    
    def get_best_strategy(self) -> Optional[Dict]:
        """Get the best performing strategy from cache"""
        if not self.strategy_cache["strategies"]:
            return None
        
        # Return highest PnL strategy
        best = max(
            self.strategy_cache["strategies"],
            key=lambda x: x.get('pnl', 0)
        )
        
        self.current_best = best
        return best
    
    def get_active_strategies(self) -> List[Dict]:
        """Get all active (non-archived) strategies"""
        return [
            s for s in self.strategy_cache["strategies"]
            if s.get('status') == 'active'
        ]
    
    def deploy_strategy(self, strategy_id: str) -> bool:
        """Mark a strategy as active/deployed"""
        for strategy in self.strategy_cache["strategies"]:
            if strategy["id"] == strategy_id:
                strategy["status"] = "active"
                strategy["deployed_at"] = datetime.now().isoformat()
                self._save_strategy_cache()
                logger.info(f"Deployed strategy: {strategy_id}")
                return True
        return False
    
    def archive_strategy(self, strategy_id: str) -> bool:
        """Archive a strategy (mark as inactive)"""
        for strategy in self.strategy_cache["strategies"]:
            if strategy["id"] == strategy_id:
                strategy["status"] = "archived"
                strategy["archived_at"] = datetime.now().isoformat()
                self._save_strategy_cache()
                logger.info(f"Archived strategy: {strategy_id}")
                return True
        return False
    
    def record_performance(self, strategy_id: str, performance: Dict):
        """Record live trading performance for a strategy"""
        for strategy in self.strategy_cache["strategies"]:
            if strategy["id"] == strategy_id:
                if "performance" not in strategy:
                    strategy["performance"] = []
                strategy["performance"].append({
                    "timestamp": datetime.now().isoformat(),
                    **performance
                })
                self._save_strategy_cache()
                return
        logger.warning(f"Strategy not found for performance tracking: {strategy_id}")
    
    def get_indicator_count(self) -> int:
        """Get total number of available indicators"""
        return NUM_INDICATORS
    
    def list_indicators(self) -> Dict:
        """List all available indicators"""
        return {
            "total": NUM_INDICATORS,
            "names": {str(k): v for k, v in INDICATOR_NAMES.items()}
        }
    
    def validate_genome(self, genome: Dict) -> Dict:
        """Validate a genome structure"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if "params" not in genome:
            validation["valid"] = False
            validation["errors"].append("Missing 'params' field")
        else:
            params = genome["params"]
            if "sl_pct" not in params:
                validation["warnings"].append("Missing 'sl_pct', using default")
            if "tp_pct" not in params:
                validation["warnings"].append("Missing 'tp_pct', using default")
        
        # Check entry rules
        if "entry_rules" not in genome:
            validation["valid"] = False
            validation["errors"].append("Missing 'entry_rules'")
        else:
            for i, rule in enumerate(genome["entry_rules"]):
                if "indicator" not in rule:
                    validation["errors"].append(f"Rule {i}: Missing 'indicator'")
                if "threshold" not in rule:
                    validation["warnings"].append(f"Rule {i}: Missing 'threshold', using default")
        
        return validation
    
    def export_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Export a strategy for deployment"""
        for strategy in self.strategy_cache["strategies"]:
            if strategy["id"] == strategy_id:
                return {
                    "genome": strategy["genome"],
                    "encoder": "GenomeEncoder",
                    "backtester": "solana_backtester.py",
                    "exported_at": datetime.now().isoformat(),
                    "config": {
                        "sl_pct": strategy["genome"]["params"]["sl_pct"],
                        "tp_pct": strategy["genome"]["params"]["tp_pct"]
                    }
                }
        return None
    
    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            "agent": "StrategyAgent",
            "status": "active",
            "strategies_cached": len(self.strategy_cache["strategies"]),
            "strategies_active": len(self.get_active_strategies()),
            "current_best": self.current_best["id"] if self.current_best else None,
            "config": {
                "population_size": self.config.population_size,
                "generations": self.config.generations,
                "min_pnl": self.config.min_pnl,
                "min_win_rate": self.config.min_win_rate
            },
            "available_indicators": NUM_INDICATORS
        }


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from data.historical_data import HistoricalDataManager
    
    print("=" * 70)
    print("🎯 STRATEGY AGENT DEMO")
    print("=" * 70)
    
    # Initialize agent
    agent = StrategyAgent()
    
    # Show status
    status = agent.get_status()
    print(f"\n📊 Agent Status:")
    print(f"   Strategies cached: {status['strategies_cached']}")
    print(f"   Available indicators: {status['available_indicators']}")
    
    # Load data
    print(f"\n📥 Loading historical data...")
    manager = HistoricalDataManager()
    df = manager.get_historical_data("SOL", timeframe="1h", days=30)
    print(f"   Loaded {len(df)} candles")
    
    if len(df) < 100:
        print("⚠️  Not enough data, using sample data...")
        from backtesting.solana_backtester import generate_sample_data
        df = generate_sample_data(n_candles=2000)
        print(f"   Generated {len(df)} sample candles")
    
    # Optimize strategy
    print(f"\n🧬 Optimizing strategy...")
    result = agent.optimize_strategy(
        df,
        population_size=20,
        generations=10,
        verbose=True
    )
    
    # Backtest best strategy
    print(f"\n🔍 Backtesting best strategy...")
    backtest = agent.backtest_strategy(df, result['best_genome'], verbose=True)
    
    # Show indicators
    print(f"\n📈 Available indicators:")
    indicators = agent.list_indicators()
    print(f"   Total: {indicators['total']}")
    
    # Export strategy
    if result['best_pnl'] > 0:
        print(f"\n📦 Exporting best strategy...")
        exported = agent.export_strategy(result['best_genome'])
        if exported:
            print(f"   Exported successfully")
            print(f"   Genome size: {len(exported['genome'])} rules")
    
    print(f"\n" + "=" * 70)
    print("✅ Strategy Agent Demo Complete")
    print("=" * 70)
