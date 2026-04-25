#!/usr/bin/env python3
"""
Agent Brain - Self-Improving Strategy Discovery System
======================================================
Runs alongside agent_runner.py to continuously:
- Scout best tokens to trade
- Collect historical market data
- Backtest strategies against real data
- Optimize strategies via genetic algorithm
- Learn from past performance
- Deploy winning strategies to live trading

Usage:
    python3 agent_brain.py                # Standard mode
    python3 agent_brain.py --interval 300 # 5-min base cycle
    python3 agent_brain.py --fast         # 2-min cycles (dev)
"""

import os
import sys
import json
import copy
import asyncio
import logging
import signal
import random
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import JupiterClient, SOL, USDC, USDT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "agent_brain.log")
    ]
)
logger = logging.getLogger("agent_brain")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
WATCHLIST_FILE = DATA_DIR / "token_watchlist.json"
ACTIVE_STRATEGIES_FILE = DATA_DIR / "active_strategies.json"
OPTIMIZED_FILE = DATA_DIR / "optimized_strategies.json"
ACCUMULATION_FILE = DATA_DIR / "accumulation_target.json"
BRAIN_STATE_FILE = PROJECT_ROOT / "brain_state.json"
DB_PATH = str(DATA_DIR / "genetic_results.db")

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_DIR.mkdir(exist_ok=True)

# ============================================================================
# PROFIT TARGETS - Brain optimizes strategies to hit these
# ============================================================================
PROFIT_TARGETS = {
    "daily_min_pct": 5.0,        # 5% daily minimum
    "weekly_target_pct": 100.0,  # Double every week (ideal)
    "monthly_min_pct": 100.0,    # Double every month (minimum)
    "min_backtest_pnl": 0.05,    # 5% PnL in backtest to deploy
    "min_trades_backtest": 5,    # Enough trades to be statistically meaningful
    "min_win_rate": 0.45,        # 45% win rate with good R:R
}


# ============================================================================
# TOKEN SCOUT AGENT
# ============================================================================

class TokenScoutAgent:
    """Scans and ranks tokens for trading opportunities."""

    # Core Solana ecosystem tokens (always monitored)
    # Addresses verified against Jupiter Search + Price V3 API
    CORE_TOKENS = {
        "SOL":   "So11111111111111111111111111111111111111112",
        "ETH":   "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
        "cbBTC": "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij",
        "JUP":   "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "BONK":  "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "JLP":   "27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4",
        "RAY":   "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "JTO":   "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
        "WIF":   "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    }

    # Stablecoins to exclude from trading candidates
    STABLECOINS = {"USDC", "USDT", "USD1", "PYUSD", "DAI", "BUSD", "jlUSDC", "CASH"}

    def __init__(self, jupiter: JupiterClient):
        self.jupiter = jupiter
        self.last_watchlist: List[Dict] = []

    async def scout(self) -> List[Dict]:
        """Scan trending + core + search tokens, filter, and rank."""
        logger.info("[Scout] Scanning for tokens...")
        seen_symbols = set()
        candidates = []

        def _add(symbol, address, name, source):
            if symbol in seen_symbols or symbol in self.STABLECOINS:
                return
            seen_symbols.add(symbol)
            candidates.append({
                "symbol": symbol, "address": address,
                "name": name, "source": source,
            })

        # 1. Core ecosystem tokens (always included)
        for symbol, address in self.CORE_TOKENS.items():
            _add(symbol, address, symbol, "core")

        # 2. Trending tokens from Jupiter (valid intervals: 1h, 6h, 24h)
        for interval in ["1h", "6h", "24h"]:
            try:
                trending = await self.jupiter.get_trending_tokens(interval)
                if trending:
                    for t in trending[:20]:
                        addr = t.get("id", t.get("address", t.get("mint", "")))
                        symbol = t.get("symbol", "?")
                        name = t.get("name", symbol)
                        if addr:
                            _add(symbol, addr, name, f"trending_{interval}")
            except Exception as e:
                logger.warning(f"[Scout] Trending {interval} error: {e}")

        # 3. Search for major crypto wrapped on Solana
        for query in ["BTC", "ETH", "MATIC", "AVAX", "LINK"]:
            try:
                results = await self.jupiter.search_tokens(query)
                if results:
                    # Take first verified-looking result
                    for t in results[:2]:
                        addr = t.get("id", t.get("address", ""))
                        symbol = t.get("symbol", "?")
                        name = t.get("name", symbol)
                        if addr and len(symbol) <= 10:
                            _add(symbol, addr, name, "search")
            except Exception:
                pass

        # 4. Get prices for ALL candidates
        ranked = []
        # Process in batches of 20 (API limit)
        for i in range(0, len(candidates), 20):
            batch = candidates[i:i+20]
            mints = [c["address"] for c in batch]
            try:
                price_resp = await self.jupiter.get_price(mints)
                for c in batch:
                    price_data = price_resp.get(c["address"], {})
                    if isinstance(price_data, dict):
                        # Jupiter Price V3 uses "usdPrice"
                        price = float(price_data.get("usdPrice", price_data.get("price", 0)))
                    else:
                        price = 0
                    if price > 0:
                        c["price"] = price
                        c["price_change_24h"] = price_data.get("priceChange24h", 0) if isinstance(price_data, dict) else 0
                        c["timestamp"] = datetime.now().isoformat()
                        ranked.append(c)
            except Exception as e:
                logger.warning(f"[Scout] Price batch error: {e}")

        # Sort: core first, then trending_1h, then others
        priority = {"core": 0, "trending_1h": 1, "trending_6h": 2, "trending_24h": 3, "search": 4}
        ranked.sort(key=lambda x: (priority.get(x.get("source", ""), 9), -x.get("price", 0)))

        # Keep top 25
        ranked = ranked[:25]
        self.last_watchlist = ranked

        # Save watchlist
        watchlist_data = {
            "updated": datetime.now().isoformat(),
            "tokens": ranked,
            "count": len(ranked),
        }
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist_data, f, indent=2)

        logger.info(f"[Scout] Found {len(ranked)} tokens: {[t['symbol'] for t in ranked[:15]]}")
        return ranked


# ============================================================================
# DATA COLLECTOR AGENT
# ============================================================================

class DataCollectorAgent:
    """Downloads and maintains historical OHLCV data."""

    def __init__(self):
        self.data_dir = DATA_DIR

    async def collect(self, watchlist: List[Dict]) -> Dict:
        """Collect data for tokens in watchlist."""
        logger.info(f"[DataCollector] Collecting data for {len(watchlist)} tokens...")
        summary = {"collected": [], "errors": [], "total_candles": 0}

        # Import here to avoid circular
        sys.path.insert(0, str(PROJECT_ROOT))
        from data.historical_data import HistoricalDataManager
        manager = HistoricalDataManager()

        tokens_to_fetch = [t["symbol"] for t in watchlist if t["symbol"] in manager.TOKENS]

        for token in tokens_to_fetch:
            try:
                df = manager.get_historical_data(token, timeframe="1h", days=30)
                candles = len(df) if df is not None else 0
                summary["collected"].append({
                    "token": token,
                    "candles": candles,
                    "timeframe": "1h",
                })
                summary["total_candles"] += candles
                logger.info(f"[DataCollector] {token}: {candles} candles (1h)")
            except Exception as e:
                summary["errors"].append({"token": token, "error": str(e)})
                logger.warning(f"[DataCollector] {token} error: {e}")

        logger.info(f"[DataCollector] Total: {summary['total_candles']} candles, "
                    f"{len(summary['errors'])} errors")
        return summary


# ============================================================================
# BACKTEST AGENT
# ============================================================================

class BacktestAgent:
    """Runs backtests using Numba JIT accelerated engine."""

    def __init__(self):
        self.results_cache: List[Dict] = []

    async def backtest(self, strategies: List[Dict], tokens: List[str] = None) -> List[Dict]:
        """Backtest strategies against available data."""
        logger.info(f"[Backtest] Testing {len(strategies)} strategies...")

        from data.historical_data import HistoricalDataManager
        from backtesting.solana_backtester import (
            precompute_indicators, evaluate_genome_python,
            generate_sample_data,
        )

        manager = HistoricalDataManager()
        results = []

        # Default to SOL if no tokens specified
        test_tokens = tokens or ["SOL"]

        for token in test_tokens:
            try:
                df = manager.get_historical_data(token, timeframe="1h", days=30)
                if df is None or len(df) < 50:
                    df = generate_sample_data(n_candles=720)
                    logger.info(f"[Backtest] Using sample data for {token}")

                indicators = precompute_indicators(df)

                for strat in strategies:
                    genome = self._strategy_to_genome(strat)
                    result = evaluate_genome_python(indicators, genome, initial_balance=1.0)

                    entry = {
                        "strategy_name": strat.get("name", "Unknown"),
                        "token": token,
                        "pnl": result["pnl"],
                        "trades": result["trades"],
                        "wins": result["wins"],
                        "losses": result["losses"],
                        "win_rate": result["win_rate"],
                        "candles_tested": len(df),
                        "timestamp": datetime.now().isoformat(),
                    }
                    results.append(entry)
                    logger.info(f"[Backtest] {strat.get('name','?')} on {token}: "
                               f"PnL={result['pnl']:.4f} WR={result['win_rate']:.1%} "
                               f"Trades={result['trades']}")

            except Exception as e:
                logger.warning(f"[Backtest] {token} error: {e}")

        self.results_cache = results
        return results

    def _strategy_to_genome(self, strategy: Dict) -> np.ndarray:
        """Convert strategy dict to genome array for backtester."""
        genome = np.zeros(GENOME_SIZE_LOCAL, dtype=np.float64)

        # Extract params from strategy (indices: 0=sl_pct, 1=tp_pct, 2=num_rules)
        params = strategy.get("params", {})
        genome[0] = params.get("sl_pct", 0.03)
        genome[1] = params.get("tp_pct", 0.06)
        genome[2] = params.get("num_rules", 1)

        # Entry rule encoding
        rules = strategy.get("entry_rules", [])
        if rules:
            rule = rules[0]
            ind = rule.get("indicator", "RSI")
            # Map indicator to index (RSI=4, SMA=10, EMA=16)
            ind_map = {"RSI": 4, "SMA": 10, "EMA": 16, "BB": 4}
            period_map = {10: 0, 14: 1, 20: 2, 50: 3, 100: 4, 200: 5}

            period = rule.get("period", 14)
            base_ind = ind_map.get(ind, 4)
            period_offset = period_map.get(period, 1)

            genome[3] = base_ind + period_offset  # indicator index
            genome[4] = rule.get("threshold", 30)
            genome[5] = 0 if rule.get("operator", "<") == ">" else 1

        return genome


# ============================================================================
# OPTIMIZER AGENT
# ============================================================================

GENOME_SIZE_LOCAL = 18  # Local constant

class OptimizerAgent:
    """Evolves strategies using genetic algorithm."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.population: List[Dict] = []
        self.generation = 0
        self.best_ever: Optional[Dict] = None

        # GA parameters
        self.pop_size = 20
        self.elite_count = 5
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8

        # Building blocks - wider ranges for aggressive trading
        self.indicators = ["RSI", "SMA", "EMA"]
        self.periods = [10, 14, 20, 50, 100]  # Must match backtester RSI/SMA/EMA_PERIODS
        self.rsi_thresholds = [15, 20, 25, 30, 35, 40, 60, 65, 70, 75, 80, 85]  # Wider range

    async def optimize(self, backtest_results: List[Dict]) -> Dict:
        """Run one generation of genetic optimization."""
        logger.info(f"[Optimizer] Generation {self.generation}...")

        from data.historical_data import HistoricalDataManager
        from backtesting.solana_backtester import (
            precompute_indicators, evaluate_genome_python,
            generate_sample_data,
        )

        # Get test data
        manager = HistoricalDataManager()
        try:
            df = manager.get_historical_data("SOL", timeframe="1h", days=30)
            if df is None or len(df) < 50:
                raise ValueError("Insufficient data")
        except Exception:
            df = generate_sample_data(n_candles=720)

        indicators = precompute_indicators(df)

        # Initialize population if empty
        if not self.population:
            self.population = [self._random_strategy() for _ in range(self.pop_size)]

        # Evaluate all - fitness rewards high returns AND trade frequency
        # Target: strategies that can deliver 5%+ daily
        evaluated = []
        for strat in self.population:
            genome = self._to_genome(strat)
            result = evaluate_genome_python(indicators, genome, initial_balance=1.0)

            # Composite fitness: PnL * frequency bonus * win rate bonus
            pnl = result["pnl"]
            trades = result["trades"]
            win_rate = result["win_rate"]

            # Bonus for strategies that trade frequently (need many trades for 5%/day)
            freq_bonus = min(trades / 10.0, 2.0) if trades > 0 else 0.1
            # Bonus for decent win rate (above 45%)
            wr_bonus = 1.0 + max(0, (win_rate - 0.45)) * 2.0
            # Penalty for strategies that don't trade
            if trades < PROFIT_TARGETS["min_trades_backtest"]:
                freq_bonus *= 0.3

            fitness = pnl * freq_bonus * wr_bonus

            strat["fitness"] = fitness
            strat["backtest"] = {
                "pnl": result["pnl"],
                "trades": result["trades"],
                "wins": result["wins"],
                "win_rate": result["win_rate"],
                "raw_pnl": pnl,
                "freq_bonus": round(freq_bonus, 2),
                "wr_bonus": round(wr_bonus, 2),
            }
            evaluated.append(strat)

        # Sort by fitness
        evaluated.sort(key=lambda x: x.get("fitness", -999), reverse=True)

        # Track best
        if evaluated:
            best = evaluated[0]
            if self.best_ever is None or best.get("fitness", -999) > self.best_ever.get("fitness", -999):
                self.best_ever = best.copy()
                logger.info(f"[Optimizer] New best! PnL={best['fitness']:.4f} "
                           f"WR={best['backtest']['win_rate']:.1%}")

        # Elite selection
        elite = evaluated[:self.elite_count]

        # Create next generation (deep copy elite to prevent mutation corruption)
        new_pop = [copy.deepcopy(s) for s in elite]

        while len(new_pop) < self.pop_size:
            if random.random() < self.crossover_rate and len(elite) >= 2:
                p1, p2 = random.sample(elite, 2)
                child = self._crossover(p1, p2)
            else:
                child = self._random_strategy()

            child = self._mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1

        # Save to DB
        self._save_generation(evaluated)

        # Export top 3 as optimized
        top3 = evaluated[:3]
        optimized = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "strategies": top3,
            "best_pnl": top3[0].get("fitness", 0) if top3 else 0,
        }
        with open(OPTIMIZED_FILE, "w") as f:
            json.dump(optimized, f, indent=2, default=str)

        logger.info(f"[Optimizer] Gen {self.generation}: Best PnL={top3[0].get('fitness', 0):.4f}, "
                    f"Pop={len(self.population)}")

        return optimized

    def _random_strategy(self) -> Dict:
        """Generate random strategy."""
        ind = random.choice(self.indicators)
        period = random.choice(self.periods)

        if ind == "RSI":
            threshold = random.choice(self.rsi_thresholds)
            operator = "<" if threshold < 50 else ">"
        else:
            # SMA/EMA: threshold = % deviation from indicator
            # "<" means buy when price is below SMA (buy the dip)
            # ">" means buy when price is above SMA (trend follow)
            threshold = round(random.uniform(0.5, 3.0), 1)
            operator = random.choice(["<", ">"])

        return {
            "name": f"{ind}_{period}_{operator}{threshold}",
            "entry_rules": [{
                "indicator": ind,
                "period": period,
                "operator": operator,
                "threshold": threshold,
            }],
            "params": {
                "sl_pct": round(random.uniform(0.01, 0.05), 3),  # Tight stops (1-5%)
                "tp_pct": round(random.uniform(0.02, 0.15), 3),  # Wide targets (2-15%)
                "num_rules": 1,
            },
            "fitness": 0,
        }

    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        """Crossover two strategies."""
        source = p1 if random.random() < 0.5 else p2
        child = {
            "entry_rules": copy.deepcopy(source["entry_rules"]),
            "params": {},
        }
        # Mix params
        for key in ["sl_pct", "tp_pct"]:
            v1 = p1.get("params", {}).get(key, 0.05)
            v2 = p2.get("params", {}).get(key, 0.05)
            child["params"][key] = round((v1 + v2) / 2, 4)
        child["params"]["num_rules"] = 1
        child["name"] = f"cross_gen{self.generation}"
        return child

    def _mutate(self, strat: Dict) -> Dict:
        """Mutate a strategy (deep copies to avoid corrupting source)."""
        if random.random() > self.mutation_rate:
            return strat
        strat = copy.deepcopy(strat)

        params = strat.get("params", {})
        if random.random() < 0.5:
            # Tight stops: 0.5% - 5% (aggressive risk management)
            params["sl_pct"] = round(min(0.05, max(0.005, params.get("sl_pct", 0.02) + random.uniform(-0.01, 0.01))), 4)
        else:
            # Wide targets: 1% - 20% (let winners run for 5%+ daily)
            params["tp_pct"] = round(min(0.20, max(0.01, params.get("tp_pct", 0.05) + random.uniform(-0.03, 0.03))), 4)

        # Sometimes mutate indicator
        if random.random() < 0.3 and strat.get("entry_rules"):
            rule = strat["entry_rules"][0]
            rule["period"] = random.choice(self.periods)
            if rule.get("indicator") == "RSI":
                rule["threshold"] = random.choice(self.rsi_thresholds)

        strat["name"] = f"mut_gen{self.generation}"
        return strat

    def _to_genome(self, strat: Dict) -> np.ndarray:
        """Convert strategy to genome array."""
        genome = np.zeros(GENOME_SIZE_LOCAL, dtype=np.float64)
        params = strat.get("params", {})
        genome[0] = params.get("sl_pct", 0.03)
        genome[1] = params.get("tp_pct", 0.06)
        genome[2] = 1

        rules = strat.get("entry_rules", [])
        if rules:
            rule = rules[0]
            ind_map = {"RSI": 4, "SMA": 10, "EMA": 16}
            period_map = {10: 0, 14: 1, 20: 2, 50: 3, 100: 4, 200: 5}
            genome[3] = ind_map.get(rule.get("indicator", "RSI"), 4) + period_map.get(rule.get("period", 14), 1)
            genome[4] = rule.get("threshold", 30)
            genome[5] = 0 if rule.get("operator", "<") == ">" else 1

        return genome

    def _save_generation(self, evaluated: List[Dict]):
        """Save generation results to SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("""CREATE TABLE IF NOT EXISTS brain_generations (
                    id INTEGER PRIMARY KEY,
                    generation INTEGER,
                    timestamp TEXT,
                    best_pnl REAL,
                    best_strategy TEXT,
                    population_size INTEGER,
                    avg_fitness REAL
                )""")

                best = evaluated[0] if evaluated else {}
                avg_fit = sum(s.get("fitness", 0) for s in evaluated) / max(len(evaluated), 1)

                c.execute("INSERT INTO brain_generations VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                          (self.generation, datetime.now().isoformat(),
                           best.get("fitness", 0), json.dumps(best, default=str),
                           len(evaluated), avg_fit))
                conn.commit()
        except Exception as e:
            logger.warning(f"[Optimizer] DB save error: {e}")


# ============================================================================
# LEARNING AGENT
# ============================================================================

class LearningAgent:
    """Analyzes performance and builds knowledge base."""

    def __init__(self):
        self.knowledge: Dict = self._load_knowledge()

    def _load_knowledge(self) -> Dict:
        """Load existing knowledge."""
        perf_file = KNOWLEDGE_DIR / "strategy_performance.json"
        if perf_file.exists():
            try:
                return json.loads(perf_file.read_text())
            except Exception:
                pass
        return {
            "strategies": {},
            "market_patterns": [],
            "lessons": [],
            "last_updated": None,
        }

    async def learn(self, backtest_results: List[Dict], optimizer_results: Dict) -> Dict:
        """Analyze results and update knowledge base."""
        logger.info("[Learning] Analyzing performance...")

        # Track strategy performance over time
        for result in backtest_results:
            name = result.get("strategy_name", "unknown")
            if name not in self.knowledge["strategies"]:
                self.knowledge["strategies"][name] = {
                    "total_tests": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "best_pnl": float("-inf"),
                    "worst_pnl": float("inf"),
                    "avg_win_rate": 0,
                    "history": [],
                }

            stats = self.knowledge["strategies"][name]
            stats["total_tests"] += 1
            pnl = result.get("pnl", 0)
            stats["total_pnl"] += pnl
            stats["avg_pnl"] = stats["total_pnl"] / stats["total_tests"]
            stats["best_pnl"] = max(stats["best_pnl"], pnl)
            stats["worst_pnl"] = min(stats["worst_pnl"], pnl)

            # Rolling average win rate
            wr = result.get("win_rate", 0)
            n = stats["total_tests"]
            stats["avg_win_rate"] = ((stats["avg_win_rate"] * (n - 1)) + wr) / n

            stats["history"].append({
                "pnl": pnl,
                "win_rate": wr,
                "trades": result.get("trades", 0),
                "token": result.get("token", "?"),
                "timestamp": result.get("timestamp", ""),
            })
            # Keep last 50 history entries
            stats["history"] = stats["history"][-50:]

        # Analyze optimizer progress
        if optimizer_results:
            gen = optimizer_results.get("generation", 0)
            best_pnl = optimizer_results.get("best_pnl", 0)
            self.knowledge["market_patterns"].append({
                "generation": gen,
                "best_pnl": best_pnl,
                "timestamp": datetime.now().isoformat(),
            })
            # Keep last 100
            self.knowledge["market_patterns"] = self.knowledge["market_patterns"][-100:]

        # Generate lessons
        self._generate_lessons()

        # Save knowledge
        self.knowledge["last_updated"] = datetime.now().isoformat()
        self._save_knowledge()

        logger.info(f"[Learning] Tracked {len(self.knowledge['strategies'])} strategies, "
                    f"{len(self.knowledge['lessons'])} lessons")
        return self.knowledge

    def _generate_lessons(self):
        """Extract lessons from accumulated data.

        Evaluates against PROFIT_TARGETS: 5%+ daily, 2x monthly.
        """
        lessons = []
        target_daily = PROFIT_TARGETS["daily_min_pct"] / 100.0  # 0.05

        for name, stats in self.knowledge["strategies"].items():
            if stats["total_tests"] >= 3:
                if stats["avg_pnl"] >= target_daily:
                    lessons.append(f"STRONG: '{name}' hits daily target "
                                   f"(avg PnL: {stats['avg_pnl']:.4f} >= {target_daily:.2f})")
                elif stats["avg_pnl"] > 0.01:
                    lessons.append(f"OK: '{name}' profitable but below 5% target "
                                   f"(avg PnL: {stats['avg_pnl']:.4f})")
                elif stats["avg_pnl"] < -0.01:
                    lessons.append(f"DROP: '{name}' losing money "
                                   f"(avg PnL: {stats['avg_pnl']:.4f}) - remove")

                if stats["avg_win_rate"] > 0.6:
                    lessons.append(f"HIGH WR: '{name}' ({stats['avg_win_rate']:.0%})")

        # Check optimizer progress
        patterns = self.knowledge.get("market_patterns", [])
        if len(patterns) >= 3:
            recent = [p["best_pnl"] for p in patterns[-5:]]
            if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                lessons.append("Optimizer improving - evolving toward 5%+ daily target")
            elif len(recent) >= 3 and recent[-1] < recent[-3]:
                lessons.append("Optimizer stalled - may need population reset for 5%+ target")

        self.knowledge["lessons"] = lessons[-20:]

    def _save_knowledge(self):
        """Save knowledge to files."""
        # JSON data
        perf_file = KNOWLEDGE_DIR / "strategy_performance.json"
        with open(perf_file, "w") as f:
            json.dump(self.knowledge, f, indent=2, default=str)

        # Human-readable lessons
        lessons_file = KNOWLEDGE_DIR / "lessons_learned.md"
        lines = [
            f"# Trading Lessons Learned",
            f"*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            f"## Strategy Performance\n",
        ]

        for name, stats in self.knowledge["strategies"].items():
            emoji = "+" if stats["avg_pnl"] > 0 else "-"
            lines.append(f"- **{name}**: PnL={stats['avg_pnl']:.4f} "
                        f"WR={stats['avg_win_rate']:.0%} "
                        f"Tests={stats['total_tests']}")

        lines.append(f"\n## Lessons\n")
        for lesson in self.knowledge.get("lessons", []):
            lines.append(f"- {lesson}")

        lessons_file.write_text("\n".join(lines))


# ============================================================================
# DEPLOYMENT AGENT
# ============================================================================

class DeploymentAgent:
    """Deploys winning strategies to live trading."""

    def __init__(self):
        self.deployed: List[Dict] = []
        self.deploy_history: List[Dict] = []

    async def deploy(self, optimized: Dict, knowledge: Dict) -> Dict:
        """Evaluate and deploy winning strategies.

        Deployment criteria aligned with PROFIT_TARGETS:
        - PnL must be positive (>5% preferred)
        - Win rate >= 45% (with good R:R ratio)
        - Must generate enough trades (frequency matters for 5%/day target)
        """
        logger.info("[Deploy] Evaluating candidates for deployment...")

        candidates = optimized.get("strategies", [])
        if not candidates:
            logger.info("[Deploy] No candidates available")
            return {"deployed": 0, "reason": "No candidates"}

        approved = []
        for strat in candidates:
            bt = strat.get("backtest", {})
            pnl = bt.get("pnl", bt.get("raw_pnl", strat.get("fitness", -1)))
            win_rate = bt.get("win_rate", 0)
            trades = bt.get("trades", 0)

            # Deployment criteria - aggressive but filtered
            min_pnl = PROFIT_TARGETS["min_backtest_pnl"]
            reasons = []
            if pnl < min_pnl:
                reasons.append(f"PnL too low ({pnl:.4f}, need {min_pnl}+)")
            if win_rate < PROFIT_TARGETS["min_win_rate"]:
                reasons.append(f"Low win rate ({win_rate:.0%})")
            if trades < PROFIT_TARGETS["min_trades_backtest"]:
                reasons.append(f"Too few trades ({trades}, need {PROFIT_TARGETS['min_trades_backtest']}+)")

            if not reasons:
                # Convert to format agent_runner.py understands
                deployed_strat = self._format_for_runner(strat)
                approved.append(deployed_strat)
                logger.info(f"[Deploy] Approved: {strat.get('name', '?')} "
                           f"PnL={pnl:.4f} WR={win_rate:.0%}")
            else:
                logger.info(f"[Deploy] Rejected {strat.get('name', '?')}: {', '.join(reasons)}")

        # Keep max 3 active strategies
        approved = approved[:3]

        if approved:
            # Write for agent_runner.py to consume
            deploy_data = {
                "deployed_at": datetime.now().isoformat(),
                "generation": optimized.get("generation", 0),
                "strategies": approved,
                "count": len(approved),
            }
            with open(ACTIVE_STRATEGIES_FILE, "w") as f:
                json.dump(deploy_data, f, indent=2, default=str)

            self.deployed = approved
            self.deploy_history.append({
                "time": datetime.now().isoformat(),
                "count": len(approved),
                "names": [s.get("name", "?") for s in approved],
            })

            logger.info(f"[Deploy] Deployed {len(approved)} strategies to live trading")
        else:
            logger.info("[Deploy] No strategies passed deployment criteria")

        return {"deployed": len(approved), "strategies": approved}

    def _format_for_runner(self, strat: Dict) -> Dict:
        """Format strategy for agent_runner.py's StrategyAgent."""
        params = strat.get("params", {})
        rules = strat.get("entry_rules", [{}])
        rule = rules[0] if rules else {}

        return {
            "name": strat.get("name", f"Brain_Gen{strat.get('generation', '?')}"),
            "description": f"Auto-evolved: {rule.get('indicator', 'RSI')} "
                          f"{rule.get('operator', '<')}{rule.get('threshold', 30)} "
                          f"P{rule.get('period', 14)} "
                          f"SL={params.get('sl_pct', 0.03)*100:.1f}% "
                          f"TP={params.get('tp_pct', 0.06)*100:.1f}%",
            "buy_threshold": -params.get("sl_pct", 0.03),
            "sell_threshold": params.get("tp_pct", 0.06),
            "lookback": rule.get("period", 14),
            "score": int(strat.get("fitness", 0) * 10000),
            "trades": strat.get("backtest", {}).get("trades", 0),
            "wins": strat.get("backtest", {}).get("wins", 0),
            # Keep original for reference
            "brain_params": params,
            "brain_rules": rules,
        }


# ============================================================================
# ACCUMULATION AGENT
# ============================================================================

class AccumulationAgent:
    """Decides optimal BTC vs SOL accumulation based on real-time market data.

    Analyzes multiple factors to determine which asset to accumulate more:
    - Relative strength (24h performance comparison)
    - Buy-the-dip detection (larger drops = bigger opportunity)
    - Trend momentum (consecutive price direction)
    - Historical volatility ratio
    - Knowledge base insights from past trades

    Outputs allocation percentages (e.g. SOL=65%, BTC=35%) to
    data/accumulation_target.json, which agent_runner.py reads.
    """

    # cbBTC mint on Solana
    BTC_MINT = "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij"
    SOL_MINT = "So11111111111111111111111111111111111111112"

    def __init__(self, jupiter: JupiterClient):
        self.jupiter = jupiter
        self.history: List[Dict] = []
        self.price_snapshots: List[Dict] = []  # Rolling price window
        self.current_target: Dict = {"SOL": 0.60, "BTC": 0.40}

    async def decide(self, watchlist: List[Dict], knowledge: Dict = None) -> Dict:
        """Analyze BTC vs SOL and decide accumulation allocation."""
        logger.info("[Accumulator] Analyzing BTC vs SOL allocation...")

        # Get data from watchlist first (already fetched by Scout)
        sol_data = next((t for t in watchlist if t["symbol"] == "SOL"), None)
        btc_data = next((t for t in watchlist if t["symbol"] == "cbBTC"), None)

        # Fallback: fetch directly if not in watchlist
        if not sol_data or not btc_data:
            try:
                mints = [self.SOL_MINT, self.BTC_MINT]
                prices = await self.jupiter.get_price(mints)

                if not sol_data:
                    pd_sol = prices.get(self.SOL_MINT, {})
                    sol_data = {
                        "symbol": "SOL",
                        "price": float(pd_sol.get("usdPrice", 0)) if isinstance(pd_sol, dict) else 0,
                        "price_change_24h": pd_sol.get("priceChange24h", 0) if isinstance(pd_sol, dict) else 0,
                    }
                if not btc_data:
                    pd_btc = prices.get(self.BTC_MINT, {})
                    btc_data = {
                        "symbol": "cbBTC",
                        "price": float(pd_btc.get("usdPrice", 0)) if isinstance(pd_btc, dict) else 0,
                        "price_change_24h": pd_btc.get("priceChange24h", 0) if isinstance(pd_btc, dict) else 0,
                    }
            except Exception as e:
                logger.warning(f"[Accumulator] Price fetch error: {e}")
                return self.current_target

        sol_price = sol_data.get("price", 0)
        btc_price = btc_data.get("price", 0)
        sol_change = sol_data.get("price_change_24h", 0) or 0
        btc_change = btc_data.get("price_change_24h", 0) or 0

        if sol_price <= 0 or btc_price <= 0:
            logger.warning("[Accumulator] Missing price data, keeping current target")
            return self.current_target

        # Track price snapshots for momentum analysis
        self.price_snapshots.append({
            "time": datetime.now().isoformat(),
            "sol": sol_price,
            "btc": btc_price,
        })
        self.price_snapshots = self.price_snapshots[-60:]  # Keep ~5h of 5-min snapshots

        # === SCORING SYSTEM ===
        sol_score = 0.0
        btc_score = 0.0
        reasons = []

        # Factor 1: Buy-the-dip (contrarian) - favor whichever dropped more
        # Bigger drops = bigger accumulation opportunity
        if sol_change < btc_change:
            dip_bonus = min(abs(sol_change - btc_change) * 0.5, 3.0)
            sol_score += 2.0 + dip_bonus
            reasons.append(f"SOL dipped more ({sol_change:+.1f}% vs {btc_change:+.1f}%)")
        elif btc_change < sol_change:
            dip_bonus = min(abs(btc_change - sol_change) * 0.5, 3.0)
            btc_score += 2.0 + dip_bonus
            reasons.append(f"BTC dipped more ({btc_change:+.1f}% vs {sol_change:+.1f}%)")

        # Factor 2: Absolute drop detection (flash crash = opportunity)
        if sol_change < -5:
            sol_score += 4.0
            reasons.append(f"SOL flash crash ({sol_change:+.1f}%) - strong accumulate")
        elif sol_change < -3:
            sol_score += 2.0
            reasons.append(f"SOL significant drop ({sol_change:+.1f}%)")
        elif sol_change < -1:
            sol_score += 0.5

        if btc_change < -5:
            btc_score += 4.0
            reasons.append(f"BTC flash crash ({btc_change:+.1f}%) - strong accumulate")
        elif btc_change < -3:
            btc_score += 2.0
            reasons.append(f"BTC significant drop ({btc_change:+.1f}%)")
        elif btc_change < -1:
            btc_score += 0.5

        # Factor 3: Momentum (if enough snapshots, check short-term trend)
        if len(self.price_snapshots) >= 6:
            recent = self.price_snapshots[-6:]
            sol_mom = (recent[-1]["sol"] - recent[0]["sol"]) / recent[0]["sol"]
            btc_mom = (recent[-1]["btc"] - recent[0]["btc"]) / recent[0]["btc"]

            # Favor the one with negative momentum (buy low)
            if sol_mom < -0.01 and sol_mom < btc_mom:
                sol_score += 1.5
                reasons.append(f"SOL short-term downtrend ({sol_mom:+.2%})")
            elif btc_mom < -0.01 and btc_mom < sol_mom:
                btc_score += 1.5
                reasons.append(f"BTC short-term downtrend ({btc_mom:+.2%})")

            # But also reward strong uptrend (ride the wave for the other)
            if sol_mom > 0.03:
                sol_score += 1.0
                reasons.append(f"SOL strong momentum ({sol_mom:+.2%})")
            if btc_mom > 0.03:
                btc_score += 1.0
                reasons.append(f"BTC strong momentum ({btc_mom:+.2%})")

        # Factor 4: BTC is digital gold - slight safety bias
        btc_score += 1.0
        reasons.append("BTC safety premium (+1)")

        # Factor 5: SOL ecosystem activity (if many trending tokens, SOL ecosystem is hot)
        sol_ecosystem_tokens = sum(1 for t in watchlist if t.get("source", "").startswith("trending"))
        if sol_ecosystem_tokens >= 10:
            sol_score += 1.5
            reasons.append(f"Hot SOL ecosystem ({sol_ecosystem_tokens} trending tokens)")
        elif sol_ecosystem_tokens >= 5:
            sol_score += 0.5

        # Factor 6: Learn from knowledge base
        if knowledge and knowledge.get("strategies"):
            # If strategies are profitable, SOL ecosystem is working well
            profitable = sum(1 for s in knowledge["strategies"].values()
                           if s.get("avg_pnl", 0) > 0)
            if profitable >= 3:
                sol_score += 1.0
                reasons.append(f"SOL strategies profitable ({profitable} winning)")

        # === CALCULATE ALLOCATION ===
        total = sol_score + btc_score
        if total == 0:
            sol_pct, btc_pct = 0.50, 0.50
        else:
            sol_pct = sol_score / total
            btc_pct = btc_score / total

        # Enforce bounds: minimum 20% each, maximum 80%
        sol_pct = max(0.20, min(0.80, sol_pct))
        btc_pct = 1.0 - sol_pct

        recommendation = "SOL" if sol_pct > btc_pct else "BTC"
        confidence = abs(sol_pct - btc_pct) / 0.60  # 0-1 scale (0.60 max spread)

        target = {
            "SOL": round(sol_pct, 2),
            "BTC": round(btc_pct, 2),
            "recommendation": recommendation,
            "confidence": round(min(confidence, 1.0), 2),
            "reasoning": {
                "sol_price": round(sol_price, 2),
                "btc_price": round(btc_price, 2),
                "sol_24h_change": round(sol_change, 2),
                "btc_24h_change": round(btc_change, 2),
                "sol_score": round(sol_score, 1),
                "btc_score": round(btc_score, 1),
                "factors": reasons,
            },
            "updated": datetime.now().isoformat(),
        }

        self.current_target = target
        self.history.append({
            "time": datetime.now().isoformat(),
            "sol_pct": sol_pct,
            "btc_pct": btc_pct,
            "rec": recommendation,
        })
        self.history = self.history[-100:]

        # Save for runner to consume
        with open(ACCUMULATION_FILE, "w") as f:
            json.dump(target, f, indent=2)

        logger.info(f"[Accumulator] Target: SOL={sol_pct:.0%} BTC={btc_pct:.0%} "
                    f"| Recommend: {recommendation} "
                    f"| Conf: {confidence:.0%} "
                    f"| {'; '.join(reasons[:3])}")

        return target


# ============================================================================
# BRAIN COORDINATOR
# ============================================================================

class BrainCoordinator:
    """Orchestrates all brain agents in continuous loop."""

    def __init__(self, interval: int = 300):
        self.interval = interval  # Base cycle in seconds (5 min)
        self.running = False
        self.cycle = 0
        self.start_time = None

        self.jupiter = JupiterClient()
        self.scout = TokenScoutAgent(self.jupiter)
        self.collector = DataCollectorAgent()
        self.backtester = BacktestAgent()
        self.optimizer = OptimizerAgent()
        self.learner = LearningAgent()
        self.deployer = DeploymentAgent()
        self.accumulator = AccumulationAgent(self.jupiter)

        self.last_results: Dict = {}

    async def run_cycle(self) -> Dict:
        """Run one brain cycle."""
        self.cycle += 1
        cycle_result = {
            "cycle": self.cycle,
            "time": datetime.now().isoformat(),
            "agents": {},
        }

        # 1. Scout (every cycle)
        try:
            watchlist = await self.scout.scout()
            cycle_result["agents"]["scout"] = {
                "status": "ok",
                "tokens": len(watchlist),
                "names": [t["symbol"] for t in watchlist[:5]],
            }
        except Exception as e:
            logger.error(f"[Brain] Scout error: {e}")
            watchlist = self.scout.last_watchlist or []
            cycle_result["agents"]["scout"] = {"status": "error", "error": str(e)}

        # 2. Accumulation decision (every cycle - needs fresh price data)
        try:
            accum_target = await self.accumulator.decide(
                watchlist,
                knowledge=self.learner.knowledge if self.learner.knowledge else None,
            )
            cycle_result["agents"]["accumulator"] = {
                "status": "ok",
                "recommendation": accum_target.get("recommendation", "?"),
                "sol_pct": accum_target.get("SOL", 0.5),
                "btc_pct": accum_target.get("BTC", 0.5),
                "confidence": accum_target.get("confidence", 0),
            }
        except Exception as e:
            logger.error(f"[Brain] Accumulator error: {e}")
            cycle_result["agents"]["accumulator"] = {"status": "error", "error": str(e)}

        # 3. Data collection (every 3 cycles = ~15 min)
        if self.cycle % 3 == 1 or self.cycle == 1:
            try:
                data_summary = await self.collector.collect(watchlist)
                cycle_result["agents"]["data_collector"] = {
                    "status": "ok",
                    "candles": data_summary.get("total_candles", 0),
                }
            except Exception as e:
                logger.error(f"[Brain] DataCollector error: {e}")
                cycle_result["agents"]["data_collector"] = {"status": "error", "error": str(e)}

        # 3. Backtest (every 2 cycles = ~10 min)
        backtest_results = []
        if self.cycle % 2 == 0 or self.cycle == 1:
            try:
                # Test current strategies
                strategies = self.optimizer.population[:5] if self.optimizer.population else [
                    self.optimizer._random_strategy() for _ in range(3)
                ]
                backtest_results = await self.backtester.backtest(strategies)
                cycle_result["agents"]["backtester"] = {
                    "status": "ok",
                    "tested": len(backtest_results),
                    "best_pnl": max((r.get("pnl", 0) for r in backtest_results), default=0),
                }
            except Exception as e:
                logger.error(f"[Brain] Backtest error: {e}")
                cycle_result["agents"]["backtester"] = {"status": "error", "error": str(e)}

        # 4. Optimizer (every 6 cycles = ~30 min)
        optimizer_results = {}
        if self.cycle % 6 == 0 or self.cycle == 1:
            try:
                optimizer_results = await self.optimizer.optimize(backtest_results)
                cycle_result["agents"]["optimizer"] = {
                    "status": "ok",
                    "generation": optimizer_results.get("generation", 0),
                    "best_pnl": optimizer_results.get("best_pnl", 0),
                }
            except Exception as e:
                logger.error(f"[Brain] Optimizer error: {e}")
                cycle_result["agents"]["optimizer"] = {"status": "error", "error": str(e)}

        # 5. Learning (every 3 cycles = ~15 min)
        knowledge = {}
        if self.cycle % 3 == 0 or self.cycle == 1:
            try:
                knowledge = await self.learner.learn(backtest_results, optimizer_results)
                cycle_result["agents"]["learner"] = {
                    "status": "ok",
                    "strategies_tracked": len(knowledge.get("strategies", {})),
                    "lessons": len(knowledge.get("lessons", [])),
                }
            except Exception as e:
                logger.error(f"[Brain] Learning error: {e}")
                cycle_result["agents"]["learner"] = {"status": "error", "error": str(e)}

        # 6. Deployment (every 2 cycles = ~10 min, but only if optimizer ran)
        if (self.cycle % 2 == 0 or self.cycle == 1) and OPTIMIZED_FILE.exists():
            try:
                opt_data = json.loads(OPTIMIZED_FILE.read_text())
                deploy_result = await self.deployer.deploy(opt_data, knowledge)
                cycle_result["agents"]["deployer"] = {
                    "status": "ok",
                    "deployed": deploy_result.get("deployed", 0),
                }
            except Exception as e:
                logger.error(f"[Brain] Deploy error: {e}")
                cycle_result["agents"]["deployer"] = {"status": "error", "error": str(e)}

        self.last_results = cycle_result
        self._write_state()
        return cycle_result

    async def run(self):
        """Main continuous loop."""
        self.running = True
        self.start_time = datetime.now()

        print("=" * 70)
        print("  AGENT BRAIN - AGGRESSIVE STRATEGY DISCOVERY")
        print("=" * 70)
        print(f"  Base Interval: {self.interval}s")
        print(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"  PROFIT TARGETS:")
        print(f"    Daily min:    {PROFIT_TARGETS['daily_min_pct']}%")
        print(f"    Weekly goal:  2x (100%)")
        print(f"    Monthly min:  2x (100%)")
        print(f"    Deploy if:    PnL>{PROFIT_TARGETS['min_backtest_pnl']:.0%} "
              f"WR>{PROFIT_TARGETS['min_win_rate']:.0%} "
              f"Trades>{PROFIT_TARGETS['min_trades_backtest']}")
        print()
        print("  Agents:")
        print("    TokenScout     - Token scanning (every cycle)")
        print("    Accumulator    - BTC vs SOL allocation (every cycle)")
        print("    DataCollector  - Historical data (every 3 cycles)")
        print("    Backtester     - Strategy testing (every 2 cycles)")
        print("    Optimizer      - GA evolution (every 6 cycles)")
        print("    Learner        - Knowledge building (every 3 cycles)")
        print("    Deployer       - Strategy deployment (every 2 cycles)")
        print("=" * 70)

        while self.running:
            try:
                result = await self.run_cycle()
                active = [k for k, v in result.get("agents", {}).items() if v.get("status") == "ok"]
                best_str = f"{self.optimizer.best_ever.get('fitness', 0):.4f}" if self.optimizer.best_ever else "N/A"
                print(f"\n--- Brain Cycle {result['cycle']} | "
                      f"Active: {', '.join(active)} | "
                      f"Gen: {self.optimizer.generation} | "
                      f"Best: {best_str} ---")

            except Exception as e:
                logger.error(f"Brain cycle error: {e}")
                print(f"\n  ERROR in brain cycle {self.cycle}: {e}")

            for i in range(self.interval):
                if not self.running:
                    break
                await asyncio.sleep(1)

        await self.jupiter.close()
        print("\nBrain stopped.")

    def _write_state(self):
        """Write brain state for dashboard."""
        state = {
            "running": self.running,
            "cycle": self.cycle,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_update": datetime.now().isoformat(),
            "interval": self.interval,
            "optimizer": {
                "generation": self.optimizer.generation,
                "population_size": len(self.optimizer.population),
                "best_ever": self.optimizer.best_ever.get("fitness", 0) if self.optimizer.best_ever else None,
                "best_strategy": self.optimizer.best_ever.get("name", "None") if self.optimizer.best_ever else "None",
            },
            "scout": {
                "watchlist": [t["symbol"] for t in self.scout.last_watchlist[:5]],
            },
            "deployer": {
                "deployed_count": len(self.deployer.deployed),
                "history": self.deployer.deploy_history[-10:],
            },
            "accumulator": {
                "current_target": self.accumulator.current_target,
                "history": self.accumulator.history[-20:],
            },
            "learner": {
                "strategies_tracked": len(self.learner.knowledge.get("strategies", {})),
                "lessons_count": len(self.learner.knowledge.get("lessons", [])),
                "lessons": self.learner.knowledge.get("lessons", [])[:5],
            },
            "last_cycle": self.last_results,
        }
        with open(BRAIN_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def stop(self):
        self.running = False


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent Brain - Strategy Discovery")
    parser.add_argument("--interval", type=int, default=300, help="Base cycle interval in seconds (default: 300)")
    parser.add_argument("--fast", action="store_true", help="Fast mode (120s cycles)")
    args = parser.parse_args()

    interval = 120 if args.fast else args.interval

    coordinator = BrainCoordinator(interval=interval)

    def signal_handler(sig, frame):
        print("\nStopping brain...")
        coordinator.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(coordinator.run())


if __name__ == "__main__":
    main()
