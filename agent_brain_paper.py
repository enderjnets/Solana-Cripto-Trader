#!/usr/bin/env python3
"""
Agent Brain - Paper Trading Mode with Advanced Auto-Improvement
================================================================
Self-improving strategy discovery system with ML, backtesting, and genetic optimization.

Features:
- ML Signal Generation (RSI + EMA + Momentum + Trend)
- Historical Backtesting (data/*.csv files)
- Performance Tracking (Win Rate, PF, Drawdown per strategy)
- Auto-Optimization (Adjust SL/TP/Risk based on history)
- Genetic Algorithm (Test multiple parameter combinations)

Usage:
    python3 agent_brain_paper.py              # Standard mode
    python3 agent_brain_paper.py --fast       # 2-min cycles (dev)
    python3 agent_brain_paper.py --backtest   # Run backtest only
    python3 agent_brain_paper.py --genetic     # Run genetic optimization
"""

import os
import sys
import json
import asyncio
import traceback
import logging
import random
import sqlite3
import numpy as np
from datetime import datetime, timedelta

class DateTimeEncoder(json.JSONEncoder):
    """Handle datetime serialization."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import csv

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import JupiterClient, SOL, USDC
from paper_trading_engine import PaperTradingEngine
from config.hardbit_schedule import get_active_profile, is_night_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "agent_brain_paper.log")
    ]
)
logger = logging.getLogger("agent_brain_paper")

# Constants
CYCLE_INTERVAL = 60  # seconds
INITIAL_BALANCE = 500  # USD (paper trading capital)

# Get dynamic settings from HARDBIT config
def get_trade_params():
    """Get trade parameters from HARDBIT config"""
    profile = get_active_profile()
    return {
        "risk_pct": profile["max_position_pct"],
        "stop_loss": profile["stop_loss_pct"],
        "take_profit": profile["take_profit_pct"],
        "max_concurrent": profile["max_concurrent_positions"],
        "mode": "NIGHT" if is_night_time() else "DAY"
    }

# Dynamic parameters (updated each cycle)
TRADE_PARAMS = get_trade_params()

# Genetic Algorithm Constants
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.2
ELITE_COUNT = 3


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class StrategyMetrics:
    """Metrics for a trading strategy."""
    name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "last_updated": self.last_updated
        }


@dataclass
class Genome:
    """Individual genome for genetic algorithm."""
    stop_loss: float
    take_profit: float
    risk_pct: float
    rsi_period: int
    ema_fast: int
    ema_slow: int
    momentum_period: int
    score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_pct": self.risk_pct,
            "rsi_period": self.rsi_period,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "momentum_period": self.momentum_period,
            "score": self.score
        }


class TechnicalIndicators:
    """Calculate technical indicators for signal generation."""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 70.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1] if prices else 0

        alpha = 2 / (period + 1)
        ema = prices[-1]
        for price in reversed(prices[:-1]):
            ema = alpha * price + (1 - alpha) * ema
        return ema

    @staticmethod
    def calculate_momentum(prices: List[float], period: int = 10) -> float:
        """Calculate momentum percentage."""
        if len(prices) < period:
            return 100.0

        current = prices[-1]
        past = prices[-period]
        return (current / past) * 100

    @staticmethod
    def calculate_trend(prices: List[float], period: int = 20) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(prices) < period:
            return 0.0

        recent = np.mean(prices[-period//2:])
        older = np.mean(prices[:period//2])
        return (recent - older) / older if older > 0 else 0.0

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 14) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(prices) < period:
            return 0.02

        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)


class HistoricalDataLoader:
    """Load historical data from CSV files."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or str(PROJECT_ROOT / "data")

    def load_csv(self, symbol: str) -> List[Dict]:
        """Load CSV data for a symbol."""
        filepath = os.path.join(self.data_dir, f"{symbol.lower()}_1h.csv")
        if not os.path.exists(filepath):
            logger.warning(f"No data file for {symbol}: {filepath}")
            return []

        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'timestamp': row.get('timestamp', ''),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0)),
                    'volume': float(row.get('volume', 0))
                })

        return data

    def get_all_symbols(self) -> List[str]:
        """Get all available symbols."""
        symbols = []
        for f in os.listdir(self.data_dir):
            if f.endswith('_1h.csv'):
                symbols.append(f.replace('_1h.csv', '').upper())
        return symbols


class MLSignalGenerator:
    """ML-based signal generator using technical indicators."""

    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        
        # Load seed prices for immediate ML functionality
        try:
            import json
            from pathlib import Path
            seed_file = Path(__file__).parent / 'data' / 'seed_prices.json'
            if seed_file.exists():
                seed_data = json.loads(seed_file.read_text())
                for symbol, prices in seed_data.items():
                    self.price_history[symbol] = list(prices)
                print(f"   Seed prices loaded: {len(seed_data)} symbols")
        except Exception as e:
            pass
        # Get HARDBIT parameters
        params = get_trade_params()
        
        self.genome = Genome(
            stop_loss=params["stop_loss"],
            take_profit=params["take_profit"],
            risk_pct=params["risk_pct"],
            rsi_period=14,
            ema_fast=9,
            ema_slow=21,
            momentum_period=10
        )

    def set_genome(self, genome: Genome):
        """Update signal generator with optimized genome."""
        self.genome = genome
        logger.info(f"Genome updated: SL={genome.stop_loss:.2%}, TP={genome.take_profit:.2%}")

    def add_price(self, symbol: str, price: float):
        """Add price to history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

        # Keep last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

    def generate_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate ML signal based on technical indicators."""
        symbol = market_data.get("symbol", "SOL")
        price = market_data.get("price", 86.0)

        # Add price to history
        self.add_price(symbol, price)
        history = self.price_history.get(symbol, [price])

        if len(history) < 10:
            return None  # Need more data

        # Calculate indicators
        rsi = TechnicalIndicators.calculate_rsi(history, self.genome.rsi_period)
        ema_fast = TechnicalIndicators.calculate_ema(history, self.genome.ema_fast)
        ema_slow = TechnicalIndicators.calculate_ema(history, self.genome.ema_slow)
        momentum = TechnicalIndicators.calculate_momentum(history, self.genome.momentum_period)
        trend = TechnicalIndicators.calculate_trend(history, 20)
        volatility = TechnicalIndicators.calculate_volatility(history, 14)

        # Generate composite score
        scores = []

        # RSI component (mean reversion)
        if rsi < 30:
            scores.append(0.4)
        elif rsi > 70:
            scores.append(-0.4)
        else:
            scores.append((rsi - 50) / 50)

        # EMA crossover component
        if ema_fast > ema_slow:
            scores.append(0.25)
        else:
            scores.append(-0.25)

        # Momentum component
        if momentum > 102:
            scores.append(0.2)
        elif momentum < 98:
            scores.append(-0.2)
        else:
            scores.append((momentum - 100) / 20)

        # Trend component
        scores.append(np.tanh(trend * 10))

        # Volatility penalty
        vol_penalty = min(volatility * 2, 0.15)
        total_score = sum(scores) - vol_penalty
        score = np.clip(total_score, -1, 1)

        # Generate signal if threshold met
        if abs(score) > 0.08:
            direction = "long" if score > 0 else "short"
            vol_multiplier = 1.0 + (volatility * 5)
            sl = self.genome.stop_loss * vol_multiplier
            tp = self.genome.take_profit * vol_multiplier

            # Build reason string
            reasons = []
            if rsi < 30:
                reasons.append(f"RSI({rsi:.0f})OS")
            elif rsi > 70:
                reasons.append(f"RSI({rsi:.0f})OB")
            if ema_fast > ema_slow:
                reasons.append("EMA+")
            else:
                reasons.append("EMA-")
            if momentum > 102:
                reasons.append(f"Mom+{momentum-100:.0f}%")

            # Get current balance and confidence from paper engine
            current_balance = self.paper.state.balance_usd if hasattr(self.paper, 'state') else INITIAL_BALANCE
            confidence = min(abs(score), 0.95)
            
            # Position size scales with confidence (bet more on high confidence signals)
            # Base: balance × risk_pct × confidence multiplier (0.3 to 1.0)
            confidence_multiplier = 0.3 + (confidence * 0.7)  # 0.3 to 0.95
            position_size = current_balance * self.genome.risk_pct * confidence_multiplier

            return {
                "symbol": symbol,
                "direction": direction,
                "price": price,
                "size": position_size,
                "stop_loss": price * (1 - sl),
                "take_profit": price * (1 + tp),
                "confidence": confidence,
                "indicators": {
                    "rsi": rsi,
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "momentum": momentum,
                    "trend": trend,
                    "volatility": volatility
                },
                "reason": " | ".join(reasons[:3])
            }

        return None


class BacktestEngine:
    """Backtesting engine using historical data."""

    def __init__(self, initial_capital: float = 500.0):
        self.initial_capital = initial_capital
        self.data_loader = HistoricalDataLoader()

    def run_backtest(self, genome: Genome, symbol: str = "SOL",
                     data: List[Dict] = None) -> Dict:
        """Run backtest with given genome parameters."""
        if data is None:
            data = self.data_loader.load_csv(symbol)

        if not data:
            return {"error": f"No data for {symbol}"}

        prices = [d['close'] for d in data]
        prices_list = prices

        rsi = TechnicalIndicators.calculate_rsi(prices_list, genome.rsi_period)
        ema_fast = TechnicalIndicators.calculate_ema(prices_list, genome.ema_fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices_list, genome.ema_slow)
        volatility = TechnicalIndicators.calculate_volatility(prices_list, 14)

        trades = []
        balance = self.initial_capital
        position = 0
        entry_price = 0

        vol_multiplier = 1.0 + (volatility * 5)
        sl = genome.stop_loss * vol_multiplier
        tp = genome.take_profit * vol_multiplier

        for i in range(20, len(prices)):
            current_price = prices[i]
            current_rsi = TechnicalIndicators.calculate_rsi(prices_list[:i+1], genome.rsi_period)
            current_ema_fast = TechnicalIndicators.calculate_ema(prices_list[:i+1], genome.ema_fast)
            current_ema_slow = TechnicalIndicators.calculate_ema(prices_list[:i+1], genome.ema_slow)

            score = 0
            if current_rsi < 35:
                score += 0.4
            elif current_rsi > 65:
                score -= 0.4
            if current_ema_fast > current_ema_slow:
                score += 0.25
            else:
                score -= 0.25

            if position == 0 and abs(score) > 0.3:
                position = 1 if score > 0 else -1
                entry_price = current_price

            if position != 0:
                pnl_pct = (current_price - entry_price) / entry_price * position
                if pnl_pct <= -sl or pnl_pct >= tp:
                    balance += balance * genome.risk_pct * pnl_pct
                    trades.append({
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl_pct": pnl_pct * 100,
                        "direction": "long" if position > 0 else "short"
                    })
                    position = 0

        winning = [t for t in trades if t["pnl_pct"] > 0]
        losing = [t for t in trades if t["pnl_pct"] <= 0]

        total_pnl = balance - self.initial_capital
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t["pnl_pct"] for t in winning]) if winning else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losing]) if losing else 0
        pf = abs(sum(t["pnl_pct"] for t in winning) / sum(t["pnl_pct"] for t in losing)) if losing and sum(t["pnl_pct"] for t in losing) != 0 else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.initial_capital * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": pf,
            "final_balance": balance,
            "trades": trades
        }


class GeneticOptimizer:
    """Genetic algorithm for parameter optimization."""

    def __init__(self, population_size: int = POPULATION_SIZE,
                 generations: int = GENERATIONS,
                 mutation_rate: float = MUTATION_RATE):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.backtester = BacktestEngine()
        self.db_path = PROJECT_ROOT / "genetic_results.db"
        self._init_db()

    def _init_db(self):
        """Initialize genetic results database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                generation INTEGER,
                genome TEXT,
                total_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                profit_factor REAL,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def create_random_genome(self) -> Genome:
        """Create a random genome."""
        return Genome(
            stop_loss=random.uniform(0.02, 0.10),
            take_profit=random.uniform(0.05, 0.20),
            risk_pct=random.uniform(0.05, 0.20),
            rsi_period=random.randint(7, 21),
            ema_fast=random.randint(5, 15),
            ema_slow=random.randint(15, 30),
            momentum_period=random.randint(5, 20)
        )

    def mutate(self, genome: Genome) -> Genome:
        """Mutate a genome."""
        return Genome(
            stop_loss=max(0.02, min(0.15, genome.stop_loss + random.uniform(-0.02, 0.02))),
            take_profit=max(0.05, min(0.30, genome.take_profit + random.uniform(-0.03, 0.03))),
            risk_pct=max(0.02, min(0.25, genome.risk_pct + random.uniform(-0.02, 0.02))),
            rsi_period=max(7, min(21, genome.rsi_period + random.randint(-3, 3))),
            ema_fast=max(3, min(15, genome.ema_fast + random.randint(-2, 2))),
            ema_slow=max(10, min(40, genome.ema_slow + random.randint(-5, 5))),
            momentum_period=max(3, min(30, genome.momentum_period + random.randint(-3, 3)))
        )

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Crossover two genomes."""
        return Genome(
            stop_loss=random.choice([parent1.stop_loss, parent2.stop_loss]),
            take_profit=random.choice([parent1.take_profit, parent2.take_profit]),
            risk_pct=random.choice([parent1.risk_pct, parent2.risk_pct]),
            rsi_period=random.choice([parent1.rsi_period, parent2.rsi_period]),
            ema_fast=random.choice([parent1.ema_fast, parent2.ema_fast]),
            ema_slow=random.choice([parent1.ema_slow, parent2.ema_slow]),
            momentum_period=random.choice([parent1.momentum_period, parent2.momentum_period])
        )

    def evaluate_fitness(self, genome: Genome) -> float:
        """Evaluate fitness of a genome."""
        result = self.backtester.run_backtest(genome, "SOL")
        if "error" in result:
            return 0.0

        fitness = (
            (result["win_rate"] / 100) * 0.3 +
            min(result["profit_factor"], 3) / 3 * 0.4 +
            min(result["total_trades"], 50) / 50 * 0.2 +
            0.1
        )
        if result["total_pnl"] < 0:
            fitness *= 0.5
        return fitness

    def save_result(self, generation: int, genome: Genome):
        """Save genome result to database."""
        result = self.backtester.run_backtest(genome, "SOL")
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results (generation, genome, total_trades, win_rate, total_pnl, profit_factor, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            generation,
            json.dumps(genome.to_dict(), cls=DateTimeEncoder),
            result.get("total_trades", 0),
            result.get("win_rate", 0),
            result.get("total_pnl", 0),
            result.get("profit_factor", 0),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

    def optimize(self) -> Tuple[Genome, List[Dict]]:
        """Run genetic optimization."""
        logger.info(f"\nGenetic Optimization: Pop={self.population_size}, Gens={self.generations}")
        
        population = [self.create_random_genome() for _ in range(self.population_size)]
        best_genome = None
        best_fitness = -float('inf')
        history = []

        for gen in range(self.generations):
            for genome in population:
                genome.score = self.evaluate_fitness(genome)

            population.sort(key=lambda g: g.score, reverse=True)

            if population[0].score > best_fitness:
                best_fitness = population[0].score
                best_genome = population[0]

            self.save_result(gen, population[0])

            logger.info(f"   Gen {gen+1}/{self.generations}: Best Score={population[0].score:.3f}")

            next_population = population[:ELITE_COUNT]
            while len(next_population) < self.population_size:
                candidates = random.sample(population[:10], 3)
                parent1 = max(candidates, key=lambda g: g.score)
                candidates = random.sample(population[:10], 3)
                parent2 = max(candidates, key=lambda g: g.score)
                child = self.crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                next_population.append(child)
            population = next_population

        result = self.backtester.run_backtest(best_genome, "SOL")
        logger.info(f"Best Genome: SL={best_genome.stop_loss:.2%}, TP={best_genome.take_profit:.2%}")

        return best_genome, history


class PerformanceTracker:
    """Track performance metrics for all strategies."""

    def __init__(self):
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.paper_engine = PaperTradingEngine()

    def update_from_paper_trades(self):
        """Update metrics from paper trading state."""
        state = self.paper_engine.state
        if not state.trades:
            return

        closed_trades = [t for t in state.trades if t["status"] == "closed"]
        if not closed_trades:
            return

        total = len(closed_trades)
        wins = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
        total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
        wins_pnl = [t.get("pnl", 0) for t in closed_trades if t.get("pnl", 0) > 0]
        losses_pnl = [t.get("pnl", 0) for t in closed_trades if t.get("pnl", 0) <= 0]

        metrics = StrategyMetrics(
            name="current_strategy",
            total_trades=total,
            winning_trades=wins,
            losing_trades=total - wins,
            total_pnl=total_pnl,
            avg_win=np.mean(wins_pnl) if wins_pnl else 0,
            avg_loss=np.mean(losses_pnl) if losses_pnl else 0,
            win_rate=wins / total * 100 if total > 0 else 0,
            profit_factor=abs(sum(wins_pnl) / sum(losses_pnl)) if losses_pnl and sum(losses_pnl) != 0 else 0,
            last_updated=datetime.now().isoformat()
        )
        self.strategies["current_strategy"] = metrics


    def should_adjust_parameters(self) -> Tuple[bool, Dict]:
        """Check if parameters should be adjusted based on performance."""
        metrics = self.strategies.get("current_strategy", None)
        if not metrics:
            return False, {"reason": "No metrics yet"}
        
        # Adjust if win rate is too low or too high (overfitting risk)
        if metrics.win_rate < 35:
            return True, {"reason": "Low win rate", "win_rate": metrics.win_rate}
        elif metrics.win_rate > 85:
            return True, {"reason": "Win rate suspiciously high", "win_rate": metrics.win_rate}
        
        # Adjust if profit factor is low
        if hasattr(metrics, 'profit_factor') and metrics.profit_factor < 1.0:
            return True, {"reason": "Low profit factor", "pf": metrics.profit_factor}
        
        return False, {"reason": "Performance acceptable"}
    
    def should_adjust_parameters(self) -> Tuple[bool, Dict]:
        """Check if parameters should be adjusted based on performance."""
        metrics = self.strategies.get("current_strategy", None)
        if not metrics:
            return False, {"reason": "No metrics yet"}
        
        # Adjust if win rate is too low
        if metrics.win_rate < 35:
            return True, {"reason": "Low win rate", "win_rate": metrics.win_rate}
        elif metrics.win_rate > 85:
            return True, {"reason": "Win rate suspiciously high", "win_rate": metrics.win_rate}
        
        return False, {"reason": "Performance acceptable"}
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        self.update_from_paper_trades()
        return {k: v.to_dict() for k, v in self.strategies.items()}


class AutoOptimizer:
    """Auto-optimize parameters based on performance history."""

    def __init__(self):
        self.tracker = PerformanceTracker()
        # Get HARDBIT parameters
        params = get_trade_params()
        self.current_genome = Genome(
            stop_loss=params["stop_loss"],
            take_profit=params["take_profit"],
            risk_pct=params["risk_pct"],
            rsi_period=14,
            ema_fast=9,
            ema_slow=21,
            momentum_period=10
        )

    def optimize_based_on_performance(self) -> Genome:
        """Optimize parameters based on recent performance."""
        should_adjust, adjustments = self.tracker.should_adjust_parameters()
        if not should_adjust:
            return self.current_genome

        logger.info(f"\nAuto-Optimization Triggered: {adjustments.get('reason', 'Performance review')}")
        metrics = self.tracker.strategies.get("current_strategy", None)

        if metrics and metrics.win_rate < 40:
            self.current_genome.risk_pct = max(0.02, self.current_genome.risk_pct * 0.8)
            self.current_genome.stop_loss = min(0.12, self.current_genome.stop_loss * 1.1)
            logger.info(f"  Reduced risk to {self.current_genome.risk_pct:.2%}")
            logger.info(f"  Widened stop loss to {self.current_genome.stop_loss:.2%}")

        return self.current_genome


class PaperAgentBrain:
    """Agent brain running in paper trading mode with auto-improvement."""

    def __init__(self):
        self.client = JupiterClient()
        self.paper = PaperTradingEngine()
        self.running = False
        self.cycle_count = 0
        self.ml_generator = MLSignalGenerator()
        self.auto_optimizer = AutoOptimizer()
        self.tracker = PerformanceTracker()

    async def get_market_data(self, symbol: str = "SOL") -> Dict:
        """Get current market data."""
        try:
            price = await self.client.get_token_price(SOL)
            trending = await self.client.get_trending_tokens("1h")
            return {
                "symbol": symbol,
                "price": price,
                "change_24h": 0,
                "volume_24h": 0,
                "trending": [t.get("symbol") for t in trending[:5]],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return {"symbol": symbol, "price": 86.0, "change_24h": 0, "volume_24h": 0, "trending": [], "timestamp": datetime.now().isoformat()}

    def generate_ml_signal(self, market_data: Dict) -> Optional[Dict]:
        """Generate ML signal using technical indicators."""
        return self.ml_generator.generate_signal(market_data)

    def execute_paper_trade(self, signal: Dict) -> bool:
        """Execute trade in paper mode."""
        trade = self.paper.execute_signal(signal)
        if trade:
            logger.info(f"PAPER TRADE: {signal['direction'].upper()} {signal['symbol']} @ ${signal['price']:.2f}")
            logger.info(f"  Size: ${signal['size']:.2f} (conf: {signal['confidence']:.2f}) | SL: ${signal['stop_loss']:.2f} | TP: ${signal['take_profit']:.2f}")
            if "reason" in signal:
                logger.info(f"  Reason: {signal['reason']}")
            return True
        return False

    async def run_cycle(self):
        """Run a single trading cycle."""
        self.cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE {self.cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")

        if self.cycle_count % 10 == 0:
            genome = self.auto_optimizer.optimize_based_on_performance()
            self.ml_generator.set_genome(genome)

        market_data = await self.get_market_data()
        logger.info(f"Market: {market_data['symbol']} @ ${market_data['price']:.2f}")

        signal = self.generate_ml_signal(market_data)
        if signal:
            logger.info(f"ML SIGNAL: {signal['direction'].upper()} | Confidence: {signal['confidence']:.0%}")
            self.execute_paper_trade(signal)
        else:
            logger.info("No ML signal generated")

        metrics = self.tracker.get_metrics()
        if "current_strategy" in metrics:
            m = metrics["current_strategy"]
            logger.info(f"\nPERFORMANCE: {m['total_trades']} trades | WR: {m['win_rate']:.1f}% | PnL: ${m['total_pnl']:.2f} | PF: {m['profit_factor']:.2f}")

    async def run_backtest_only(self):
        """Run backtest using historical data."""
        logger.info("\nBACKTEST MODE")
        data_loader = HistoricalDataLoader()
        symbols = data_loader.get_all_symbols()
        logger.info(f"Available symbols: {', '.join(symbols)}")

        genome = Genome(
            stop_loss=0.05, take_profit=0.10, risk_pct=0.10,
            rsi_period=14, ema_fast=9, ema_slow=21, momentum_period=10
        )
        backtester = BacktestEngine()

        for symbol in symbols:
            data = data_loader.load_csv(symbol)
            if data:
                result = backtester.run_backtest(genome, symbol)
                logger.info(f"\n{symbol}: {result['total_trades']} trades, {result['win_rate']:.1f}% WR, PnL: ${result['total_pnl']:.2f}, PF: {result['profit_factor']:.2f}")

    async def run_genetic_only(self):
        """Run genetic optimization only."""
        logger.info("\nGENETIC OPTIMIZATION MODE")
        optimizer = GeneticOptimizer(population_size=10, generations=5)
        best_genome, history = optimizer.optimize()
        logger.info(f"\nBest Genome: SL={best_genome.stop_loss:.2%}, TP={best_genome.take_profit:.2%}, Risk={best_genome.risk_pct:.2%}")

    async def run(self):
        """Main loop."""
        logger.info("\n" + "="*60)
        logger.info("AGENT BRAIN - PAPER TRADING WITH AUTO-IMPROVEMENT")
        logger.info("="*60)
        logger.info(f"  Initial Balance: ${INITIAL_BALANCE:,}")
        logger.info(f"  ML Indicators: RSI + EMA + Momentum + Trend")
        logger.info(f"  Auto-Optimization: Enabled")
        logger.info(f"  Genetic Algorithm: Enabled")
        logger.info("="*60)

        self.paper.start()
        self.running = True
        while self.running:
            try:
                await self.run_cycle()
                await asyncio.sleep(CYCLE_INTERVAL)
            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                self.paper.stop()
                self.running = False
            except Exception as e:
                logger.error(f"Cycle error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(10)

        await self.client.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Brain - Paper Trading")
    parser.add_argument("--fast", action="store_true", help="Fast mode")
    parser.add_argument("--slow", action="store_true", help="Slow mode")
    parser.add_argument("--backtest", action="store_true", help="Run backtest only")
    parser.add_argument("--genetic", action="store_true", help="Run genetic optimization only")

    args = parser.parse_args()

    global CYCLE_INTERVAL
    if args.fast:
        CYCLE_INTERVAL = 60
    elif args.slow:
        CYCLE_INTERVAL = 300

    brain = PaperAgentBrain()

    if args.backtest:
        asyncio.run(brain.run_backtest_only())
    elif args.genetic:
        asyncio.run(brain.run_genetic_only())
    else:
        asyncio.run(brain.run())


if __name__ == "__main__":
    main()
