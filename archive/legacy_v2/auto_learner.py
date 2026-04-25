#!/usr/bin/env python3
"""
AUTO-LEARNING SYSTEM v1.0
=========================
Sistema de auto-aprendizaje completo para trading.

Características:
1. Genetic Algorithm Evolution - Evolución de estrategias óptimas
2. Reinforcement Learning - Adaptación en tiempo real
3. Adaptive Parameters - Ajuste dinámico de parámetros
4. Performance Feedback Loop - Mejora continua basada en resultados
5. Risk Management Adaptativo - Gestión de riesgo inteligente

Objetivo: 5% diario con max 10% drawdown
"""

import json
import time
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import threading
from collections import deque
import pickle

# Import existing genetic miner
from strategies.genetic_miner import StrategyMiner, Genome

logger = logging.getLogger("auto_learner")

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = Path("data/auto_learning.db")
STATE_FILE = Path("data/learner_state.json")
PERFORMANCE_HISTORY_SIZE = 1000  # Keep last 1000 trades for analysis

# Daily Target
DAILY_TARGET = 0.05  # 5%
MAX_DRAWDOWN = 0.10  # 10%

# Learning Parameters
EVOLUTION_INTERVAL = 3600  # Evolve strategies every 1 hour
ADAPTATION_INTERVAL = 300  # Adapt parameters every 5 minutes
MIN_TRADES_FOR_EVOLUTION = 50  # Minimum trades before evolving
PERFORMANCE_WINDOW = 100  # Look at last 100 trades for performance

# Genetic Algorithm
POPULATION_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.7

# Reinforcement Learning
STATE_SIZE = 10  # Market state features
ACTION_SIZE = 3   # Buy, Sell, Hold
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 0.3
EXPLORATION_DECAY = 0.995

# Adaptive Parameters
PARAM_ADAPTATION_RATE = 0.1  # How fast to adapt parameters
MIN_CONFIDENCE = 0.6  # Minimum confidence to execute trade


# ============================================================================
# DATABASE
# ============================================================================
class AutoLearningDB:
    """Persistent storage for learning data"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Performance history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    trade_id TEXT,
                    strategy_id TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    win BOOLEAN,
                    holding_time REAL,
                    entry_params TEXT,
                    exit_params TEXT,
                    market_state TEXT,
                    confidence REAL,
                    metadata TEXT
                )
            """)
            
            # Strategy performance
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT UNIQUE,
                    genome TEXT,
                    generation INTEGER,
                    fitness REAL,
                    win_rate REAL,
                    avg_pnl REAL,
                    sharpe REAL,
                    max_dd REAL,
                    total_trades INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_used DATETIME,
                    active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Learning state
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    current_generation INTEGER,
                    best_fitness REAL,
                    exploration_rate REAL,
                    total_trades INTEGER,
                    total_pnl REAL,
                    daily_pnl REAL,
                    active_strategy TEXT,
                    adaptation_count INTEGER,
                    metadata TEXT
                )
            """)
            
            # Market states for RL
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    state_vector TEXT,
                    action INTEGER,
                    reward REAL,
                    next_state TEXT,
                    done BOOLEAN
                )
            """)
            
            conn.commit()
    
    def record_performance(self, trade_data: Dict):
        """Record trade performance"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance 
                (trade_id, strategy_id, pnl, pnl_pct, win, holding_time,
                 entry_params, exit_params, market_state, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get("trade_id"),
                trade_data.get("strategy_id"),
                trade_data.get("pnl"),
                trade_data.get("pnl_pct"),
                trade_data.get("win"),
                trade_data.get("holding_time"),
                json.dumps(trade_data.get("entry_params", {})),
                json.dumps(trade_data.get("exit_params", {})),
                json.dumps(trade_data.get("market_state", {})),
                trade_data.get("confidence"),
                json.dumps(trade_data.get("metadata", {}))
            ))
            conn.commit()
    
    def get_recent_performance(self, n: int = PERFORMANCE_WINDOW) -> pd.DataFrame:
        """Get last N trades for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM performance ORDER BY timestamp DESC LIMIT {n}",
                conn
            )
        return df
    
    def save_strategy(self, genome: Genome, generation: int, metrics: Dict):
        """Save evolved strategy"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO strategies
                (strategy_id, genome, generation, fitness, win_rate, avg_pnl,
                 sharpe, max_dd, total_trades, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"gen_{generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                json.dumps(genome.to_dict()),
                generation,
                metrics.get("fitness", 0),
                metrics.get("win_rate", 0),
                metrics.get("avg_pnl", 0),
                metrics.get("sharpe", 0),
                metrics.get("max_dd", 0),
                metrics.get("total_trades", 0),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def get_best_strategy(self) -> Optional[Dict]:
        """Get best performing strategy"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM strategies 
                WHERE active = TRUE 
                ORDER BY fitness DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        return None
    
    def get_all_active_strategies(self) -> List[Dict]:
        """Get all active strategies"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM strategies 
                WHERE active = TRUE 
                ORDER BY fitness DESC
            """)
            rows = cursor.fetchall()
            if rows:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        return []


# ============================================================================
# REINFORCEMENT LEARNING AGENT
# ============================================================================
@dataclass
class MarketState:
    """Market state for RL"""
    rsi: float
    ema_diff: float  # EMA fast - EMA slow
    volatility: float
    volume_ratio: float
    price_momentum: float
    bb_position: float  # Where price is in Bollinger Bands
    trend_strength: float
    recent_pnl: float
    trade_count: int
    time_of_day: float  # Normalized hour (0-1)


class RLAgent:
    """Reinforcement Learning agent for real-time adaptation"""
    
    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EXPLORATION_RATE
        self.epsilon_decay = EXPLORATION_DECAY
        self.epsilon_min = 0.01
        
        # Q-table for simple tabular RL (can upgrade to DQN later)
        self.q_table = {}
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Performance tracking
        self.total_reward = 0
        self.episode_count = 0
        
        self.db = AutoLearningDB()
    
    def get_state_key(self, state: MarketState) -> str:
        """Discretize state for Q-table"""
        # Discretize continuous values into bins
        rsi_bin = int(state.rsi / 10)  # 0-10
        ema_bin = int((state.ema_diff + 1) * 5)  # -5 to +5
        vol_bin = int(min(state.volatility * 10, 10))  # 0-10
        mom_bin = int(min(abs(state.price_momentum) * 100, 10))  # 0-10
        
        return f"{rsi_bin}_{ema_bin}_{vol_bin}_{mom_bin}"
    
    def choose_action(self, state: MarketState, explore: bool = True) -> Tuple[int, float]:
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        # Exploration
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
            confidence = 1.0 / self.action_size
        else:
            # Exploitation - use Q-values
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            
            q_values = self.q_table[state_key]
            action = np.argmax(q_values)
            confidence = (q_values[action] - q_values.mean()) / (q_values.std() + 1e-8)
            confidence = 1 / (1 + np.exp(-confidence))  # Sigmoid to 0-1
        
        return action, confidence
    
    def remember(self, state: MarketState, action: int, reward: float, 
                 next_state: MarketState, done: bool):
        """Store experience for replay"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        """Update Q-values from experience"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            # Q-learning update
            current_q = self.q_table[state_key][action]
            
            if done:
                target_q = reward
            else:
                max_next_q = np.max(self.q_table[next_state_key])
                target_q = reward + self.gamma * max_next_q
            
            # Update with learning rate
            self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.episode_count += 1
    
    def save_q_table(self, path: Path):
        """Save Q-table to file"""
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, path: Path):
        """Load Q-table from file"""
        if path.exists():
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)


# ============================================================================
# ADAPTIVE PARAMETER OPTIMIZER
# ============================================================================
class AdaptiveParameterOptimizer:
    """Dynamically adjusts trading parameters based on performance"""
    
    def __init__(self):
        self.params = {
            "sl_pct": 0.025,  # 2.5% stop loss
            "tp_pct": 0.05,   # 5% take profit
            "position_size": 0.1,  # 10% of capital
            "leverage": 5.0,
            "confidence_threshold": 0.6,
            "max_trades_per_day": 10,
            "risk_per_trade": 0.05  # 5% risk per trade
        }
        
        self.param_history = {k: deque(maxlen=100) for k in self.params.keys()}
        self.performance_history = deque(maxlen=100)
        
        self.db = AutoLearningDB()
    
    def update_parameters(self, recent_trades: pd.DataFrame):
        """Adapt parameters based on recent performance"""
        if len(recent_trades) < 10:
            return self.params
        
        # Calculate performance metrics
        win_rate = recent_trades['win'].mean()
        avg_pnl = recent_trades['pnl_pct'].mean()
        total_pnl = recent_trades['pnl_pct'].sum()
        
        # Store performance
        self.performance_history.append({
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "timestamp": datetime.now().isoformat()
        })
        
        # Adapt stop loss and take profit based on performance
        if win_rate > 0.5:
            # Winning more than 50% - can be more aggressive
            self.params["sl_pct"] *= (1 + PARAM_ADAPTATION_RATE * 0.5)
            self.params["tp_pct"] *= (1 + PARAM_ADAPTATION_RATE * 0.5)
        else:
            # Losing more than 50% - need to be more conservative
            self.params["sl_pct"] *= (1 - PARAM_ADAPTATION_RATE * 0.5)
            self.params["tp_pct"] *= (1 - PARAM_ADAPTATION_RATE * 0.3)
        
        # Adapt position size based on recent performance
        if total_pnl > 0:
            # Making money - can increase position size slightly
            self.params["position_size"] = min(
                self.params["position_size"] * (1 + PARAM_ADAPTATION_RATE * 0.3),
                0.2  # Max 20% position size
            )
        else:
            # Losing money - reduce position size
            self.params["position_size"] = max(
                self.params["position_size"] * (1 - PARAM_ADAPTATION_RATE * 0.5),
                0.05  # Min 5% position size
            )
        
        # Adapt leverage based on win rate
        if win_rate > 0.6:
            self.params["leverage"] = min(self.params["leverage"] + 0.5, 10.0)
        elif win_rate < 0.4:
            self.params["leverage"] = max(self.params["leverage"] - 0.5, 1.0)
        
        # Adapt confidence threshold based on performance
        if win_rate > 0.55:
            self.params["confidence_threshold"] = max(
                self.params["confidence_threshold"] - 0.05,
                0.5
            )
        else:
            self.params["confidence_threshold"] = min(
                self.params["confidence_threshold"] + 0.05,
                0.8
            )
        
        # Store parameter history
        for key, value in self.params.items():
            self.param_history[key].append(value)
        
        return self.params
    
    def get_current_parameters(self) -> Dict:
        """Get current optimized parameters"""
        return self.params.copy()


# ============================================================================
# AUTO-LEARNING ORCHESTRATOR
# ============================================================================
class AutoLearningOrchestrator:
    """Main orchestrator for the auto-learning system"""
    
    def __init__(self, data_manager=None):
        self.db = AutoLearningDB()
        self.rl_agent = RLAgent()
        self.param_optimizer = AdaptiveParameterOptimizer()
        
        self.data_manager = data_manager
        self.genetic_miner = None
        
        # State
        self.current_generation = 0
        self.best_strategy = None
        self.total_trades = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Threading
        self.running = False
        self.evolution_thread = None
        self.adaptation_thread = None
        
        # Load previous state
        self._load_state()
        
        # Load RL Q-table if exists
        q_table_path = Path("data/q_table.pkl")
        self.rl_agent.load_q_table(q_table_path)
    
    def _load_state(self):
        """Load previous learning state"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.current_generation = state.get("current_generation", 0)
                self.total_trades = state.get("total_trades", 0)
                self.daily_pnl = state.get("daily_pnl", 0.0)
                self.total_pnl = state.get("total_pnl", 0.0)
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save current learning state"""
        state = {
            "current_generation": self.current_generation,
            "total_trades": self.total_trades,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "last_updated": datetime.now().isoformat(),
            "exploration_rate": self.rl_agent.epsilon
        }
        
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Also save Q-table
        q_table_path = Path("data/q_table.pkl")
        self.rl_agent.save_q_table(q_table_path)
    
    def initialize_genetic_miner(self, df: pd.DataFrame):
        """Initialize genetic algorithm miner with historical data"""
        self.genetic_miner = StrategyMiner(
            df=df,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS
        )
        logger.info(f"Genetic miner initialized with {len(df)} candles")
    
    def evolve_strategies(self):
        """Run genetic algorithm evolution"""
        if self.total_trades < MIN_TRADES_FOR_EVOLUTION:
            logger.info(f"Not enough trades for evolution ({self.total_trades}/{MIN_TRADES_FOR_EVOLUTION})")
            return None
        
        if self.genetic_miner is None:
            logger.error("Genetic miner not initialized")
            return None
        
        try:
            logger.info(f"Starting evolution generation {self.current_generation + 1}")
            
            # Run evolution
            results = self.genetic_miner.evolve()
            
            if results:
                best_genome = results["best_genome"]
                metrics = results["metrics"]
                
                # Save best strategy
                self.db.save_strategy(best_genome, self.current_generation + 1, metrics)
                
                # Update current best
                self.best_strategy = {
                    "genome": best_genome,
                    "metrics": metrics,
                    "generation": self.current_generation + 1
                }
                
                self.current_generation += 1
                
                logger.info(f"Evolution complete. Best fitness: {metrics['fitness']:.4f}")
                logger.info(f"Win rate: {metrics['win_rate']:.2%}, Sharpe: {metrics['sharpe']:.2f}")
                
                self._save_state()
                
                return results
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def get_market_state(self, df: pd.DataFrame) -> MarketState:
        """Extract current market state for RL"""
        if len(df) < 30:
            return None
        
        recent = df.tail(30)
        current = df.iloc[-1]
        
        # Calculate features
        rsi = current.get('RSI', 50)
        
        ema_fast = current.get(f'EMA_{EMA_FAST}', current['close'])
        ema_slow = current.get(f'EMA_{EMA_SLOW}', current['close'])
        ema_diff = (ema_fast - ema_slow) / current['close']
        
        returns = recent['close'].pct_change()
        volatility = returns.std()
        
        volume_ratio = current['volume'] / recent['volume'].mean() if recent['volume'].mean() > 0 else 1
        
        price_momentum = (current['close'] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Bollinger Bands position
        bb_high = current.get('BB_HIGH', current['close'])
        bb_low = current.get('BB_LOW', current['close'])
        bb_range = bb_high - bb_low
        bb_position = (current['close'] - bb_low) / bb_range if bb_range > 0 else 0.5
        
        # Trend strength (ADX-like)
        trend_strength = abs(ema_diff) * 100
        
        # Recent PnL
        recent_perf = self.db.get_recent_performance(10)
        recent_pnl = recent_perf['pnl_pct'].sum() if len(recent_perf) > 0 else 0
        
        # Trade count today
        today = datetime.now().date()
        today_trades = len(recent_perf[
            pd.to_datetime(recent_perf['timestamp']).dt.date == today
        ]) if len(recent_perf) > 0 else 0
        
        # Time of day (0-1)
        hour = datetime.now().hour + datetime.now().minute / 60
        time_of_day = hour / 24
        
        return MarketState(
            rsi=rsi,
            ema_diff=ema_diff,
            volatility=volatility,
            volume_ratio=volume_ratio,
            price_momentum=price_momentum,
            bb_position=bb_position,
            trend_strength=trend_strength,
            recent_pnl=recent_pnl,
            trade_count=today_trades,
            time_of_day=time_of_day
        )
    
    def should_trade(self, state: MarketState) -> Tuple[bool, float, str]:
        """Determine if we should trade using RL"""
        action, confidence = self.rl_agent.choose_action(state, explore=True)
        
        # 0 = Buy, 1 = Sell, 2 = Hold
        should_trade = action in [0, 1]
        
        # Get current parameters
        params = self.param_optimizer.get_current_parameters()
        
        # Check confidence threshold
        if confidence < params["confidence_threshold"]:
            return False, confidence, "hold"
        
        # Check daily limits
        if self.daily_pnl >= DAILY_TARGET:
            return False, confidence, "daily_target_reached"
        
        # Check drawdown
        if self.daily_pnl <= -MAX_DRAWDOWN:
            return False, confidence, "max_drawdown_reached"
        
        direction = "long" if action == 0 else "short"
        
        return should_trade, confidence, direction
    
    def record_trade_result(self, trade_data: Dict):
        """Record trade result and learn from it"""
        # Record to database
        self.db.record_performance(trade_data)
        
        self.total_trades += 1
        self.daily_pnl += trade_data.get("pnl_pct", 0)
        self.total_pnl += trade_data.get("pnl", 0)
        
        # Create market state from trade data
        market_state_data = trade_data.get("market_state", {})
        state = MarketState(**market_state_data) if market_state_data else None
        
        if state:
            # Calculate reward for RL
            reward = trade_data.get("pnl_pct", 0) * 100  # Scale reward
            
            # Remember for replay
            self.rl_agent.remember(
                state=state,
                action=0 if trade_data.get("direction") == "long" else 1,
                reward=reward,
                next_state=state,  # Would need actual next state in production
                done=False
            )
            
            # Learn from experience
            self.rl_agent.learn()
        
        # Adapt parameters based on recent performance
        recent_trades = self.db.get_recent_performance(PERFORMANCE_WINDOW)
        self.param_optimizer.update_parameters(recent_trades)
        
        # Save state
        self._save_state()
    
    def _evolution_loop(self):
        """Background thread for periodic evolution"""
        while self.running:
            time.sleep(EVOLUTION_INTERVAL)
            if self.running:
                logger.info("Running scheduled evolution...")
                self.evolve_strategies()
    
    def _adaptation_loop(self):
        """Background thread for parameter adaptation"""
        while self.running:
            time.sleep(ADAPTATION_INTERVAL)
            if self.running:
                # Get recent performance and adapt
                recent_trades = self.db.get_recent_performance(PERFORMANCE_WINDOW)
                if len(recent_trades) > 0:
                    params = self.param_optimizer.update_parameters(recent_trades)
                    logger.info(f"Parameters adapted: SL={params['sl_pct']:.3f}, "
                               f"TP={params['tp_pct']:.3f}, Size={params['position_size']:.2f}")
    
    def start(self):
        """Start the auto-learning system"""
        self.running = True
        
        # Start evolution thread
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.evolution_thread.start()
        
        # Start adaptation thread
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        logger.info("Auto-learning system started")
        logger.info(f"Evolution interval: {EVOLUTION_INTERVAL}s")
        logger.info(f"Adaptation interval: {ADAPTATION_INTERVAL}s")
    
    def stop(self):
        """Stop the auto-learning system"""
        self.running = False
        
        if self.evolution_thread:
            self.evolution_thread.join(timeout=5)
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=5)
        
        # Save final state
        self._save_state()
        
        logger.info("Auto-learning system stopped")
    
    def get_status(self) -> Dict:
        """Get current status of the auto-learning system"""
        return {
            "running": self.running,
            "current_generation": self.current_generation,
            "total_trades": self.total_trades,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "exploration_rate": self.rl_agent.epsilon,
            "current_parameters": self.param_optimizer.get_current_parameters(),
            "best_strategy": {
                "generation": self.best_strategy["generation"] if self.best_strategy else 0,
                "fitness": self.best_strategy["metrics"]["fitness"] if self.best_strategy else 0,
                "win_rate": self.best_strategy["metrics"]["win_rate"] if self.best_strategy else 0,
            } if self.best_strategy else None
        }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("AUTO-LEARNING SYSTEM v1.0")
    print("="*80)
    
    # Initialize
    orchestrator = AutoLearningOrchestrator()
    
    # Get status
    status = orchestrator.get_status()
    print("\n📊 Current Status:")
    print(f"  Generation: {status['current_generation']}")
    print(f"  Total Trades: {status['total_trades']}")
    print(f"  Daily PnL: {status['daily_pnl']:.2%}")
    print(f"  Total PnL: ${status['total_pnl']:.2f}")
    print(f"  Exploration Rate: {status['exploration_rate']:.3f}")
    
    if status['best_strategy']:
        print(f"\n🧬 Best Strategy:")
        print(f"  Generation: {status['best_strategy']['generation']}")
        print(f"  Fitness: {status['best_strategy']['fitness']:.4f}")
        print(f"  Win Rate: {status['best_strategy']['win_rate']:.2%}")
    
    print("\n⚙️  Current Parameters:")
    params = status['current_parameters']
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n✅ System ready to start")
    print("Call orchestrator.start() to begin auto-learning")
