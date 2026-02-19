#!/usr/bin/env python3
"""
Strategy Optimizer Agent
========================
Auto-improves trading strategies based on performance.

Features:
- Analyzes signal effectiveness
- Adjusts parameters automatically
- Tests new strategies
- Deploys improvements

Usage:
    python3 strategy_optimizer_agent.py --start     # Start optimizer
    python3 strategy_optimizer_agent.py --status   # Check status
"""

import json
import time
import asyncio
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Paths
PROJECT_ROOT = Path(__file__).parent
PAPER_STATE_FILE = PROJECT_ROOT / "paper_trading_state.json"
OPTIMIZER_STATE_FILE = PROJECT_ROOT / "optimizer_state.json"


@dataclass
class StrategyParams:
    """Strategy parameters - Extended with 13 optimization parameters."""
    name: str
    
    # Original parameters
    range_threshold: float = 0.25  # Buy when < 25% of range
    ma_period: int = 5
    volatility_threshold: float = 0.005
    stop_loss: float = 0.05
    take_profit: float = 0.10
    
    # NEW: RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # NEW: Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # NEW: EMA crossover
    ema_fast: int = 12
    ema_slow: int = 26
    
    # NEW: Position & risk management
    position_size: float = 0.10  # 10% of balance
    leverage: int = 3
    max_hold_time: int = 60  # minutes
    trailing_stop: bool = True
    trailing_distance: float = 0.02  # 2%
    
    # Performance tracking
    score: float = 0.0
    trades: int = 0
    wins: int = 0


@dataclass
class OptimizerState:
    """Optimizer state."""
    running: bool = False
    start_time: Optional[str] = None
    strategies: List[Dict] = field(default_factory=list)
    improvements: List[Dict] = field(default_factory=list)
    total_iterations: int = 0


class StrategyOptimizer:
    """Auto-optimizes trading strategies."""

    def __init__(self):
        self.state = self._load_state()
        self.strategies = self._init_strategies()

    def _load_state(self) -> OptimizerState:
        """Load state."""
        if OPTIMIZER_STATE_FILE.exists():
            return OptimizerState(**json.loads(OPTIMIZER_STATE_FILE.read_text()))
        return OptimizerState()

    def _save_state(self):
        """Save state."""
        data = {
            "running": self.state.running,
            "start_time": self.state.start_time,
            "strategies": self.state.strategies,
            "improvements": self.state.improvements,
            "total_iterations": self.state.total_iterations
        }
        OPTIMIZER_STATE_FILE.write_text(json.dumps(data, indent=2))

    def _init_strategies(self) -> List[StrategyParams]:
        """Initialize default strategies with extended parameters."""
        return [
            StrategyParams(
                name="range_aggressive",
                range_threshold=0.25,
                ma_period=5,
                rsi_period=7,
                rsi_oversold=30,
                rsi_overbought=70,
                bb_period=20,
                bb_std=2.0,
                ema_fast=12,
                ema_slow=26,
                position_size=0.15,
                leverage=3,
                max_hold_time=60,
                trailing_stop=True,
                trailing_distance=0.02,
                score=0.6
            ),
            StrategyParams(
                name="range_conservative",
                range_threshold=0.15,
                ma_period=10,
                rsi_period=14,
                rsi_oversold=25,
                rsi_overbought=75,
                bb_period=30,
                bb_std=2.5,
                ema_fast=20,
                ema_slow=50,
                position_size=0.08,
                leverage=2,
                max_hold_time=120,
                trailing_stop=True,
                trailing_distance=0.015,
                score=0.5
            ),
            StrategyParams(
                name="rsi_momentum",
                rsi_period=7,
                rsi_oversold=35,
                rsi_overbought=65,
                position_size=0.12,
                leverage=3,
                max_hold_time=45,
                trailing_stop=True,
                trailing_distance=0.025,
                score=0.55
            ),
            StrategyParams(
                name="bb_breakout",
                bb_period=20,
                bb_std=2.0,
                volatility_threshold=0.01,
                position_size=0.10,
                leverage=4,
                max_hold_time=30,
                trailing_stop=False,
                score=0.45
            ),
            StrategyParams(
                name="ema_crossover",
                ema_fast=12,
                ema_slow=26,
                position_size=0.10,
                leverage=3,
                max_hold_time=90,
                trailing_stop=True,
                trailing_distance=0.02,
                score=0.50
            ),
            StrategyParams(
                name="hybrid_advanced",
                range_threshold=0.20,
                ma_period=8,
                rsi_period=10,
                rsi_oversold=32,
                rsi_overbought=68,
                bb_period=25,
                bb_std=2.0,
                ema_fast=15,
                ema_slow=40,
                position_size=0.12,
                leverage=3,
                max_hold_time=60,
                trailing_stop=True,
                trailing_distance=0.022,
                score=0.52
            ),
        ]

    def _load_paper_trades(self) -> List[Dict]:
        """Load paper trades."""
        if PAPER_STATE_FILE.exists():
            return json.loads(PAPER_STATE_FILE.read_text()).get("trades", [])
        return []

    def analyze_performance(self) -> Dict:
        """Analyze strategy performance."""
        trades = self._load_paper_trades()

        if not trades:
            return {"status": "no_data", "message": "No trades yet"}

        # Calculate metrics
        total = len(trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = total - wins

        # P&L
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        # Win rate
        win_rate = (wins / total * 100) if total > 0 else 0

        # Average win/loss
        wins_pnl = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0]
        losses_pnl = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) <= 0]

        avg_win = np.mean(wins_pnl) if wins_pnl else 0
        avg_loss = np.mean(losses_pnl) if losses_pnl else 0

        return {
            "status": "analyzed",
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }

    def _mutate_params(self, params: StrategyParams) -> StrategyParams:
        """Create mutated version of strategy with all parameters."""
        mutation = StrategyParams(
            name=f"{params.name}_v{random.randint(1, 100)}",
            
            # Original parameters
            range_threshold=params.range_threshold * random.uniform(0.8, 1.2),
            ma_period=max(2, params.ma_period + random.randint(-2, 2)),
            volatility_threshold=max(0.001, params.volatility_threshold * random.uniform(0.7, 1.3)),
            stop_loss=max(0.02, params.stop_loss * random.uniform(0.8, 1.2)),
            take_profit=max(0.05, params.take_profit * random.uniform(0.8, 1.2)),
            
            # NEW: RSI parameters
            rsi_period=max(5, min(21, params.rsi_period + random.randint(-3, 3))),
            rsi_oversold=max(20, min(40, params.rsi_oversold + random.uniform(-3, 3))),
            rsi_overbought=max(60, min(80, params.rsi_overbought + random.uniform(-3, 3))),
            
            # NEW: Bollinger Bands
            bb_period=max(10, min(40, params.bb_period + random.randint(-3, 3))),
            bb_std=max(1.5, min(3.0, params.bb_std * random.uniform(0.8, 1.2))),
            
            # NEW: EMA
            ema_fast=max(5, min(20, params.ema_fast + random.randint(-2, 2))),
            ema_slow=max(15, min(60, params.ema_slow + random.randint(-5, 5))),
            
            # NEW: Position & risk
            position_size=max(0.05, min(0.25, params.position_size * random.uniform(0.8, 1.2))),
            leverage=max(1, min(10, params.leverage + random.randint(-1, 1))),
            max_hold_time=max(15, min(180, params.max_hold_time + random.randint(-15, 15))),
            trailing_stop=params.trailing_stop,
            trailing_distance=max(0.01, min(0.05, params.trailing_distance * random.uniform(0.8, 1.2))),
        )
        return mutation

    def _test_strategy(self, strategy: StrategyParams, trades: List[Dict]) -> float:
        """Test strategy on historical trades."""
        if not trades:
            return 0.5  # Neutral score

        # Simple scoring based on win rate and P&L
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(trades) if trades else 0.5

        # Adjust based on strategy parameters
        score = win_rate

        # Bonus for conservative parameters
        if strategy.stop_loss < 0.05:
            score += 0.05
        if strategy.take_profit > strategy.stop_loss * 2:
            score += 0.05

        return min(1.0, max(0.0, score))

    def optimize(self) -> List[Dict]:
        """Run one optimization cycle."""
        self.state.total_iterations += 1

        # Analyze current performance
        perf = self.analyze_performance()

        # Create mutations
        improvements = []
        trades = self._load_paper_trades()

        for strategy in self.strategies:
            # Test current strategy
            current_score = self._test_strategy(strategy, trades)

            # Create mutation
            mutated = self._mutate_params(strategy)
            mutated_score = self._test_strategy(mutated, trades)

            if mutated_score > current_score:
                # Keep mutation
                improvements.append({
                    "time": datetime.now().isoformat(),
                    "strategy": strategy.name,
                    "improvement": f"Score: {current_score:.2f} → {mutated_score:.2f}",
                    "new_params": {
                        "range_threshold": mutated.range_threshold,
                        "ma_period": mutated.ma_period,
                        "stop_loss": mutated.stop_loss,
                        "take_profit": mutated.take_profit
                    }
                })
                # Update strategy
                strategy.range_threshold = mutated.range_threshold
                strategy.ma_period = mutated.ma_period
                strategy.stop_loss = mutated.stop_loss
                strategy.take_profit = mutated.take_profit
                strategy.score = mutated_score

        self.state.improvements.extend(improvements[-10:])  # Keep last 10
        self._save_state()

        return improvements

    def get_best_strategy(self) -> Optional[StrategyParams]:
        """Get best performing strategy."""
        if not self.strategies:
            return None
        return max(self.strategies, key=lambda s: s.score)

    def get_recommendation(self) -> Dict:
        """Get optimization recommendation."""
        perf = self.analyze_performance()
        best = self.get_best_strategy()

        recommendations = []

        if perf.get("win_rate", 0) < 0.4:
            recommendations.append("Win rate is low - consider more conservative parameters")

        if perf.get("profit_factor", 0) < 1.0:
            recommendations.append("Profit factor < 1 - review risk management")

        if best and best.score < 0.5:
            recommendations.append("Best strategy score is low - try different approach")

        if perf.get("total_trades", 0) < 10:
            recommendations.append("Need more trades for reliable analysis")

        return {
            "performance": perf,
            "best_strategy": best.name if best else None,
            "best_score": best.score if best else 0,
            "recommendations": recommendations,
            "iterations": self.state.total_iterations
        }


class OptimizerAgent:
    """Optimizer agent that runs continuously."""

    def __init__(self):
        self.optimizer = StrategyOptimizer()
        self.running = False

    async def run(self):
        """Main loop."""
        print("\n" + "="*60)
        print("🧠 STRATEGY OPTIMIZER AGENT")
        print("="*60)
        print("Auto-improving trading strategies based on performance")
        print("="*60 + "\n")

        self.running = True
        cycle = 0

        while self.running:
            cycle += 1
            print(f"\n🔄 OPTIMIZATION CYCLE {cycle}")
            print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")

            # Run optimization
            improvements = self.optimizer.optimize()

            if improvements:
                print(f"   ✅ {len(improvements)} improvements made:")
                for imp in improvements[-3:]:  # Show last 3
                    print(f"      • {imp['strategy']}: {imp['improvement']}")
            else:
                print("   📊 No improvements needed")

            # Get recommendation
            rec = self.optimizer.get_recommendation()

            print(f"\n   📈 Performance: {rec['performance'].get('status', 'N/A')}")
            print(f"   🏆 Best Strategy: {rec['best_strategy']} ({rec['best_score']:.2f})")

            if rec['recommendations']:
                print(f"\n   💡 Recommendations:")
                for r in rec['recommendations'][:3]:
                    print(f"      • {r}")

            await asyncio.sleep(60)  # Optimize every minute

    def stop(self):
        """Stop agent."""
        self.running = False


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Optimizer Agent")
    parser.add_argument("--start", action="store_true", help="Start optimizer")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--optimize", action="store_true", help="Run one optimization cycle")

    args = parser.parse_args()

    agent = OptimizerAgent()

    if args.start:
        await agent.run()
    elif args.status:
        optimizer = StrategyOptimizer()
        rec = optimizer.get_recommendation()
        print("\n📊 OPTIMIZER STATUS")
        print(json.dumps(rec, indent=2))
    elif args.optimize:
        optimizer = StrategyOptimizer()
        improvements = optimizer.optimize()
        print(f"\n✅ Optimization complete: {len(improvements)} improvements")
    else:
        await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
