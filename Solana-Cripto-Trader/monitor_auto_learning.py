#!/usr/bin/env python3
"""
AUTO-LEARNING MONITOR DASHBOARD
================================
Dashboard en tiempo real para monitorear el progreso del aprendizaje.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

sys.path.insert(0, str(Path(__file__).parent))

from auto_learner import AutoLearningOrchestrator
import sqlite3

logger = logging.getLogger("auto_learning_monitor")


# ============================================================================
# LEARNING MONITOR
# ============================================================================
class LearningMonitor:
    """Monitor del sistema de auto-aprendizaje"""
    
    def __init__(self):
        self.learner = AutoLearningOrchestrator()
        self.db = self.learner.db
    
    def get_performance_report(self, hours: int = 24) -> dict:
        """Get performance report for last N hours"""
        conn = sqlite3.connect(self.learner.db.db_path)
        
        since = datetime.now() - timedelta(hours=hours)
        
        # Get trades
        df = conn.execute("""
            SELECT * FROM performance
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (since.isoformat(),)).fetchall()
        
        if not df:
            return {"error": "No trades in selected period"}
        
        columns = [desc[0] for desc in conn.execute("PRAGMA table_info(performance)").fetchall()]
        
        import pandas as pd
        df = pd.DataFrame(df, columns=columns)
        
        # Calculate metrics
        total_trades = len(df)
        wins = df[df['win'] == 1]
        losses = df[df['win'] == 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        total_pnl = df['pnl_pct'].sum()
        profit_factor = abs(wins['pnl_pct'].sum() / losses['pnl_pct'].sum()) if len(losses) > 0 and losses['pnl_pct'].sum() != 0 else 0
        
        # Best and worst trades
        best_trade = df.loc[df['pnl_pct'].idxmax()] if len(df) > 0 else None
        worst_trade = df.loc[df['pnl_pct'].idxmin()] if len(df) > 0 else None
        
        # Sharpe ratio (simplified)
        returns = df['pnl_pct']
        sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + df['pnl_pct']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        conn.close()
        
        return {
            "period_hours": hours,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "best_trade": {
                "pnl_pct": best_trade['pnl_pct'] if best_trade is not None else 0,
                "symbol": json.loads(best_trade['metadata']).get('symbol', 'unknown') if best_trade is not None else 'unknown'
            } if best_trade is not None else None,
            "worst_trade": {
                "pnl_pct": worst_trade['pnl_pct'] if worst_trade is not None else 0,
                "symbol": json.loads(worst_trade['metadata']).get('symbol', 'unknown') if worst_trade is not None else 'unknown'
            } if worst_trade is not None else None
        }
    
    def get_strategy_performance(self) -> list:
        """Get performance of all strategies"""
        conn = sqlite3.connect(self.learner.db.db_path)
        
        strategies = conn.execute("""
            SELECT strategy_id, fitness, win_rate, avg_pnl, sharpe, 
                   max_dd, total_trades, generation, created_at
            FROM strategies
            WHERE active = TRUE
            ORDER BY fitness DESC
            LIMIT 10
        """).fetchall()
        
        result = []
        for s in strategies:
            result.append({
                "strategy_id": s[0],
                "fitness": s[1],
                "win_rate": s[2],
                "avg_pnl": s[3],
                "sharpe": s[4],
                "max_dd": s[5],
                "total_trades": s[6],
                "generation": s[7],
                "created_at": s[8]
            })
        
        conn.close()
        return result
    
    def get_learning_progress(self) -> dict:
        """Get learning progress over time"""
        status = self.learner.get_status()
        
        return {
            "current_generation": status['current_generation'],
            "total_trades": status['total_trades'],
            "exploration_rate": status['exploration_rate'],
            "daily_progress": {
                "target": 0.05,
                "current": status['daily_pnl'],
                "remaining": 0.05 - status['daily_pnl'],
                "progress_pct": (status['daily_pnl'] / 0.05) * 100 if status['daily_pnl'] > 0 else 0
            },
            "parameters": status['current_parameters']
        }
    
    def print_dashboard(self):
        """Print comprehensive dashboard"""
        print("\n" + "="*80)
        print("📊 AUTO-LEARNING MONITOR DASHBOARD")
        print("="*80)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Learning Progress
        print("\n🧠 LEARNING PROGRESS")
        print("-" * 80)
        progress = self.get_learning_progress()
        print(f"  Generación actual: {progress['current_generation']}")
        print(f"  Total trades: {progress['total_trades']}")
        print(f"  Exploration rate: {progress['exploration_rate']:.3f}")
        
        daily = progress['daily_progress']
        print(f"\n  📈 Progreso Diario:")
        print(f"     Target: {daily['target']:.2%}")
        print(f"     Current: {daily['current']:.2%}")
        print(f"     Remaining: {daily['remaining']:.2%}")
        print(f"     Progress: {daily['progress_pct']:.1f}%")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * min(daily['progress_pct'] / 100, 1))
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"     [{bar}] {daily['progress_pct']:.1f}%")
        
        # Performance Report
        print("\n💰 PERFORMANCE (Last 24h)")
        print("-" * 80)
        perf = self.get_performance_report(hours=24)
        
        if "error" not in perf:
            print(f"  Total trades: {perf['total_trades']}")
            print(f"  Win rate: {perf['win_rate']:.2%}")
            print(f"  Profit factor: {perf['profit_factor']:.2f}")
            print(f"  Sharpe ratio: {perf['sharpe_ratio']:.2f}")
            print(f"  Max drawdown: {perf['max_drawdown']:.2%}")
            print(f"  Total PnL: {perf['total_pnl']:.2%}")
            
            if perf['best_trade']:
                print(f"\n  ✅ Best trade: {perf['best_trade']['pnl_pct']:+.2%} ({perf['best_trade']['symbol']})")
            if perf['worst_trade']:
                print(f"  ❌ Worst trade: {perf['worst_trade']['pnl_pct']:+.2%} ({perf['worst_trade']['symbol']})")
        else:
            print(f"  {perf['error']}")
        
        # Top Strategies
        print("\n🧬 TOP STRATEGIES")
        print("-" * 80)
        strategies = self.get_strategy_performance()
        
        if strategies:
            for i, s in enumerate(strategies[:5], 1):
                print(f"  #{i} - Gen {s['generation']} | "
                      f"Fitness: {s['fitness']:.4f} | "
                      f"Win: {s['win_rate']:.1%} | "
                      f"Sharpe: {s['sharpe']:.2f} | "
                      f"Trades: {s['total_trades']}")
        else:
            print("  No strategies evolved yet")
        
        # Current Parameters
        print("\n⚙️  CURRENT PARAMETERS")
        print("-" * 80)
        params = progress['parameters']
        print(f"  Stop Loss: {params['sl_pct']:.2%}")
        print(f"  Take Profit: {params['tp_pct']:.2%}")
        print(f"  Position Size: {params['position_size']:.1%}")
        print(f"  Leverage: {params['leverage']:.1f}x")
        print(f"  Confidence Threshold: {params['confidence_threshold']:.2f}")
        print(f"  Risk per Trade: {params['risk_per_trade']:.1%}")
        
        print("\n" + "="*80 + "\n")
    
    def monitor_loop(self, interval: int = 60):
        """Continuous monitoring loop"""
        print("\n🔍 Starting continuous monitoring...")
        print(f"   Updates every {interval} seconds")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.print_dashboard()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Learning Monitor")
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    parser.add_argument('--once', action='store_true', help='Show dashboard once and exit')
    
    args = parser.parse_args()
    
    monitor = LearningMonitor()
    
    if args.once:
        monitor.print_dashboard()
    else:
        monitor.monitor_loop(args.interval)
