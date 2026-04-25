#!/usr/bin/env python3
"""
Crypto Worker - Distributed Backtesting Client
==============================================
Worker client that polls coordinator and executes backtests.

Features:
- Polls coordinator for work
- Executes genetic algorithm backtests
- Submits results to coordinator
- Supports multiple instances per machine

Usage:
    python crypto_worker.py

Environment Variables:
    COORDINATOR_URL - Coordinator URL (default: http://localhost:5001)
    WORKER_INSTANCE - Instance number (default: 1)
    NUM_WORKERS - Total workers on this machine (default: 1)
    INTERVAL - Poll interval in seconds (default: 5)
"""

import os
import sys
import json
import time
import uuid
import socket
import platform
import logging
import requests
import threading
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("worker")

# ============================================================================
# CONFIGURATION
# ============================================================================
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:5001")
WORKER_INSTANCE = int(os.environ.get("WORKER_INSTANCE", 1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 1))
POLL_INTERVAL = int(os.environ.get("INTERVAL", 5))
USE_RAY = os.environ.get("USE_RAY", "false").lower() == "true"

# Generate unique worker ID
WORKER_ID = f"{socket.gethostname()}_{platform.system()}_W{WORKER_INSTANCE}"


# ============================================================================
# WORKER CLASS
# ============================================================================
class CryptoWorker:
    """
    Worker that executes backtesting tasks from coordinator.
    """
    
    def __init__(self):
        self.worker_id = WORKER_ID
        self.coordinator_url = COORDINATOR_URL
        self.running = True
        self.work_completed = 0
        self.total_execution_time = 0
        self.current_work = None
        
        # Import backtesting modules
        sys.path.insert(0, '.')
        self._init_backtester()
    
    def _init_backtester(self):
        """Initialize backtesting capabilities"""
        try:
            from data.historical_data import HistoricalDataManager
            self.manager = HistoricalDataManager()
            logger.info("✅ HistoricalDataManager loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load data manager: {e}")
            self.manager = None
        
        # Try to import genetic miner
        try:
            from strategies.genetic_miner import StrategyMiner
            self.has_miner = True
            logger.info("✅ StrategyMiner loaded")
        except Exception as e:
            self.has_miner = False
            logger.warning(f"⚠️ Could not load genetic miner: {e}")
    
    def register(self):
        """Register with coordinator"""
        try:
            resp = requests.post(
                f"{self.coordinator_url}/api/register_worker",
                json={
                    "worker_id": self.worker_id,
                    "hostname": socket.gethostname(),
                    "platform": platform.system()
                },
                timeout=10
            )
            if resp.status_code == 200:
                logger.info(f"✅ Registered with coordinator: {self.worker_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to register: {e}")
        return False
    
    def get_work(self):
        """Poll coordinator for work"""
        try:
            resp = requests.get(
                f"{self.coordinator_url}/api/get_work",
                params={"worker_id": self.worker_id},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("work")
        except Exception as e:
            logger.error(f"❌ Failed to get work: {e}")
        return None
    
    def execute_backtest(self, work):
        """Execute a backtest work unit"""
        start_time = time.time()
        work_type = work.get("work_type", "backtest")
        params = work.get("params", {})
        
        logger.info(f"🔄 Executing {work_type}: {json.dumps(params)[:100]}...")
        
        result = {
            "work_unit_id": work.get("id"),
            "worker_id": self.worker_id,
            "pnl": 0,
            "win_rate": 0,
            "sharpe": 0,
            "max_dd": 0,
            "trades": 0,
            "data": {}
        }
        
        try:
            if work_type == "genome_eval" and self.has_miner:
                # Evaluate a single genome
                result = self._eval_genome(params)
            elif work_type == "backtest":
                # Run full backtest
                result = self._run_backtest(params)
            elif work_type == "genetic_run":
                # Run genetic algorithm
                result = self._run_genetic(params)
            else:
                logger.warning(f"⚠️ Unknown work type: {work_type}")
                result["data"] = {"error": f"Unknown work type: {work_type}"}
        
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            result["data"] = {"error": str(e)}
        
        elapsed = time.time() - start_time
        result["execution_time"] = elapsed
        
        return result
    
    def _eval_genome(self, params):
        """Evaluate a single genome"""
        from strategies.genetic_miner import Genome, StrategyMiner
        
        genome = Genome.from_dict(params.get("genome", {}))
        
        # Get price data
        token = params.get("token", "SOL")
        df = self.manager.get_historical_data(token, timeframe="1h", days=90)
        
        # Create miner and evaluate
        miner = StrategyMiner(df, population_size=1, generations=1)
        pnl, win_rate, sharpe = miner.evaluate(genome)
        
        return {
            "work_unit_id": params.get("work_id"),
            "worker_id": self.worker_id,
            "pnl": pnl,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "trades": miner.evaluate(genome)[0] or 0,
            "data": {"genome": params.get("genome")}
        }
    
    def _run_backtest(self, params):
        """Run a full backtest"""
        token = params.get("token", "SOL")
        strategy = params.get("strategy", "rsi")
        days = params.get("days", 90)
        tp_pct = params.get("tp_pct", 0.05)
        sl_pct = params.get("sl_pct", 0.03)
        
        # Get data
        df = self.manager.get_historical_data(token, timeframe="1h", days=days)
        
        # Run backtest
        from strategies import StrategyFactory, backtest_strategy
        strat = StrategyFactory.create(strategy, params.get("strategy_params", {}))
        result = backtest_strategy(strat, df, tp_pct=tp_pct, sl_pct=sl_pct)
        
        return {
            "work_unit_id": params.get("id"),
            "worker_id": self.worker_id,
            "pnl": result.get("total_return", 0),
            "win_rate": result.get("win_rate", 0),
            "sharpe": 0,
            "max_dd": result.get("min_pnl", 0),
            "trades": result.get("total_trades", 0),
            "data": result
        }
    
    def _run_genetic(self, params):
        """Run genetic algorithm"""
        from strategies.genetic_miner import StrategyMiner
        
        token = params.get("token", "SOL")
        population = params.get("population", 20)
        generations = params.get("generations", 10)
        
        df = self.manager.get_historical_data(token, timeframe="1h", days=90)
        
        miner = StrategyMiner(df, population_size=population, generations=generations)
        result = miner.evolve(verbose=False)
        
        return {
            "work_unit_id": params.get("id"),
            "worker_id": self.worker_id,
            "pnl": result.get("best_pnl", 0),
            "win_rate": result.get("win_rate", 0),
            "sharpe": result.get("sharpe", 0),
            "trades": result.get("best_pnl", 0) or 0,
            "data": result
        }
    
    def submit_result(self, result):
        """Submit result to coordinator"""
        try:
            resp = requests.post(
                f"{self.coordinator_url}/api/submit_result",
                json=result,
                timeout=30
            )
            if resp.status_code == 200:
                self.work_completed += 1
                logger.info(f"✅ Submitted result #{self.work_completed}: PnL={result.get('pnl', 0):.4f}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to submit result: {e}")
        return False
    
    def heartbeat(self):
        """Send heartbeat to coordinator"""
        try:
            requests.post(
                f"{self.coordinator_url}/api/heartbeat",
                json={"worker_id": self.worker_id},
                timeout=10
            )
        except:
            pass
    
    def run(self):
        """Main worker loop"""
        logger.info(f"🚀 Worker {self.worker_id} starting...")
        
        # Register with coordinator
        if not self.register():
            logger.error("❌ Failed to register, exiting")
            return
        
        # Main loop
        while self.running:
            try:
                # Get work
                work = self.get_work()
                
                if work:
                    # Execute
                    result = self.execute_backtest(work)
                    
                    # Submit
                    self.submit_result(result)
                else:
                    # No work available, wait
                    time.sleep(POLL_INTERVAL)
                
                # Heartbeat
                self.heartbeat()
                
            except KeyboardInterrupt:
                logger.info("👋 Worker shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                time.sleep(POLL_INTERVAL)
        
        logger.info(f"👋 Worker {self.worker_id} stopped")


# ============================================================================
# MULTI-WORKER LAUNCHER
# ============================================================================
def launch_workers(n: int):
    """Launch N worker instances"""
    import subprocess
    
    processes = []
    
    for i in range(1, n + 1):
        env = os.environ.copy()
        env["WORKER_INSTANCE"] = str(i)
        env["NUM_WORKERS"] = str(n)
        
        cmd = [sys.executable, __file__]
        
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        logger.info(f"🚀 Launched worker {i}/{n} (PID: {p.pid})")
        time.sleep(2)  # Stagger launches
    
    return processes


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solana Trading Worker")
    parser.add_argument("--coordinator", "-c", help="Coordinator URL")
    parser.add_argument("--instance", "-i", type=int, help="Worker instance number")
    parser.add_argument("--workers", "-w", type=int, help="Number of workers to launch")
    parser.add_argument("--test", action="store_true", help="Run single test backtest")
    args = parser.parse_args()
    
    if args.coordinator:
        os.environ["COORDINATOR_URL"] = args.coordinator
    
    if args.instance:
        os.environ["WORKER_INSTANCE"] = str(args.instance)
    
    if args.test:
        # Single test run
        worker = CryptoWorker()
        test_result = worker.execute_backtest({
            "work_type": "backtest",
            "params": {"token": "SOL", "strategy": "rsi", "days": 30}
        })
        print(f"\n📊 Test Result:")
        print(json.dumps(test_result, indent=2))
    elif args.workers and args.workers > 1:
        # Launch multiple workers
        launch_workers(args.workers)
    else:
        # Single worker mode
        worker = CryptoWorker()
        worker.run()
