#!/usr/bin/env python3
"""
Auto-Learning Trading System - Simplified (No Threading)
========================================================
Versión simplificada sin threads daemon para evitar deadlocks.
"""

import sys
import time
import json
import signal
import logging
from pathlib import Path
from datetime import datetime

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/auto_learning_simplified.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("auto_learning_simple")

# Global variables
running = True
master = None
auto_learner = None
integration = None

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    global running, integration, auto_learner
    logger.info("🛑 Shutdown signal received...")
    running = False

    try:
        # Stop integration
        if integration:
            integration.running = False

        # Stop and save auto-learner
        if auto_learner:
            auto_learner.running = False
            auto_learner._save_state()

        logger.info("✅ System stopped gracefully")
    except Exception as e:
        logger.error(f"❌ Error stopping: {e}")

    sys.exit(0)

def main():
    global running, master, auto_learner, integration

    print("="*80)
    print("🚀 AUTO-LEARNING SIMPLIFIED (NO THREADING)")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Import without thread dependencies
        print("📦 Importing modules...")
        from master_orchestrator import MasterOrchestrator
        from auto_learner import AutoLearningOrchestrator
        print("✅ Modules imported")

        # Initialize master
        print("\n📊 Initializing master orchestrator...")
        master = MasterOrchestrator()
        print("✅ Master orchestrator initialized")

        # Initialize auto-learner
        print("\n🧠 Initializing auto-learning...")
        auto_learner = AutoLearningOrchestrator()
        auto_learner.running = True  # Set running without starting threads
        print("✅ Auto-learning initialized")

        # Create simple integration wrapper
        class SimpleIntegration:
            def __init__(self, master, learner):
                self.master = master
                self.learner = learner
                self.running = True
                self.trades_since_evolution = 0
                self.last_adaptation = time.time()
                self.last_evolution = time.time()

            def get_status(self):
                learner_status = self.learner.get_status()
                return {
                    "running": self.running,
                    "generation": learner_status.get("current_generation", 0),
                    "total_trades": learner_status.get("total_trades", 0),
                    "daily_pnl": learner_status.get("daily_pnl", 0.0),
                    "total_pnl": learner_status.get("total_pnl", 0.0),
                    "exploration_rate": learner_status.get("exploration_rate", 0.3),
                    "trades_since_evolution": self.trades_since_evolution,
                    "best_strategy": learner_status.get("best_strategy")
                }

            def get_config(self):
                return {
                    "capital": 500.0,
                    "target_daily": 0.05,
                    "max_drawdown": 0.10,
                    "evolution_interval": 3600,
                    "adaptation_interval": 300,
                }

            def get_current_parameters(self):
                return self.learner.param_optimizer.get_current_parameters()

            def run_loop(self):
                """Main loop without threads"""
                while self.running:
                    time.sleep(10)  # Check every 10 seconds

                    # Adaptation every 5 minutes
                    if time.time() - self.last_adaptation >= 300:
                        logger.info("🔄 Running adaptation...")
                        self.learner.param_optimizer.optimize_parameters(
                            self.learner.get_recent_trades(50)
                        )
                        self.last_adaptation = time.time()

                    # Evolution every hour
                    if time.time() - self.last_evolution >= 3600:
                        logger.info("🧬 Running evolution...")
                        self.learner.evolve_strategies()
                        self.trades_since_evolution = 0
                        self.last_evolution = time.time()

                    # Check performance
                    status = self.learner.get_status()
                    if status['daily_pnl'] >= 0.05:
                        logger.info(f"🎉 Daily target reached! PnL: {status['daily_pnl']:.2%}")
                    elif status['daily_pnl'] <= -0.10:
                        logger.warning(f"⚠️ Max drawdown! PnL: {status['daily_pnl']:.2%}")

        integration = SimpleIntegration(master, auto_learner)
        print("✅ Integration created (simplified, no threads)")

        # Show initial status
        print("\n📈 System Status:")
        status = integration.get_status()
        print(f"  Running: {status['running']}")
        print(f"  Generation: {status['generation']}")
        print(f"  Total Trades: {status['total_trades']}")
        print(f"  Daily PnL: {status['daily_pnl']:.2%}")
        print(f"  Exploration Rate: {status['exploration_rate']:.3f}")

        print("\n🎯 Configuration:")
        config = integration.get_config()
        print(f"  Capital: ${config['capital']}")
        print(f"  Target Daily: {config['target_daily']*100:.1f}%")
        print(f"  Max Drawdown: {config['max_drawdown']*100:.1f}%")

        print("\n🧠 Auto-Learning Features:")
        print("  ✅ Genetic Algorithm Evolution (simplified)")
        print("  ✅ Reinforcement Learning")
        print("  ✅ Adaptive Parameters")
        print("  ✅ Performance Feedback Loop")
        print("  ✅ Risk Management")

        print("\n📊 Trading Strategy:")
        params = integration.get_current_parameters()
        print(f"  Stop Loss: {params['sl_pct']*100:.1f}%")
        print(f"  Take Profit: {params['tp_pct']*100:.1f}%")
        print(f"  Position Size: {params['position_size']*100:.1f}%")
        print(f"  Leverage: {params['leverage']:.1f}x")

        print()
        print("="*80)
        print("✅ SYSTEM ACTIVE AND RUNNING (NO THREADING)")
        print("="*80)
        print()
        print("📊 The system is now:")
        print("  • Running in single thread (no daemon threads)")
        print("  • Learning from every trade")
        print("  • Adapting parameters every 5 minutes")
        print("  • Evolving strategies every hour")
        print("  • Monitoring performance continuously")
        print()
        print("🎯 Target: 5% daily")
        print("🛡️ Max Drawdown: 10%")
        print("🧬 Auto-learning: ACTIVE")
        print()
        print("📊 Monitor commands:")
        print("  tail -f /tmp/auto_learning_simplified.log")
        print()
        print("🛑 Press Ctrl+C to stop")
        print("="*80)
        print()

        # Main loop
        update_count = 0
        last_update = time.time()

        while running:
            try:
                # Run integration loop
                integration.run_loop()

                # Update every 5 minutes
                if time.time() - last_update >= 300:
                    update_count += 1
                    last_update = time.time()

                    status = integration.get_status()

                    print(f"\n{'='*80}")
                    print(f"📊 UPDATE #{update_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*80}")
                    print(f"  Generation: {status['generation']}")
                    print(f"  Total Trades: {status['total_trades']}")
                    print(f"  Daily PnL: {status['daily_pnl']:.2%}")
                    print(f"  Total PnL: ${status['total_pnl']:.2f}")
                    print(f"  Exploration: {status['exploration_rate']:.3f}")
                    print(f"  Best Strategy: {status.get('best_strategy', 'None')}")

                    if status['daily_pnl'] >= 0.05:
                        print("\n🎉 DAILY TARGET REACHED!")

                    if status['daily_pnl'] <= -0.10:
                        print("\n⚠️ MAX DRAWDOWN WARNING!")

                    print(f"{'='*80}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received...")
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    print("\n✅ System shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
