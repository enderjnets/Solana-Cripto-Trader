#!/usr/bin/env python3
"""
Auto-Learning System - NO THREADING VERSION
============================================
Versión completamente sin threads daemon.
Usa un loop principal simple que ejecuta evolución y adaptación periódicamente.
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/auto_learning_no_threads.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("auto_learning_simple")

# Global variables
running = True
master = None
auto_learner = None

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    global running
    logger.info("🛑 Shutdown signal received...")
    running = False

def main():
    global running, master, auto_learner

    print("="*80)
    print("🚀 AUTO-LEARNING SYSTEM - NO THREADING VERSION")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Import modules
        print("📦 Importing modules...")
        from master_orchestrator import MasterOrchestrator
        from auto_learner import AutoLearningOrchestrator
        print("✅ Modules imported")

        # Initialize master
        print("\n📊 Initializing master orchestrator...")
        master = MasterOrchestrator()
        print("✅ Master orchestrator initialized")

        # Initialize auto-learner (WITHOUT starting threads)
        print("\n🧠 Initializing auto-learning...")
        auto_learner = AutoLearningOrchestrator()
        auto_learner.running = True  # Set flag but don't start threads
        print("✅ Auto-learning initialized (no threads)")

        # Show initial status
        print("\n📈 Initial Status:")
        status = auto_learner.get_status()
        print(f"  Generation: {status['current_generation']}")
        print(f"  Total Trades: {status['total_trades']}")
        print(f"  Daily PnL: {status['daily_pnl']:.2%}")
        print(f"  Exploration Rate: {status['exploration_rate']:.3f}")

        print("\n🎯 Configuration:")
        print(f"  Target Daily: 5%")
        print(f"  Max Drawdown: 10%")
        print(f"  Evolution Interval: 1 hour")
        print(f"  Adaptation Interval: 5 minutes")

        print()
        print("="*80)
        print("✅ SYSTEM ACTIVE (NO THREADING)")
        print("="*80)
        print()
        print("📊 The system is now:")
        print("  • Running in main thread (no daemon threads)")
        print("  • Adapting parameters every 5 minutes")
        print("  • Evolving strategies every hour")
        print("  • Monitoring performance continuously")
        print()
        print("🎯 Target: 5% daily")
        print("🛡️ Max Drawdown: 10%")
        print("🧬 Auto-learning: ACTIVE")
        print()
        print("📊 Monitor commands:")
        print("  tail -f /tmp/auto_learning_no_threads.log")
        print()
        print("🛑 Press Ctrl+C to stop")
        print("="*80)
        print()

        # Main loop - run evolution and adaptation in main thread
        last_adaptation = time.time()
        last_evolution = time.time()
        update_count = 0

        while running:
            # Sleep for 10 seconds
            time.sleep(10)

            # Adaptation every 5 minutes
            if time.time() - last_adaptation >= 300:
                logger.info("🔄 Running adaptation...")
                try:
                    # Get recent trades
                    recent_trades = auto_learner.get_recent_trades(50)
                    if recent_trades:
                        auto_learner.param_optimizer.optimize_parameters(recent_trades)
                        logger.info("✅ Adaptation completed")
                    else:
                        logger.info("⚠️ No recent trades for adaptation")
                except Exception as e:
                    logger.error(f"❌ Adaptation error: {e}")

                last_adaptation = time.time()

            # Evolution every hour
            if time.time() - last_evolution >= 3600:
                logger.info("🧬 Running evolution...")
                try:
                    result = auto_learner.evolve_strategies()
                    if result:
                        logger.info("✅ Evolution completed")
                    else:
                        logger.info("⚠️ Evolution not executed")
                except Exception as e:
                    logger.error(f"❌ Evolution error: {e}")

                last_evolution = time.time()

            # Status update every 5 minutes
            if time.time() - last_adaptation >= 300:
                update_count += 1
                status = auto_learner.get_status()

                print(f"\n{'='*80}")
                print(f"📊 UPDATE #{update_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")
                print(f"  Generation: {status['generation']}")
                print(f"  Total Trades: {status['total_trades']}")
                print(f"  Daily PnL: {status['daily_pnl']:.2%}")
                print(f"  Total PnL: ${status['total_pnl']:.2f}")
                print(f"  Exploration: {status['exploration_rate']:.3f}")

                if status['daily_pnl'] >= 0.05:
                    print("\n🎉 DAILY TARGET REACHED!")

                if status['daily_pnl'] <= -0.10:
                    print("\n⚠️ MAX DRAWDOWN WARNING!")

                print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received...")
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    # Cleanup
    if auto_learner:
        try:
            auto_learner.running = False
            auto_learner._save_state()
            print("✅ Auto-learner stopped")
        except Exception as e:
            logger.error(f"Error stopping auto-learner: {e}")

    print("\n✅ System shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
