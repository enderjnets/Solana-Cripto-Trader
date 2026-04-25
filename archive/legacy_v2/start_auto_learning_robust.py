#!/usr/bin/env python3
"""
Auto-Learning Trading System - Robust Startup Script
=====================================================
Versión mejorada con mejor manejo de señales y menos bloqueo.
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

# Configure logging - CRITICAL only to reduce log spam
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/auto_learning_trading_robust.log'),
    ]
)

logger = logging.getLogger("auto_learning_startup")

# Global variables
integration = None
running = True
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    global running, shutdown_requested, integration
    logger.info(f"🛑 Shutdown signal received ({signum})...")
    shutdown_requested = True

    if integration:
        try:
            integration.stop()
            logger.info("✅ Integration stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping integration: {e}")

    running = False
    sys.exit(0)

def main():
    global integration, running, shutdown_requested

    print("="*80)
    print("🚀 AUTO-LEARNING TRADING SYSTEM - STARTUP (ROBUST)")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Import with error handling
        from integrate_auto_learner import AutoLearningIntegration
        from master_orchestrator import MasterOrchestrator

        # Initialize master
        print("📊 Initializing master orchestrator...")
        try:
            master = MasterOrchestrator()
            print("✅ Master orchestrator initialized")
        except Exception as e:
            print(f"❌ Failed to initialize master: {e}")
            return 1

        # Initialize integration
        print()
        print("🧠 Initializing auto-learning integration...")
        try:
            integration = AutoLearningIntegration(master)
            print("✅ Auto-learning integration initialized")
        except Exception as e:
            print(f"❌ Failed to initialize integration: {e}")
            return 1

        # Start integration
        print()
        print("🔄 Starting auto-learning trading system...")
        try:
            integration.start()
            print("✅ Auto-learning trading system ACTIVE")
        except Exception as e:
            print(f"❌ Failed to start integration: {e}")
            return 1

        # Show initial status
        print()
        print("📈 System Status:")
        try:
            status = integration.get_status()
            print(f"  Running: {status['running']}")
            print(f"  Generation: {status['generation']}")
            print(f"  Total Trades: {status['total_trades']}")
            print(f"  Daily PnL: {status['daily_pnl']:.2%}")
            print(f"  Exploration Rate: {status['exploration_rate']:.3f}")
        except Exception as e:
            print(f"  ⚠️  Could not get status: {e}")

        print()
        print("🎯 Configuration:")
        try:
            config = integration.get_config()
            print(f"  Capital: ${config['capital']}")
            print(f"  Target Daily: {config['target_daily']*100:.1f}%")
            print(f"  Max Drawdown: {config['max_drawdown']*100:.1f}%")
        except Exception as e:
            print(f"  ⚠️  Could not get config: {e}")

        print()
        print("🧠 Auto-Learning Features:")
        print("  ✅ Genetic Algorithm Evolution")
        print("  ✅ Reinforcement Learning")
        print("  ✅ Adaptive Parameters")
        print("  ✅ Performance Feedback Loop")
        print("  ✅ Risk Management")

        print()
        print("📊 Trading Strategy:")
        try:
            params = integration.get_current_parameters()
            print(f"  Stop Loss: {params['sl_pct']*100:.1f}%")
            print(f"  Take Profit: {params['tp_pct']*100:.1f}%")
            print(f"  Position Size: {params['position_size']*100:.1f}%")
            print(f"  Leverage: {params['leverage']:.1f}x")
        except Exception as e:
            print(f"  ⚠️  Could not get parameters: {e}")

        print()
        print("="*80)
        print("✅ SYSTEM ACTIVE AND TRADING")
        print("="*80)
        print()
        print("📊 The system is now:")
        print("  • Executing trades with auto-learning")
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
        print("  tail -f /tmp/auto_learning_trading_robust.log")
        print("  python3 monitor_auto_learning.py")
        print()
        print("🛑 Press Ctrl+C to stop")
        print("="*80)
        print()

        # Main loop - use shorter sleeps for better responsiveness
        update_count = 0
        last_update = time.time()

        while running and not shutdown_requested:
            # Short sleep for better signal handling
            time.sleep(10)  # Check every 10 seconds

            if shutdown_requested:
                break

            # Update every 5 minutes
            if time.time() - last_update >= 300:
                update_count += 1
                last_update = time.time()

                try:
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

                    # Check for important events
                    if status['daily_pnl'] >= 0.05:
                        print("\n🎉 DAILY TARGET REACHED!")

                    if status['daily_pnl'] <= -0.10:
                        print("\n⚠️ MAX DRAWDOWN WARNING!")

                    print(f"{'='*80}\n")

                except Exception as e:
                    print(f"⚠️  Error getting status: {e}")
                    logger.error(f"Error getting status: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received...")
        if integration:
            integration.stop()
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    # Cleanup
    if integration:
        try:
            integration.stop()
            print("✅ Integration stopped")
        except:
            pass

    print("\n✅ System shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
