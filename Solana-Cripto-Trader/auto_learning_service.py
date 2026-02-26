#!/usr/bin/env python3
"""
Simple Auto-Learning Service
============================
Versión minimalista que FUNCIONA sin threads daemon.
Implementa auto-learning básico de forma síncrona.
"""

import sys
import time
import json
import signal
import logging
from pathlib import Path
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/auto_learning_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("auto_learning_service")

# Global variables
running = True

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    global running
    logger.info("🛑 Shutdown signal received...")
    running = False

def load_state():
    """Load current trading state"""
    state_file = Path.home() / ".config/solana-jupiter-bot/master_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}

def save_learner_state(state):
    """Save learner state"""
    state_file = Path(__file__).parent / "data/learner_state.json"
    state_file.parent.mkdir(exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def simple_adaptation(state):
    """Simple parameter adaptation based on performance"""
    daily_pnl = state.get('daily_pnl', 0)

    # Adjust exploration rate based on performance
    if daily_pnl > 0.03:  # 3% gain
        exploration_rate = max(0.1, 0.3 - (daily_pnl * 2))
    elif daily_pnl < -0.05:  # 5% loss
        exploration_rate = min(0.5, 0.3 + abs(daily_pnl))
    else:
        exploration_rate = 0.3

    return {
        'exploration_rate': exploration_rate,
        'last_adaptation': datetime.now().isoformat()
    }

def simple_evolution(state):
    """Simple strategy evolution"""
    win_rate = state.get('stats', {}).get('win_rate', 50)

    # Adjust parameters based on win rate
    if win_rate > 60:
        logger.info("🧬 Evolution: High win rate - keeping current strategy")
    elif win_rate < 40:
        logger.info("🧬 Evolution: Low win rate - increasing exploration")
    else:
        logger.info("🧬 Evolution: Moderate win rate - fine-tuning")

    return {
        'generation': state.get('generation', 0) + 1,
        'last_evolution': datetime.now().isoformat()
    }

def main():
    global running

    print("="*80)
    print("🚀 AUTO-LEARNING SERVICE - SIMPLE VERSION")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize state
    learner_state = {
        'generation': 0,
        'exploration_rate': 0.3,
        'total_adaptations': 0,
        'total_evolutions': 0,
        'started_at': datetime.now().isoformat()
    }

    save_learner_state(learner_state)
    logger.info("✅ Auto-learning service initialized")
    logger.info(f"📊 Initial exploration rate: {learner_state['exploration_rate']}")

    # Main loop
    last_adaptation = time.time()
    last_evolution = time.time()
    loop_count = 0

    print("\n🎯 Target: 5% daily")
    print("🛡️ Max Drawdown: 10%")
    print("🧬 Auto-learning: ACTIVE")
    print("\n📊 Monitor: tail -f /tmp/auto_learning_service.log")
    print("🛑 Press Ctrl+C to stop")
    print("="*80)
    print()

    while running:
        try:
            loop_count += 1

            # Load current trading state
            trading_state = load_state()

            # Status update every loop (60 seconds)
            if loop_count % 6 == 0:  # Every minute
                capital = trading_state.get('paper_capital', 0)
                daily_pnl = trading_state.get('daily_pnl', 0)
                cycles = trading_state.get('cycles', 0)
                win_rate = trading_state.get('stats', {}).get('win_rate', 0)

                # Daily PnL is already in percentage format from master_orchestrator
                # (e.g., -0.37 means -0.37%, not -37%)
                logger.info(f"📊 Status - Capital: ${capital:.2f} | Daily PnL: {daily_pnl:.2f}% | Cycles: {cycles} | Win Rate: {win_rate}%")
                logger.info(f"🧬 Generation: {learner_state['generation']} | Exploration: {learner_state['exploration_rate']:.3f}")

                # Check targets (daily_pnl is in percentage: 5.0 = 5%)
                if daily_pnl >= 5.0:
                    logger.info("🎉🎉🎉 DAILY TARGET REACHED! +5% achieved!")

                if daily_pnl <= -10.0:
                    logger.warning("⚠️⚠️⚠️ MAX DRAWDOWN WARNING! -10% reached!")

            # Adaptation every 5 minutes
            if time.time() - last_adaptation >= 300:
                logger.info("🔄 Running parameter adaptation...")
                adaptation = simple_adaptation(trading_state)
                learner_state.update(adaptation)
                learner_state['total_adaptations'] += 1
                save_learner_state(learner_state)
                logger.info(f"✅ Adaptation completed - Exploration: {adaptation['exploration_rate']:.3f}")
                last_adaptation = time.time()

            # Evolution every hour
            if time.time() - last_evolution >= 3600:
                logger.info("🧬 Running strategy evolution...")
                evolution = simple_evolution(trading_state)
                learner_state.update(evolution)
                learner_state['total_evolutions'] += 1
                save_learner_state(learner_state)
                logger.info(f"✅ Evolution completed - Generation: {evolution['generation']}")
                last_evolution = time.time()

            # Sleep for 10 seconds
            time.sleep(10)

        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            time.sleep(30)  # Wait before retrying

    # Cleanup
    logger.info("💾 Saving final state...")
    learner_state['stopped_at'] = datetime.now().isoformat()
    save_learner_state(learner_state)
    logger.info("✅ Auto-learning service stopped")

    return 0

if __name__ == "__main__":
    sys.exit(main())
