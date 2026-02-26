#!/usr/bin/env python3
"""
Auto-Learning Trading System - Startup Script
==============================================
Inicia el sistema de trading con auto-aprendizaje integrado.
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
        logging.FileHandler('/tmp/auto_learning_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("auto_learning_startup")

# Global variables
integration = None
running = True

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    global running, integration
    logger.info("🛑 Shutdown signal received...")
    running = False
    if integration:
        integration.stop()
    logger.info("✅ Stopped gracefully")
    sys.exit(0)

def main():
    global integration, running
    
    print("="*80)
    print("🚀 AUTO-LEARNING TRADING SYSTEM - STARTUP")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Import integration
        from integrate_auto_learner import AutoLearningIntegration
        from master_orchestrator import MasterOrchestrator
        
        print("📊 Initializing master orchestrator...")
        master = MasterOrchestrator()
        print("✅ Master orchestrator initialized")
        
        print()
        print("🧠 Initializing auto-learning integration...")
        integration = AutoLearningIntegration(master)
        print("✅ Auto-learning integration initialized")
        
        print()
        print("🔄 Starting auto-learning trading system...")
        integration.start()
        print("✅ Auto-learning trading system ACTIVE")
        
        print()
        print("📈 System Status:")
        status = integration.get_status()
        print(f"  Running: {status['running']}")
        print(f"  Generation: {status['generation']}")
        print(f"  Total Trades: {status['total_trades']}")
        print(f"  Daily PnL: {status['daily_pnl']:.2%}")
        print(f"  Exploration Rate: {status['exploration_rate']:.3f}")
        
        print()
        print("🎯 Configuration:")
        config = integration.get_config()
        print(f"  Capital: ${config['capital']}")
        print(f"  Target Daily: {config['target_daily']*100:.1f}%")
        print(f"  Max Drawdown: {config['max_drawdown']*100:.1f}%")
        
        print()
        print("🧠 Auto-Learning Features:")
        print("  ✅ Genetic Algorithm Evolution")
        print("  ✅ Reinforcement Learning")
        print("  ✅ Adaptive Parameters")
        print("  ✅ Performance Feedback Loop")
        print("  ✅ Risk Management")
        
        print()
        print("📊 Trading Strategy:")
        params = integration.get_current_parameters()
        print(f"  Stop Loss: {params['sl_pct']*100:.1f}%")
        print(f"  Take Profit: {params['tp_pct']*100:.1f}%")
        print(f"  Position Size: {params['position_size']*100:.1f}%")
        print(f"  Leverage: {params['leverage']:.1f}x")
        
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
        print("  tail -f /tmp/auto_learning_trading.log")
        print("  python3 monitor_auto_learning.py")
        print()
        print("🛑 Press Ctrl+C to stop")
        print("="*80)
        print()
        
        # Keep running and show updates
        update_count = 0
        while running:
            time.sleep(300)  # Update every 5 minutes
            update_count += 1
            
            if not running:
                break
            
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
        logger.error(f"❌ Error starting system: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
