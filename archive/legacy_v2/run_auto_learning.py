#!/usr/bin/env python3
"""
Auto-Learning System Launcher
==============================
Script dedicado para iniciar el sistema de auto-aprendizaje.
"""

import sys
import time
import signal
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from auto_learning_wrapper import get_wrapper

# Global wrapper
wrapper = None

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    global wrapper
    print('\n\n🛑 Shutdown signal received...')
    if wrapper:
        wrapper.stop()
        print('✅ Auto-learning stopped gracefully')
    sys.exit(0)

def main():
    global wrapper
    
    print('='*80)
    print('🚀 AUTO-LEARNING SYSTEM - STARTING')
    print('='*80)
    print(f'📅 {time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get wrapper
    print('\n📊 Initializing auto-learning wrapper...')
    wrapper = get_wrapper()
    print('✅ Wrapper initialized')
    
    # Start learning
    print('\n🧠 Starting auto-learning system...')
    wrapper.start()
    print('✅ Auto-learning ACTIVE')
    
    # Show initial status
    print('\n📈 Initial Status:')
    status = wrapper.get_status()
    print(f'  Generation: {status["current_generation"]}')
    print(f'  Total Trades: {status["total_trades"]}')
    print(f'  Daily PnL: {status["daily_pnl"]:.2%}')
    print(f'  Exploration Rate: {status["exploration_rate"]:.3f}')
    
    print('\n🎯 Configuration:')
    print(f'  Target Daily: 5%')
    print(f'  Max Drawdown: 10%')
    print(f'  Evolution Interval: 1 hour')
    print(f'  Adaptation Interval: 5 min')
    
    print('\n✅ System ACTIVE and learning...')
    print('📊 Monitoring enabled')
    print('🔄 Press Ctrl+C to stop\n')
    print('='*80)
    
    # Keep running and update periodically
    update_count = 0
    while True:
        time.sleep(300)  # Update every 5 minutes
        update_count += 1
        
        status = wrapper.get_status()
        print(f"\n{'='*80}")
        print(f"📊 UPDATE #{update_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"  Generation: {status['current_generation']}")
        print(f"  Total Trades: {status['total_trades']}")
        print(f"  Daily PnL: {status['daily_pnl']:.2%}")
        print(f"  Total PnL: ${status['total_pnl']:.2f}")
        print(f"  Exploration: {status['exploration_rate']:.3f}")
        
        # Check targets
        if status['daily_pnl'] >= 0.05:
            print("\n🎉 DAILY TARGET REACHED!")
        
        if status['daily_pnl'] <= -0.10:
            print("\n⚠️ MAX DRAWDOWN WARNING!")
        
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
