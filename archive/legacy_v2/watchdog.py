#!/usr/bin/env python3
"""
Trading System Watchdog
Automatically restarts the trading system if it crashes
"""
import subprocess
import time
import sys
import os
import signal

SCRIPT_PATH = "/home/enderj/.openclaw/workspace/solana-jupiter-bot/unified_trading_system.py"
LOG_FILE = "/home/enderj/.openclaw/workspace/solana-jupiter-bot/watchdog.log"
CHECK_INTERVAL = 30  # Check every 30 seconds

def log(msg):
    """Log message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_msg)
    print(log_msg.strip())

def is_process_running():
    """Check if trading system is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python3 unified_trading"],
            capture_output=True,
            text=True
        )
        # Filter out watchdog process
        pids = [p for p in result.stdout.strip().split('\n') if p and int(p) != os.getpid()]
        return len(pids) > 0
    except:
        return False

def start_system():
    """Start the trading system"""
    log("🚀 Starting trading system...")
    subprocess.Popen(
        ["python3", SCRIPT_PATH, "--start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    time.sleep(5)  # Wait for system to initialize
    
def stop_system():
    """Stop the trading system"""
    log("🛑 Stopping trading system...")
    subprocess.run(["pkill", "-f", "unified_trading_system.py"], capture_output=True)

def main():
    log("🐕 Trading System Watchdog STARTED")
    log(f"Checking every {CHECK_INTERVAL} seconds")
    
    consecutive_failures = 0
    max_failures = 3
    
    while True:
        try:
            if not is_process_running():
                consecutive_failures += 1
                log(f"⚠️ System not running! Failure #{consecutive_failures}")
                
                if consecutive_failures >= max_failures:
                    log("🔄 Restarting system...")
                    start_system()
                    consecutive_failures = 0
            else:
                if consecutive_failures > 0:
                    log("✅ System is running")
                consecutive_failures = 0
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            log("🛑 Watchdog stopped by user")
            break
        except Exception as e:
            log(f"❌ Watchdog error: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
