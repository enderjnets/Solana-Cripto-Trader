#!/usr/bin/env python3
"""
Watchdog for unified_trading_system.py
Automatically restarts the trading system if it crashes
"""
import os
import sys
import time
import subprocess
import signal

SCRIPT_PATH = "/home/enderj/.openclaw/workspace/solana-jupiter-bot/unified_trading_system.py"
LOG_PATH = "/home/enderj/.openclaw/workspace/solana-jupiter-bot/unified_trading_system.log"
PID_FILE = "/home/enderj/.openclaw/workspace/solana-jupiter-bot/watchdog.pid"
CHECK_INTERVAL = 60  # seconds

def is_running():
    """Check if the trading system is running"""
    try:
        result = subprocess.run(["pgrep", "-f", "unified_trading_system.py"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def start_system():
    """Start the trading system"""
    print(f"[WATCHDOG] Starting trading system...")
    log_file = open(LOG_PATH, "a")
    subprocess.Popen(
        ["python3", "-u", SCRIPT_PATH, "--continuous"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(SCRIPT_PATH),
        start_new_session=True
    )
    log_file.close()
    print(f"[WATCHDOG] Trading system started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    print(f"[WATCHDOG] Starting watchdog...")
    print(f"[WATCHDOG] Monitoring: {SCRIPT_PATH}")
    print(f"[WATCHDOG] Check interval: {CHECK_INTERVAL}s")
    
    consecutive_failures = 0
    max_failures = 3
    
    while True:
        if not is_running():
            consecutive_failures += 1
            print(f"[WATCHDOG] System not running! Failure #{consecutive_failures}")
            
            if consecutive_failures >= max_failures:
                print(f"[WATCHDOG] Multiple failures detected, restarting...")
                start_system()
                consecutive_failures = 0
            else:
                print(f"[WATCHDOG] Waiting {CHECK_INTERVAL}s before restart...")
                time.sleep(CHECK_INTERVAL)
        else:
            if consecutive_failures > 0:
                print(f"[WATCHDOG] System is healthy")
            consecutive_failures = 0
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
