#!/usr/bin/env python3
"""
Auto-restart wrapper for trading system
"""
import subprocess
import time
import sys
import os

SCRIPT = "/home/enderj/.openclaw/workspace/solana-jupiter-bot/unified_trading_system.py"
LOG = "/home/enderj/.openclaw/workspace/solana-jupiter-bot/autorestart.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(LOG, "a") as f:
        f.write(line)
    print(line.strip())

def main():
    log("🚀 Auto-restart wrapper started")
    
    while True:
        log("Starting trading system...")
        
        proc = subprocess.Popen(
            ["python3", "-u", SCRIPT, "--start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        log(f"Process started with PID {proc.pid}")
        
        # Wait for process to exit
        try:
            proc.wait()
        except KeyboardInterrupt:
            log("Stopping...")
            proc.terminate()
            break
        
        log(f"Process died with code {proc.returncode}")
        log("Waiting 5 seconds before restart...")
        time.sleep(5)

if __name__ == "__main__":
    main()
