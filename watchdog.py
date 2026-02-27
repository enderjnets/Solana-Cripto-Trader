#!/usr/bin/env python3
"""
Trading Watchdog - Monitors and restarts trading system if it crashes
"""

import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Configuration
WATCHDOG_INTERVAL = 30  # Check every 30 seconds
TRADING_SCRIPT = Path(__file__).parent / "unified_trading_system.py"
LOG_FILE = Path(__file__).parent / "logs" / "watchdog.log"
HEARTBEAT_FILE = Path(__file__).parent / "data" / "heartbeat.txt"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("watchdog")

class TradingWatchdog:
    """Monitors trading system and restarts if crashed"""
    
    def __init__(self):
        self.running = True
        self.process = None
        self.start_time = None
        self.restart_count = 0
        self.max_restarts = 10
        self.restart_window = 3600  # 1 hour
        self.restarts = []
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Watchdog received signal {signum}, shutting down...")
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        sys.exit(0)
    
    def start_trading(self):
        """Start trading system"""
        try:
            logger.info("🚀 Starting trading system...")
            self.process = subprocess.Popen(
                [sys.executable, str(TRADING_SCRIPT), "--continuous"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=TRADING_SCRIPT.parent
            )
            self.start_time = time.time()
            logger.info(f"✅ Trading system started (PID: {self.process.pid})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            return False
    
    def check_trading(self):
        """Check if trading system is running"""
        if not self.process:
            return False
        
        # Check if process is alive
        poll = self.process.poll()
        if poll is not None:
            logger.warning(f"⚠️ Trading system exited with code {poll}")
            return False
        
        # Check heartbeat file
        if HEARTBEAT_FILE.exists():
            try:
                heartbeat_time = datetime.fromisoformat(HEARTBEAT_FILE.read_text().strip().split('\n')[0])
                age = (datetime.now() - heartbeat_time).total_seconds()
                if age > 60:  # No heartbeat for 60 seconds
                    logger.warning(f"⚠️ No heartbeat for {age:.0f}s, process may be hung")
                    return False
            except:
                pass
        
        return True
    
    def restart_trading(self):
        """Restart trading system"""
        # Clean up old restarts
        now = time.time()
        self.restarts = [r for r in self.restarts if now - r < self.restart_window]
        
        # Check if too many restarts
        if len(self.restarts) >= self.max_restarts:
            logger.error(f"❌ Too many restarts ({len(self.restarts)} in last hour), stopping watchdog")
            return False
        
        # Kill old process if exists
        if self.process and self.process.poll() is None:
            logger.info("🛑 Stopping old trading process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        # Start new process
        self.restarts.append(now)
        self.restart_count += 1
        logger.info(f"🔄 Restarting trading system (attempt {self.restart_count})...")
        return self.start_trading()
    
    def run(self):
        """Main watchdog loop"""
        logger.info("🦅 Trading Watchdog started")
        logger.info(f"📊 Monitoring interval: {WATCHDOG_INTERVAL}s")
        logger.info(f"📊 Max restarts: {self.max_restarts}/hour")
        
        # Start trading initially
        if not self.start_trading():
            logger.error("❌ Failed to start trading system, exiting")
            return
        
        # Main monitoring loop
        while self.running:
            try:
                time.sleep(WATCHDOG_INTERVAL)
                
                if not self.check_trading():
                    logger.warning("⚠️ Trading system not responding, restarting...")
                    if not self.restart_trading():
                        break
                else:
                    logger.debug("✅ Trading system running normally")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Watchdog error: {e}")
                time.sleep(5)
        
        logger.info("🛑 Watchdog stopped")

if __name__ == "__main__":
    watchdog = TradingWatchdog()
    watchdog.run()
