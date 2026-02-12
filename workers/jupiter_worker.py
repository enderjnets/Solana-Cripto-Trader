#!/usr/bin/env python3
"""
Jupiter Worker for Solana Trading Bot
===================================
Worker process that monitors prices and executes trading tasks.

Features:
- Real-time price monitoring
- Price alert system
- Auto-swap execution
- Health checks
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("jupiter_worker")


@dataclass
class WorkerState:
    """State of a worker"""
    worker_id: str
    status: str = "stopped"  # stopped, running, error
    last_run: Optional[datetime] = None
    price_alerts: Dict = field(default_factory=dict)
    last_price: float = 0.0
    swap_count: int = 0


class PriceMonitor:
    """Monitor token prices in real-time"""
    
    def __init__(self, tokens: Dict[str, str]):
        self.tokens = tokens
        self.prices: Dict[str, float] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable] = []
    
    def start(self, interval: float = 5.0):
        """Start monitoring prices"""
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self._thread.start()
        logger.info(f"Price monitor started for {len(self.tokens)} tokens")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Price monitor stopped")
    
    def _monitor_loop(self, interval: float):
        """Background price monitoring loop"""
        import httpx
        
        while self.running:
            try:
                # Fetch prices from Jupiter
                ids = "%2C".join(self.tokens.values())
                url = f"https://lite-api.jup.ag/price/v3?ids={ids}"
                
                resp = httpx.get(url, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    for symbol, addr in self.tokens.items():
                        if addr in data:
                            self.prices[symbol] = data[addr].get('usdPrice', 0)
                    
                    # Notify callbacks
                    for cb in self._callbacks:
                        try:
                            cb(self.prices)
                        except Exception as e:
                            logger.debug(f"Callback error: {e}")
                
            except Exception as e:
                logger.error(f"Price fetch error: {e}")
            
            time.sleep(interval)
    
    def add_callback(self, callback: Callable):
        """Add callback for price updates"""
        self._callbacks.append(callback)
    
    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        return self.prices.get(symbol, 0.0)


class JupiterWorker:
    """
    Worker that monitors prices and can execute trades.
    """
    
    # Token addresses
    TOKENS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYW",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP",
        "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
        "PYTH": "HZ1JovNiBEgZ1W7E2hKQzF8Tz3G6fZ6K3jKGn1c3bY7V",
    }
    
    def __init__(self, worker_id: str = None):
        self.worker_id = worker_id or f"worker_{os.getpid()}"
        self.state = WorkerState(worker_id=self.worker_id)
        self.price_monitor = PriceMonitor(self.TOKENS)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info(f"Worker {self.worker_id} shutting down...")
        self.stop()
    
    def start(self):
        """Start the worker"""
        if self._running:
            logger.warning(f"Worker {self.worker_id} already running")
            return
        
        self._running = True
        self.state.status = "running"
        self.state.last_run = datetime.now()
        
        # Start price monitor
        self.price_monitor.start(interval=5.0)
        
        # Start worker thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop the worker"""
        self._running = False
        self.state.status = "stopped"
        self.price_monitor.stop()
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _run_loop(self):
        """Main worker loop"""
        import httpx
        
        while self._running:
            try:
                # Get latest prices
                prices = self.price_monitor.prices
                self.state.last_price = prices.get("SOL", 0)
                
                # Check price alerts
                for token, alert_price in self.state.price_alerts.items():
                    current_price = prices.get(token, 0)
                    if current_price > 0:
                        logger.debug(f"{token}: ${current_price:.4f} (alert: ${alert_price:.4f})")
                
                # Sleep
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(5)
    
    def get_status(self) -> Dict:
        """Get worker status"""
        return {
            "worker_id": self.worker_id,
            "status": self.state.status,
            "last_run": self.state.last_run.isoformat() if self.state.last_run else None,
            "last_price": self.state.last_price,
            "prices": self.price_monitor.prices,
            "swap_count": self.state.swap_count
        }
    
    def add_price_alert(self, token: str, price: float):
        """Add price alert"""
        self.state.price_alerts[token] = price
        logger.info(f"Price alert added: {token} @ ${price:.4f}")
    
    def get_quote(self, input_token: str, output_token: str, amount: float) -> Optional[Dict]:
        """Get swap quote"""
        import httpx
        
        try:
            input_mint = self.TOKENS.get(input_token)
            output_mint = self.TOKENS.get(output_token)
            
            if not input_mint or not output_mint:
                logger.error(f"Unknown token: {input_token} or {output_token}")
                return None
            
            # Convert amount
            decimals = 9 if input_token == "SOL" else 6
            amount_lamports = int(amount * (10 ** decimals))
            
            url = f"https://lite-api.jup.ag/ultra/v1/order?inputMint={input_mint}&outputMint={output_mint}&amount={amount_lamports}"
            resp = httpx.get(url, timeout=30.0)
            
            if resp.status_code == 200:
                return resp.json()
            return None
            
        except Exception as e:
            logger.error(f"Quote error: {e}")
            return None


# Global worker instance for dashboard
_worker_instance: Optional[JupiterWorker] = None
_worker_lock = threading.Lock()


def get_worker() -> JupiterWorker:
    """Get or create global worker instance"""
    global _worker_instance
    with _worker_lock:
        if _worker_instance is None:
            _worker_instance = JupiterWorker()
        return _worker_instance


def worker_status() -> Dict:
    """Get worker status"""
    return get_worker().get_status()


def worker_start():
    """Start worker"""
    get_worker().start()
    return "Worker started"


def worker_stop():
    """Stop worker"""
    get_worker().stop()
    return "Worker stopped"


def worker_quote(input_token: str, output_token: str, amount: float) -> Dict:
    """Get quote from worker"""
    return get_worker().get_quote(input_token, output_token, amount)


if __name__ == "__main__":
    print("=" * 60)
    print("Jupiter Worker - Solana Trading Bot")
    print("=" * 60)
    
    worker = get_worker()
    
    import argparse
    parser = argparse.ArgumentParser(description="Jupiter Worker")
    parser.add_argument("--start", action="store_true", help="Start worker")
    parser.add_argument("--stop", action="store_true", help="Stop worker")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--quote", nargs=3, metavar=("IN", "OUT", "AMT"), help="Get quote")
    
    args = parser.parse_args()
    
    if args.start:
        worker.start()
        print("Worker started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(10)
                status = worker.get_status()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                print(f"  Status: {status['status']}")
                print(f"  SOL Price: ${status['last_price']:.2f}")
                print(f"  Prices: {len(status['prices'])} tokens")
        except KeyboardInterrupt:
            worker.stop()
    
    elif args.stop:
        worker.stop()
        print("Worker stopped")
    
    elif args.status:
        import json
        print(json.dumps(worker.get_status(), indent=2, default=str))
    
    elif args.quote:
        input_token, output_token, amount = args.quote
        amount = float(amount)
        result = worker.get_quote(input_token, output_token, amount)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Failed to get quote")
