#!/usr/bin/env python3
"""
WebSocket Price Feed - Real-Time Prices via Binance WebSocket
=============================================================
Provides <1s latency price updates using Binance combined streams.
Falls back gracefully to REST price_feed if WebSocket is unavailable.

Usage:
    from api.websocket_feed import get_ws_price_feed
    
    feed = get_ws_price_feed()
    price = feed.get_price("SOLUSD")  # Compatible with existing interface
    
Architecture:
    - Background asyncio thread runs WebSocket connection
    - Thread-safe price cache updated in real-time
    - get_price() returns instantly from cache
    - Automatic reconnection on disconnect
    - Falls back to REST if WS hasn't received data
"""

import asyncio
import json
import logging
import threading
import time as time_module
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("websocket_feed")

# ============================================================================
# SYMBOL MAPPING
# ============================================================================

# Map our internal symbols to Binance stream names
# Internal format: "SOLUSD", "ETHUSD", etc.
# Binance format: "solusdt", "ethusdt", etc.

SYMBOL_TO_BINANCE = {
    "SOL": "solusdt",
    "ETH": "ethusdt",
    "BTC": "btcusdt",
    "BONK": "bonkusdt",
    "WIF": "wifusdt",
    "JUP": "jupusdt",
    "RAY": "rayusdt",
    "JTO": "jtousdt",
    "POPCAT": "popcatusdt",
}

# Reverse mapping: binance stream symbol → our symbol
BINANCE_TO_SYMBOL = {}
for sym, bsym in SYMBOL_TO_BINANCE.items():
    BINANCE_TO_SYMBOL[bsym.replace("usdt", "").upper()] = sym

# Build combined stream URL
BINANCE_STREAMS = [f"{s}@ticker" for s in SYMBOL_TO_BINANCE.values()]
BINANCE_WS_URL = f"wss://stream.binance.com:9443/stream?streams={'/'.join(BINANCE_STREAMS)}"


# ============================================================================
# PRICE CACHE (thread-safe)
# ============================================================================

class PriceCache:
    """Thread-safe in-memory price cache updated by WebSocket"""
    
    def __init__(self):
        self._prices: Dict[str, float] = {}
        self._timestamps: Dict[str, float] = {}
        self._changes_24h: Dict[str, float] = {}
        self._volumes: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def update(self, symbol: str, price: float, change_24h: float = 0.0, volume: float = 0.0):
        """Update price for a symbol (called from WS thread)"""
        with self._lock:
            self._prices[symbol] = price
            self._timestamps[symbol] = time_module.time()
            self._changes_24h[symbol] = change_24h
            self._volumes[symbol] = volume
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price. Returns None if stale (>120s) or missing."""
        with self._lock:
            if symbol not in self._prices:
                return None
            # Check staleness — 120s max
            age = time_module.time() - self._timestamps.get(symbol, 0)
            if age > 120:
                return None
            return self._prices[symbol]
    
    def get_age(self, symbol: str) -> float:
        """Get age of cached price in seconds"""
        with self._lock:
            ts = self._timestamps.get(symbol, 0)
            return time_module.time() - ts if ts else float('inf')
    
    def get_change_24h(self, symbol: str) -> float:
        """Get 24h change percentage"""
        with self._lock:
            return self._changes_24h.get(symbol, 0.0)
    
    def get_all(self) -> Dict[str, Dict]:
        """Get all cached prices with metadata"""
        with self._lock:
            result = {}
            now = time_module.time()
            for sym in self._prices:
                result[sym] = {
                    "price": self._prices[sym],
                    "age_s": now - self._timestamps.get(sym, 0),
                    "change_24h": self._changes_24h.get(sym, 0),
                    "volume": self._volumes.get(sym, 0),
                }
            return result


# ============================================================================
# WEBSOCKET FEED
# ============================================================================

class WebSocketPriceFeed:
    """
    Real-time price feed using Binance WebSocket.
    
    Runs a background thread with asyncio event loop for the WS connection.
    Exposes synchronous get_price() for the trading system.
    Falls back to REST price_feed when WS data is unavailable.
    """
    
    def __init__(self):
        self._cache = PriceCache()
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reconnect_count = 0
        self._max_reconnects = 50
        self._messages_received = 0
        self._start_time: Optional[float] = None
        
        # REST fallback
        self._rest_fallback = None
    
    def start(self):
        """Start WebSocket connection in background thread"""
        if self._running:
            logger.info("📡 WebSocket feed already running")
            return
        
        self._running = True
        self._start_time = time_module.time()
        self._thread = threading.Thread(target=self._run_ws_loop, daemon=True, name="ws-price-feed")
        self._thread.start()
        logger.info(f"🚀 WebSocket price feed started (streams: {len(BINANCE_STREAMS)} tokens)")
        logger.info(f"📡 URL: {BINANCE_WS_URL[:80]}...")
    
    def stop(self):
        """Stop WebSocket connection"""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._connected = False
        logger.info("🛑 WebSocket price feed stopped")
    
    def _run_ws_loop(self):
        """Run asyncio event loop in background thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_listen())
        except Exception as e:
            logger.error(f"❌ WS loop error: {e}")
        finally:
            self._loop.close()
            self._connected = False
    
    async def _ws_listen(self):
        """Main WebSocket listener with auto-reconnection"""
        try:
            import websockets
        except ImportError:
            logger.error("❌ 'websockets' library not installed. Run: pip install websockets")
            return
        
        while self._running and self._reconnect_count < self._max_reconnects:
            try:
                logger.info(f"🔌 Connecting to Binance WebSocket... (attempt {self._reconnect_count + 1})")
                
                async with websockets.connect(
                    BINANCE_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._connected = True
                    self._reconnect_count = 0
                    logger.info("✅ Binance WebSocket connected!")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        self._process_message(message)
                
            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                wait = min(5 * self._reconnect_count, 60)
                logger.warning(f"⚠️ WS disconnected: {e}. Reconnecting in {wait}s... "
                             f"(attempt {self._reconnect_count}/{self._max_reconnects})")
                await asyncio.sleep(wait)
        
        if self._reconnect_count >= self._max_reconnects:
            logger.error("❌ Max reconnection attempts reached. WebSocket feed stopped.")
        self._connected = False
    
    def _process_message(self, raw_message: str):
        """Process a Binance combined stream message"""
        try:
            data = json.loads(raw_message)
            
            # Combined stream format: {"stream": "solusdt@ticker", "data": {...}}
            stream = data.get("stream", "")
            ticker = data.get("data", {})
            
            if not ticker or "@ticker" not in stream:
                return
            
            # Extract symbol from stream name (e.g., "solusdt@ticker" → "SOL")
            binance_pair = stream.split("@")[0]  # "solusdt"
            base_symbol = binance_pair.replace("usdt", "").upper()  # "SOL"
            
            # Map back to our internal symbol
            symbol = BINANCE_TO_SYMBOL.get(base_symbol, base_symbol)
            
            # Binance 24hr ticker fields:
            # c = last price, P = price change percent, v = volume
            price = float(ticker.get("c", 0))
            change_24h = float(ticker.get("P", 0))
            volume = float(ticker.get("v", 0))
            
            if price > 0:
                self._cache.update(symbol, price, change_24h, volume)
                self._messages_received += 1
                
                # Log every 100th message to avoid spam
                if self._messages_received % 100 == 0:
                    logger.debug(f"📊 WS update #{self._messages_received}: {symbol}=${price:.4f}")
        
        except Exception as e:
            logger.error(f"❌ Error processing WS message: {e}")
    
    # ====================================================================
    # PUBLIC API — Compatible with existing price_feed interface
    # ====================================================================
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get price for a symbol. Compatible with existing interface.
        
        Accepts formats: "SOLUSD", "SOL/USD", "SOL", "SOLUSDT"
        Falls back to REST if WS price is unavailable.
        """
        # Normalize symbol
        clean = symbol.upper().replace("/", "").replace("USD", "").replace("USDT", "").replace("USDC", "")
        
        # Try WebSocket cache first
        ws_price = self._cache.get_price(clean)
        if ws_price is not None:
            return ws_price
        
        # Fallback to REST
        logger.debug(f"📡 WS cache miss for {clean}, falling back to REST")
        return self._get_rest_price(symbol)
    
    def get_price_change_24h(self, symbol: str) -> float:
        """Get 24h price change percentage"""
        clean = symbol.upper().replace("/", "").replace("USD", "").replace("USDT", "").replace("USDC", "")
        return self._cache.get_change_24h(clean)
    
    def _get_rest_price(self, symbol: str) -> Optional[float]:
        """Fallback to REST price feed"""
        try:
            if self._rest_fallback is None:
                from api.price_feed import get_price_feed
                self._rest_fallback = get_price_feed()
            
            price = self._rest_fallback.get_price_sync(symbol)
            if price and price > 0:
                return price
        except Exception as e:
            logger.error(f"❌ REST fallback failed for {symbol}: {e}")
        
        return None
    
    def get_status(self) -> Dict:
        """Get feed status for monitoring"""
        uptime = time_module.time() - self._start_time if self._start_time else 0
        return {
            "connected": self._connected,
            "running": self._running,
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            "uptime_seconds": round(uptime, 1),
            "cached_prices": self._cache.get_all(),
        }
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# ============================================================================
# SINGLETON
# ============================================================================

_ws_feed: Optional[WebSocketPriceFeed] = None
_ws_feed_lock = threading.Lock()


def get_ws_price_feed() -> WebSocketPriceFeed:
    """Get or create the singleton WebSocket price feed"""
    global _ws_feed
    with _ws_feed_lock:
        if _ws_feed is None:
            _ws_feed = WebSocketPriceFeed()
            _ws_feed.start()
        return _ws_feed


def stop_ws_price_feed():
    """Stop the singleton WebSocket price feed"""
    global _ws_feed
    with _ws_feed_lock:
        if _ws_feed:
            _ws_feed.stop()
            _ws_feed = None


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    print("=" * 60)
    print("🧪 WebSocket Price Feed Test")
    print("=" * 60)
    
    feed = get_ws_price_feed()
    
    print("\n⏳ Waiting 5 seconds for prices...\n")
    time_module.sleep(5)
    
    # Test prices
    for symbol in ["SOL", "ETH", "BTC", "BONK", "WIF", "JUP"]:
        price = feed.get_price(f"{symbol}USD")
        change = feed.get_price_change_24h(symbol)
        age = feed._cache.get_age(symbol)
        if price:
            print(f"   {symbol:6s}: ${price:>12.4f}  ({change:+.2f}%)  [age: {age:.1f}s]")
        else:
            print(f"   {symbol:6s}: NO DATA (fallback to REST)")
    
    print(f"\n📡 Status: {json.dumps(feed.get_status(), indent=2, default=str)}")
    
    stop_ws_price_feed()
    print("\n✅ Test complete")
