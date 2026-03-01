#!/usr/bin/env python3
"""
WebSocket Price Feed - Real-Time Prices via Kraken WebSocket v2
===============================================================
Provides <1s latency price updates using Kraken ticker stream.
Falls back gracefully to REST price_feed if WebSocket is unavailable.

Usage:
    from api.websocket_feed import get_ws_price_feed
    
    feed = get_ws_price_feed()
    price = feed.get_price("SOLUSD")  # Compatible with existing interface
"""

import asyncio
import json
import logging
import threading
import time as time_module
from typing import Dict, Optional

logger = logging.getLogger("websocket_feed")

# ============================================================================
# SYMBOL MAPPING — Kraken format
# ============================================================================

# Kraken WebSocket v2 uses "SOL/USD" style symbols
KRAKEN_SYMBOLS = [
    "SOL/USD", "BTC/USD", "ETH/USD",
    "BONK/USD", "WIF/USD", "JUP/USD",
    "RAY/USD", "JTO/USD",
]

# Normalize Kraken pair names to our internal format
# Kraken sometimes returns XBT instead of BTC, etc.
KRAKEN_NORMALIZE = {
    "XBT": "BTC",
    "XXBT": "BTC",
    "XETH": "ETH",
}

KRAKEN_WS_URL = "wss://ws.kraken.com/v2"


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
        with self._lock:
            self._prices[symbol] = price
            self._timestamps[symbol] = time_module.time()
            self._changes_24h[symbol] = change_24h
            self._volumes[symbol] = volume
    
    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            if symbol not in self._prices:
                return None
            age = time_module.time() - self._timestamps.get(symbol, 0)
            if age > 120:
                return None
            return self._prices[symbol]
    
    def get_age(self, symbol: str) -> float:
        with self._lock:
            ts = self._timestamps.get(symbol, 0)
            return time_module.time() - ts if ts else float('inf')
    
    def get_change_24h(self, symbol: str) -> float:
        with self._lock:
            return self._changes_24h.get(symbol, 0.0)
    
    def get_all(self) -> Dict[str, Dict]:
        with self._lock:
            result = {}
            now = time_module.time()
            for sym in self._prices:
                result[sym] = {
                    "price": self._prices[sym],
                    "age_s": round(now - self._timestamps.get(sym, 0), 1),
                    "change_24h": self._changes_24h.get(sym, 0),
                    "volume": self._volumes.get(sym, 0),
                }
            return result


# ============================================================================
# WEBSOCKET FEED — Kraken v2
# ============================================================================

class WebSocketPriceFeed:
    """
    Real-time price feed using Kraken WebSocket v2.
    
    Runs a background daemon thread with asyncio event loop.
    Exposes synchronous get_price() for the trading system.
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
        self._rest_fallback = None
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time_module.time()
        self._thread = threading.Thread(target=self._run_ws_loop, daemon=True, name="ws-price-feed")
        self._thread.start()
        logger.info(f"🚀 Kraken WebSocket price feed started ({len(KRAKEN_SYMBOLS)} symbols)")
    
    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._connected = False
        logger.info("🛑 WebSocket price feed stopped")
    
    def _run_ws_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_listen())
        except Exception as e:
            logger.error(f"❌ WS loop error: {e}")
        finally:
            try:
                self._loop.close()
            except:
                pass
            self._connected = False
    
    async def _ws_listen(self):
        try:
            import websockets
        except ImportError:
            logger.error("❌ 'websockets' not installed. pip install websockets")
            return
        
        while self._running and self._reconnect_count < self._max_reconnects:
            try:
                async with websockets.connect(
                    KRAKEN_WS_URL,
                    ping_interval=30,
                    ping_timeout=15,
                    close_timeout=5,
                ) as ws:
                    self._connected = True
                    self._reconnect_count = 0
                    
                    # Subscribe to ticker for all symbols
                    sub_msg = {
                        "method": "subscribe",
                        "params": {
                            "channel": "ticker",
                            "symbol": KRAKEN_SYMBOLS,
                        }
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info("✅ Kraken WebSocket connected & subscribed")
                    
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
                if self._running:
                    await asyncio.sleep(wait)
        
        if self._reconnect_count >= self._max_reconnects:
            logger.error("❌ Max reconnects reached. WS feed stopped.")
        self._connected = False
    
    def _process_message(self, raw_message: str):
        try:
            data = json.loads(raw_message)
            
            # Kraken v2 ticker format:
            # {"channel": "ticker", "type": "update", "data": [{"symbol": "SOL/USD", "last": 83.15, ...}]}
            if data.get("channel") != "ticker":
                return
            
            for item in data.get("data", []):
                pair = item.get("symbol", "")  # "SOL/USD"
                last_price = item.get("last", 0)
                
                if not pair or not last_price:
                    continue
                
                # Extract base symbol: "SOL/USD" → "SOL"
                base = pair.split("/")[0].upper()
                base = KRAKEN_NORMALIZE.get(base, base)
                
                # Volume and 24h change
                volume = float(item.get("volume", 0))
                change_24h = float(item.get("change_pct", 0))
                
                self._cache.update(base, float(last_price), change_24h, volume)
                self._messages_received += 1
                
                if self._messages_received % 200 == 0:
                    logger.debug(f"📊 WS #{self._messages_received}: {base}=${last_price:.4f}")
        
        except Exception as e:
            logger.error(f"❌ WS message error: {e}")
    
    # ====================================================================
    # PUBLIC API
    # ====================================================================
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get price for a symbol. Compatible with existing interface.
        Accepts: "SOLUSD", "SOL/USD", "SOL", "SOLUSDT"
        """
        clean = symbol.upper().replace("/", "").replace("USD", "").replace("USDT", "").replace("USDC", "")
        clean = KRAKEN_NORMALIZE.get(clean, clean)
        
        ws_price = self._cache.get_price(clean)
        if ws_price is not None:
            return ws_price
        
        # Fallback to REST
        return self._get_rest_price(symbol)
    
    def get_price_change_24h(self, symbol: str) -> float:
        clean = symbol.upper().replace("/", "").replace("USD", "").replace("USDT", "").replace("USDC", "")
        clean = KRAKEN_NORMALIZE.get(clean, clean)
        return self._cache.get_change_24h(clean)
    
    def _get_rest_price(self, symbol: str) -> Optional[float]:
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
    global _ws_feed
    with _ws_feed_lock:
        if _ws_feed is None:
            _ws_feed = WebSocketPriceFeed()
            _ws_feed.start()
        return _ws_feed


def stop_ws_price_feed():
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
    print("🧪 Kraken WebSocket Price Feed Test")
    print("=" * 60)
    
    feed = get_ws_price_feed()
    
    print("\n⏳ Waiting 5 seconds for prices...\n")
    time_module.sleep(5)
    
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
