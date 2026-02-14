#!/usr/bin/env python3
"""
Redis Cache Manager
==================
Fast state sharing and caching using Redis.

Features:
- Price caching
- Trade state sharing
- Pub/Sub for real-time updates
- TTL-based expiration

Usage:
    from cache.redis_manager import RedisSimulator
    cache = RedisSimulator()
"""

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class RedisSimulator:
    """
    Simulates Redis using file-based storage.
    For production, replace with actual redis-py client.
    """

    def __init__(self, cache_dir: str = ".redis_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.subscribers: Dict[str, List] = {}

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        file_path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
        if file_path.exists():
            data = json.loads(file_path.read_text())
            if data.get("expires", 0) > datetime.now().timestamp():
                return data.get("value")
            file_path.unlink()
        return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL."""
        file_path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
        data = {
            "value": value,
            "expires": datetime.now().timestamp() + ttl
        }
        file_path.write_text(json.dumps(data))
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        file_path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    async def keys(self, pattern: str = "*") -> List[str]:
        """List all keys matching pattern."""
        keys = []
        for f in self.cache_dir.glob("*.json"):
            key = f.stem
            keys.append(key)
        return keys

    async def publish(self, channel: str, message: str):
        """Publish message to channel."""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        for callback in self.subscribers[channel]:
            await callback(channel, message)

    async def subscribe(self, channel: str, callback):
        """Subscribe to channel."""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(callback)


class PriceCache:
    """Fast price caching layer."""

    def __init__(self, redis: RedisSimulator):
        self.redis = redis
        self.local_cache: Dict[str, Dict] = {}

    async def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price."""
        if symbol in self.local_cache:
            cached = self.local_cache[symbol]
            if cached["expires"] > datetime.now().timestamp():
                return cached["value"]

        key = f"price:{symbol}"
        value = await self.redis.get(key)
        if value:
            self.local_cache[symbol] = {
                "value": float(value),
                "expires": datetime.now().timestamp() + 60
            }
            return float(value)
        return None

    async def set_price(self, symbol: str, price: float, ttl: int = 60):
        """Cache price."""
        key = f"price:{symbol}"
        await self.redis.set(key, str(price), ttl)
        self.local_cache[symbol] = {
            "value": price,
            "expires": datetime.now().timestamp() + ttl
        }

    async def get_all_prices(self) -> Dict[str, float]:
        """Get all cached prices."""
        prices = {}
        for symbol in ["SOL", "ETH", "JUP", "WIF", "BONK", "RAY", "JTO", "cbBTC"]:
            price = await self.get_price(symbol)
            if price:
                prices[symbol] = price
        return prices


class TradeStateManager:
    """Manages trade state with Redis."""

    def __init__(self, redis: RedisSimulator):
        self.redis = redis

    async def save_trade(self, trade_id: str, data: Dict):
        """Save trade state."""
        key = f"trade:{trade_id}"
        await self.redis.set(key, json.dumps(data), ttl=3600)

    async def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Get trade state."""
        key = f"trade:{trade_id}"
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    async def get_all_trades(self) -> List[Dict]:
        """Get all active trades."""
        trades = []
        for key in await self.redis.keys("trade:*"):
            trade = await self.get_trade(key)
            if trade:
                trades.append(trade)
        return trades


class MarketDataCache:
    """Caches market data for ML models."""

    def __init__(self, redis: RedisSimulator):
        self.redis = redis
        self.price_history: Dict[str, List[float]] = {}

    async def add_price_point(self, symbol: str, price: float):
        """Add price point to history."""
        key = f"history:{symbol}"
        history = await self.redis.get(key)
        if history:
            data = json.loads(history)
        else:
            data = []

        data.append({
            "price": price,
            "timestamp": datetime.now().isoformat()
        })

        data = data[-50:]
        await self.redis.set(key, json.dumps(data), ttl=3600)

        self.price_history[symbol] = [d["price"] for d in data]

    async def get_history(self, symbol: str, limit: int = 50) -> List[float]:
        """Get price history for ML."""
        if symbol in self.price_history:
            return self.price_history[symbol]

        key = f"history:{symbol}"
        data = await self.redis.get(key)
        if data:
            parsed = json.loads(data)
            self.price_history[symbol] = [d["price"] for d in parsed]
            return self.price_history[symbol]
        return []


async def main():
    """Test Redis simulator."""
    print("\nTesting Redis Simulator")
    print("="*50)

    redis = RedisSimulator()
    cache = PriceCache(redis)
    state = TradeStateManager(redis)
    market = MarketDataCache(redis)

    # Test price caching
    print("\nPrice Caching:")
    await cache.set_price("SOL", 87.50)
    price = await cache.get_price("SOL")
    print(f"   SOL price: ${price}")

    # Test trade state
    print("\nTrade State:")
    await state.save_trade("trade_001", {
        "symbol": "SOL",
        "direction": "BUY",
        "entry_price": 87.0,
        "pnl_pct": 0.5
    })
    trade = await state.get_trade("trade_001")
    print(f"   Trade: {trade}")

    print("\nRedis Simulator working!")


if __name__ == "__main__":
    asyncio.run(main())
