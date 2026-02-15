"""Real-time price feed from CoinGecko"""
import aiohttp
import asyncio
from typing import Dict, Optional

COINGECKO_API = "https://api.coingecko.com/api/v3"

# Token ID mapping (CoinGecko uses different IDs)
TOKEN_IDS = {
    'SOL': 'solana',
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'JUP': 'jupiter',
    'BONK': 'bonk',
    'WIF': 'wif',
    'RAY': 'raydium',
    'JTO': 'jto-token',
    'METEO': 'meteora',
    'ORCA': 'orca',
    'USDC': 'usd-coin',
    'USDT': 'tether',
}

class PriceFeed:
    """Real-time price feed"""
    
    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 60  # 60 seconds cache
    
    async def get_price(self, symbol: str) -> float:
        """Get price for a single token"""
        # Check cache first
        if symbol.upper() in self._cache:
            if (asyncio.get_event_loop().time() - self._cache_time.get(symbol.upper(), 0)) < self._cache_ttl:
                return self._cache[symbol.upper()]
        
        # Get from API
        token_id = TOKEN_IDS.get(symbol.upper(), symbol.lower())
        url = f"{COINGECKO_API}/simple/price?ids={token_id}&vs_currencies=usd"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = data.get(token_id, {}).get('usd', 0)
                        if price:
                            self._cache[symbol.upper()] = price
                            self._cache_time[symbol.upper()] = asyncio.get_event_loop().time()
                            return price
        except Exception as e:
            print(f"Price fetch error: {e}")
        
        return self._cache.get(symbol.upper(), 0)
    
    async def get_prices(self, symbols: list) -> Dict[str, float]:
        """Get prices for multiple tokens"""
        # Get unique token IDs
        token_ids = []
        symbol_map = {}
        for s in symbols:
            tid = TOKEN_IDS.get(s.upper(), s.lower())
            token_ids.append(tid)
            symbol_map[tid] = s.upper()
        
        ids_str = ",".join(set(token_ids))
        url = f"{COINGECKO_API}/simple/price?ids={ids_str}&vs_currencies=usd"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = {}
                        for tid, symbol in symbol_map.items():
                            price = data.get(tid, {}).get('usd', 0)
                            result[symbol] = price
                            if price:
                                self._cache[symbol] = price
                                self._cache_time[symbol] = asyncio.get_event_loop().time()
                        return result
        except Exception as e:
            print(f"Price fetch error: {e}")
        
        return {s.upper(): self._cache.get(s.upper(), 0) for s in symbols}
    
    def get_price_sync(self, symbol: str) -> float:
        """Synchronous price getter"""
        try:
            return asyncio.run(self.get_price(symbol))
        except:
            return self._cache.get(symbol.upper(), 0)

# Singleton instance
_price_feed: Optional[PriceFeed] = None

def get_price_feed() -> PriceFeed:
    global _price_feed
    if _price_feed is None:
        _price_feed = PriceFeed()
    return _price_feed
