"""Real-time price feed from Kraken API"""
import requests
from typing import Dict, Optional
import time

KRAKEN_PAIRS = {
    'SOL': 'SOLUSD',
    'BTC': 'XBTUSD', 
    'ETH': 'ETHUSD',
    'JUP': 'SOLUSD',  # Fallback to SOL
    'BONK': 'SOLUSD',
    'WIF': 'SOLUSD',
}

class KrakenPriceFeed:
    """Real-time price feed from Kraken"""
    
    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 30  # 30 seconds cache
    
    def get_price(self, symbol: str) -> float:
        """Get price for a symbol"""
        symbol = symbol.upper()
        
        # Check cache
        if symbol in self._cache:
            if time.time() - self._cache_time.get(symbol, 0) < self._cache_ttl:
                return self._cache[symbol]
        
        # Get from Kraken
        pair = KRAKEN_PAIRS.get(symbol, 'SOLUSD')
        url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
        
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for k, v in data.get('result', {}).items():
                    price = float(v.get('c', [0])[0])
                    if price > 0:
                        self._cache[symbol] = price
                        self._cache_time[symbol] = time.time()
                        return price
        except:
            pass
        
        return self._cache.get(symbol, 0)
    
    def get_prices(self, symbols: list) -> Dict[str, float]:
        """Get prices for multiple symbols"""
        result = {}
        for sym in symbols:
            result[sym] = self.get_price(sym)
        return result

# Singleton
_price_feed: Optional[KrakenPriceFeed] = None

def get_kraken_price() -> KrakenPriceFeed:
    global _price_feed
    if _price_feed is None:
        _price_feed = KrakenPriceFeed()
    return _price_feed
