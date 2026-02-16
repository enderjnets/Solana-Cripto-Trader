"""Real-time price feed from CryptoCompare (primary) + CoinGecko (fallback)"""
import aiohttp
import asyncio
import json
from typing import Dict, Optional

CRYPTOCOMPARE_API = "https://min-api.cryptocompare.com/data/pricemulti"
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

# CryptoCompare symbols (different from token names)
CC_SYMBOLS = {
    # Major
    'SOL': 'SOL',
    'BTC': 'BTC',
    'ETH': 'ETH',
    # Stablecoins (not traded but needed for reference)
    'USDC': 'USDC',
    'USDT': 'USDT',
    # Top Solana tokens
    'WIF': 'WIF',
    'BONK': 'BONK',
    'PUMP': 'PUMP',
    'JTO': 'JTO',
    'JUP': 'JUP',
    'RAY': 'RAY',
    'ORCA': 'ORCA',
    'HNT': 'HNT',
    'WEN': 'WEN',
    'POPCAT': 'POPCAT',
    'MEW': 'MEW',
    'BOME': 'BOME',
    'PNUT': 'PNUT',
    'JLP': 'JLP',
    'TRUMP': 'TRUMP',
    # More tokens
    'SOLI': 'SOLI',
    'SLERF': 'SLERF',
    'CATO': 'CATO',
    'DYOR': 'DYOR',
    'GME': 'GME',
    'BOOK': 'BOOK',
    'AI16Z': 'AI16Z',
    # Additional popular tokens
    'MOON': 'MOON',
    'MER': 'MER',
    'BLZE': 'BLZE',
    'REAL': 'REAL',
    'DEXI': 'DEXI',
    'UXD': 'UXD',
    'STSOL': 'STSOL',
    'MSOL': 'MSOL',
    'LDO': 'LDO',
    'SBR': 'SBR',
    'STEP': 'STEP',
    'FIDA': 'FIDA',
    'PORT': 'PORT',
    'MAP': 'MAP',
    'ATLAS': 'ATLAS',
    'POLIS': 'POLIS',
    'RUN': 'RUN',
    'GRASS': 'GRASS',
    'KVNX': 'KVNX',
    'HAWK': 'HAWK',
    'FORM': 'FORM',
    'CHRIS': 'CHRIS',
    'ZERU': 'ZERU',
    'NAS': 'NAS',
    'PAAL': 'PAAL',
}

class PriceFeed:
    """Real-time price feed using CryptoCompare"""
    
    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes cache to reduce memory
    
    async def get_price(self, symbol: str) -> float:
        """Get price for a single token"""
        symbol = symbol.upper()
        
        # Check cache first
        if symbol in self._cache:
            if (asyncio.get_event_loop().time() - self._cache_time.get(symbol, 0)) < self._cache_ttl:
                return self._cache[symbol]
        
        # Get from CryptoCompare
        cc_symbol = CC_SYMBOLS.get(symbol, symbol)
        url = f"{CRYPTOCOMPARE_API}?fsyms={cc_symbol}&tsyms=USD"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = data.get(cc_symbol, {}).get('USD', 0)
                        if price:
                            self._cache[symbol] = price
                            self._cache_time[symbol] = asyncio.get_event_loop().time()
                            print(f"💹 {symbol}: \${price}")
                            return price
        except Exception as e:
            print(f"Price fetch error: {e}")
        
        return self._cache.get(symbol, 0)
    
    async def get_prices(self, symbols: list) -> Dict[str, float]:
        """Get prices for multiple tokens"""
        cc_symbols = [CC_SYMBOLS.get(s.upper(), s.upper()) for s in symbols]
        fsyms = ",".join(set(cc_symbols))
        url = f"{CRYPTOCOMPARE_API}?fsyms={fsyms}&tsyms=USD"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = {}
                        for s in symbols:
                            cc_s = CC_SYMBOLS.get(s.upper(), s.upper())
                            price = data.get(cc_s, {}).get('USD', 0)
                            result[s.upper()] = price
                            if price:
                                self._cache[s.upper()] = price
                                self._cache_time[s.upper()] = asyncio.get_event_loop().time()
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
