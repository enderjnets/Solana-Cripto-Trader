#!/usr/bin/env python3
"""
Solana Trading Bot - Multi Coin Scanner (Available on Jupiter Lite)
"""
import asyncio
import json
import requests
from pathlib import Path
from datetime import datetime
from solana.rpc.api import Client
from solders.pubkey import Pubkey

# Config
WALLET_ADDRESS = "H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a"
RPC_URL = "https://api.devnet.solana.com"

# Tokens available on Jupiter Lite API (free, no auth)
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tFm7mkBZKaUUTwJWS3d",  # Try anyway
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP",
    "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
    "PEPE": "HZ1JovNiVvGrGNiiYvEozEVgZ58xa3kPfYoBKRJiNfnh",
    "SHIB": "Gx6C6F1wPm8oTWqRrCKDkN6b2TqQtqJiHKKqK4RD9Gq",
    "DOGE": "Ez2zQv7vL8WJ5K8h1mY5r9Y3pF4xT6wK2jN8qR3vL5mP",
}

# Try to get BONK and WIF from alternative source
ALT_TOKENS = {
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP",
    "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
}

STATE_FILE = Path("~/.config/solana-jupiter-bot/state.json").expanduser()
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

class TradingBot:
    def __init__(self):
        self.client = Client(RPC_URL)
        self.wallet = Pubkey.from_string(WALLET_ADDRESS)
        self.load_state()
    
    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                self.state = json.load(f)
        else:
            self.state = {
                "capital_usd": 500,
                "positions": {},
                "trades": [],
                "tracked_coins": list(TOKENS.keys())
            }
    
    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_prices_jupiter(self):
        """Get prices from Jupiter Lite"""
        token_ids = list(TOKENS.values())
        ids_str = ",".join(token_ids)
        
        url = "https://lite-api.jup.ag/price/v3"
        params = {"ids": ids_str}
        
        prices = {}
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            
            for symbol, mint in TOKENS.items():
                if mint in data:
                    try:
                        prices[symbol] = {
                            "price": float(data[mint]["usdPrice"]),
                            "change_24h": float(data[mint].get("priceChange24h", 0)),
                            "source": "jupiter"
                        }
                    except:
                        pass
        except Exception as e:
            print(f"⚠️ Jupiter error: {e}")
        
        return prices
    
    def get_prices_coingecko(self):
        """Get prices from CoinGecko (free tier)"""
        prices = {}
        try:
            # Use CoinGecko free API
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": "solana,bitcoin,ethereum,tether,bonk,dogwifhat,pepe,shiba-inu",
                "vs_currencies": "usd",
                "include_24hr_change": "true"
            }
            
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            
            mapping = {
                "solana": "SOL",
                "bitcoin": "BTC",
                "ethereum": "ETH",
                "tether": "USDT",
                "bonk": "BONK",
                "dogwifhat": "WIF",
                "pepe": "PEPE",
                "shiba-inu": "SHIB"
            }
            
            for coingecko_id, symbol in mapping.items():
                if coingecko_id in data:
                    prices[symbol] = {
                        "price": data[coingecko_id]["usd"],
                        "change_24h": data[coingecko_id].get("usd_24h_change", 0),
                        "source": "coingecko"
                    }
        except Exception as e:
            print(f"⚠️ CoinGecko error: {e}")
        
        return prices
    
    def get_all_prices(self):
        """Combine prices from multiple sources"""
        prices = {}
        
        # Try Jupiter first
        jup_prices = self.get_prices_jupiter()
        prices.update(jup_prices)
        
        # Fill missing from CoinGecko
        cg_prices = self.get_prices_coingecko()
        for symbol, data in cg_prices.items():
            if symbol not in prices:
                prices[symbol] = data
        
        return prices
    
    def get_balance(self):
        resp = self.client.get_balance(self.wallet)
        return resp.value / 1e9
    
    def find_opportunities(self, prices):
        """Find trading opportunities"""
        opportunities = []
        
        for symbol, data in prices.items():
            change = data.get("change_24h", 0)
            
            # Buy signal: dip > 5%
            if change < -5:
                opportunities.append({
                    "symbol": symbol,
                    "signal": "BUY",
                    "reason": f"Dip: {change:+.2f}%",
                    "price": data["price"]
                })
            # Sell signal: pump > 10%
            elif change > 10:
                opportunities.append({
                    "symbol": symbol,
                    "signal": "SELL",
                    "reason": f"Pump: {change:+.2f}%",
                    "price": data["price"]
                })
        
        return opportunities
    
    async def run(self):
        print("="*70)
        print("🚀 SOLANA MULTI-COIN SCANNER (Jupiter + CoinGecko)")
        print("="*70)
        print(f"Wallet: {WALLET_ADDRESS[:10]}...")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                prices = self.get_all_prices()
                sol_balance = self.get_balance()
                opportunities = self.find_opportunities(prices)
                
                print(f"\n{'='*70}")
                print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Scan #{iteration}")
                print(f"💰 Wallet: {sol_balance:.4f} SOL")
                print(f"{'='*70}")
                
                # Sort by 24h change
                sorted_prices = sorted(
                    prices.items(), 
                    key=lambda x: x[1].get("change_24h", 0), 
                    reverse=True
                )
                
                print(f"\n📊 MARKET SCAN ({len(prices)} coins):")
                print("-"*50)
                for symbol, data in sorted_prices:
                    change = data.get("change_24h", 0)
                    emoji = "🟢" if change >= 0 else "🔴"
                    source = data.get("source", "?")
                    price_str = f"${data['price']:.6f}" if data['price'] < 1 else f"${data['price']:.2f}"
                    print(f"  {emoji} {symbol:5}: {price_str:>14}  ({change:>+7.2f}%) [{source}]")
                
                # Opportunities
                if opportunities:
                    print(f"\n🎯 OPPORTUNITIES ({len(opportunities)}):")
                    print("-"*50)
                    for opp in opportunities:
                        emoji = "🟢" if opp["signal"] == "BUY" else "🔴"
                        price_str = f"${opp['price']:.6f}" if opp['price'] < 1 else f"${opp['price']:.2f}"
                        print(f"  {emoji} {opp['signal']:4} {opp['symbol']:5} @ {price_str} - {opp['reason']}")
                else:
                    print(f"\n⏳ Sin oportunidades (esperando movimientos...)")
                
                print(f"\n💎 Capital: ${self.state['capital_usd']:.2f}")
                print(f"🔄 Trades: {len(self.state['trades'])}")
                
                self.save_state()
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            await asyncio.sleep(30)

async def main():
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
