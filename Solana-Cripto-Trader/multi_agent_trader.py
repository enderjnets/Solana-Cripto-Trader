#!/usr/bin/env python3
"""
Multi-Agent Trading - MAX TOKENS + AUTO DISCOVERY
"""
import asyncio
import json
import random
import requests
from pathlib import Path
from datetime import datetime
from solana.rpc.api import Client
from solders.pubkey import Pubkey

WALLET_ADDRESS = "H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a"
RPC_URL = "https://api.devnet.solana.com"
STATE_FILE = Path("~/.config/solana-jupiter-bot/multi_agent_state.json").expanduser()

# MAX TOKENS - All major + meme
TOKENS = {
    # Major
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tFm7mkBZKaUUTwJWS3d",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW",
    
    # DeFi
    "RAY": "4k3DyjzvzpLhG1hGLbo2duNZf1kWQqawqjJHbDkPkrm",
    "SRM": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",
    "MNGO": "MangoCzV36M1c9AMgdk841qGZ8EfYsKKF9LRcUsQh3m",
    "ORCA": "orcaEKTdK7LKz57vaAYr9QeLsV6XEZ9rJEM7TKu5Sing",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2",
    "AA": "ALTDEZZnV4GQ41nEof1UGNMVxXC4MnMGfRUKL6NZFwRx",
    "MNDE": "MNDEFzGByWmUCG7F2C5MNKKGHYynKzxeCWNYmKmShUX",
    
    # Memes
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP",
    "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
    "PEPE": "HZ1JovNiVvGrGNiiYvEozEVgZ58xa3kPfYoBKRJiNfnh",
    "DOGE": "Ez2zQv7vL8WJ5K8h1mY5r9Y3pF4xT6wK2jN8qR3vL5mP",
    "SHIB": "Gx6C6F1wPm8oTWqRrCKDkN6b2TqQtqJiHKKqK4RD9Gq",
    "FLOKI": "FLEniGBX6aLQJ9JGC5m1N3xKmBYL3z6S4VqV7XWDTpo",
    "SAMO": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
    "POG": "PoGvxXFJy1J7hK2JJ3w1jNapKkNdZ9N1vNqC6NqVXq",
    "MUMU": "MuMu12wMTB4Tr5iQf4JxM6E25N4p3v2HqWvN1oKxM",
    "CATO": "CaP7XqvFfi7DSjM4w3hY7X5YfYvY9qQ1vN3dFgHjK",
    "BODEN": "7dHbkkEbrJUyFLkUTfY1gfX7s7dYqJYXLZBvVNZGqZC",
    "AI16Z": "AiZ6j9nFK1t1X6g6Y4X5Z8q7N3J2L9M4K7P0R2T5Y",
}

class MarketScanner:
    def __init__(self):
        self.scan_count = 0
        self.discovered = {}
    
    def scan(self):
        self.scan_count += 1
        prices = {}
        
        # Get prices for base tokens
        try:
            ids = ",".join([TOKENS[k] for k in ["SOL", "BTC", "ETH", "USDC", "USDT"]])
            resp = requests.get("https://lite-api.jup.ag/price/v3", params={"ids": ids}, timeout=10)
            data = resp.json()
            for sym, mint in TOKENS.items():
                if mint in data and isinstance(data[mint], dict):
                    prices[sym] = {"price": float(data[mint].get("usdPrice", 0)), 
                                   "change": float(data[mint].get("priceChange24h", 0))}
        except:
            pass
        
        # Fill from CoinGecko
        try:
            cg_ids = "solana,bitcoin,ethereum,tether,raydium,orca-token,jupiter-inu,bonk,dogwifhat,pepe,shiba-inu,floki,samoyed"
            resp = requests.get("https://api.coingecko.com/api/v3/simple/price", 
                params={"ids": cg_ids, "vs_currencies": "usd", "include_24hr_change": "true"}, timeout=10)
            data = resp.json()
            mapping = {
                "solana": "SOL", "bitcoin": "BTC", "ethereum": "ETH", 
                "tether": "USDT", "raydium": "RAY", "orca-token": "ORCA",
                "jupiter-inu": "JUP", "bonk": "BONK", "dogwifhat": "WIF", 
                "pepe": "PEPE", "shiba-inu": "SHIB", "floki": "FLOKI", "samoyed": "SAMO"
            }
            for cg, sym in mapping.items():
                if cg in data and isinstance(data[cg], dict) and sym not in prices:
                    prices[sym] = {"price": float(data[cg].get("usd", 0)), 
                                   "change": float(data[cg].get("usd_24h_change", 0))}
        except:
            pass
        
        # AUTO DISCOVERY every 10 cycles
        if self.scan_count % 10 == 0:
            print("\n🌐 [SCANNER] Discovering new tokens...")
            try:
                # DEX Screener
                resp = requests.get("https://api.dexscreener.com/latest/dex/tokens/solana", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for pair in data.get("pairs", [])[:20]:
                        token = pair.get("baseToken", {})
                        if token:
                            sym = token.get("symbol", "")
                            addr = token.get("address", "")
                            if sym and addr and sym not in prices:
                                price = float(pair.get("priceUsd", 0))
                                if price > 0:
                                    prices[sym] = {"price": price, 
                                                   "change": float(pair.get("priceChange", {}).get("h24", 0)),
                                                   "source": "discovered"}
                                    print(f"   🪙 {sym}: ${price:.6f} ({prices[sym]['change']:+.1f}%)")
                                    self.discovered[sym] = addr
            except Exception as e:
                print(f"   ⚠️ Discovery error: {e}")
        
        return prices

class Analyst:
    """Finds opportunities on MAX tokens"""
    
    def analyze(self, prices):
        opps = []
        
        for sym, d in prices.items():
            if not isinstance(d, dict):
                continue
            
            ch = d.get("change", 0)
            
            # Trade on ANY movement
            if ch < -1:
                opps.append({"symbol": sym, "action": "BUY", "strength": abs(ch) * 2, 
                            "reason": f"Dip {ch:+.1f}%"})
            elif ch < -0.3:
                opps.append({"symbol": sym, "action": "BUY", "strength": abs(ch), 
                            "reason": f"Tiny dip {ch:+.1f}%"})
            
            if ch > 0.5:
                opps.append({"symbol": sym, "action": "SELL", "strength": ch,
                            "reason": f"Pump {ch:+.1f}%"})
        
        return sorted(opps, key=lambda x: x["strength"], reverse=True)

class RiskManager:
    def __init__(self):
        self.max_pos = 8
        self.tp = 1.5
        self.sl = 1.0
    
    def validate_entry(self, opp, state):
        if len(state.get("positions", {})) >= self.max_pos:
            return False, "Max"
        if opp["symbol"] in state.get("positions", {}):
            return False, "Has it"
        return True, "OK"
    
    def check_exits(self, positions, prices):
        exits = []
        for sym, data in positions.items():
            if not isinstance(data, dict):
                continue
            if sym not in prices:
                continue
            price_info = prices[sym]
            if not isinstance(price_info, dict):
                continue
            
            current = price_info.get("price", 0)
            entry = data.get("entry_price", 0)
            amt = data.get("amount", 0)
            
            if entry > 0 and amt > 0 and current > 0:
                pnl = ((current - entry) / entry) * 100
                
                if pnl >= self.tp:
                    exits.append({"symbol": sym, "action": "TAKE_PROFIT", 
                                 "reason": f"+{pnl:.1f}%", "amount": amt, 
                                 "price": current, "pnl": pnl})
                elif pnl <= -self.sl:
                    exits.append({"symbol": sym, "action": "STOP_LOSS", 
                                 "reason": f"{pnl:.1f}%", "amount": amt, 
                                 "price": current, "pnl": pnl})
        return exits

class Trader:
    def __init__(self):
        self.client = Client(RPC_URL)
        self.wallet = Pubkey.from_string(WALLET_ADDRESS)
        self.trade_size_pct = 0.15
    
    def get_wallet(self):
        return self.client.get_balance(self.wallet).value / 1e9
    
    def execute_entry(self, sym, price, state):
        amount = state["capital_usd"] * self.trade_size_pct / price
        cost = state["capital_usd"] * self.trade_size_pct
        
        if sym not in state["positions"]:
            state["positions"][sym] = {"amount": 0, "entry_price": price}
        
        state["positions"][sym]["amount"] += amount
        state["positions"][sym]["entry_price"] = price
        state["capital_usd"] -= cost
        
        state["trades"].append({"time": datetime.now().isoformat(), "action": "BUY", 
                               "symbol": sym, "price": price, "amount": amount, "cost": cost})
        return True
    
    def execute_exit(self, exit_info, state):
        sym = exit_info["symbol"]
        amt = exit_info["amount"]
        price = exit_info["price"]
        pnl = exit_info["pnl"]
        
        proceeds = amt * price
        state["capital_usd"] += proceeds
        state["trades"].append({"time": datetime.now().isoformat(), "action": exit_info["action"], 
                               "symbol": sym, "price": price, "proceeds": proceeds, "pnl": pnl})
        state["today_pnl"] = state.get("today_pnl", 0) + pnl
        del state["positions"][sym]
        return True

class CEO:
    def __init__(self):
        self.daily_target = 5.0
    
    def should_trade(self, state):
        total = state.get("capital_usd", 500)
        positions = state.get("positions", {})
        
        current_total = total + len(positions) * 20
        pnl_pct = ((current_total - 500) / 500) * 100
        
        print(f"\n👑 [CEO] Progress: {pnl_pct:+.2f}% / {self.daily_target}%")
        
        if pnl_pct < self.daily_target:
            print(f"   ⚡ MAX AGGRESSIVE MODE")
        return True

class Orchestrator:
    def __init__(self):
        self.scanner = MarketScanner()
        self.analyst = Analyst()
        self.risk = RiskManager()
        self.trader = Trader()
        self.ceo = CEO()
        self.load_state()
    
    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                self.state = json.load(f)
        else:
            self.state = {"capital_usd": 500, "positions": {}, "trades": [], "today_pnl": 0}
        
        last_date = self.state.get("last_date", "")
        today = datetime.now().strftime("%Y-%m-%d")
        if last_date != today:
            self.state["today_pnl"] = 0
            self.state["last_date"] = today
    
    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    async def cycle(self, n):
        print(f"\n{'='*55}\n🔄 CYCLE {n} - {datetime.now().strftime('%H:%M:%S')}")
        
        prices = self.scanner.scan()
        print(f"   📊 Scanning {len(prices)} tokens")
        
        opps = self.analyst.analyze(prices)
        self.ceo.should_trade(self.state)
        
        # Check exits
        exits = self.risk.check_exits(self.state["positions"], prices)
        for ex in exits:
            self.trader.execute_exit(ex, self.state)
            emoji = "🎯" if ex["action"] == "TAKE_PROFIT" else "🛑"
            print(f"   {emoji} {ex['action']} {ex['symbol']}: {ex['reason']}")
        
        # Execute entries
        if opps:
            trades_made = 0
            for o in opps:
                if trades_made >= 5:
                    break
                    
                if o["action"] == "BUY":
                    ok, msg = self.risk.validate_entry(o, self.state)
                    price_info = prices.get(o["symbol"])
                    
                    if ok and price_info and isinstance(price_info, dict):
                        self.trader.execute_entry(o["symbol"], price_info.get("price", 0), self.state)
                        print(f"   ✅ BUY {o['symbol']} @ ${price_info.get('price', 0):.4f} - {o['reason']}")
                        trades_made += 1
        
        # Summary
        sol_price = prices.get("SOL", {}).get("price", 0) if isinstance(prices.get("SOL"), dict) else 0
        
        print(f"\n{'='*55}")
        print(f"💵 SOL: ${sol_price:.2f} | 💰 Capital: ${self.state['capital_usd']:.2f}")
        
        total_pos = 0
        for sym, d in self.state["positions"].items():
            if not isinstance(d, dict):
                continue
            cur = prices.get(sym, {}).get("price", 0) if isinstance(prices.get(sym), dict) else 0
            val = d.get("amount", 0) * cur
            pnl = ((cur - d.get("entry_price", 0)) / d.get("entry_price", 1) * 100) if d.get("entry_price", 0) > 0 else 0
            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"   📈 {sym}: {emoji} {pnl:+.1f}%")
            total_pos += val
        
        total = self.state["capital_usd"] + total_pos
        pnl = total - 500
        print(f"💎 Total: ${total:.2f} | P&L: ${pnl:+.2f} ({(pnl/500)*100:+.2f}%)")
        print(f"🔄 Trades: {len(self.state['trades'])} | 📊 Tokens: {len(prices)}")
        
        self.save_state()
    
    async def run(self):
        print("="*55)
        print("🚀 MAX TOKENS + AUTO DISCOVERY TRADING")
        print("="*55)
        
        n = 0
        while True:
            n += 1
            try:
                await self.cycle(n)
            except Exception as e:
                print(f"❌ {e}")
            await asyncio.sleep(20)

asyncio.run(Orchestrator().run())
