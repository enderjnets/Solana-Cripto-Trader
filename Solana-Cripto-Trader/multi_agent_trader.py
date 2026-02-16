#!/usr/bin/env python3
"""
Multi-Agent Trading System - AGGRESSIVE 5% DAILY GOAL
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

# Target: 5% daily
DAILY_GOAL_PCT = 5.0

TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP",
    "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
    "PEPE": "HZ1JovNiVvGrGNiiYvEozEVgZ58xa3kPfYoBKRJiNfnh",
}

class MarketScanner:
    def __init__(self):
        self.name = "Market Scanner"
        self.scan_count = 0
    
    def scan(self):
        self.scan_count += 1
        prices = {}
        
        # Jupiter + CoinGecko
        try:
            ids = ",".join(TOKENS.values())
            resp = requests.get("https://lite-api.jup.ag/price/v3", params={"ids": ids}, timeout=10)
            data = resp.json()
            for sym, mint in TOKENS.items():
                if mint in data and isinstance(data[mint], dict):
                    prices[sym] = {"price": float(data[mint].get("usdPrice", 0)), 
                                   "change": float(data[mint].get("priceChange24h", 0))}
        except:
            pass
        
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/simple/price", 
                params={"ids": "solana,bitcoin,ethereum,tether,bonk,dogwifhat,pepe", "vs_currencies": "usd", "include_24hr_change": "true"}, timeout=10)
            data = resp.json()
            mapping = {"solana": "SOL", "bitcoin": "BTC", "tether": "USDT", "bonk": "BONK", "dogwifhat": "WIF", "pepe": "PEPE"}
            for cg, sym in mapping.items():
                if cg in data and isinstance(data[cg], dict) and sym not in prices:
                    prices[sym] = {"price": float(data[cg].get("usd", 0)), 
                                   "change": float(data[cg].get("usd_24h_change", 0))}
        except:
            pass
        
        return prices

class Analyst:
    """Aggressive analyst - finds MORE opportunities"""
    
    def __init__(self):
        self.name = "Analyst"
    
    def analyze(self, prices):
        opps = []
        
        for sym, d in prices.items():
            if not isinstance(d, dict):
                continue
            
            ch = d.get("change", 0)
            
            # Very aggressive signals
            if ch < -3:  # Even small dip = opportunity
                opps.append({"symbol": sym, "action": "BUY", "strength": abs(ch), 
                            "reason": f"Dip {ch:+.1f}%"})
            elif ch > 5:  # Take profit on pumps
                opps.append({"symbol": sym, "action": "SELL", "strength": ch,
                            "reason": f"Pump {ch:+.1f}%"})
            
            # Also look for momentum
            if ch > 2:
                opps.append({"symbol": sym, "action": "BUY_MOMENTUM", "strength": ch,
                            "reason": f"Momentum {ch:+.1f}%"})
        
        return sorted(opps, key=lambda x: x["strength"], reverse=True)

class RiskManager:
    """Aggressive risk management - 5% daily goal"""
    
    def __init__(self):
        self.max_pos = 5  # More positions
        self.tp = 3.0     # Take profit at 3%
        self.sl = 2.0      # Stop loss at 2%
        self.daily_goal = 5.0  # 5% daily target
    
    def validate_entry(self, opp, state):
        if len(state.get("positions", {})) >= self.max_pos:
            return False, "Max positions"
        if opp["symbol"] in state.get("positions", {}):
            return False, "Already in position"
        
        # Check daily progress toward goal
        today_pnl = state.get("today_pnl", 0)
        if today_pnl >= self.daily_goal:
            return False, "Daily goal reached!"
        
        return True, "Approved"
    
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
                
                # Take profit at 3%
                if pnl >= self.tp:
                    exits.append({"symbol": sym, "action": "TAKE_PROFIT", 
                                 "reason": f"+{pnl:.1f}%", "amount": amt, 
                                 "price": current, "pnl": pnl})
                # Stop loss at 2%
                elif pnl <= -self.sl:
                    exits.append({"symbol": sym, "action": "STOP_LOSS", 
                                 "reason": f"{pnl:.1f}%", "amount": amt, 
                                 "price": current, "pnl": pnl})
        
        return exits

class Trader:
    """Aggressive trader - more trades, bigger size"""
    
    def __init__(self):
        self.client = Client(RPC_URL)
        self.wallet = Pubkey.from_string(WALLET_ADDRESS)
        self.trade_size_pct = 0.15  # 15% per trade (more aggressive)
    
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
        
        # Track daily PnL
        state["today_pnl"] = state.get("today_pnl", 0) + pnl
        
        del state["positions"][sym]
        return True

class CEO:
    """CEO Agent - Strategic decisions"""
    
    def __init__(self):
        self.name = "CEO"
        self.daily_target = 5.0
    
    def should_trade(self, state, opportunities):
        """Decide if we should trade aggressively"""
        capital = state.get("capital_usd", 500)
        positions = state.get("positions", {})
        today_pnl = state.get("today_pnl", 0)
        
        # Calculate current total
        total = capital + sum(p.get("amount", 0) * 0.00001 for p in positions.values())
        current_pnl_pct = ((total - 500) / 500) * 100
        
        print(f"\n👑 [CEO] Daily Progress: {current_pnl_pct:+.2f}% / {self.daily_target}%")
        
        # If we're behind, be more aggressive
        if current_pnl_pct < self.daily_target:
            print(f"   📈 Behind target - AGGRESSIVE MODE")
            return True
        
        print(f"   ✅ On track or ahead!")
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
        
        # Reset daily PnL if new day
        last_date = self.state.get("last_date", "")
        today = datetime.now().strftime("%Y-%m-%d")
        if last_date != today:
            self.state["today_pnl"] = 0
            self.state["last_date"] = today
    
    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    async def cycle(self, n):
        print(f"\n{'='*60}\n🔄 CYCLE {n} - {datetime.now().strftime('%H:%M:%S')}")
        
        # CEO decides strategy
        prices = self.scanner.scan()
        opps = self.analyst.analyze(prices)
        
        # CEO strategic decision
        self.ceo.should_trade(self.state, opps)
        
        # Check exits first
        exits = self.risk.check_exits(self.state["positions"], prices)
        for ex in exits:
            self.trader.execute_exit(ex, self.state)
            emoji = "🎯" if ex["action"] == "TAKE_PROFIT" else "🛑"
            print(f"   {emoji} {ex['action']} {ex['symbol']}: {ex['reason']}")
        
        # Execute entries (AGGRESSIVE)
        if opps:
            trades_made = 0
            for o in opps:
                if trades_made >= 3:  # Up to 3 trades per cycle
                    break
                    
                if o["action"] in ["BUY", "BUY_MOMENTUM"]:
                    ok, msg = self.risk.validate_entry(o, self.state)
                    price_info = prices.get(o["symbol"])
                    
                    if ok and price_info and isinstance(price_info, dict):
                        self.trader.execute_entry(o["symbol"], price_info.get("price", 0), self.state)
                        print(f"   ✅ {o['action']} {o['symbol']} @ ${price_info.get('price', 0):.6f} - {o['reason']}")
                        trades_made += 1
                    # else:
                        # print(f"   ❌ {o['symbol']}: {msg}")
        
        # Summary
        sol_price = prices.get("SOL", {}).get("price", 0) if isinstance(prices.get("SOL"), dict) else 0
        
        print(f"\n{'='*60}")
        print(f"💵 SOL: ${sol_price:.2f}")
        print(f"💰 Capital: ${self.state['capital_usd']:.2f}")
        
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
        pnl_pct = (pnl / 500) * 100
        
        # Daily goal progress
        daily_pnl = self.state.get("today_pnl", 0)
        
        print(f"💎 Total: ${total:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"🎯 Daily Goal: {daily_pnl:+.2f}% / {self.ceo.daily_target}%")
        print(f"🔄 Trades: {len(self.state['trades'])}")
        
        self.save_state()
    
    async def run(self):
        print("="*60)
        print("🚀 MULTI-AGENT SOLANA TRADING (AGGRESSIVE - 5% DAILY)")
        print("="*60)
        
        n = 0
        while True:
            n += 1
            try:
                await self.cycle(n)
            except Exception as e:
                print(f"❌ {e}")
            await asyncio.sleep(30)  # Faster cycles (30s)

asyncio.run(Orchestrator().run())
