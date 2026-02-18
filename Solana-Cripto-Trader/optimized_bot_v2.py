#!/usr/bin/env python3
"""
SOLANA TRADING BOT - ESTRATEGIA OPTIMIZADA V2
Objetivo: 5% diario con gestión de riesgo mejorada
"""
import asyncio
import json
import requests
import time
from pathlib import Path
from datetime import datetime
from solana.rpc.api import Client
from solders.pubkey import Pubkey

WALLET_ADDRESS = "H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a"
RPC_URL = "https://api.devnet.solana.com"
STATE_FILE = Path("~/.config/solana-jupiter-bot/optimized_state.json").expanduser()
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Tokens principales - SOLO LOS MEJORES
TOP_TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tFm7mkBZKaUTwJWS3d",
    "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP",
}


class OptimizedStrategy:
    """
    Estrategia optimizada para el objetivo 5% diario
    
    Cambios vs V1:
    - TP: 5% (antes 2.5%) - cubre comisiones + gain
    - SL: 2% (antes 1%) - menos whipsaws
    - Solo 2 posiciones max (antes 4)
    - Solo entradas fuertes (>3% cambio)
    - Tamaño: 5% del capital (antes 1%)
    - Cooldown: 30 min (antes 5 min)
    """
    
    def __init__(self):
        # PARÁMETROS OPTIMIZADOS
        self.max_positions = 2  # Reducido de 4
        self.trade_size_pct = 0.05  # 5% del capital (antes 1%)
        self.tp_pct = 5.0  # 5% take profit
        self.sl_pct = 2.0  # 2% stop loss (antes 1%)
        self.min_momentum = 3.0  # Solo entradas fuertes (antes 1.5)
        self.cooldown_minutes = 30  # 30 min cooldown
        self.max_daily_trades = 8  # Máximo trades por día
        
        # Tracking
        self.last_entry_time = {}
        self.daily_trades = 0
        self.last_reset = ""
        
        # Stats
        self.win_rate = 0.5
        self.total_wins = 0
        self.total_losses = 0
    
    def reset_daily(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_reset != today:
            self.daily_trades = 0
            self.last_reset = today
    
    def get_prices(self):
        """Obtener precios de Jupiter + CoinGecko"""
        prices = {}
        
        # Jupiter
        try:
            ids = ",".join(TOP_TOKENS.values())
            resp = requests.get("https://lite-api.jup.ag/price/v3", 
                              params={"ids": ids}, timeout=10)
            data = resp.json()
            for sym, mint in TOP_TOKENS.items():
                if mint in data and isinstance(data[mint], dict):
                    prices[sym] = {
                        "price": float(data[mint].get("usdPrice", 0)),
                        "change": float(data[mint].get("priceChange24h", 0))
                    }
        except Exception as e:
            print(f"   ⚠️ Jupiter error: {e}")
        
        # CoinGecko fallback
        try:
            cg_ids = "solana,bitcoin,ethereum,dogwifhat,bonk"
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_ids, "vs_currencies": "usd", "include_24hr_change": "true"},
                timeout=10
            )
            data = resp.json()
            mapping = {"solana": "SOL", "bitcoin": "BTC", "ethereum": "ETH", 
                      "dogwifhat": "WIF", "bonk": "BONK"}
            for cg, sym in mapping.items():
                if cg in data and sym not in prices:
                    prices[sym] = {
                        "price": float(data[cg].get("usd", 0)),
                        "change": float(data[cg].get("usd_24h_change", 0))
                    }
        except Exception as e:
            print(f"   ⚠️ CoinGecko error: {e}")
        
        return prices
    
    def find_opportunities(self, prices):
        """Encontrar oportunidades de compra"""
        opps = []
        
        for sym, data in prices.items():
            if not isinstance(data, dict):
                continue
            
            change = data.get("change", 0)
            
            # SOLO comprar en dips significativos
            if change < -self.min_momentum:
                opps.append({
                    "symbol": sym,
                    "action": "BUY",
                    "strength": abs(change),
                    "price": data.get("price", 0),
                    "reason": f"Dip: {change:+.1f}%"
                })
        
        return sorted(opps, key=lambda x: x["strength"], reverse=True)
    
    def validate_entry(self, symbol, state):
        """Validar si podemos entrar"""
        positions = state.get("positions", {})
        
        # Check max positions
        if len(positions) >= self.max_positions:
            return False, "Max positions reached"
        
        # Check already has
        if symbol in positions:
            return False, "Already have position"
        
        # Check daily limit
        if self.daily_trades >= self.max_daily_trades:
            return False, "Daily limit reached"
        
        # Check cooldown
        last = self.last_entry_time.get(symbol, 0)
        if time.time() - last < self.cooldown_minutes * 60:
            return False, f"Cooldown active ({int((time.time()-last)/60)}min)"
        
        return True, "OK"
    
    def check_exits(self, positions, prices):
        """Check exits for TP/SL"""
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
                pnl_pct = ((current - entry) / entry) * 100
                
                if pnl_pct >= self.tp_pct:
                    exits.append({
                        "symbol": sym,
                        "action": "TAKE_PROFIT",
                        "pnl": pnl_pct,
                        "amount": amt,
                        "price": current,
                        "reason": f"TP: +{pnl_pct:.1f}%"
                    })
                elif pnl_pct <= -self.sl_pct:
                    exits.append({
                        "symbol": sym,
                        "action": "STOP_LOSS",
                        "pnl": pnl_pct,
                        "amount": amt,
                        "price": current,
                        "reason": f"SL: {pnl_pct:.1f}%"
                    })
        
        return exits
    
    def execute_entry(self, symbol, price, state):
        """Ejecutar entrada"""
        cost_pct = self.trade_size_pct
        cost = state["capital_usd"] * cost_pct
        amount = cost / price if price > 0 else 0
        
        state["positions"][symbol] = {
            "amount": amount,
            "entry_price": price,
            "entry_time": datetime.now().isoformat()
        }
        
        state["capital_usd"] -= cost
        state["trades"].append({
            "time": datetime.now().isoformat(),
            "action": "BUY",
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "cost": cost
        })
        
        self.last_entry_time[symbol] = time.time()
        self.daily_trades += 1
        
        return True
    
    def execute_exit(self, exit_info, state):
        """Ejecutar salida"""
        sym = exit_info["symbol"]
        amt = exit_info["amount"]
        price = exit_info["price"]
        pnl = exit_info["pnl"]
        
        proceeds = amt * price
        state["capital_usd"] += proceeds
        
        action = exit_info["action"]
        
        state["trades"].append({
            "time": datetime.now().isoformat(),
            "action": action,
            "symbol": sym,
            "price": price,
            "proceeds": proceeds,
            "pnl": pnl
        })
        
        state["today_pnl"] = state.get("today_pnl", 0) + pnl
        
        # Update stats
        if action == "TAKE_PROFIT":
            self.total_wins += 1
        else:
            self.total_losses += 1
        
        self.win_rate = self.total_wins / (self.total_wins + self.total_losses) if (self.total_wins + self.total_losses) > 0 else 0.5
        
        del state["positions"][sym]
        
        return True
    
    def save_state(self, state):
        """Guardar estado"""
        state["stats"] = {
            "win_rate": self.win_rate,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "tp": self.tp_pct,
            "sl": self.sl_pct,
            "size_pct": self.trade_size_pct
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)


class TradingBot:
    def __init__(self):
        self.client = Client(RPC_URL)
        self.wallet = Pubkey.from_string(WALLET_ADDRESS)
        self.strategy = OptimizedStrategy()
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
                "today_pnl": 0
            }
        
        # Reset diario
        self.strategy.reset_daily()
        
        # Cargar stats
        stats = self.state.get("stats", {})
        self.strategy.total_wins = stats.get("total_wins", 0)
        self.strategy.total_losses = stats.get("total_losses", 0)
        self.strategy.win_rate = stats.get("win_rate", 0.5)
    
    def save_state(self):
        self.strategy.save_state(self.state)
    
    async def cycle(self, n):
        print(f"\n{'='*55}")
        print(f"🔄 CYCLE {n} | {datetime.now().strftime('%H:%M:%S')}")
        
        # Get prices
        prices = self.strategy.get_prices()
        print(f"   📊 Prices loaded: {len(prices)} tokens")
        
        # Mostrar precios
        for sym, data in sorted(prices.items(), key=lambda x: x[1].get("change", 0)):
            change = data.get("change", 0)
            emoji = "🟢" if change >= 0 else "🔴"
            price_str = f"${data['price']:.4f}" if data['price'] < 1 else f"${data['price']:.2f}"
            print(f"   {emoji} {sym}: {price_str} ({change:+.1f}%)")
        
        # Check exits
        exits = self.strategy.check_exits(self.state["positions"], prices)
        for ex in exits:
            self.strategy.execute_exit(ex, self.state)
            emoji = "🎯" if ex["action"] == "TAKE_PROFIT" else "🛑"
            print(f"   {emoji} {ex['action']} {ex['symbol']}: {ex['reason']}")
        
        # Find opportunities
        opps = self.strategy.find_opportunities(prices)
        
        if opps:
            print(f"\n   🎯 Opportunities: {len(opps)}")
            for o in opps[:3]:
                print(f"      {o['symbol']}: {o['reason']}")
        
        # Execute entries
        if opps and len(self.state["positions"]) < self.strategy.max_positions:
            for o in opps:
                if len(self.state["positions"]) >= self.strategy.max_positions:
                    break
                
                ok, msg = self.strategy.validate_entry(o["symbol"], self.state)
                
                if not ok:
                    print(f"   ❌ {o['symbol']}: {msg}")
                    continue
                
                price = o.get("price", 0)
                if price <= 0:
                    continue
                
                self.strategy.execute_entry(o["symbol"], price, self.state)
                print(f"   ✅ BUY {o['symbol']} @ ${price:.4f} ({self.strategy.trade_size_pct*100}% capital)")
        
        # Summary
        total_pos_value = 0
        for sym, data in self.state["positions"].items():
            if isinstance(data, dict):
                price = prices.get(sym, {}).get("price", 0) if isinstance(prices.get(sym), dict) else 0
                val = data.get("amount", 0) * price
                entry = data.get("entry_price", 0)
                pnl = ((price - entry) / entry * 100) if entry > 0 and price > 0 else 0
                emoji = "🟢" if pnl >= 0 else "🔴"
                print(f"   📊 {sym}: {emoji} {pnl:+.1f}% (${val:.2f})")
                total_pos_value += val
        
        total = self.state["capital_usd"] + total_pos_value
        pnl = total - 500
        pnl_pct = (pnl / 500) * 100
        
        print(f"\n{'='*55}")
        print(f"💰 Capital: ${self.state['capital_usd']:.2f}")
        print(f"📈 Positions: ${total_pos_value:.2f}")
        print(f"💎 Total: ${total:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"🎯 Win Rate: {self.strategy.win_rate*100:.0f}% | Today: {self.strategy.daily_trades} trades")
        print(f"⚙️ TP: {self.strategy.tp_pct}% | SL: {self.strategy.sl_pct}% | Size: {self.strategy.trade_size_pct*100}%")
        
        self.save_state()
    
    async def run(self):
        print("="*55)
        print("🚀 SOLANA BOT V2 - OPTIMIZADO 5%/DÍA")
        print("="*55)
        print(f"📊 Max Positions: {self.strategy.max_positions}")
        print(f"💰 Trade Size: {self.strategy.trade_size_pct*100}%")
        print(f"🎯 TP: {self.strategy.tp_pct}% | 🛑 SL: {self.strategy.sl_pct}%")
        print(f"🔒 Min Momentum: {self.strategy.min_momentum}%")
        
        n = 0
        while True:
            n += 1
            try:
                await self.cycle(n)
            except KeyboardInterrupt:
                print("\n🛑 Bot detenido")
                break
            except Exception as e:
                import traceback
                print(f"❌ Error: {e}")
                print(traceback.format_exc(limit=2))
            
            await asyncio.sleep(30)  # 30 segundos entre ciclos


if __name__ == "__main__":
    try:
        asyncio.run(TradingBot().run())
    except KeyboardInterrupt:
        print("\n🛑 Detenido")
