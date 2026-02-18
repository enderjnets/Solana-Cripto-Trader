#!/usr/bin/env python3
"""
SOLANA TRADING BOT V4 - SCAN MERCADO COMPLETO + PROTECCIONES
Escanea TODO el mercado y elige los mejores según momentum

NUEVO EN V4:
- ✅ Filtro de tendencia (no comprar en mercado bajista)
- ✅ Auto-pause después de 2 SLs seguidos
- ✅ Cooldown de 2 horas después de SL
- ✅ Cooldown general de 90 minutos
"""
import asyncio
import json
import requests
import time
from pathlib import Path
from datetime import datetime

STATE_FILE = Path("~/.config/solana-jupiter-bot/v3_state.json").expanduser()
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


class MarketScanner:
    """Escanea el mercado completo"""
    
    def __init__(self):
        self.max_tokens = 50  # Top 50 tokens
    
    def get_all_prices(self):
        """Obtiene precios de múltiples fuentes"""
        prices = {}
        
        # 1. CoinGecko - Top cryptos
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 50,
                    "page": 1,
                    "sparkline": "false"
                },
                timeout=15
            )
            data = resp.json()
            if isinstance(data, list):
                for coin in data:
                    if isinstance(coin, dict):
                        sym = coin.get("symbol", "").upper()
                        if sym:
                            prices[sym] = {
                                "price": coin.get("current_price", 0) or 0,
                                "change": coin.get("price_change_percentage_24h", 0) or 0,
                                "volume": coin.get("total_volume", 0) or 0,
                                "market_cap": coin.get("market_cap", 0) or 0,
                                "source": "coingecko"
                            }
        except Exception as e:
            print(f"   ⚠️ CoinGecko error: {e}")
        
        # 2. DEX Screener - Solana tokens
        try:
            resp = requests.get(
                "https://api.dexscreener.com/latest/dex/tokens/solana",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    pairs = data.get("pairs", [])
                    if isinstance(pairs, list):
                        for pair in pairs[:50]:
                            if isinstance(pair, dict):
                                token = pair.get("baseToken", {})
                                if token and isinstance(token, dict):
                                    sym = token.get("symbol", "").upper()
                                    if sym:
                                        price_usd = pair.get("priceUsd")
                                        if price_usd and sym not in prices:
                                            price_change = pair.get("priceChange", {})
                                            if isinstance(price_change, dict):
                                                change = price_change.get("h24", 0)
                                            else:
                                                change = 0
                                            prices[sym] = {
                                                "price": float(price_usd),
                                                "change": float(change) if change else 0,
                                                "volume": float(pair.get("volume", {}).get("h24", 0)) if isinstance(pair.get("volume"), dict) else 0,
                                                "market_cap": 0,
                                                "source": "dexscreener"
                                            }
        except Exception as e:
            print(f"   ⚠️ DEX Screener error: {e}")
        
        return prices
    
    def find_best_opportunities(self, prices, top_n=5):
        """Encuentra las mejores oportunidades de compra (dips)"""
        opportunities = []
        
        for sym, data in prices.items():
            if not isinstance(data, dict):
                continue
            
            change = data.get("change", 0)
            
            # Skip stablecoins
            if sym in ["USDT", "USDC", "DAI", "FRAX", "BUSD", "USD", "UST"]:
                continue
            
            # BUY signal: dip > 4% (subido para mejor calidad)
            if change < -4:
                opportunities.append({
                    "symbol": sym,
                    "action": "BUY",
                    "strength": abs(change),
                    "price": data.get("price", 0),
                    "change": change,
                    "volume": data.get("volume", 0),
                    "reason": f"Dip: {change:.1f}%"
                })
            
            # SELL signal: pump > 5%
            elif change > 5:
                opportunities.append({
                    "symbol": sym,
                    "action": "SELL",
                    "strength": change,
                    "price": data.get("price", 0),
                    "change": change,
                    "volume": data.get("volume", 0),
                    "reason": f"Pump: {change:.1f}%"
                })
        
        # Sort by strength (absolute change)
        opportunities.sort(key=lambda x: x["strength"], reverse=True)
        
        return opportunities[:top_n]


class TradingStrategy:
    """
    Estrategia OPTIMIZADA V4 - Con protecciones
    """
    
    def __init__(self):
        self.max_positions = 2
        self.trade_size_pct = 0.10  # 10%
        self.tp_pct = 5.0  # 5%
        self.sl_pct = 3.0  # 3%
        
        # 🔒 PROTECCIONES NUEVAS
        self.cooldown_minutes = 90  # 90 min después de SL (era 30)
        self.sl_cooldown_hours = 2  # 2 horas después de SL antes de re-entry
        self.max_sl_in_a_row = 2  # Auto-pause después de 2 SLs seguidos
        
        self.last_entry = {}
        self.last_sl_time = {}  # Track cuando fue el último SL por token
        self.sl_count = 0  # Contador de SLs consecutivos
        self.is_paused = False  # Auto-pause mode
        self.pause_until = 0  # Tiempo hasta que se activa el bot
        
        self.daily_trades = 0
        self.last_reset = ""
        
        # Stats
        self.wins = 0
        self.losses = 0
    
    def reset_daily(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_reset != today:
            self.daily_trades = 0
            self.last_reset = today
            # Reset SL count cada día nuevo
            self.sl_count = 0
            self.is_paused = False
    
    def check_pause(self):
        """Verifica si el bot debe estar en pause"""
        if self.is_paused:
            if time.time() < self.pause_until:
                return True, f"Paused until {datetime.fromtimestamp(self.pause_until).strftime('%H:%M')}"
            else:
                # Fin del pause
                self.is_paused = False
                self.sl_count = 0
                print("   ✅ Bot reactivado después de pause")
        return False, ""
    
    def trigger_pause(self, reason):
        """Activa el modo pause"""
        self.is_paused = True
        self.pause_until = time.time() + (60 * 60)  # 1 hora de pause
        print(f"   ⚠️ BOT PAUSADO: {reason}")
        print(f"   ⏰ Reactivando en 1 hora...")
    
    def validate_entry(self, symbol, positions, market_trend="neutral"):
        """Validación MEJORADA con protecciones"""
        
        # 1. Check pause
        paused, msg = self.check_pause()
        if paused:
            return False, f"Bot paused: {msg}"
        
        # 2. Check max positions
        if len(positions) >= self.max_positions:
            return False, "Max positions"
        
        # 3. Check if already have
        if symbol in positions:
            return False, "Already have"
        
        # 4. Check cooldown normal
        last = self.last_entry.get(symbol, 0)
        if time.time() - last < self.cooldown_minutes * 60:
            return False, f"Cooldown active ({int((time.time()-last)/60)}min)"
        
        # 5. Check SL cooldown específico
        last_sl = self.last_sl_time.get(symbol, 0)
        if time.time() - last_sl < self.sl_cooldown_hours * 3600:
            return False, f"SL cooldown active ({int((time.time()-last_sl)/3600)}h)"
        
        # 6. 🆕 FILTRO DE TENDENCIA - Solo comprar si mercado está UP
        if market_trend == "down":
            return False, "Market down - no entries"
        
        return True, "OK"
    
    def check_exits(self, positions, prices):
        exits = []
        for sym, data in positions.items():
            if sym not in prices:
                continue
            
            current = prices[sym].get("price", 0)
            entry = data.get("entry_price", 0)
            amt = data.get("amount", 0)
            
            if entry > 0 and amt > 0 and current > 0:
                pnl_pct = ((current - entry) / entry) * 100
                
                if pnl_pct >= self.tp_pct:
                    exits.append({
                        "symbol": sym, "action": "TAKE_PROFIT",
                        "pnl": pnl_pct, "amount": amt, "price": current
                    })
                elif pnl_pct <= -self.sl_pct:
                    exits.append({
                        "symbol": sym, "action": "STOP_LOSS",
                        "pnl": pnl_pct, "amount": amt, "price": current
                    })
        
        return exits
    
    def execute_entry(self, symbol, price, state):
        cost = state["capital_usd"] * self.trade_size_pct
        amount = cost / price if price > 0 else 0
        
        state["positions"][symbol] = {
            "amount": amount,
            "entry_price": price,
            "entry_time": datetime.now().isoformat()
        }
        
        state["capital_usd"] -= cost
        state["trades"].append({
            "time": datetime.now().isoformat(),
            "action": "BUY", "symbol": symbol,
            "price": price, "amount": amount, "cost": cost
        })
        
        self.last_entry[symbol] = time.time()
        self.daily_trades += 1
    
    def execute_exit(self, exit_info, state):
        sym = exit_info["symbol"]
        amt = exit_info["amount"]
        price = exit_info["price"]
        pnl = exit_info["pnl"]
        
        proceeds = amt * price
        state["capital_usd"] += proceeds
        
        if exit_info["action"] == "TAKE_PROFIT":
            self.wins += 1
            # Reset SL counter after win
            self.sl_count = 0
            print(f"   🎯 WIN! Reset SL counter")
        else:
            self.losses += 1
            # Track SL time para cooldown
            self.last_sl_time[sym] = time.time()
            # Increment SL counter
            self.sl_count += 1
            print(f"   🛑 SL #{self.sl_count}")
            
            # 🚨 AUTO-PAUSE si 2 SLs seguidos
            if self.sl_count >= self.max_sl_in_a_row:
                self.trigger_pause(f"{self.sl_count} SLs in a row")
        
        state["trades"].append({
            "time": datetime.now().isoformat(),
            "action": exit_info["action"], "symbol": sym,
            "price": price, "proceeds": proceeds, "pnl": pnl
        })
        
        state["today_pnl"] = state.get("today_pnl", 0) + pnl
        del state["positions"][sym]
    
    def save_state(self, state):
        state["stats"] = {
            "win_rate": self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0.5,
            "wins": self.wins,
            "losses": self.losses,
            "sl_count": self.sl_count,
            "is_paused": self.is_paused
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)


class TradingBot:
    def __init__(self):
        self.scanner = MarketScanner()
        self.strategy = TradingStrategy()
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
        
        self.strategy.reset_daily()
        
        stats = self.state.get("stats", {})
        self.strategy.wins = stats.get("wins", 0)
        self.strategy.losses = stats.get("losses", 0)
        self.strategy.sl_count = stats.get("sl_count", 0)
        
        # Si hay 2+ SLs, pausar inmediatamente
        if self.strategy.sl_count >= self.strategy.max_sl_in_a_row:
            self.strategy.trigger_pause(f"Loaded with {self.strategy.sl_count} SLs")
    
    def save_state(self):
        self.strategy.save_state(self.state)
    
    async def cycle(self, n):
        print(f"\n{'='*60}")
        print(f"🔄 CYCLE {n} | {datetime.now().strftime('%H:%M:%S')}")
        
        # ESCANEAR MERCADO COMPLETO
        prices = self.scanner.get_all_prices()
        print(f"   📊 Mercado: {len(prices)} tokens escaneados")
        
        # Mostrar top movimientos
        sorted_prices = sorted(
            prices.items(),
            key=lambda x: x[1].get("change", 0) if isinstance(x[1], dict) else 0,
            reverse=True
        )[:10]
        
        print(f"\n   🔺 TOP GAINERS:")
        for sym, data in sorted_prices[:5]:
            if isinstance(data, dict):
                print(f"      {sym}: ${data.get('price', 0):.4f} ({data.get('change', 0):+.1f}%)")
        
        print(f"\n   🔻 TOP LOSERS:")
        for sym, data in sorted_prices[-5:]:
            if isinstance(data, dict):
                print(f"      {sym}: ${data.get('price', 0):.4f} ({data.get('change', 0):+.1f}%)")
        
        # 🆕 DETECTAR TENDENCIA DEL MERCADO
        top10_changes = [data.get("change", 0) for sym, data in sorted_prices[:10] if isinstance(data, dict)]
        avg_change = sum(top10_changes) / len(top10_changes) if top10_changes else 0
        
        if avg_change > 1:
            market_trend = "up"
            trend_emoji = "📈"
        elif avg_change < -1:
            market_trend = "down"
            trend_emoji = "📉"
        else:
            market_trend = "neutral"
            trend_emoji = "➡️"
        
        print(f"\n   {trend_emoji} Mercado: {market_trend.upper()} ({avg_change:+.1f}% avg)")
        
        # 🆕 MOSTRAR STATUS DE PAUSE
        if self.strategy.is_paused:
            print(f"   ⚠️ BOT PAUSADO: Esperando para reanudar...")
        
        # Encontrar mejores oportunidades
        opportunities = self.scanner.find_best_opportunities(prices, top_n=5)
        
        if opportunities:
            buys = [o for o in opportunities if o["action"] == "BUY"]
            sells = [o for o in opportunities if o["action"] == "SELL"]
            
            if buys:
                print(f"\n   🎯 BUY Signals ({len(buys)}):")
                for o in buys[:3]:
                    print(f"      {o['symbol']}: {o['reason']} @ ${o['price']:.4f}")
            
            if sells:
                print(f"\n   💰 SELL Signals ({len(sells)}):")
                for o in sells[:3]:
                    print(f"      {o['symbol']}: {o['reason']} @ ${o['price']:.4f}")
        
        # Check exits
        exits = self.strategy.check_exits(self.state["positions"], prices)
        for ex in exits:
            self.strategy.execute_exit(ex, self.state)
            emoji = "🎯" if ex["action"] == "TAKE_PROFIT" else "🛑"
            print(f"\n   {emoji} {ex['action']} {ex['symbol']}: {ex['pnl']:+.1f}%")
        
        # Execute entries - elegir mejor oportunidad
        if opportunities:
            for o in opportunities:
                if len(self.state["positions"]) >= self.strategy.max_positions:
                    break
                
                if o["action"] != "BUY":
                    continue
                
                ok, msg = self.strategy.validate_entry(o["symbol"], self.state["positions"], market_trend)
                if not ok:
                    continue
                
                price = o.get("price", 0)
                if price <= 0:
                    continue
                
                self.strategy.execute_entry(o["symbol"], price, self.state)
                print(f"\n   ✅ BUY {o['symbol']} @ ${price:.4f} ({self.strategy.trade_size_pct*100}% capital)")
        
        # Summary
        total_pos_value = 0
        print(f"\n   📊 POSITIONS:")
        for sym, data in self.state["positions"].items():
            price = prices.get(sym, {}).get("price", 0) if isinstance(prices.get(sym), dict) else 0
            val = data.get("amount", 0) * price
            entry = data.get("entry_price", 0)
            pnl = ((price - entry) / entry * 100) if entry > 0 and price > 0 else 0
            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"      {sym}: {emoji} {pnl:+.1f}% (${val:.2f})")
            total_pos_value += val
        
        total = self.state["capital_usd"] + total_pos_value
        pnl = total - 500
        pnl_pct = (pnl / 500) * 100
        wr = self.strategy.wins / (self.strategy.wins + self.strategy.losses) * 100 if (self.strategy.wins + self.strategy.losses) > 0 else 50
        
        print(f"\n{'='*60}")
        print(f"💰 Capital: ${self.state['capital_usd']:.2f}")
        print(f"📈 Positions: ${total_pos_value:.2f}")
        print(f"💎 Total: ${total:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"🎯 WR: {wr:.0f}% | Trades: {self.strategy.daily_trades}")
        
        self.save_state()
    
    async def run(self):
        print("="*60)
        print("🚀 SOLANA BOT V3 - MERCADO COMPLETO")
        print("="*60)
        print(f"📊 Escanea: TODO el mercado")
        print(f"💰 Trade Size: {self.strategy.trade_size_pct*100}%")
        print(f"🎯 TP: {self.strategy.tp_pct}% | 🛑 SL: {self.strategy.sl_pct}%")
        
        n = 0
        while True:
            n += 1
            try:
                await self.cycle(n)
            except KeyboardInterrupt:
                print("\n🛑 Detenido")
                break
            except Exception as e:
                import traceback
                print(f"❌ Error: {e}")
                print(traceback.format_exc(limit=2))
            
            await asyncio.sleep(30)


if __name__ == "__main__":
    try:
        asyncio.run(TradingBot().run())
    except KeyboardInterrupt:
        print("\n🛑 Detenido")
