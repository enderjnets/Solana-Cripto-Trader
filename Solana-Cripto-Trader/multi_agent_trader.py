#!/usr/bin/env python3
"""
Multi-Agent Trading - MAX TOKENS + AUTO DISCOVERY + AI ANALYSIS
"""
import asyncio
import json
import random
import requests
import os
import time
from pathlib import Path
from datetime import datetime
from solana.rpc.api import Client
from solders.pubkey import Pubkey

WALLET_ADDRESS = "H9GF6t5hdypfH5PsDhS42sb9ybFWEh5zD5Nht9rQX19a"
RPC_URL = "https://api.devnet.solana.com"
STATE_FILE = Path("~/.config/solana-jupiter-bot/multi_agent_state.json").expanduser()

# MiniMax config
MINIMAX_API_URL = os.getenv("MINIMAX_API_URL", "http://localhost:8090/v1")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")

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
    
    # Exclude stablecoins from trading
    STABLECOINS = {"USDC", "USDT", "DAI", "FRAX", "UST", "BUSD"}
    
    def analyze(self, prices):
        opps = []
        
        for sym, d in prices.items():
            if not isinstance(d, dict):
                continue
            
            # Skip stablecoins - they're not for trading
            if sym in self.STABLECOINS:
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


class AIAnalyzer:
    """Analiza oportunidades con IA (MiniMax)"""
    
    def __init__(self):
        self.api_url = MINIMAX_API_URL
        self.api_key = MINIMAX_API_KEY
        self.group_id = MINIMAX_GROUP_ID
        self.enabled = bool(self.api_key and self.api_key != "your-api-key-here")
        self.cache = {}  # Cache de análisis por símbolo
        self.cache_duration = 300  # 5 minutos
    
    def analyze_opportunity(self, opp, prices, state):
        """Analiza una oportunidad de trade con IA"""
        if not self.enabled:
            return {"approved": True, "confidence": 0.5, "reason": "AI disabled"}
        
        sym = opp["symbol"]
        
        # Cache check
        if sym in self.cache:
            cached = self.cache[sym]
            if time.time() - cached["time"] < self.cache_duration:
                return cached["result"]
        
        # Preparar contexto
        price_data = prices.get(sym, {})
        current_price = price_data.get("price", 0)
        change_24h = price_data.get("change", 0)
        
        # Estado del portfolio
        capital = state.get("capital_usd", 500)
        positions = state.get("positions", {})
        position_value = sum(
            p.get("amount", 0) * prices.get(s, {}).get("price", 0)
            for s, p in positions.items()
        )
        
        # Construir prompt
        prompt = self._build_prompt({
            "symbol": sym,
            "current_price": current_price,
            "change_24h": change_24h,
            "momentum": opp.get("strength", 0),
            "capital": capital,
            "position_value": position_value,
            "num_positions": len(positions),
            "total_trades": len(state.get("trades", [])),
        })
        
        # Llamar a MiniMax
        result = self._call_minimax(prompt)
        
        # Cache resultado
        self.cache[sym] = {
            "result": result,
            "time": time.time()
        }
        
        return result
    
    def _build_prompt(self, data):
        return f"""Eres un experto en trading de criptomonedas. Analiza esta oportunidad de compra:

Símbolo: {data['symbol']}
Precio actual: ${data['current_price']:.4f}
Cambio 24h: {data['change_24h']:+.2f}%
Momentum: {data['momentum']:.2f}

Tu capital: ${data['capital']:.2f}
Valor en posiciones: ${data['position_value']:.2f}
Posiciones abiertas: {data['num_positions']}
Total trades: {data['total_trades']}

Responde en JSON:
{{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "reason": "explicación breve",
  "suggested_size": 0.1-0.2 (porcentaje del capital),
  "stop_loss": 0.5-2.0 (%),
  "take_profit": 1.0-3.0 (%)
}}"""

    def _call_minimax(self, prompt):
        """Llama a la API de MiniMax"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "MiniMax-M2.1",
                "messages": [
                    {"role": "system", "content": "Eres un experto en trading de criptomonedas. Respondes siempre en JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON de la respuesta
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            return {"approved": True, "confidence": 0.5, "reason": "API error, default approval"}
        
        except Exception as e:
            print(f"   ⚠️ AI Error: {e}")
            return {"approved": True, "confidence": 0.5, "reason": f"Error: {str(e)[:20]}"}


class AdaptiveRiskManager:
    """RiskManager con autoaprendizaje"""
    def __init__(self):
        self.max_pos = 6
        self.max_pos_per_cycle = 2
        self.min_momentum = 0.8
        self.max_capital_pct = 0.20
        self.tp = 2.0  # 2% take profit
        self.sl = 1.0  # 1% stop loss (risk)
        self.last_entry_time = {}
        self.cooldown_minutes = 3
        
        # Autoaprendizaje
        self.trade_history = []  # [ {"symbol": "SOL", "result": "TP" | "SL", "pnl": 1.5, "time": timestamp}]
        self.learn_cycle = 10  # Analizar cada 10 trades
        self.token_stats = {}  # { "SOL": {"wins": 3, "losses": 2, "avg_pnl": 1.2} }
        self.best_params = {"tp": 1.5, "sl": 1.0}
        self.adaptive_enabled = True
    
    def record_trade_result(self, symbol, result, pnl):
        """Registra resultado de un trade para aprendizaje"""
        self.trade_history.append({
            "symbol": symbol,
            "result": result,  # "TP" o "SL"
            "pnl": pnl,
            "time": import_time()
        })
        
        # Actualizar stats por token
        if symbol not in self.token_stats:
            self.token_stats[symbol] = {"wins": 0, "losses": 0, "total_pnl": 0}
        
        if result == "TP":
            self.token_stats[symbol]["wins"] += 1
            self.token_stats[symbol]["total_pnl"] += pnl
        else:
            self.token_stats[symbol]["losses"] += 1
            self.token_stats[symbol]["total_pnl"] += pnl
        
        # Feedback loop: analizar cada N trades
        if len(self.trade_history) % self.learn_cycle == 0:
            self.analyze_and_adapt()
    
    def analyze_and_adapt(self):
        """Analiza rendimiento y ajusta parámetros"""
        print(f"\n🧠 [ADAPTIVE] Analizando últimos {len(self.trade_history)} trades...")
        
        # Calcular win rate global
        wins = sum(1 for t in self.trade_history[-self.learn_cycle:] if t["result"] == "TP")
        total = self.learn_cycle
        win_rate = wins / total * 100
        
        # Calcular PnL promedio
        recent_pnl = sum(t["pnl"] for t in self.trade_history[-self.learn_cycle:])
        
        print(f"   Win Rate: {win_rate:.1f}% | PnL: ${recent_pnl:.2f}")
        
        # AJUSTE 1: Si win rate < 40%, aumentar SL y reducir TP
        if win_rate < 40:
            self.sl = max(0.5, self.sl - 0.1)  # Más ajustado
            self.tp = min(3.0, self.tp + 0.2)  # Necesita más recompensa
            print(f"   📉 Ajustando: SL={self.sl}% (más ajustado), TP={self.tp}% (mayor)")
        
        # AJUSTE 2: Si win rate > 70%, ser más agresivo
        elif win_rate > 70:
            self.sl = min(1.5, self.sl + 0.1)
            self.tp = max(1.0, self.tp - 0.1)
            print(f"   📈 Ajustando: SL={self.sl}%, TP={self.tp}% (más agresivo)")
        
        # AJUSTE 3: Analizar tokens específicos
        for sym, stats in self.token_stats.items():
            if stats["wins"] + stats["losses"] >= 3:
                sym_wr = stats["wins"] / (stats["wins"] + stats["losses"]) * 100
                if sym_wr < 30:
                    print(f"   ⚠️ {sym} tiene {sym_wr:.0f}% win rate - evitar trades")
                    # No hacer nada, el validate_entry ya tiene momentum check
        
        # Guardar mejores parámetros
        if recent_pnl > 0:
            self.best_params = {"tp": self.tp, "sl": self.sl}
            print(f"   ✅ Mejores parámetros: TP={self.best_params['tp']}%, SL={self.best_params['sl']}%")
        
        return {"win_rate": win_rate, "pnl": recent_pnl, "tp": self.tp, "sl": self.sl}
    
    def get_token_confidence(self, symbol):
        """Retorna confianza para tradear un token específico"""
        if symbol not in self.token_stats:
            return 0.5  # Neutral
        
        stats = self.token_stats[symbol]
        total = stats["wins"] + stats["losses"]
        if total < 3:
            return 0.5  # No hay suficientes datos
        
        return stats["wins"] / total
    
    def validate_entry(self, opp, state):
        positions = state.get("positions", {})
        capital = state.get("capital_usd", 500)
        
        # Check max positions
        if len(positions) >= self.max_pos:
            return False, "Max positions"
        
        # Check if already has this token
        if opp["symbol"] in positions:
            return False, "Already has"
        
        # Check momentum strength
        strength = opp.get("strength", 0)
        if strength < self.min_momentum:
            return False, f"Weak momentum ({strength:.1f})"
        
        # Check capital per position
        max_per_position = capital * self.max_capital_pct
        if opp.get("cost", 50) > max_per_position:
            return False, "Exceeds cap"
        
        # Check cooldown
        last_time = self.last_entry_time.get(opp["symbol"], 0)
        if import_time() - last_time < self.cooldown_minutes * 60:
            return False, "Cooldown"
        
        # AJUSTE 3: Feedback loop - no entrar si token tiene mal historial
        if self.adaptive_enabled:
            confidence = self.get_token_confidence(opp["symbol"])
            if confidence < 0.25 and self.token_stats.get(opp["symbol"], {}).get("wins", 0) + self.token_stats.get(opp["symbol"], {}).get("losses", 0) >= 3:
                return False, f"Low confidence ({confidence:.0%})"
        
        return True, "OK"
    
    def record_entry(self, symbol):
        self.last_entry_time[symbol] = import_time()
    
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


def import_time():
    import time
    return time.time()


# Alias para compatibilidad
RiskManager = AdaptiveRiskManager


class Trader:
    def __init__(self):
        self.client = Client(RPC_URL)
        self.wallet = Pubkey.from_string(WALLET_ADDRESS)
        self.trade_size_pct = 0.01  # 1% del capital por trade
    
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
        self.ai_analyzer = AIAnalyzer()  # 🤖 IA para análisis profundo
        self.risk = RiskManager()
        self.trader = Trader()
        self.ceo = CEO()
        self.load_state()
        
        # Estado del AI
        if self.ai_analyzer.enabled:
            print(f"\n🤖 AI Analyzer: ACTIVADO (MiniMax)")
        else:
            print(f"\n🤖 AI Analyzer: DESACTIVADO (configura MINIMAX_API_KEY)")
    
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
            
            # Autoaprendizaje: registrar resultado
            result = "TP" if ex["action"] == "TAKE_PROFIT" else "SL"
            self.risk.record_trade_result(ex["symbol"], result, ex.get("pnl", 0))
            
            emoji = "🎯" if ex["action"] == "TAKE_PROFIT" else "🛑"
            print(f"   {emoji} {ex['action']} {ex['symbol']}: {ex['reason']}")
        
        # Execute entries
        if opps:
            trades_made = 0
            for o in opps:
                if trades_made >= self.risk.max_pos_per_cycle:  # Max 2 per cycle
                    break
                    
                if o["action"] == "BUY":
                    ok, msg = self.risk.validate_entry(o, self.state)
                    
                    if not ok:
                        print(f"   ❌ {o['symbol']}: {msg}")
                        continue
                    
                    price_info = prices.get(o["symbol"])
                    
                    if not (price_info and isinstance(price_info, dict)):
                        continue
                    
                    # 🤖 ANÁLISIS IA antes de ejecutar
                    if self.ai_analyzer.enabled:
                        ai_result = self.ai_analyzer.analyze_opportunity(o, prices, self.state)
                        print(f"   🧠 AI Analysis {o['symbol']}: confidence={ai_result.get('confidence', 0):.0%} - {ai_result.get('reason', '')}")
                        
                        if not ai_result.get("approved", True):
                            print(f"   ❌ AI REJECTED {o['symbol']}: {ai_result.get('reason', 'No reason')}")
                            continue
                        
                        # Usar sugerencias de IA si están disponibles
                        if "suggested_size" in ai_result:
                            o["ai_size"] = ai_result["suggested_size"]
                    
                    self.trader.execute_entry(o["symbol"], price_info.get("price", 0), self.state)
                    self.risk.record_entry(o["symbol"])  # Record for cooldown
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
