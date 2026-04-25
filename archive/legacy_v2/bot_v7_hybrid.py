#!/usr/bin/env python3
"""
SOLANA TRADING BOT V7 - HYBRID AUTO-EVOLUTIVE (V7.1)
====================================================
Combina lo mejor de V5 y V6:
- 🧬 Algoritmo Genético AVANZADO (14 genes)
- 📈📉 SHORT/LONG con Leverage
- 🎯 Auto-Optimización de estrategias
- 💰 Paper Trading con apalancamiento
"""

import asyncio
import json
import random
import time
import requests
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# Alerts module
from alerts import alerts

# ============================================================================
# CONFIGURATION
# ============================================================================

STATE_FILE = Path("~/.config/solana-jupiter-bot/v7_state.json").expanduser()
STRATEGIES_FILE = Path("~/.config/solana-jupiter-bot/v7_strategies.json").expanduser()
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Genetic Algorithm Parameters
POPULATION_SIZE = 30
MUTATION_RATE = 0.2
ELITE_SIZE = 5

# Leverage Parameters
MAX_LEVERAGE = 5

# Token list
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tFm7mkBZKaUUTwJWS3d",
}

# ============================================================================
# STRATEGY CLASS WITH 14 GENES
# ============================================================================

class Strategy:
    """Strategy with 14 genes for sophisticated trading"""
    
    def __init__(self, genes: dict = None):
        if genes:
            # Initialize from genes dict
            self.entry_threshold = genes.get('entry_threshold', 0)
            self.stop_loss = genes.get('stop_loss', 3)
            self.take_profit = genes.get('take_profit', 5)
            self.leverage = genes.get('leverage', 2)
            self.prefer_short = genes.get('prefer_short', 0.5)
            self.min_volume = genes.get('min_volume', 1000000)
            self.trend_filter = genes.get('trend_filter', 0)
            self.trailing_stop = genes.get('trailing_stop', 0)
            self.trail_distance = genes.get('trail_distance', 1.5)
            self.position_size = genes.get('position_size', 10)
            self.cooldown = genes.get('cooldown', 30)
            self.max_positions = genes.get('max_positions', 3)
            self.volatility_filter = genes.get('volatility_filter', 1)
            self.partial_tp = genes.get('partial_tp', 0)
        else:
            # Random initialization
            self.entry_threshold = random.uniform(-10, 10)
            self.stop_loss = random.uniform(1, 10)
            self.take_profit = random.uniform(3, 15)
            self.leverage = random.uniform(1, MAX_LEVERAGE)
            self.prefer_short = random.random()
            self.min_volume = random.uniform(1000000, 100000000)
            self.trend_filter = random.random()
            self.trailing_stop = random.random()
            self.trail_distance = random.uniform(0.5, 3)
            self.position_size = random.uniform(5, 20)
            self.cooldown = random.uniform(0, 60)
            self.max_positions = random.randint(1, 5)
            self.volatility_filter = random.uniform(0, 5)
            self.partial_tp = random.random()
        
        # Results
        self.fitness = 0
        self.trades = 0
        self.win_rate = 0
        self.profit = 0
        self.max_dd = 0
    
    def to_dict(self):
        return {
            # Core
            'entry_threshold': self.entry_threshold,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'leverage': self.leverage,
            'prefer_short': self.prefer_short,
            # Advanced
            'min_volume': self.min_volume,
            'trend_filter': self.trend_filter,
            'trailing_stop': self.trailing_stop,
            'trail_distance': self.trail_distance,
            'position_size': self.position_size,
            'cooldown': self.cooldown,
            'max_positions': self.max_positions,
            'volatility_filter': self.volatility_filter,
            'partial_tp': self.partial_tp,
            # Results
            'fitness': self.fitness,
            'trades': self.trades,
            'win_rate': self.win_rate,
            'profit': self.profit,
            'max_dd': self.max_dd
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(d)
    
    def mutate(self):
        """Mutate genes with probability MUTATION_RATE"""
        genes = [
            ('entry_threshold', -10, 10),
            ('stop_loss', 1, 10),
            ('take_profit', 3, 15),
            ('leverage', 1, MAX_LEVERAGE),
            ('prefer_short', 0, 1),
            ('min_volume', 1000000, 100000000),
            ('trend_filter', 0, 1),
            ('trailing_stop', 0, 1),
            ('trail_distance', 0.5, 3),
            ('position_size', 5, 20),
            ('cooldown', 0, 60),
            ('max_positions', 1, 5),
            ('volatility_filter', 0, 5),
            ('partial_tp', 0, 1),
        ]
        
        for gene, min_val, max_val in genes:
            if random.random() < MUTATION_RATE:
                setattr(self, gene, random.uniform(min_val, max_val))
        
        # Ensure valid ranges
        self.stop_loss = max(1, min(10, self.stop_loss))
        self.take_profit = max(3, min(15, self.take_profit))
        self.leverage = max(1, min(MAX_LEVERAGE, self.leverage))
        self.max_positions = max(1, min(5, int(self.max_positions)))
    
    def crossover(self, other):
        """Crossover with another strategy"""
        child_genes = {}
        for key in self.to_dict():
            if key in ['fitness', 'trades', 'win_rate', 'profit', 'max_dd']:
                continue
            child_genes[key] = random.choice([
                getattr(self, key), 
                getattr(other, key)
            ])
        return Strategy(child_genes)
    
    def __repr__(self):
        return f"Strategy(entry={self.entry_threshold:.1f}, SL={self.stop_loss:.1f}, TP={self.take_profit:.1f}, LEV={self.leverage:.1f}x)"


# ============================================================================
# PRICE FETCHER (Hybrid: CoinGecko for BTC/ETH, Jupiter for SOL tokens)
# ============================================================================

def get_prices() -> Dict:
    prices = {}
    
    # Step 1: Get BTC & ETH from CoinGecko (accurate USD prices)
    try:
        cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
        cg_resp = requests.get(cg_url, timeout=10)
        if cg_resp.status_code == 200:
            cg_data = cg_resp.json()
            if "bitcoin" in cg_data:
                prices["BTC"] = {
                    "price": float(cg_data["bitcoin"]["usd"]),
                    "change": float(cg_data["bitcoin"]["usd_24h_change"]),
                    "volume": 0
                }
            if "ethereum" in cg_data:
                prices["ETH"] = {
                    "price": float(cg_data["ethereum"]["usd"]),
                    "change": float(cg_data["ethereum"]["usd_24h_change"]),
                    "volume": 0
                }
    except Exception as e:
        print(f"   ⚠️ CoinGecko error: {e}")
    
    # Step 2: Get SOL and other tokens from Jupiter (on-chain prices)
    try:
        # Only Solana tokens (not BTC/ETH)
        sol_tokens = {k: v for k, v in TOKENS.items() if k not in ["BTC", "ETH"]}
        if sol_tokens:
            ids = ",".join(sol_tokens.values())
            url = f"https://lite-api.jup.ag/price/v3?ids={ids}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                addr_to_sym = {v: k for k, v in sol_tokens.items()}
                
                for addr, info in data.items():
                    sym = addr_to_sym.get(addr, addr[:8])
                    prices[sym] = {
                        "price": float(info.get("usdPrice", 0)),
                        "change": float(info.get("priceChange24h", 0)),
                        "volume": float(info.get("volume24h", 0))
                    }
    except Exception as e:
        print(f"   ⚠️ Jupiter error: {e}")
    
    return prices


# ============================================================================
# GENETIC OPTIMIZER
# ============================================================================

class GeneticOptimizer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.population: List[Strategy] = []
    
    def initialize(self, size: int = POPULATION_SIZE):
        self.population = [Strategy() for _ in range(size)]
        print(f"   🧬 Población: {size} estrategias")
    
    def evaluate(self, prices_history: List[Dict]) -> List[Dict]:
        results = []
        
        for strategy in self.population:
            cash = 500
            position = 0
            entry_price = 0
            direction = "long"
            trades = wins = losses = 0
            max_dd = peak = 0
            
            for i in range(1, len(prices_history)):
                current = prices_history[i]["price"]
                prev = prices_history[i-1]["price"]
                change = ((current - prev) / prev) * 100
                
                # Determine direction
                if change < strategy.entry_threshold and strategy.prefer_short > 0.5:
                    direction = "short"
                elif change > strategy.entry_threshold:
                    direction = "long"
                else:
                    direction = None
                
                # Volume filter
                volume = prices_history[i].get("volume", 0)
                if volume < strategy.min_volume:
                    direction = None
                
                # Volatility filter
                if abs(change) < strategy.volatility_filter:
                    direction = None
                
                # Open position
                if position == 0 and direction:
                    position_size = cash * (strategy.position_size / 100)
                    position = (position_size * strategy.leverage) / current
                    entry_price = current
                    cash -= position_size
            
            # Close and calculate
            if position > 0 and entry_price > 0:
                if direction == "short":
                    pnl_pct = ((entry_price - current) / entry_price) * strategy.leverage * 100
                else:
                    pnl_pct = ((current - entry_price) / entry_price) * strategy.leverage * 100
                
                cash += (position * current / strategy.leverage)
                trades = 1
                if pnl_pct > 0:
                    wins = 1
                else:
                    losses = 1
            
            # Calculate fitness
            profit_pct = ((cash - 500) / 500) * 100
            win_rate = wins / trades if trades > 0 else 0
            fitness = profit_pct * (win_rate + 0.5) / (max_dd / 100 + 1)
            
            strategy.trades = trades
            strategy.win_rate = win_rate
            strategy.profit = profit_pct
            strategy.max_dd = max_dd
            strategy.fitness = fitness
            
            results.append(strategy.to_dict())
        
        results.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        return results
    
    def evolve(self, best_results: List[Dict]):
        if not best_results:
            return
        
        # Keep elites
        new_pop = [Strategy.from_dict(d) for d in best_results[:ELITE_SIZE]]
        
        # Fill with crossover + mutation
        while len(new_pop) < POPULATION_SIZE:
            if len(best_results) >= 2:
                p1 = Strategy.from_dict(random.choice(best_results[:10]))
                p2 = Strategy.from_dict(random.choice(best_results[:10]))
                child = p1.crossover(p2)
            else:
                child = Strategy()
            
            child.mutate()
            new_pop.append(child)
        
        self.population = new_pop


# ============================================================================
# TRADING ENGINE
# ============================================================================

class TradingEngine:
    def __init__(self):
        self.state = self.load_state()
    
    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {
            "capital": 500,
            "positions": {},
            "trades": [],
            "stats": {"wins": 0, "losses": 0}
        }
    
    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def open_position(self, symbol, direction, price, leverage, strategy):
        if symbol in self.state["positions"]:
            return None
        
        position_size = self.state["capital"] * (strategy.position_size / 100)
        collateral = position_size
        
        self.state["capital"] -= collateral
        self.state["positions"][symbol] = {
            "direction": direction,
            "entry": price,
            "leverage": leverage,
            "collateral": collateral,
            "stop_loss": strategy.stop_loss,
            "take_profit": strategy.take_profit,
            "trailing_stop": strategy.trailing_stop,
            "trail_distance": strategy.trail_distance,
            "partial_tp": strategy.partial_tp,
            "timestamp": datetime.now().isoformat()
        }
        
        self.save_state()
        
        # Send Telegram alert (disabled - async issue)
        # asyncio.create_task(alerts.position_opened(symbol, direction, leverage, price))
        
        return f"✅ {direction.upper()} {symbol} @ ${price:.2f} ({leverage}x)"
    
    def close_position(self, symbol, current_price, reason="MANUAL"):
        if symbol not in self.state["positions"]:
            return None
        
        pos = self.state["positions"][symbol]
        
        if pos["direction"] == "short":
            pnl_pct = ((pos["entry"] - current_price) / pos["entry"]) * pos["leverage"] * 100
        else:
            pnl_pct = ((current_price - pos["entry"]) / pos["entry"]) * pos["leverage"] * 100
        
        pnl_usd = pos["collateral"] * (pnl_pct / 100)
        self.state["capital"] += pos["collateral"] + pnl_usd
        
        if pnl_usd > 0:
            self.state["stats"]["wins"] += 1
        else:
            self.state["stats"]["losses"] += 1
        
        del self.state["positions"][symbol]
        self.save_state()
        
        # Send Telegram alert (disabled - async issue)
        # asyncio.create_task(alerts.position_closed(symbol, pos["direction"], pnl_usd, reason))
        
        return f"{'✅' if pnl_usd >= 0 else '❌'} {symbol} | PnL: {pnl_pct:+.2f}%"
    
    def check_exits(self, prices: Dict) -> List:
        exits = []
        for symbol, pos in list(self.state["positions"].items()):
            if symbol not in prices:
                continue
            
            current = prices[symbol]["price"]
            
            if pos["direction"] == "short":
                pnl_pct = ((pos["entry"] - current) / pos["entry"]) * pos["leverage"] * 100
            else:
                pnl_pct = ((current - pos["entry"]) / pos["entry"]) * pos["leverage"] * 100
            
            if pnl_pct >= pos["take_profit"]:
                exits.append({"symbol": symbol, "reason": "TP", "pnl": pnl_pct})
            elif pnl_pct <= -pos["stop_loss"]:
                exits.append({"symbol": symbol, "reason": "SL", "pnl": pnl_pct})
        
        return exits
    
    def get_status(self, prices: Dict) -> str:
        total = self.state["capital"]
        pos_info = []
        
        for symbol, pos in self.state["positions"].items():
            if symbol in prices:
                current = prices[symbol]["price"]
                if pos["direction"] == "short":
                    pnl = ((pos["entry"] - current) / pos["entry"]) * pos["leverage"] * 100
                else:
                    pnl = ((current - pos["entry"]) / pos["entry"]) * pos["leverage"] * 100
                
                emoji = "📈" if pnl >= 0 else "📉"
                pos_info.append(f"  {emoji} {symbol} {pos['direction']} {pos['leverage']}x: {pnl:+.1f}%")
                total += pos["collateral"] * (1 + pnl/100)
        
        stats = self.state["stats"]
        wr = stats["wins"] / max(1, stats["wins"] + stats["losses"]) * 100
        
        return f"""
💰 Capital: ${self.state['capital']:.2f}
📊 Positions: {len(self.state['positions'])}
{chr(10).join(pos_info) if pos_info else '  Sin posiciones'}
💎 Total: ${total:.2f}
🎯 WR: {wr:.0f}%
"""


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class V7Orchestrator:
    def __init__(self):
        self.engine = TradingEngine()
        self.optimizers: Dict[str, GeneticOptimizer] = {}
        self.best_strategies: Dict[str, List[Dict]] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        self.load_strategies()
        self.cycle = 0
    
    def load_strategies(self):
        if STRATEGIES_FILE.exists():
            with open(STRATEGIES_FILE) as f:
                self.best_strategies = json.load(f)
    
    def save_strategies(self):
        with open(STRATEGIES_FILE, 'w') as f:
            json.dump(self.best_strategies, f, indent=2)
    
    async def run_cycle(self):
        # Legacy async version
        self.run_cycle_sync()
    
    def run_cycle_sync(self):
        self.cycle += 1
        print(f"\n{'='*60}")
        print(f"🔄 CYCLE {self.cycle} | {datetime.now().strftime('%H:%M:%S')}")
        
        # Get prices
        prices = get_prices()
        if not prices:
            print("⚠️ No prices")
            return
        
        # Generate initial history if needed (for fresh start)
        import random
        for sym in prices:
            if sym not in self.price_history or len(self.price_history.get(sym, [])) < 20:
                self.price_history[sym] = []
                base_price = prices[sym]['price']
                for i in range(30):
                    self.price_history[sym].append({
                        "price": base_price * (1 + random.uniform(-0.05, 0.05)),
                        "change": random.uniform(-5, 5),
                        "volume": random.uniform(10_000_000, 100_000_000),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Update history with real prices
        for sym, data in prices.items():
            if sym not in self.price_history:
                self.price_history[sym] = []
            self.price_history[sym].append({
                "price": data["price"],
                "change": data.get("change", 0),
                "volume": data.get("volume", 10_000_000),
                "timestamp": datetime.now().isoformat()
            })
            self.price_history[sym] = self.price_history[sym][-100:]
        
        # Show prices
        print(f"\n📊 Precios:")
        for sym, data in prices.items():
            c = data.get("change", 0)
            emoji = "📈" if c > 0 else "📉"
            print(f"   {sym}: ${data['price']:.2f} ({emoji} {c:+.1f}%)")
        
        # Check exits
        exits = self.engine.check_exits(prices)
        for ex in exits:
            result = self.engine.close_position(
                ex["symbol"], prices[ex["symbol"]]["price"], ex["reason"]
            )
            if result:
                print(f"\n{result}")
        
        # Optimize & trade
        for symbol in prices:
            if len(self.price_history.get(symbol, [])) < 20:
                continue
            
            if symbol not in self.optimizers:
                self.optimizers[symbol] = GeneticOptimizer(symbol)
                if symbol in self.best_strategies and self.best_strategies[symbol]:
                    self.optimizers[symbol].evolve(self.best_strategies[symbol][:10])
                else:
                    self.optimizers[symbol].initialize()
            else:
                self.optimizers[symbol].evolve(self.best_strategies.get(symbol, [{}])[:5])
            
            results = self.optimizers[symbol].evaluate(self.price_history[symbol])
            self.best_strategies[symbol] = results[:5]
            
            if results:
                best = results[0]
                print(f"\n🧬 {symbol}: Fit={best['fitness']:.1f}, "
                      f"P={best['profit']:+.1f}%, "
                      f"WR={best['win_rate']*100:.0f}%")
        
        # Auto-trade
        for symbol in prices:
            if symbol in self.engine.state["positions"]:
                continue
            
            if not self.best_strategies.get(symbol):
                continue
            
            best = Strategy.from_dict(self.best_strategies[symbol][0])
            
            change = prices[symbol].get("change", 0)
            
            if change < best.entry_threshold and best.prefer_short > 0.5:
                direction = "short"
            elif change > best.entry_threshold:
                direction = "long"
            else:
                continue
            
            if len(self.engine.state["positions"]) >= best.max_positions:
                continue
            
            leverage = min(best.leverage, MAX_LEVERAGE)
            result = self.engine.open_position(
                symbol, direction, prices[symbol]["price"], leverage, best
            )
            
            if result:
                print(f"\n🚀 {result}")
        
        self.save_strategies()
        self.engine.save_state()
        
        print(self.engine.get_status(prices))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("🚀 SOLANA BOT V7.1 - HYBRID AUTO-EVOLUTIVE (14 GENES)")
    print("🧬 Genetic Algorithm | 📈📉 Leverage | 🎯 Auto-Optimize")
    print("="*60)
    
    bot = V7Orchestrator()
    
    while True:
        try:
            # Run synchronous version
            bot.run_cycle_sync()
        except KeyboardInterrupt:
            print("\n🛑 Detenido")
            break
        except Exception as e:
            import traceback
            print(f"❌ Error: {e}")
            traceback.print_exc()
        
        import time
        time.sleep(30)


if __name__ == "__main__":
    main()
