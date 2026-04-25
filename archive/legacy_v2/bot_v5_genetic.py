#!/usr/bin/env python3
"""
SOLANA TRADING BOT V5 - AUTO-EVOLUTIVO
======================================
Sistema con agentes que:
1. Escogen monedas
2. Descargan datos
3. Crean estrategias
4. Hacen backtest
5. Optimizan con algoritmos genéticos
6. Evolucionan continuamente
"""

import asyncio
import random
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import requests
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

STATE_FILE = Path("~/.config/solana-jupiter-bot/v5_state.json").expanduser()
DATA_DIR = Path("~/.config/solana-jupiter-bot/data").expanduser()
STRATEGIES_FILE = Path("~/.config/solana-jupiter-bot/strategies.json").expanduser()

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GENETIC ALGORITHM PARAMETERS
# ============================================================================

POPULATION_SIZE = 10
MUTATION_RATE = 0.1
ELITE_SIZE = 5
GENERATIONS = 100

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Strategy:
    """Estrategia trading"""
    buy_threshold: float      # -10 to 0 (buy when drop X%)
    sell_threshold: float    # 0 to 10 (sell when rise X%)
    stop_loss: float        # 1 to 10
    take_profit: float      # 1 to 20
    position_size: float    # 0.05 to 0.5
    
    # Results
    fitness: float = 0
    trades: int = 0
    win_rate: float = 0
    profit: float = 0
    max_dd: float = 0
    
    def to_dict(self):
        return {
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'fitness': self.fitness,
            'trades': self.trades,
            'win_rate': self.win_rate,
            'profit': self.profit,
            'max_dd': self.max_dd
        }
    
    @classmethod
    def from_dict(cls, d):
        s = cls(
            buy_threshold=d['buy_threshold'],
            sell_threshold=d['sell_threshold'],
            stop_loss=d['stop_loss'],
            take_profit=d['take_profit'],
            position_size=d['position_size']
        )
        s.fitness = d.get('fitness', 0)
        s.trades = d.get('trades', 0)
        s.win_rate = d.get('win_rate', 0)
        s.profit = d.get('profit', 0)
        s.max_dd = d.get('max_dd', 0)
        return s
    
    def mutate(self):
        """Mutate strategy parameters"""
        if random.random() < MUTATION_RATE:
            self.buy_threshold += random.uniform(-1, 1)
            self.buy_threshold = max(-15, min(0, self.buy_threshold))
        
        if random.random() < MUTATION_RATE:
            self.sell_threshold += random.uniform(-1, 1)
            self.sell_threshold = max(0, min(15, self.sell_threshold))
        
        if random.random() < MUTATION_RATE:
            self.stop_loss += random.uniform(-0.5, 0.5)
            self.stop_loss = max(1, min(10, self.stop_loss))
        
        if random.random() < MUTATION_RATE:
            self.take_profit += random.uniform(-1, 1)
            self.take_profit = max(1, min(20, self.take_profit))
        
        if random.random() < MUTATION_RATE:
            self.position_size += random.uniform(-0.05, 0.05)
            self.position_size = max(0.05, min(0.5, self.position_size))
    
    def crossover(self, other):
        """Crossover with another strategy"""
        child = Strategy(
            buy_threshold=random.choice([self.buy_threshold, other.buy_threshold]),
            sell_threshold=random.choice([self.sell_threshold, other.sell_threshold]),
            stop_loss=random.choice([self.stop_loss, other.stop_loss]),
            take_profit=random.choice([self.take_profit, other.take_profit]),
            position_size=random.choice([self.position_size, other.position_size])
        )
        return child


# ============================================================================
# AGENT 1: MARKET SCANNER - Escoge mejores monedas
# ============================================================================

class MarketScannerAgent:
    """Agente que escanea el mercado y escoge las mejores monedas"""
    
    def __init__(self):
        self.watchlist = []
    
    async def scan_market(self) -> List[Dict]:
        """Escanea mercado y retorna monedas con momentum"""
        # Fallback data when API fails
        fallback = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'price_change_percentage_24h': 2.5, 'total_volume': 30000000000, 'market_cap': 1000000000000, 'current_price': 67000},
            {'symbol': 'ETH', 'name': 'Ethereum', 'price_change_percentage_24h': 3.2, 'total_volume': 15000000000, 'market_cap': 300000000000, 'current_price': 3500},
            {'symbol': 'SOL', 'name': 'Solana', 'price_change_percentage_24h': -5.2, 'total_volume': 2000000000, 'market_cap': 40000000000, 'current_price': 82},
            {'symbol': 'XRP', 'name': 'Ripple', 'price_change_percentage_24h': -3.1, 'total_volume': 1500000000, 'market_cap': 30000000000, 'current_price': 1.42},
            {'symbol': 'ZEC', 'name': 'Zcash', 'price_change_percentage_24h': -8.5, 'total_volume': 500000000, 'market_cap': 1000000000, 'current_price': 265},
            {'symbol': 'TAO', 'name': 'Bittensor', 'price_change_percentage_24h': -6.2, 'total_volume': 300000000, 'market_cap': 2000000000, 'current_price': 183},
            {'symbol': 'BNB', 'name': 'BNB', 'price_change_percentage_24h': 1.5, 'total_volume': 800000000, 'market_cap': 80000000000, 'current_price': 610},
            {'symbol': 'ADA', 'name': 'Cardano', 'price_change_percentage_24h': -2.8, 'total_volume': 600000000, 'market_cap': 15000000000, 'current_price': 0.42},
            {'symbol': 'DOGE', 'name': 'Dogecoin', 'price_change_percentage_24h': -4.1, 'total_volume': 800000000, 'market_cap': 8000000000, 'current_price': 0.11},
            {'symbol': 'AVAX', 'name': 'Avalanche', 'price_change_percentage_24h': -3.5, 'total_volume': 400000000, 'market_cap': 3000000000, 'current_price': 35},
        ]
        
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={"vs_currency": "usd", "order": "volume_desc", "per_page": 50, "page": 1},
                timeout=10
            )
            coins = resp.json()
            
            if not isinstance(coins, list):
                print("   ⚠️ API error, using fallback data")
                coins = fallback
        except Exception as e:
            print(f"   ⚠️ API unavailable, using fallback: {e}")
            coins = fallback
        
        opportunities = []
        for coin in coins:
            try:
                change = coin.get('price_change_percentage_24h', 0) or 0
                volume = coin.get('total_volume', 0) or 0
                market_cap = coin.get('market_cap', 0) or 0
                
                if market_cap < 10_000_000:
                    continue
                
                opportunities.append({
                    'symbol': coin.get('symbol', '').upper(),
                    'name': coin.get('name', ''),
                    'price': coin.get('current_price', 0),
                    'change_24h': change,
                    'volume': volume,
                    'market_cap': market_cap,
                    'score': abs(change) * np.log10(volume + 1)
                })
            except:
                continue
        
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        print(f"   📊 Market scan: {len(opportunities)} oportunidades")
        return opportunities[:20]


# ============================================================================
# AGENT 2: DATA MANAGER - Descarga datos históricos
# ============================================================================

class DataAgent:
    """Agente que descarga y gestiona datos históricos"""
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
    
    def generate_synthetic_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Genera datos sintéticos cuando la API falla"""
        import numpy as np
        
        np.random.seed(hash(symbol) % 100000)
        
        n = days * 24  # Hourly candles
        base_price = {
            'BTC': 67000, 'ETH': 3500, 'SOL': 82, 'XRP': 1.42,
            'BNB': 610, 'ADA': 0.42, 'DOGE': 0.11, 'AVAX': 35,
            'DOT': 7, 'MATIC': 0.85, 'LINK': 15, 'UNI': 7,
            'LTC': 75, 'ATOM': 9, 'XLM': 0.12, 'ALGO': 0.18,
            'VET': 0.025, 'FIL': 5, 'TRX': 0.11, 'NEAR': 5,
            'ZEC': 265, 'TAO': 183, 'MANTLE': 0.5, 'WIF': 2.5,
            'PEPE': 0.000001, 'BONK': 0.00001
        }.get(symbol, 100)
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0002, 0.02, n)  # Small positive trend
        prices = [base_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        prices = prices[:n]  # Ensure same length
        
        # Create OHLC
        df = pd.DataFrame({
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=n, freq='h'),
            'open': prices,
            'close': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': [np.random.uniform(1000000, 10000000) for _ in range(n)]
        })
        
        # Save
        filepath = self.data_dir / f"{symbol}.csv"
        df.to_csv(filepath, index=False)
        
        print(f"   📊 Generated {symbol}: {len(df)} candles, ${df['close'].iloc[0]:.0f}-${df['close'].iloc[-1]:.0f}")
        return df
    
    def download_data(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Descarga datos históricos de CoinGecko"""
        # Check cache first
        filepath = self.data_dir / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        # Try API first
        try:
            id_map = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
                'XRP': 'ripple', 'BNB': 'binancecoin', 'ADA': 'cardano',
                'DOGE': 'dogecoin', 'DOT': 'polkadot', 'MATIC': 'matic-network',
                'LTC': 'litecoin', 'AVAX': 'avalanche-2', 'LINK': 'chainlink',
                'UNI': 'uniswap', 'ATOM': 'cosmos', 'XLM': 'stellar',
            }
            
            coin_id = id_map.get(symbol, symbol.lower())
            
            resp = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                params={'vs_currency': 'usd', 'days': str(days)},
                timeout=30
            )
            
            data = resp.json()
            
            if 'prices' not in data:
                return self.generate_synthetic_data(symbol, days)
            
            prices = data['prices']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['close']
            df['high'] = df['close'] * 1.01
            df['low'] = df['close'] * 0.99
            df['volume'] = 1_000_000
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            filepath = self.data_dir / f"{symbol}.csv"
            df.to_csv(filepath, index=False)
            
            print(f"   ✅ Downloaded {symbol}: {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"   ⚠️ API failed for {symbol}, generating synthetic data")
            return self.generate_synthetic_data(symbol, days)


# ============================================================================
# AGENT 3: STRATEGY GENERATOR - Crea estrategias
# ============================================================================

class StrategyGeneratorAgent:
    """Agente que genera nuevas estrategias"""
    
    def __init__(self):
        self.population: List[Strategy] = []
    
    def generate_random(self) -> Strategy:
        """Genera estrategia aleatoria"""
        return Strategy(
            buy_threshold=random.uniform(-15, -2),
            sell_threshold=random.uniform(2, 15),
            stop_loss=random.uniform(1, 5),
            take_profit=random.uniform(2, 15),
            position_size=random.uniform(0.05, 0.3)
        )
    
    def initialize_population(self, size: int = POPULATION_SIZE):
        """Inicializa población aleatoria"""
        self.population = [self.generate_random() for _ in range(size)]
        print(f"   🧬 Población inicial: {size} estrategias")
    
    def evolve(self, best_results: List[Dict]):
        """Evoluciona población basándose en resultados"""
        # Sort by fitness
        if best_results:
            best_results.sort(key=lambda x: x.get('fitness', 0), reverse=True)
            
            # Keep elites
            new_population = []
            for d in best_results[:ELITE_SIZE]:
                new_population.append(Strategy.from_dict(d))
            
            # Fill rest with crossover + mutation
            while len(new_population) < POPULATION_SIZE:
                if len(best_results) >= 2:
                    parent1 = random.choice(best_results[:10])
                    parent2 = random.choice(best_results[:10])
                    
                    child = Strategy(
                        buy_threshold=random.choice([parent1['buy_threshold'], parent2['buy_threshold']]),
                        sell_threshold=random.choice([parent1['sell_threshold'], parent2['sell_threshold']]),
                        stop_loss=random.choice([parent1['stop_loss'], parent2['stop_loss']]),
                        take_profit=random.choice([parent1['take_profit'], parent2['take_profit']]),
                        position_size=random.choice([parent1['position_size'], parent2['position_size']])
                    )
                else:
                    child = self.generate_random()
                
                child.mutate()
                new_population.append(child)
            
            self.population = new_population
            print(f"   🧬 Población evolucionada: {len(self.population)} estrategias")
    
    def evaluate_population(self, df: pd.DataFrame) -> List[Dict]:
        """Evalúa toda la población"""
        results = []
        
        for strategy in self.population:
            # Simple backtest
            cash = 500
            position = 0
            entry_price = 0
            trades = 0
            wins = 0
            losses = 0
            max_dd = 0
            peak = cash
            
            for i in range(1, len(df)):
                current = df['close'].iloc[i]
                prev = df['close'].iloc[i-1]
                low = df['low'].iloc[i]
                high = df['high'].iloc[i]
                
                dip_pct = ((low - prev) / prev) * 100
                
                if position == 0:
                    if dip_pct <= strategy.buy_threshold:
                        position = (cash * strategy.position_size) / current
                        entry_price = current
                        cash -= (position * current)
                else:
                    pnl_pct = ((current - entry_price) / entry_price) * 100
                    
                    if pnl_pct >= strategy.take_profit or pnl_pct <= -strategy.stop_loss:
                        cash += (position * current)
                        trades += 1
                        
                        if pnl_pct > 0:
                            wins += 1
                        else:
                            losses += 1
                        
                        position = 0
                        
                        if cash > peak:
                            peak = cash
                        dd = (peak - cash) / peak * 100 if peak > 0 else 0
                        if dd > max_dd:
                            max_dd = dd
            
            if position > 0:
                cash += position * df['close'].iloc[-1]
            
            profit_pct = ((cash - 500) / 500) * 100
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            fitness = profit_pct * (win_rate + 0.5) / (max_dd / 100 + 1)
            
            strategy.trades = trades
            strategy.win_rate = win_rate
            strategy.profit = profit_pct
            strategy.max_dd = max_dd
            strategy.fitness = fitness
            
            results.append(strategy.to_dict())
        
        results.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        return results


# ============================================================================
# AGENT 4: BACKTEST ENGINE - Evalúa estrategias
# ============================================================================

class BacktestAgent:
    """Agente que corre backtests"""
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, symbol: str, data_dir=DATA_DIR) -> Optional[pd.DataFrame]:
        """Carga datos desde cache o archivo"""
        if symbol in self.cache:
            return self.cache[symbol]
        
        filepath = data_dir / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.cache[symbol] = df
            return df
        
        return None
    
    def backtest(self, strategy: Strategy, df: pd.DataFrame) -> Strategy:
        """Evalúa una estrategia"""
        cash = 500
        position = 0
        entry_price = 0
        trades = 0
        wins = 0
        losses = 0
        max_dd = 0
        peak = cash
        
        for i in range(1, len(df)):
            current = df['close'].iloc[i]
            prev = df['close'].iloc[i-1]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            
            # Calculate dip
            dip_pct = ((low - prev) / prev) * 100
            pump_pct = ((high - prev) / prev) * 100
            
            if position == 0:
                # Check buy signal
                if dip_pct <= strategy.buy_threshold:
                    position = (cash * strategy.position_size) / current
                    entry_price = current
                    cash -= (position * current)
            else:
                # Check exit
                pnl_pct = ((current - entry_price) / entry_price) * 100
                
                if pnl_pct >= strategy.take_profit or pnl_pct <= -strategy.stop_loss:
                    cash += (position * current)
                    trades += 1
                    
                    if pnl_pct > 0:
                        wins += 1
                    else:
                        losses += 1
                    
                    position = 0
                    
                    # Track drawdown
                    if cash > peak:
                        peak = cash
                    dd = (peak - cash) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
        
        # Close position
        if position > 0:
            cash += position * df['close'].iloc[-1]
        
        # Calculate fitness
        profit_pct = ((cash - 500) / 500) * 100
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Fitness = profit * win_rate / (drawdown + 1)
        fitness = profit_pct * (win_rate + 0.5) / (max_dd / 100 + 1)
        
        strategy.trades = trades
        strategy.win_rate = win_rate
        strategy.profit = profit_pct
        strategy.max_dd = max_dd
        strategy.fitness = fitness
        
        return strategy
    
    def evaluate_population(self, population: List[Strategy], df: pd.DataFrame) -> List[Dict]:
        """Evalúa toda la población"""
        results = []
        
        for strategy in population:
            strategy = self.backtest(strategy, df)
            results.append(strategy.to_dict())
        
        # Sort by fitness
        results.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        
        return results


# ============================================================================
# ORCHESTRATOR - Coordina todos los agentes
# ============================================================================

class TradingOrchestrator:
    """Orquestador principal del sistema auto-evolutivo"""
    
    def __init__(self):
        self.scanner = MarketScannerAgent()
        self.data_agent = DataAgent()
        self.strategy_gen = StrategyGeneratorAgent()
        self.backtester = BacktestAgent()
        
        self.best_strategies: Dict[str, List[Dict]] = {}  # Per symbol
        self.load_strategies()
        
        # Bot state
        self.state = {
            'capital_usd': 500,
            'positions': {},
            'trades': [],
            'stats': {'wins': 0, 'losses': 0}
        }
        self.load_state()
    
    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                self.state = json.load(f)
    
    def save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def load_strategies(self):
        if STRATEGIES_FILE.exists():
            with open(STRATEGIES_FILE) as f:
                self.best_strategies = json.load(f)
    
    def save_strategies(self):
        with open(STRATEGIES_FILE, 'w') as f:
            json.dump(self.best_strategies, f, indent=2)
    
    async def run_cycle(self, cycle_num: int):
        print(f"\n{'='*60}")
        print(f"🔄 CYCLE {cycle_num} | {datetime.now().strftime('%H:%M:%S')}")
        
        # Step 1: Scan market
        print("\n📡 Agente 1: Escaneando mercado...")
        opportunities = await self.scanner.scan_market()
        
        if not opportunities:
            print("   ⚠️ No opportunities found")
            return
        
        # Step 2: Download data for top coins
        print("\n📥 Agente 2: Descargando datos...")
        for coin in opportunities[:2]:  # Top 5
            symbol = coin['symbol']
            df = self.data_agent.download_data(symbol, days=90)
            if df is not None:
                # Step 3 & 4: Generate and test strategies
                print(f"\n🧬 Agente 3+4: Evaluando estrategias para {symbol}...")
                
                # Load or init population
                if symbol in self.best_strategies and self.best_strategies[symbol]:
                    self.strategy_gen.evolve(self.best_strategies[symbol][:10])
                else:
                    self.strategy_gen.initialize_population()
                
                # Evaluate
                results = self.strategy_gen.evaluate_population(df)
                
                # Save best
                self.best_strategies[symbol] = results[:10]
                print(f"   Best: {results[0].get('fitness', 0):.2f} | "
                      f"WR: {results[0].get('win_rate', 0)*100:.0f}% | "
                      f"Profit: {results[0].get('profit', 0):+.2f}%")
        
        # Save strategies
        self.save_strategies()
        
        # Step 5: Update bot state
        self.save_state()
        
        print(f"\n✅ Cycle {cycle_num} completo")
        print(f"   Monedas optimizadas: {len(self.best_strategies)}")
        
        # Show best overall
        all_best = []
        for sym, strategies in self.best_strategies.items():
            if strategies:
                all_best.append((sym, strategies[0]))
        
        all_best.sort(key=lambda x: x[1].get('fitness', 0), reverse=True)
        
        if all_best:
            sym, best = all_best[0]
            print(f"\n🏆 MEJOR ESTRATEGIA: {sym}")
            print(f"   Fitness: {best.get('fitness', 0):.2f}")
            print(f"   WR: {best.get('win_rate', 0)*100:.0f}%")
            print(f"   Profit: {best.get('profit', 0):+.2f}%")
            print(f"   Params: Buy<{best.get('buy_threshold', 0):.1f}%, "
                  f"SL={best.get('stop_loss', 0):.1f}%, "
                  f"TP={best.get('take_profit', 0):.1f}%")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("="*60)
    print("🚀 SOLANA BOT V5 - AUTO-EVOLUTIVO")
    print("🧠 Sistema con Algoritmos Genéticos")
    print("="*60)
    
    orchestrator = TradingOrchestrator()
    
    cycle = 0
    while True:
        cycle += 1
        try:
            await orchestrator.run_cycle(cycle)
        except KeyboardInterrupt:
            print("\n🛑 Detenido")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Wait before next cycle
        await asyncio.sleep(60)  # Run every minute in demo mode


if __name__ == "__main__":
    asyncio.run(main())
