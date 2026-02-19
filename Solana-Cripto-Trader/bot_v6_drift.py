#!/usr/bin/env python3
"""
SOLANA TRADING BOT V6 - DRIFT PROTOCOL + LEVERAGE + SHORTS
========================================================
- Trading con APALANCAMIENTO (2x, 3x, 5x)
- SHORT trading (apostar a la baja)
- LONG trading (apostar al alza)
- Paper Trading simulado
- Drift Protocol integration
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

# ============================================================================
# CONFIGURATION
# ============================================================================

STATE_FILE = Path("~/.config/solana-jupiter-bot/v6_state.json").expanduser()
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# Token addresses (Solana)
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tFm7mkBZKaUUTwJWS3d",
    "BNB": "BXX阮s8dCfxVHQDq8wqzS8CQ7CGf3DjfWn3gP2N5XcP",
    "XRP": "BXX阮s8dCfxVHQDq8wqzS8CQ7CGf3DjfWn3gP2N5XcP",
    "ADA": "BXX阮s8dCfxVHQDq8wqzS8CQ7CGf3DjfWn3gP2N5XcP",
    "DOGE": "BXX阮s8dCfxVHQDq8wqzS8CQ7CGf3DjfWn3gP2N5XcP",
    "AVAX": "BXX阮s8dCfxVHQDq8wqzS8CQ7CGf3DjfWn3gP2N5XcP",
    "LINK": "BXX阮s8dCfxVHQDq8wqzS8CQ7CGf3DjfWn3gP2N5XcP",
    "DOT": "BXX阮s8dCfxVHQDq8wqzS8CQ7CGf3DjfWn3gP2N5XcP",
}

# Risk parameters
MAX_LEVERAGE = 10  # Drift allows up to 10x
DEFAULT_LEVERAGE = 3
TP_SHORT = 5  # 5% take profit for shorts
SL_SHORT = 3   # 3% stop loss for shorts
TP_LONG = 5   # 5% take profit for longs
SL_LONG = 3   # 3% stop loss for longs

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    symbol: str
    direction: str  # "long" or "short"
    amount: float   # In tokens
    entry_price: float
    leverage: float
    timestamp: str

# ============================================================================
# PRICE FETCHER - Jupiter API
# ============================================================================

def get_jupiter_prices() -> Dict:
    """Get real prices from Jupiter API"""
    try:
        ids = ",".join(TOKENS.values())
        url = f"https://lite-api.jup.ag/price/v3?ids={ids}"
        resp = requests.get(url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            prices = {}
            
            # Map back to symbols
            addr_to_sym = {v: k for k, v in TOKENS.items()}
            
            for addr, info in data.items():
                sym = addr_to_sym.get(addr, addr[:8])
                prices[sym] = {
                    "price": float(info.get("usdPrice", 0)),
                    "change": float(info.get("priceChange24h", 0))
                }
            
            return prices
    except Exception as e:
        print(f"⚠️ Jupiter API error: {e}")
    
    return {}

# ============================================================================
# DRIFT PROTOCOL - PAPER TRADING
# ============================================================================

class DriftPaperTrader:
    """
    Simulates Drift Protocol trading with leverage and shorts
    """
    
    def __init__(self):
        self.state = self.load_state()
    
    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        
        return {
            "capital": 500.0,
            "positions": {},  # {symbol: {"direction": "long/short", "amount": x, "entry": y, "leverage": z}}
            "trades": [],
            "stats": {"wins": 0, "losses": 0}
        }
    
    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def calculate_pnl(self, direction: str, entry: float, current: float, leverage: float) -> float:
        """
        Calculate PnL for LONG or SHORT
        
        LONG: Gain when price goes UP
        SHORT: Gain when price goes DOWN
        
        PnL = ((exit - entry) / entry) * leverage * 100 (for LONG)
        PnL = ((entry - exit) / entry) * leverage * 100 (for SHORT)
        """
        if direction == "long":
            return ((current - entry) / entry) * leverage * 100
        else:  # short
            return ((entry - current) / entry) * leverage * 100
    
    def open_position(self, symbol: str, direction: str, price: float, leverage: float = 2.0):
        """
        Open a position with leverage
        
        Example:
        - Capital: $100
        - Leverage: 2x
        - Position size: $200
        - If LONG and price goes up 5% -> Gain 10%
        - If SHORT and price goes down 5% -> Gain 10%
        """
        # Check if already has position
        if symbol in self.state["positions"]:
            return f"❌ Ya tienes posición en {symbol}"
        
        # Calculate position size
        position_value = self.state["capital"] * leverage  # With leverage
        token_amount = position_value / price
        
        # Reserve collateral (capital / leverage)
        collateral = self.state["capital"] * 0.1  # 10% of capital per trade
        self.state["capital"] -= collateral
        
        # Open position
        self.state["positions"][symbol] = {
            "direction": direction,
            "amount": token_amount,
            "entry": price,
            "leverage": leverage,
            "collateral": collateral,
            "timestamp": datetime.now().isoformat()
        }
        
        self.state["trades"].append({
            "time": datetime.now().isoformat(),
            "action": f"OPEN_{direction.upper()}",
            "symbol": symbol,
            "price": price,
            "leverage": leverage,
            "value": position_value
        })
        
        self.save_state()
        
        emoji = "📈" if direction == "long" else "📉"
        return f"""
{emoji} **{direction.upper()} ABIERTO** (Paper)
━━━━━━━━━━━━━━━━━━━━━
💰 Símbolo: {symbol}
📊 Dirección: {direction.upper()}
💵 Precio entrada: ${price:.4f}
🔧 Leverage: {leverage}x
💎 Valor posición: ${position_value:.2f}
💳 Colateral: ${collateral:.2f}
"""
    
    def close_position(self, symbol: str, current_price: float, reason: str = "MANUAL"):
        """Close a position and calculate PnL"""
        if symbol not in self.state["positions"]:
            return f"❌ No tienes posición en {symbol}"
        
        pos = self.state["positions"][symbol]
        direction = pos["direction"]
        entry = pos["entry"]
        leverage = pos["leverage"]
        collateral = pos["collateral"]
        
        # Calculate PnL
        pnl_pct = self.calculate_pnl(direction, entry, current_price, leverage)
        pnl_usd = collateral * (pnl_pct / 100) * leverage
        
        # Update capital (return collateral + PnL)
        self.state["capital"] += collateral + pnl_usd
        
        # Update stats
        if pnl_usd > 0:
            self.state["stats"]["wins"] += 1
        else:
            self.state["stats"]["losses"] += 1
        
        # Record trade
        self.state["trades"].append({
            "time": datetime.now().isoformat(),
            "action": f"CLOSE_{reason}",
            "symbol": symbol,
            "price": current_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd
        })
        
        # Remove position
        del self.state["positions"][symbol]
        self.save_state()
        
        emoji = "✅" if pnl_usd >= 0 else "❌"
        return f"""
{emoji} **POSICIÓN CERRADA** ({reason})
━━━━━━━━━━━━━━━━━━━━━
💰 Símbolo: {symbol}
📊 Dirección: {direction.upper()}
💵 Entry: ${entry:.4f}
💵 Exit: ${current_price:.4f}
📈 PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})
"""
    
    def check_exits(self, current_prices: Dict) -> List[Dict]:
        """Check if any positions hit TP or SL"""
        exits = []
        
        for symbol, pos in list(self.state["positions"].items()):
            if symbol not in current_prices:
                continue
            
            current = current_prices[symbol]["price"]
            direction = pos["direction"]
            entry = pos["entry"]
            leverage = pos["leverage"]
            
            pnl_pct = self.calculate_pnl(direction, entry, current, leverage)
            
            # Check TP/SL
            if direction == "long":
                if pnl_pct >= TP_LONG:
                    exits.append({"symbol": symbol, "reason": "TP", "pnl_pct": pnl_pct})
                elif pnl_pct <= -SL_LONG:
                    exits.append({"symbol": symbol, "reason": "SL", "pnl_pct": pnl_pct})
            else:  # short
                if pnl_pct >= TP_SHORT:
                    exits.append({"symbol": symbol, "reason": "TP", "pnl_pct": pnl_pct})
                elif pnl_pct <= -SL_SHORT:
                    exits.append({"symbol": symbol, "reason": "SL", "pnl_pct": pnl_pct})
        
        return exits
    
    def get_status(self, current_prices: Dict) -> str:
        """Get current status"""
        total_value = self.state["capital"]
        positions_info = []
        
        for symbol, pos in self.state["positions"].items():
            if symbol in current_prices:
                current = current_prices[symbol]["price"]
                pnl_pct = self.calculate_pnl(
                    pos["direction"], pos["entry"], current, pos["leverage"]
                )
                emoji = "📈" if pnl_pct >= 0 else "📉"
                positions_info.append(
                    f"  {emoji} {symbol}: {pos['direction']} {pos['leverage']}x | "
                    f"Entry: ${pos['entry']:.2f} | Now: ${current:.2f} | "
                    f"PnL: {pnl_pct:+.2f}%"
                )
                total_value += pos["collateral"] * (1 + pnl_pct/100 * pos["leverage"])
            else:
                positions_info.append(
                    f"  ⏸️ {symbol}: {pos['direction']} {pos['leverage']}x | "
                    f"Entry: ${pos['entry']:.2f}"
                )
        
        stats = self.state["stats"]
        total_trades = stats["wins"] + stats["losses"]
        wr = (stats["wins"] / total_trades * 100) if total_trades > 0 else 0
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━
💵 **CAPITAL**: ${self.state['capital']:.2f}
📊 **POSICIONES** ({len(self.state['positions'])}):
{chr(10).join(positions_info) if positions_info else "  Sin posiciones"}
💎 **TOTAL**: ${total_value:.2f}
📈 **Wins**: {stats['wins']} | ❌ **Losses**: {stats['losses']}
🎯 **Win Rate**: {wr:.1f}%
━━━━━━━━━━━━━━━━━━━━━
"""

# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

    def generate_signals(prices: Dict) -> Dict:
        """
        Generate trading signals based on price movement
        
        - SHORT: Price dropping strongly (<-3%)
        - LONG: Price rising strongly (>3%)
        
        Leverage tiers:
        - Strong move (>5%): 5x
        - Moderate move (3-5%): 3x
        """
        signals = {"short": [], "long": []}
        
        for symbol, data in prices.items():
            change = data.get("change", 0)
            
            # Determine leverage based on strength
            if abs(change) > 5:
                leverage = 5
            elif abs(change) > 3:
                leverage = 3
            else:
                leverage = 2
            
            # SHORT signal: strong drop
            if change < -3:
                signals["short"].append({
                    "symbol": symbol,
                    "change": change,
                    "leverage": leverage,
                    "reason": f"Caída fuerte: {change:.1f}%"
                })
            # LONG signal: strong rise
            elif change > 3:
                signals["long"].append({
                    "symbol": symbol,
                    "change": change,
                    "leverage": leverage,
                    "reason": f"Subida fuerte: {change:.1f}%"
                })
        
        # Sort by strength
        signals["short"].sort(key=lambda x: x["change"])
        signals["long"].sort(key=lambda x: x["change"], reverse=True)
        
        return signals

# ============================================================================
# MAIN BOT
# ============================================================================

class DriftBotV6:
    def __init__(self):
        self.trader = DriftPaperTrader()
        self.cycle_count = 0
    
    async def run_cycle(self):
        self.cycle_count += 1
        print(f"\n{'='*60}")
        print(f"🔄 CYCLE {self.cycle_count} | {datetime.now().strftime('%H:%M:%S')}")
        
        # Get prices
        print("\n📡 Obteniendo precios...")
        prices = get_jupiter_prices()
        
        if not prices:
            print("⚠️ No se pudieron obtener precios")
            return
        
        print(f"   ✅ {len(prices)} tokens con precio")
        
        # Show top movers
        sorted_prices = sorted(prices.items(), key=lambda x: x[1].get("change", 0), reverse=True)
        
        print("\n🔺 TOP GAINERS:")
        for sym, data in sorted_prices[:3]:
            print(f"   {sym}: ${data['price']:.4f} ({data['change']:+.1f}%)")
        
        print("\n🔻 TOP LOSERS:")
        for sym, data in sorted_prices[-3:]:
            print(f"   {sym}: ${data['price']:.4f} ({data['change']:+.1f}%)")
        
        # Check for exits (TP/SL)
        exits = self.trader.check_exits(prices)
        
        for exit_info in exits:
            symbol = exit_info["symbol"]
            reason = exit_info["reason"]
            pnl_pct = exit_info["pnl_pct"]
            current = prices[symbol]["price"]
            
            result = self.trader.close_position(symbol, current, reason)
            print(f"\n{'='*40}")
            print(result)
        
        # Generate signals
        signals = generate_signals(prices)
        
        if signals["short"]:
            print(f"\n📉 SEÑALES SHORT ({len(signals['short'])}):")
            for s in signals["short"][:3]:
                print(f"   🔥 {s['symbol']}: {s['reason']} | Leverage: {s['leverage']}x")
        
        if signals["long"]:
            print(f"\n📈 SEÑALES LONG ({len(signals['long'])}):")
            for s in signals["long"][:3]:
                print(f"   🚀 {s['symbol']}: {s['reason']} | Leverage: {s['leverage']}x")
        
        # AUTO-TRADING: Open positions based on signals
        max_positions = 2
        
        # Open SHORT positions
        if len(self.trader.state["positions"]) < max_positions:
            for signal in signals["short"][:1]:  # Take top 1
                symbol = signal["symbol"]
                if symbol not in self.trader.state["positions"]:
                    price = prices[symbol]["price"]
                    leverage = signal["leverage"]
                    
                    result = self.trader.open_position(symbol, "short", price, leverage)
                    print(f"\n🚀 Abriendo SHORT: {symbol}")
        
        # Open LONG positions
        if len(self.trader.state["positions"]) < max_positions:
            for signal in signals["long"][:1]:
                symbol = signal["symbol"]
                if symbol not in self.trader.state["positions"]:
                    price = prices[symbol]["price"]
                    leverage = signal["leverage"]
                    
                    result = self.trader.open_position(symbol, "long", price, leverage)
                    print(f"\n🚀 Abriendo LONG: {symbol}")
        
        # Show status
        print(self.trader.get_status(prices))
    
    def start(self):
        print("="*60)
        print("🚀 SOLANA BOT V6 - DRIFT PROTOCOL")
        print("📈📉 LONG & SHORT with LEVERAGE")
        print("="*60)
        
        import nest_asyncio
        nest_asyncio.apply()
        
        while True:
            try:
                asyncio.run(self.run_cycle())
            except KeyboardInterrupt:
                print("\n🛑 Detenido")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
            
            time.sleep(30)


if __name__ == "__main__":
    bot = DriftBotV6()
    bot.start()
