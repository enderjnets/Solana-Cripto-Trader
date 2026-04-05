#!/usr/bin/env python3
"""
Paper Trading Module
Simulates trading with $500 capital using real Jupiter prices
"""
import json
import os
from datetime import datetime
from pathlib import Path

PAPER_BALANCE_FILE = Path("~/.config/solana-jupiter-bot/paper_balance.json").expanduser()

# Initial capital
INITIAL_CAPITAL = 500.0  # USD

class PaperTrader:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.load_balance()
    
    def load_balance(self):
        """Load paper balance from file"""
        if PAPER_BALANCE_FILE.exists():
            with open(PAPER_BALANCE_FILE) as f:
                data = json.load(f)
                self.capital = data.get('capital', self.initial_capital)
                self.positions = data.get('positions', {})
                self.history = data.get('history', [])
        else:
            self.capital = self.initial_capital
            self.positions = {'SOL': 0}
            self.history = []
    
    def save_balance(self):
        """Save paper balance to file"""
        PAPER_BALANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PAPER_BALANCE_FILE, 'w') as f:
            json.dump({
                'capital': self.capital,
                'positions': self.positions,
                'history': self.history
            }, f, indent=2)
    
    def get_price(self):
        """Get real SOL price from Jupiter"""
        import requests
        url = "https://lite-api.jup.ag/price/v3"
        params = {"ids": "So11111111111111111111111111111111111111112"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return float(data['So11111111111111111111111111111111111111112']['usdPrice'])
        return None
    
    def buy(self, amount_usd: float) -> str:
        """Buy SOL with USD amount"""
        price = self.get_price()
        if not price:
            return "❌ No se pudo obtener el precio de SOL"
        
        sol_amount = amount_usd / price
        
        if self.capital < amount_usd:
            return f"❌ Capital insuficiente. Capital: ${self.capital:.2f}"
        
        # Execute paper trade
        self.capital -= amount_usd
        self.positions['SOL'] = self.positions.get('SOL', 0) + sol_amount
        
        # Record history
        self.history.append({
            'time': datetime.now().isoformat(),
            'action': 'BUY',
            'amount_usd': amount_usd,
            'sol_amount': sol_amount,
            'price': price
        })
        
        self.save_balance()
        
        return f"✅ **Compra ejecutada (PAPER)**\n\n"                f"💵 Cantidad: ${amount_usd:.2f}"                f"\n🪙 SOL: {sol_amount:.4f}"                f"\n💰 Precio: ${price:.2f}"                f"\n📊 Capital remaining: ${self.capital:.2f}"
    
    def sell(self, sol_amount: float) -> str:
        """Sell SOL amount"""
        price = self.get_price()
        if not price:
            return "❌ No se pudo obtener el precio de SOL"
        
        if self.positions.get('SOL', 0) < sol_amount:
            return f"❌ POSICIÓN INSUFICIENTE. Tienes: {self.positions.get('SOL', 0):.4f} SOL"
        
        usd_amount = sol_amount * price
        
        # Execute paper trade
        self.capital += usd_amount
        self.positions['SOL'] -= sol_amount
        
        # Record history
        self.history.append({
            'time': datetime.now().isoformat(),
            'action': 'SELL',
            'sol_amount': sol_amount,
            'amount_usd': usd_amount,
            'price': price
        })
        
        self.save_balance()
        
        return f"✅ **Venta ejecutada (PAPER)**\n\n"                f"🪙 Cantidad: {sol_amount:.4f} SOL"                f"\n💵 Recibido: ${usd_amount:.2f}"                f"\n💰 Precio: ${price:.2f}"                f"\n📊 Capital: ${self.capital:.2f}"
    
    def status(self) -> str:
        """Show paper trading status"""
        price = self.get_price()
        if not price:
            return "❌ No se pudo obtener el precio"
        
        sol_value = self.positions.get('SOL', 0) * price
        total = self.capital + sol_value
        pnl = total - self.initial_capital
        pnl_pct = (pnl / self.initial_capital) * 100
        
        return f"📊 **PAPER TRADING STATUS**\n\n"                f"💵 Capital: ${self.capital:.2f}"                f"\n🪙 POSICIÓN SOL: {self.positions.get('SOL', 0):.4f}"                f"\n💰 Valor POSICIÓN: ${sol_value:.2f}"                f"\n📈 Total: ${total:.2f}"                f"\n{'📈' if pnl >= 0 else '📉'} P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"                f"\n💵 Precio actual: ${price:.2f}"


def main():
    import sys
    trader = PaperTrader()
    
    if len(sys.argv) < 2:
        print(trader.status())
        return
    
    cmd = sys.argv[1]
    
    if cmd == "status":
        print(trader.status())
    elif cmd == "buy" and len(sys.argv) > 2:
        amount = float(sys.argv[2])
        print(trader.buy(amount))
    elif cmd == "sell" and len(sys.argv) > 2:
        amount = float(sys.argv[2])
        print(trader.sell(amount))
    elif cmd == "reset":
        trader.capital = trader.initial_capital
        trader.positions = {'SOL': 0}
        trader.history = []
        trader.save_balance()
        print("✅ Paper trading reset")
    else:
        print("Usage: paper_trading.py [status|buy <USD>|sell <SOL>|reset]")


if __name__ == "__main__":
    main()
