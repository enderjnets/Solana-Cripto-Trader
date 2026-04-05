#!/usr/bin/env python3
"""
Simple Backtester for Solana Bot V4 Strategy
============================================
Tests: Buy when price drops X%, Sell when rises Y%
"""

import pandas as pd
import numpy as np

def simple_backtest(df, buy_threshold=-5, sell_threshold=5, stop_loss=3, take_profit=5, position_size=0.1):
    """
    Simple backtest matching Bot V4 strategy
    Uses intrabar (high/low) to detect dips
    """
    cash = 500  # Starting capital
    position = 0  # Amount held
    entry_price = 0
    trades = []
    
    for i in range(1, len(df)):
        current_close = df['close'].iloc[i]
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        prev_close = df['close'].iloc[i-1]
        
        # Calculate dip from previous close to current low
        dip_pct = ((current_low - prev_close) / prev_close) * 100
        pump_pct = ((current_high - prev_close) / prev_close) * 100
        
        if position == 0:
            # No position - check for buy signal (dip)
            if dip_pct <= buy_threshold:
                # BUY at current close (or could use current_low for more conservative)
                position = (cash * position_size) / current_close
                entry_price = current_close
                cash -= (position * current_close)
                trades.append({
                    'type': 'BUY',
                    'price': current_close,
                    'dip': dip_pct,
                    'balance': cash
                })
        else:
            # Have position - check for sell
            pnl_pct = ((current_close - entry_price) / entry_price) * 100
            
            # Check TP/SL
            if pnl_pct >= take_profit or pnl_pct <= -stop_loss:
                action = 'SELL_TP' if pnl_pct >= take_profit else 'SELL_SL'
                cash += (position * current_close)
                trades.append({
                    'type': action,
                    'price': current_close,
                    'pnl_pct': pnl_pct,
                    'balance': cash
                })
                position = 0
                entry_price = 0
    
    # Close any open position at end
    if position > 0:
        cash += position * df['close'].iloc[-1]
    
    # Calculate stats
    sell_trades = [t for t in trades if 'BUY' not in t.get('type', '')]
    wins = len([t for t in sell_trades if t.get('pnl_pct', 0) > 0])
    losses = len([t for t in sell_trades if t.get('pnl_pct', 0) < 0])
    buy_trades = len([t for t in trades if t.get('type') == 'BUY'])
    
    return {
        'trades': len(trades),
        'sell_trades': len(sell_trades),
        'buy_trades': buy_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'final_balance': cash,
        'profit_pct': ((cash - 500) / 500) * 100
    }


if __name__ == '__main__':
    import sys
    
    # Load data
    df = pd.read_csv('data/btc_historical.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f'📊 Backtest: BTC {len(df)} candles')
    print(f'Price: ${df["close"].min():.0f} - ${df["close"].max():.0f}')
    print()
    
    # Test strategies
    strategies = [
        ('Buy Dip 3% / Sell 3%', -3, 3),
        ('Buy Dip 5% / Sell 5%', -5, 5),
        ('Buy Dip 7% / Sell 7%', -7, 7),
        ('Buy Dip 10% / Sell 10%', -10, 10),
    ]
    
    print('='*60)
    for name, buy_th, sell_th in strategies:
        result = simple_backtest(df, buy_threshold=buy_th, sell_threshold=sell_th)
        print(f'''
{name}:
  Total Trades: {result['trades']}
  Wins: {result['wins']} | Losses: {result['losses']}
  Win Rate: {result['win_rate']*100:.1f}%
  Profit: {result['profit_pct']:+.2f}%
''')
