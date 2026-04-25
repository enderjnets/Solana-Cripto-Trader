#!/usr/bin/env python3
"""Simple trading script - runs on cron"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from api.price_feed import CryptoComparePriceFeed
from paper_trading_engine import PaperTradingEngine

def main():
    # Load state
    state_file = Path(__file__).parent / "paper_trading_state.json"
    with open(state_file) as f:
        state = json.load(f)
    
    # Get prices
    pf = CryptoComparePriceFeed()
    
    # Simple check - just verify system works
    print(f"Balance: ${state['balance_usd']:.2f}")
    print(f"Trades: {len([t for t in state['trades'] if t['status']=='open'])}")
    print("OK")
    
if __name__ == "__main__":
    main()
