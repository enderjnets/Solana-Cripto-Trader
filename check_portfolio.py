#!/usr/bin/env python3
"""
Solana + USDT Portfolio Checker
Check SOL, BTC (wrapped), and USDT balances
"""
import json
import os
from solana.rpc.api import Client
from solders.pubkey import Pubkey

# Load wallet
WALLET_PATH = os.path.expanduser("~/.config/solana-jupiter-bot/wallet.json")
with open(WALLET_PATH) as f:
    wallet_data = json.load(f)

wallet_address = wallet_data["address"]
pubkey = Pubkey.from_string(wallet_address)

# Token addresses
TOKENS = {
    "SOL": "So11111111111111111111111111111111111111111112",
    "BTC": "cbbtcf3aa214zXHbiAZQwf4122FBYbraNdFqgw4iMij",  # cbBTC on Solana
    "USDT": "Es9vMFrzaCERmkhfr9WMq8i5icD4Qwpq6xS5VUUSbmE1",
}

# TARGET ALLOCATION: 50% SOL / 30% BTC / 20% USDT (optimal for +5% daily)
TARGET_ALLOCATION = {"SOL": 0.50, "BTC": 0.30, "USDT": 0.20}

# Get SOL price (mock - in production use Jupiter API)
SOL_PRICE_USD = 80.0  # Placeholder

def get_token_balance(token_mint):
    """Get token balance using RPC"""
    from spl.token.client import Token
    from spl.token.constants import TOKEN_PROGRAM_ID
    
    try:
        rpc = Client("https://api.mainnet-beta.solana.com")
        
        # Get token accounts
        response = rpc.get_token_accounts_by_owner(pubkey, {"mint": Pubkey.from_string(token_mint)})
        
        if response.value:
            account = response.value[0].pubkey
            token = Token(rpc, Pubkey.from_string(token_mint), TOKEN_PROGRAM_ID, pubkey)
            balance = token.get_balance_info(account).value
            return balance / 1e6  # USDT has 6 decimals
        return 0
    except Exception as e:
        return f"Error: {e}"

def get_portfolio_value():
    """Calculate total portfolio value"""
    rpc = Client("https://api.mainnet-beta.solana.com")
    
    # SOL balance
    sol_balance = rpc.get_balance(pubkey).value / 1e9
    sol_value = sol_balance * SOL_PRICE_USD
    
    # Placeholder for token values
    btc_value = 0  # TODO: Get BTC price and balance
    usdt_value = 0  # TODO: Get USDT balance
    
    total_value = sol_value + btc_value + usdt_value
    
    return {
        "wallet": wallet_address,
        "SOL": {"balance": sol_balance, "price_usd": SOL_PRICE_USD, "value": sol_value},
        "BTC": {"balance": 0, "price_usd": 0, "value": 0},
        "USDT": {"balance": 0, "price_usd": 1.0, "value": 0},
        "total_value_usd": total_value,
        "allocation": {
            "SOL": sol_value / total_value if total_value > 0 else 0,
            "BTC": 0,
            "USDT": 0,
        }
    }

if __name__ == "__main__":
    portfolio = get_portfolio_value()
    print("="*50)
    print("  🦞 SOLANA PORTFOLIO (SOL + BTC + USDT)")
    print("="*50)
    print(f"  Wallet: {portfolio['wallet'][:10]}...")
    print(f"")
    print(f"  💰 SOL:  {portfolio['SOL']['balance']:.4f} x ${portfolio['SOL']['price_usd']:.2f} = ${portfolio['SOL']['value']:.2f}")
    print(f"  ₿ BTC:  {portfolio['BTC']['balance']:.8f} x ${portfolio['BTC']['price_usd']:.2f} = ${portfolio['BTC']['value']:.2f}")
    print(f"  ₮ USDT: {portfolio['USDT']['balance']:.2f} x $1.00 = ${portfolio['USDT']['value']:.2f}")
    print(f"")
    print(f"  📊 TOTAL: ${portfolio['total_value_usd']:.2f}")
    print(f"")
    print(f"  Allocation:")
    print(f"    SOL:  {portfolio['allocation']['SOL']*100:.1f}%")
    print(f"    BTC:  {portfolio['allocation']['BTC']*100:.1f}%")
    print(f"    USDT: {portfolio['allocation']['USDT']*100:.1f}%")
    print("="*50)
