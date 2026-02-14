#!/usr/bin/env python3
"""
Execute Trade Script
====================
Real trading execution on Solana using Jupiter API.

Usage:
    python3 execute_trade.py --buy 0.5 USDC       # Buy SOL with USDC
    python3 execute_trade.py --sell 0.5 SOL        # Sell SOL for USDC
    python3 execute_trade.py --test               # Test without executing
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Solana libraries
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.api_integrations import JupiterClient, SOL, USDC

# RPC endpoints
RPC_DEVNET = "https://api.devnet.solana.com"
RPC_MAINNET = "https://api.mainnet-beta.solana.com"

# Wallet file
WALLET_DIR = Path.home() / ".config" / "solana-jupiter-bot"
WALLET_FILE = WALLET_DIR / "wallet.json"


def load_wallet() -> Keypair:
    """Load wallet from file."""
    if not WALLET_FILE.exists():
        raise FileNotFoundError(f"Wallet not found: {WALLET_FILE}")

    data = json.loads(WALLET_FILE.read_text())
    private_key = data.get("private_key", data.get("keypair", data))

    # Always JSON list format
    if isinstance(private_key, str) and private_key.startswith("["):
        # JSON string format
        return Keypair.from_json(private_key)
    elif isinstance(private_key, list):
        # List format
        return Keypair.from_json(json.dumps(private_key))
    else:
        # Base58 format (legacy)
        import base58
        key_bytes = base58.b58decode(private_key)
        return Keypair.from_bytes(key_bytes)


def get_balance(pubkey: Pubkey, network: str = "devnet") -> float:
    """Get wallet balance in SOL."""
    rpc_url = RPC_DEVNET if network == "devnet" else RPC_MAINNET
    client = Client(rpc_url)
    lamports = client.get_balance(pubkey).value
    return lamports / 1e9


async def execute_buy(amount_usdc: float, wallet: Keypair, network: str = "devnet"):
    """Buy SOL with USDC."""
    wallet_pubkey = str(wallet.pubkey())
    client = JupiterClient()

    print(f"\n🟢 BUY {amount_usdc} USDC → SOL")
    print(f"   Wallet: {wallet_pubkey[:20]}...")

    # Get quote - Buy SOL with USDC
    order = await client.quote_usdc_to_sol(amount_usdc, wallet_pubkey)

    print(f"   Input: {amount_usdc} USDC")
    print(f"   Output SOL: {client.micro_to_sol(int(order.out_amount)):.4f} SOL")
    print(f"   Price Impact: {float(order.price_impact_pct):.4f}%")

    if not order.transaction:
        print("   ❌ No transaction generated")
        return False

    # Sign transaction
    print("\n🔐 Signing transaction...")
    try:
        tx = Transaction.from_bytes(bytes.fromhex(order.transaction))
        signed = wallet.sign_message(tx.compile_message())
        print(f"   ✅ Signed: {signed.signature}")

        # Send to network (if not test mode)
        if network != "test":
            sol_client = Client(RPC_DEVNET if network == "devnet" else RPC_MAINNET)
            tx_id = sol_client.send_raw_transaction(signed.serialize(), Confirmed)
            print(f"   📤 Sent: {tx_id.value}")
            return True
        else:
            print("   🧪 TEST MODE - Not sent to network")
            return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    finally:
        await client.close()


async def execute_sell(amount_sol: float, wallet: Keypair, network: str = "devnet"):
    """Sell SOL for USDC."""
    wallet_pubkey = str(wallet.pubkey())
    client = JupiterClient()

    print(f"\n🔴 SELL {amount_sol} SOL → USDC")
    print(f"   Wallet: {wallet_pubkey[:20]}...")

    # Get quote - Sell SOL for USDC
    order = await client.quote_sol_to_usdc(amount_sol, wallet_pubkey)

    print(f"   Input: {amount_sol} SOL")
    print(f"   Output USDC: {client.micro_to_usdc(int(order.out_amount)):.2f} USDC")
    print(f"   Price Impact: {float(order.price_impact_pct):.4f}%")

    if not order.transaction:
        print("   ❌ No transaction generated")
        return False

    # Sign transaction
    print("\n🔐 Signing transaction...")
    try:
        tx = Transaction.from_bytes(bytes.fromhex(order.transaction))
        signed = wallet.sign_message(tx.compile_message())
        print(f"   ✅ Signed: {signed.signature}")

        # Send to network
        if network != "test":
            sol_client = Client(RPC_DEVNET if network == "devnet" else RPC_MAINNET)
            tx_id = sol_client.send_raw_transaction(signed.serialize(), Confirmed)
            print(f"   📤 Sent: {tx_id.value}")
            return True
        else:
            print("   🧪 TEST MODE - Not sent to network")
            return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    finally:
        await client.close()


async def test_quote(amount: float, direction: str):
    """Test quote without executing."""
    client = JupiterClient()
    wallet = str(load_wallet().pubkey())

    if direction == "buy":
        # Buy SOL with USDC
        order = await client.quote_usdc_to_sol(amount, wallet)
        print(f"\n📊 QUOTE TEST: {amount} USDC → SOL")
        print(f"   Output: {client.micro_to_sol(int(order.out_amount)):.4f} SOL")
    else:
        # Sell SOL for USDC
        order = await client.quote_sol_to_usdc(amount, wallet)
        print(f"\n📊 QUOTE TEST: {amount} SOL → USDC")
        print(f"   Output: {client.micro_to_usdc(int(order.out_amount)):.2f} USDC")

    print(f"   Impact: {float(order.price_impact_pct):.4f}%")
    print(f"   Route: {len(order.route_plan)} hops")
    print(f"   TX Generated: {'✅' if order.transaction else '❌'}")

    await client.close()


async def main():
    parser = argparse.ArgumentParser(description="Execute trades on Solana")
    parser.add_argument("--buy", type=float, help="Buy SOL with USDC amount")
    parser.add_argument("--sell", type=float, help="Sell SOL amount for USDC")
    parser.add_argument("--test", action="store_true", help="Test quote only")
    parser.add_argument("--network", default="devnet", choices=["devnet", "mainnet"],
                        help="Network to use")

    args = parser.parse_args()

    # Load wallet
    try:
        wallet = load_wallet()
        wallet_pubkey = wallet.pubkey()
        print(f"\n💰 WALLET LOADED")
        print(f"   Address: {str(wallet_pubkey)}")
        print(f"   Balance: {get_balance(wallet_pubkey, args.network)} SOL")
    except FileNotFoundError:
        print(f"\n❌ Wallet not found. Run: python3 tools/solana_wallet.py --generate")
        return

    # Execute based on args
    if args.test:
        if args.buy:
            await test_quote(args.buy, "buy")
        elif args.sell:
            await test_quote(args.sell, "sell")
    elif args.buy:
        await execute_buy(args.buy, wallet, args.network)
    elif args.sell:
        await execute_sell(args.sell, wallet, args.network)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
