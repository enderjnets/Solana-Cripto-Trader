#!/usr/bin/env python3
"""
Solana Wallet Manager
====================
Generate, import, and manage Solana wallets for trading.

Usage:
    python3 tools/solana_wallet.py --generate    # Generate new wallet
    python3 tools/solana_wallet.py --import     # Import from JSON
    python3 tools/solana_wallet.py --balance    # Check balance
    python3 tools/solana_wallet.py --address    # Show address
    python3 tools/solana_wallet.py --save-env   # Save to .env
"""

import os
import sys
import json
try:
    import base58  # type: ignore
except ImportError:
    base58 = None
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Solana libraries
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.api import Client

# Config
WALLET_DIR = Path.home() / ".config" / "solana-jupiter-bot"
WALLET_FILE = WALLET_DIR / "wallet.json"
ENCRYPTED_KEY_FILE = WALLET_DIR / "wallet.enc"
# Use the .env in the project root (2 levels up from tools/)
ENV_FILE = Path(__file__).parent.parent / ".env"


@dataclass
class WalletInfo:
    """Wallet information (supports both legacy and current fields)."""
    public_key: str = ""
    key_type: str = "json"
    created_at: str = ""
    last_used: str = ""
    network: str = "devnet"
    address: str = ""
    private_key: str = ""  # JSON format
    balance_sol: float = 0.0
    balance_lamports: int = 0


@dataclass
class WalletBalance:
    """Legacy wallet balance object expected by test_system.py."""
    sol_balance: float = 0.0
    usdc_balance: float = 0.0
    usdt_balance: float = 0.0
    token_balances: dict | None = None

    def total_usd_value(self, sol_price_usd: float) -> float:
        extras = sum((self.token_balances or {}).values())
        return (self.sol_balance * sol_price_usd) + self.usdc_balance + self.usdt_balance + extras


class WalletManager:
    """
    Manage Solana wallet for trading.
    
    Features:
    - Generate new wallets
    - Import from JSON/private key
    - Sign transactions
    - Check balance
    """
    
    # RPC endpoints
    RPC_DEVNET = "https://api.devnet.solana.com"
    RPC_MAINNET = "https://api.mainnet-beta.solana.com"
    
    def __init__(self, network: str = "devnet"):
        """
        Initialize wallet manager.
        
        Args:
            network: 'devnet' or 'mainnet'
        """
        self.network = network
        self.rpc_url = self.RPC_DEVNET if network == "devnet" else self.RPC_MAINNET
        self.client = Client(self.rpc_url)
        self.keypair: Optional[Keypair] = None
        self._load()
    
    def _load(self):
        """Load wallet from file or .env."""
        # Try to load from file
        if WALLET_FILE.exists():
            try:
                data = json.loads(WALLET_FILE.read_text())
                if "private_key" in data:
                    # Load from JSON list format
                    private_key = data["private_key"]
                    if isinstance(private_key, list):
                        self.keypair = Keypair.from_json(json.dumps(private_key))
                    else:
                        self.keypair = Keypair.from_json(private_key)
                    print(f"✅ Wallet loaded from {WALLET_FILE}")
                    return
            except Exception as e:
                print(f"⚠️ Error loading from file: {e}")
        
        # Try to load from .env
        env_content = ""
        if ENV_FILE.exists():
            env_content = ENV_FILE.read_text()
        
        for line in env_content.split("\n"):
            if line.startswith("HOT_WALLET_PRIVATE_KEY="):
                private_key = line.split("=", 1)[1].strip()
                try:
                    # Try JSON format first
                    if private_key.startswith("["):
                        self.keypair = Keypair.from_json(private_key)
                        print("✅ Wallet loaded from .env (JSON format)")
                        return
                    else:
                        # Try base58
                        if base58 is None:
                            raise ImportError("base58 package not installed")
                        key_bytes = base58.b58decode(private_key)
                        if len(key_bytes) == 64:
                            self.keypair = Keypair.from_bytes(key_bytes)
                            print("✅ Wallet loaded from .env (base58 format)")
                            return
                except Exception as e:
                    print(f"⚠️ Error loading from .env: {e}")
        
        print("❌ No wallet found. Use --generate or --import")
    
    def _save(self):
        """Save wallet to file."""
        if self.keypair:
            WALLET_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "address": str(self.keypair.pubkey()),
                "private_key": self.keypair.to_json(),
                "network": self.network
            }
            WALLET_FILE.write_text(json.dumps(data, indent=2))
            os.chmod(WALLET_FILE, 0o600)
            print(f"✅ Wallet saved to {WALLET_FILE}")
    
    def generate(self) -> WalletInfo:
        """Generate a new wallet."""
        self.keypair = Keypair()
        self._save()
        
        info = self.get_info()
        
        print(f"\n✅ NEW WALLET GENERATED!")
        print(f"\n📍 Address: {info.address}")
        print(f"\n🔐 Private Key (JSON format):")
        print(f"   {info.private_key[:60]}...")
        print(f"\n⚠️  IMPORTANT:")
        print(f"   - Save your private key securely!")
        print(f"   - Never share it with anyone!")
        print(f"   - File: {WALLET_FILE}")
        
        return info
    
    def import_key(self, private_key: str) -> WalletInfo:
        """
        Import wallet from private key.
        
        Args:
            private_key: JSON format or base58
        """
        try:
            # Try JSON format
            if private_key.startswith("["):
                self.keypair = Keypair.from_json(private_key)
            else:
                # Try base58
                if base58 is None:
                    raise ImportError("base58 package not installed")
                key_bytes = base58.b58decode(private_key)
                self.keypair = Keypair.from_bytes(key_bytes)
            
            self._save()
            
            info = self.get_info()
            print(f"\n✅ WALLET IMPORTED!")
            print(f"📍 Address: {info.address}")
            print(f"💰 Balance: {info.balance_sol} SOL")
            
            return info
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")
    
    def get_info(self) -> WalletInfo:
        """Get wallet information."""
        if not self.keypair:
            raise ValueError("No wallet loaded")
        
        address = str(self.keypair.pubkey())
        
        try:
            # Use SoldersPubkey for RPC call
            pubkey_obj = Pubkey.from_string(address)
            response = self.client.get_balance(pubkey_obj)
            lamports = response.value
            sol = lamports / 1e9
        except Exception as e:
            print(f"⚠️ RPC Error: {e}")
            lamports = 0
            sol = 0.0
        
        now = __import__('datetime').datetime.utcnow().isoformat()
        return WalletInfo(
            public_key=address,
            key_type='json',
            created_at=now,
            last_used=now,
            network=self.network,
            address=address,
            private_key=self.keypair.to_json(),
            balance_sol=sol,
            balance_lamports=lamports
        )
    
    def sign_message(self, message: str) -> str:
        """Sign a message."""
        if not self.keypair:
            raise ValueError("No wallet loaded")
        signature = self.keypair.sign_message(message.encode())
        return signature
    
    def get_address(self) -> str:
        """Get wallet address."""
        if not self.keypair:
            raise ValueError("No wallet loaded")
        return str(self.keypair.pubkey())
    
    def save_to_env(self):
        """Save wallet info to .env file."""
        if not self.keypair:
            raise ValueError("No wallet loaded")
        
        address = str(self.keypair.pubkey())
        private_key = self.keypair.to_json()
        
        # Read existing .env
        env_content = ""
        if ENV_FILE.exists():
            env_content = ENV_FILE.read_text()
        
        # Update lines
        lines = env_content.split("\n")
        new_lines = []
        address_set = False
        private_set = False
        
        for line in lines:
            if line.startswith("HOT_WALLET_ADDRESS="):
                new_lines.append(f"HOT_WALLET_ADDRESS={address}")
                address_set = True
            elif line.startswith("HOT_WALLET_PRIVATE_KEY="):
                new_lines.append(f"HOT_WALLET_PRIVATE_KEY={private_key}")
                private_set = True
            else:
                new_lines.append(line)
        
        if not address_set:
            new_lines.append(f"HOT_WALLET_ADDRESS={address}")
        if not private_set:
            new_lines.append(f"HOT_WALLET_PRIVATE_KEY={private_key}")
        
        ENV_FILE.write_text("\n".join(new_lines))
        print(f"✅ Updated {ENV_FILE}")


class SolanaWallet(WalletManager):
    """Backward-compatible alias for older code/tests."""
    pass


class HotWalletManager(WalletManager):
    """Backward-compatible alias for older code/tests."""
    pass


# ==================== MAIN ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Solana Wallet Manager")
    parser.add_argument("--network", "-n", default="devnet",
                       choices=["devnet", "mainnet"],
                       help="Network (default: devnet)")
    parser.add_argument("--generate", action="store_true",
                       help="Generate new wallet")
    parser.add_argument("--import", dest="import_key",
                       help="Import wallet from JSON private key")
    parser.add_argument("--address", action="store_true",
                       help="Show wallet address")
    parser.add_argument("--balance", action="store_true",
                       help="Show wallet balance")
    parser.add_argument("--save-env", action="store_true",
                       help="Save wallet to .env file")
    
    args = parser.parse_args()
    
    wallet = WalletManager(network=args.network)
    
    if args.generate:
        wallet.generate()
    
    elif args.import_key:
        wallet.import_key(args.import_key)
    
    elif args.address:
        try:
            print(wallet.get_address())
        except ValueError as e:
            print(f"❌ {e}")
    
    elif args.balance:
        try:
            info = wallet.get_info()
            print(f"📍 {info.address}")
            print(f"💰 Balance: {info.balance_sol} SOL")
            print(f"   ({info.balance_lamports:,} lamports)")
        except ValueError as e:
            print(f"❌ {e}")
    
    elif args.save_env:
        try:
            wallet.save_to_env()
        except ValueError as e:
            print(f"❌ {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
