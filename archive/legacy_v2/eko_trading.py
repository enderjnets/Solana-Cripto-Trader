#!/usr/bin/env python3
"""
Eko Trading Integration Module
===============================

This module enables Eko to process trading commands from Telegram.

Usage:
    from eko_trading import process_trading_command
    
    response = process_trading_command("Eko, mi balance")
    response = process_trading_command("Eko, precio de SOL")
    response = process_trading_command("Eko, compra 0.5 SOL")
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Configuration
SOLANA_BOT_PATH = Path("/home/enderj/.openclaw/workspace/solana-jupiter-bot")
TRADING_HANDLER = SOLANA_BOT_PATH / "trading_handler.py"


class TradingCommandProcessor:
    """Process trading commands for Eko."""
    
    def __init__(self, bot_path: Path = SOLANA_BOT_PATH):
        self.bot_path = bot_path
        self.handler = TRADING_HANDLER
    
    def run_command(self, args: list) -> str:
        """Run trading handler command."""
        try:
            result = subprocess.run(
                [sys.executable, str(self.handler)] + args,
                cwd=self.bot_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            # Filter out stderr messages
            stdout = result.stdout
            stderr = result.stderr
            if "✅ Wallet loaded" in stderr:
                pass  # Ignore wallet loaded message
            return stdout.strip()
        except subprocess.TimeoutExpired:
            return "❌ Timeout: El comando tardó demasiado"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def get_balance(self) -> str:
        """Get wallet balance."""
        return self.run_command(["--balance"])
    
    def get_price(self) -> str:
        """Get SOL price."""
        return self.run_command(["--price"])
    
    def buy(self, amount: float) -> str:
        """Buy SOL."""
        return self.run_command(["--buy", str(amount)])
    
    def sell(self, amount: float) -> str:
        """Sell SOL."""
        return self.run_command(["--sell", str(amount)])
    
    def get_status(self) -> str:
        """Get system status."""
        return self.run_command(["--status"])
    
    def get_address(self) -> str:
        """Get wallet address."""
        return self.run_command(["--address"])
    
    def get_quote(self, amount: float, side: str = "buy") -> str:
        """Get quote for swap."""
        # Quote always shows SOL -> USDC
        return self.run_command(["--quote", str(amount)])
    
    def process_message(self, message: str) -> Optional[str]:
        """
        Process a Telegram message and return trading response.
        
        Args:
            message: Raw message from Telegram
            
        Returns:
            Response string or None if no trading command detected
        """
        msg_lower = message.lower().strip()
        
        # Accept "Eko" or "Eco" as trigger words
        # If message doesn't start with Eko/Eco, still process it
        # This handles messages like "Eco, dame el precio de SOL"
        
        # Balance commands
        if "mi balance" in msg_lower or "balance" in msg_lower:
            return self.get_balance()
        
        # Price commands
        if "precio" in msg_lower and "sol" in msg_lower:
            return self.get_price()
        
        # Status commands
        if "status" in msg_lower or "estado" in msg_lower:
            return self.get_status()
        
        # Address commands
        if "direcci" in msg_lower or "address" in msg_lower or "wallet" in msg_lower:
            return self.get_address()
        
        # Buy commands - show quote
        buy_match = re.search(r"compra\s+([\d.]+)\s*sol", msg_lower)
        if buy_match:
            amount = float(buy_match.group(1))
            if 0.001 <= amount <= 10:  # Sanity check
                return self.get_quote(amount, "buy")
            else:
                return "❌ Amount must be between 0.001 and 10 SOL"
        
        # Sell commands - show quote
        sell_match = re.search(r"vende\s+([\d.]+)\s*sol", msg_lower)
        if sell_match:
            amount = float(sell_match.group(1))
            if 0.001 <= amount <= 10:
                return self.get_quote(amount, "sell")
            else:
                return "❌ Amount must be between 0.001 and 10 SOL"
        
        # Quote commands
        quote_match = re.search(r"cuanto\s+(?:es|tiene)\s+([\d.]+)\s*sol", msg_lower)
        if quote_match:
            amount = float(quote_match.group(1))
            return self.get_quote(amount, "buy")
        
        # How much is X in USDC
        conversion_match = re.search(r"([\d.]+)\s*sol\s*(?:en|a|in)\s*usdc", msg_lower)
        if conversion_match:
            amount = float(conversion_match.group(1))
            return self.get_quote(amount, "buy")
        
        return None  # No trading command detected


# Singleton instance
_processor: Optional[TradingCommandProcessor] = None


def get_processor() -> TradingCommandProcessor:
    """Get or create the trading processor singleton."""
    global _processor
    if _processor is None:
        _processor = TradingCommandProcessor()
    return _processor


def process_trading_command(message: str) -> Optional[str]:
    """
    Process a trading command from a Telegram message.
    
    Usage:
        from eko_trading import process_trading_command
        
        response = process_trading_command("Eko, mi balance")
        if response:
            send_to_telegram(response)
    
    Args:
        message: Raw message from Telegram
        
    Returns:
        Response string or None if not a trading command
    """
    processor = get_processor()
    return processor.process_message(message)


# Quick test when run directly
if __name__ == "__main__":
    print("🧪 Testing Eko Trading Integration")
    print("="*50)
    
    processor = TradingCommandProcessor()
    
    test_commands = [
        "Eko, mi balance",
        "Eko, precio de SOL",
        "Eko, status del sistema",
        "Eko, mi dirección",
    ]
    
    for cmd in test_commands:
        print(f"\n📝 Command: {cmd}")
        print("-"*40)
        response = processor.process_message(cmd)
        if response:
            print(response)
        else:
            print("⚠️ No response (not a trading command)")
    
    print("\n" + "="*50)
    print("✅ Integration test complete")
