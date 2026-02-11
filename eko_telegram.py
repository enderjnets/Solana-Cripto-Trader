#!/usr/bin/env python3
"""
Eko Telegram Trading CLI
========================

Simple CLI for testing trading commands.

Usage:
    python3 eko_telegram.py "Eko, mi balance"
    python3 eko_telegram.py "Eko, precio de SOL"
    python3 eko_telegram.py "Eko, compra 0.5 SOL"
"""

import sys
from eko_trading import process_trading_command


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 eko_telegram.py \"<message>\"")
        print("\nExamples:")
        print('  python3 eko_telegram.py "Eko, mi balance"')
        print('  python3 eko_telegram.py "Eko, precio de SOL"')
        print('  python3 eko_telegram.py "Eko, compra 0.5 SOL"')
        print('  python3 eko_telegram.py "Eko, status del sistema"')
        sys.exit(1)
    
    message = " ".join(sys.argv[1:])
    
    # Add "Eko," prefix if not present
    if not message.lower().startswith("eko"):
        message = "Eko, " + message
    
    response = process_trading_command(message)
    
    if response:
        print(response)
    else:
        print("❓ No trading command detected in message.")
        print(f"Message: {message}")


if __name__ == "__main__":
    main()
