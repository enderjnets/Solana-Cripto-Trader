#!/bin/bash
# Wrapper that keeps the trading system running
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot

while true; do
    echo "$(date): Starting trading system..." >> /tmp/trading_wrapper.log
    python3 -u unified_trading_system.py --start 2>&1 | tee -a /tmp/trading.log
    echo "$(date): Process exited, waiting 5s..." >> /tmp/trading_wrapper.log
    sleep 5
done
