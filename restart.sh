#!/bin/bash
# Simple wrapper to keep trading system running

LOG_FILE="/home/enderj/.openclaw/workspace/solana-jupiter-bot/restart.log"
SCRIPT="/home/enderj/.openclaw/workspace/solana-jupiter-bot/unified_trading_system.py"

echo "$(date): Wrapper started" >> $LOG_FILE

while true; do
    echo "$(date): Starting trading system..." >> $LOG_FILE
    python3 $SCRIPT --start >> /dev/null 2>&1
    
    # Wait for process to die
    wait $!
    
    echo "$(date): Process died, waiting 5s before restart..." >> $LOG_FILE
    sleep 5
done
