#!/bin/bash
while true; do
    cd ~/.openclaw/workspace/Solana-Cripto-Trader
    python3 -u master_orchestrator.py >> ~/.config/solana-jupiter-bot/master.log 2>&1
    echo "[WATCHDOG] Bot died, restarting..." >> ~/.config/solana-jupiter-bot/master.log
    sleep 5
done
