#!/bin/bash
# Watchdog para Master Orchestrator con Auto-Learning Integrado
# ==============================================================
# Mantiene el sistema corriendo y reinicia si muere

cd ~/.openclaw/workspace/Solana-Cripto-Trader

while true; do
    echo "[$(date)] Starting Master Orchestrator with Auto-Learning..." >> ~/.config/solana-jupiter-bot/master.log
    python3 -u start_master_with_auto_learning.py >> ~/.config/solana-jupiter-bot/master.log 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date)] Bot died with exit code $EXIT_CODE, restarting..." >> ~/.config/solana-jupiter-bot/master.log
    else
        echo "[$(date)] Bot stopped normally, exiting..." >> ~/.config/solana-jupiter-bot/master.log
        break
    fi

    sleep 5
done
