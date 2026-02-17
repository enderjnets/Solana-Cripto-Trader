#!/bin/bash
# Watchdog para mantener el bot de trading corriendo

cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
source venv/bin/activate

while true; do
    if ! pgrep -f "multi_agent_trader.py" > /dev/null; then
        echo "[$(date)] Bot muerto, reiniciando..."
        nohup python -u multi_agent_trader.py > bot.log 2>&1 &
    else
        echo "[$(date)] Bot corriendo..."
    fi
    sleep 60
done
