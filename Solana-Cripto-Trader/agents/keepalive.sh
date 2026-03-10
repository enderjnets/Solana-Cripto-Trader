#!/bin/bash
if ! pgrep -f "orchestrator.py.*--live" > /dev/null 2>&1; then
    cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents
    setsid nohup python3 -u orchestrator.py --live --interval 120 >> /home/enderj/.config/solana-jupiter-bot/modular.log 2>&1 &
fi
