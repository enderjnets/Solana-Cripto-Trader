#!/bin/bash
LOCKFILE="/tmp/solana_modular_orchestrator.lock"

if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[WATCHDOG] Already running (PID $OLD_PID), exiting."
        exit 0
    else
        echo "[WATCHDOG] Stale lock removed."
        rm -f "$LOCKFILE"
    fi
fi

echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

cd ~/.openclaw/workspace/Solana-Cripto-Trader

# Load .env for Paperclip API key
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Asegurar que chat_agent corre como background
if ! pgrep -f "chat_agent.py" > /dev/null 2>&1; then
    echo "[WATCHDOG] Starting chat_agent.py..."
    nohup python3 -u chat_agent.py >> /home/enderj/.config/solana-jupiter-bot/chat_agent.log 2>&1 &
fi

while true; do
    # Re-check chat_agent cada ciclo
    if ! pgrep -f "chat_agent.py" > /dev/null 2>&1; then
        echo "[WATCHDOG] chat_agent died, restarting..."
        nohup python3 -u chat_agent.py >> /home/enderj/.config/solana-jupiter-bot/chat_agent.log 2>&1 &
    fi
    echo "[WATCHDOG] Starting agents/orchestrator.py..."
    python3 -u agents/orchestrator.py
    echo "[WATCHDOG] Bot died (exit $?), restarting in 5s..."
    sleep 5
done
