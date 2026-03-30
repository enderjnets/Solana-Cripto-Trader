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

cd ~/.openclaw/workspace/Solana-Cripto-Trader/agents
while true; do
    echo "[WATCHDOG] Starting agents/orchestrator.py..."
    python3 -u orchestrator.py
    echo "[WATCHDOG] Bot died (exit $?), restarting in 5s..."
    sleep 5
done
