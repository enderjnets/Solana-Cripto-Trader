#!/bin/bash
LOCKFILE="/tmp/solana_master_orchestrator.lock"

# Check if already running
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[WATCHDOG] Already running (PID $OLD_PID), exiting."
        exit 0
    else
        echo "[WATCHDOG] Stale lock, removing."
        rm -f "$LOCKFILE"
    fi
fi

# Write our PID
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

cd ~/.openclaw/workspace/Solana-Cripto-Trader
while true; do
    echo "[WATCHDOG] Starting master_orchestrator.py..."
    python3 -u master_orchestrator.py
    echo "[WATCHDOG] Bot died (exit $?), restarting in 5s..."
    sleep 5
done
