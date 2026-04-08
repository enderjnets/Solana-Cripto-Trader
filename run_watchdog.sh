#!/bin/bash
LOCKFILE="/tmp/solana_modular_orchestrator.lock"
HANDOVER_FLAG="/tmp/solana_watchdog_handover.lock"
RESTART_MARKER="/tmp/solana_watchdog_restart_marker"
IS_SYSTEMD=false

# Detect if running under systemd (PPid=1 or cgroup contains system.slice)
if [ -d /sys/fs/cgroup/system.slice ] || [ "$(cat /proc/1/comm 2>/dev/null)" = "systemd" ]; then
    IS_SYSTEMD=true
fi

# --- Restart loop prevention ---
# If we see the restart marker modified within the last 60s, we're in a restart loop
# (systemd keeps restarting us because another watchdog holds the lock)
if [ -f "$RESTART_MARKER" ]; then
    LAST_MOD=$(stat -c %Y "$RESTART_MARKER" 2>/dev/null)
    NOW=$(date +%s)
    if [ $((NOW - LAST_MOD)) -lt 60 ] && [ "$IS_SYSTEMD" = true ]; then
        # Update marker to show we're still stuck
        echo "$(date +%s)" > "$RESTART_MARKER"
        echo "[WATCHDOG] Restart loop detected (another watchdog holds lock). Exiting to stop spam."
        # Write handover flag so the other watchdog knows systemd wants to yield
        echo "$(date +%s):systemd:$PPID" > "$HANDOVER_FLAG"
        exit 1
    fi
fi
# Touch restart marker
echo "$(date +%s)" > "$RESTART_MARKER"

# --- Lock acquisition ---
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        # Another watchdog is running. If we're under systemd, exit 1 to signal
        # to systemd that this is a failure (stops Restart=always from spinning).
        # If running manually, just exit 0 (normal duplicate check).
        echo "[WATCHDOG] Already running (PID $OLD_PID), exiting."
        if [ "$IS_SYSTEMD" = true ]; then
            # Write handover flag so the holding process can see systemd wants to yield
            echo "$(date +%s):systemd:$PPID" > "$HANDOVER_FLAG"
            exit 1
        fi
        echo "[WATCHDOG] NOTE: Use 'systemctl start solana-jupiter-bot.service' to manage the bot."
        exit 0
    else
        echo "[WATCHDOG] Stale lock removed."
        rm -f "$LOCKFILE"
    fi
fi

echo $$ > "$LOCKFILE"
# Signal that we successfully took the lock
rm -f "$HANDOVER_FLAG"
trap "rm -f $LOCKFILE $RESTART_MARKER" EXIT

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

    # Check for existing orchestrator before launching (fast-fail)
    # Pattern matches BOTH agents/orchestrator.py AND root-level orchestrator.py
    # Uses both pgrep (process check) and the absolute lock file for double protection
    ORCH_PID_LOCK="/tmp/solana_jupiter_orchestrator.lock"
    if pgrep -f "orchestrator\.py" > /dev/null 2>&1; then
        echo "[WATCHDOG] orchestrator already running (detected by pgrep), skipping launch"
    elif [ -f "$ORCH_PID_LOCK" ] && kill -0 "$(cat $ORCH_PID_LOCK)" 2>/dev/null; then
        echo "[WATCHDOG] orchestrator already running (lock file PID $(cat $ORCH_PID_LOCK)), skipping launch"
    else
        echo "[WATCHDOG] Starting agents/orchestrator.py..."
        python3 -u agents/orchestrator.py
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 1 ]; then
            echo "[WATCHDOG] orchestrator rejected duplicate launch (exit $EXIT_CODE), waiting 30s before retry..."
            sleep 30
            continue
        fi
        echo "[WATCHDOG] Bot died (exit $EXIT_CODE), restarting in 5s..."
    fi
    sleep 5
done
