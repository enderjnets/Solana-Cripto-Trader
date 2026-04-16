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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env for Paperclip API key
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Canonical path for chat_agent (SOLAA-25: was launching bare chat_agent.py from wrong location)
CHAT_AGENT_PATH="agents/chat_agent.py"
CHAT_AGENT_Canonical="agents/chat_agent.py"

# Backoff state: if file missing, we back off instead of tight-looping
chat_agent_backoff=0
chat_agent_missing_logged=false

# Ensure chat_agent runs as background process
# SOLAA-25 fix: use canonical path, check existence, and apply backoff on repeated failure
if ! pgrep -f "agents/chat_agent\.py" > /dev/null 2>&1; then
    if [ ! -f "$CHAT_AGENT_PATH" ]; then
        if [ "$chat_agent_missing_logged" = false ]; then
            echo "[WATCHDOG] chat_agent path missing ($CHAT_AGENT_PATH), not starting. Will retry after backoff." | tee -a ${HOME}/.config/solana-jupiter-bot/chat_agent.log
            chat_agent_missing_logged=true
        fi
        chat_agent_backoff=10
    else
        echo "[WATCHDOG] Starting agents/chat_agent.py..."
        nohup python3 -u "$CHAT_AGENT_PATH" >> ${HOME}/.config/solana-jupiter-bot/chat_agent.log 2>&1 &
        chat_agent_backoff=0
        chat_agent_missing_logged=false
    fi
fi

while true; do
    # Re-check chat_agent each cycle (SOLAA-25: use canonical path for pgrep match)
    if ! pgrep -f "agents/chat_agent\.py" > /dev/null 2>&1; then
        if [ ! -f "$CHAT_AGENT_PATH" ]; then
            if [ "$chat_agent_missing_logged" = false ]; then
                echo "[WATCHDOG] chat_agent path missing ($CHAT_AGENT_PATH), skipping restart. Backoff ${chat_agent_backoff}s." | tee -a ${HOME}/.config/solana-jupiter-bot/chat_agent.log
                chat_agent_missing_logged=true
            fi
            # Don't tight-loop: exponential backoff capped at 5 minutes
            if [ "$chat_agent_backoff" -eq 0 ]; then
                chat_agent_backoff=10
            elif [ "$chat_agent_backoff" -lt 300 ]; then
                chat_agent_backoff=$((chat_agent_backoff * 2))
            fi
            echo "[WATCHDOG] chat_agent missing backoff: waiting ${chat_agent_backoff}s before retry"
            sleep "$chat_agent_backoff"
            continue
        else
            if [ "$chat_agent_missing_logged" = true ]; then
                echo "[WATCHDOG] chat_agent path restored ($CHAT_AGENT_PATH), resuming normal restart." | tee -a ${HOME}/.config/solana-jupiter-bot/chat_agent.log
                chat_agent_missing_logged=false
                chat_agent_backoff=0
            fi
            echo "[WATCHDOG] chat_agent died, restarting..."
            nohup python3 -u "$CHAT_AGENT_PATH" >> ${HOME}/.config/solana-jupiter-bot/chat_agent.log 2>&1 &
        fi
    fi

    # Check for existing orchestrator before launching (fast-fail)
    # Pattern matches BOTH agents/orchestrator.py AND root-level orchestrator.py
    # Uses both pgrep (process check) and the absolute lock file for double protection
    ORCH_PID_LOCK="/tmp/solana_jupiter_orchestrator.lock"
    if pgrep -f "Solana-Cripto-Trader/agents/orchestrator" > /dev/null 2>&1; then
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
