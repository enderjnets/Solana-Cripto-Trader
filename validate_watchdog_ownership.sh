#!/bin/bash
# Runtime validation script for watchdog ownership
# Fails (exit 1) when service is auto-restarting while external watchdog exists.
# This script is idempotent and safe to run at any time.

set -euo pipefail

LOCKFILE="/tmp/solana_modular_orchestrator.lock"
SERVICE_NAME="solana-jupiter-bot.service"

echo "[VALIDATE] Checking watchdog ownership..."

# Get the PID holding the lock (if any)
if [ ! -f "$LOCKFILE" ]; then
    echo "[VALIDATE] No lock file exists. Checking service status..."
    # No lock - verify service is running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "[VALIDATE] OK: Service is active and no conflicting lock exists."
        exit 0
    else
        echo "[VALIDATE] WARNING: Service is not active and no lock file exists."
        exit 1
    fi
fi

LOCK_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
if [ -z "$LOCK_PID" ] || ! kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "[VALIDATE] Stale lock file exists but process is dead."
    exit 1
fi

# Get the PID of the systemd service's watchdog (if running)
SERVICE_WATCHDOG_PID=""
if systemctl is-active --quiet "$SERVICE_NAME"; then
    SERVICE_WATCHDOG_PID=$(systemctl show --property MainPID --value "$SERVICE_NAME" 2>/dev/null || echo "")
fi

# Determine if lock holder is the systemd service or external
if [ -n "$SERVICE_WATCHDOG_PID" ] && [ "$SERVICE_WATCHDOG_PID" != "0" ]; then
    if [ "$LOCK_PID" = "$SERVICE_WATCHDOG_PID" ]; then
        echo "[VALIDATE] OK: Lock held by systemd service (PID $LOCK_PID)."
        exit 0
    fi

    # Check if lock holder's parent is systemd
    LOCK_PPID=$(cat /proc/"$LOCK_PID"/stat 2>/dev/null | cut -d' ' -f4 || echo "")
    if [ "$LOCK_PPID" = "1" ]; then
        echo "[VALIDATE] OK: Lock held by systemd-spawned process (PID $LOCK_PID)."
        exit 0
    fi
fi

# At this point, lock is held by a non-systemd process
echo "[VALIDATE] FAIL: Lock held by external process (PID $LOCK_PID)."
echo "[VALIDATE] External watchdog is running outside systemd control."
echo "[VALIDATE] This will cause service restart loop (SOLAA-20)."

# Check if service restart counter is growing
RESTART_COUNT=$(systemctl show "$SERVICE_NAME" -p NRestarts --value 2>/dev/null || echo "0")
echo "[VALIDATE] Service restart count: $RESTART_COUNT"

if [ "$RESTART_COUNT" -gt 5 ]; then
    echo "[VALIDATE] CRITICAL: Service has restarted $RESTART_COUNT times."
    echo "[VALIDATE] External watchdog must be terminated to stop restart loop."
fi

exit 1
