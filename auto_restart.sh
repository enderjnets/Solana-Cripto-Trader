#!/bin/bash
# auto_restart.sh - Auto-restart Unified Trading System if heartbeat is old
# =================================================================
# This script continuously monitors the heartbeat and auto-restarts if needed.

HEARTBEAT_FILE="/home/enderj/.openclaw/workspace/solana-jupiter-bot/data/heartbeat.txt"
MAX_AGE=300  # 5 minutes
PROJECT_DIR="/home/enderj/.openclaw/workspace/solana-jupiter-bot"

echo "🚀 Auto-restart monitor started..."
echo "   Heartbeat file: $HEARTBEAT_FILE"
echo "   Max age: ${MAX_AGE}s"
echo "   Check interval: 60s"
echo ""

while true; do
    # Check if heartbeat file exists
    if [ ! -f "$HEARTBEAT_FILE" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ❌ No heartbeat file, restarting..."
        cd "$PROJECT_DIR"
        nohup python3 unified_trading_system.py --continuous > unified_trading_system_output.log 2>&1 &
        sleep 600  # Wait 10 minutes before next check
        continue
    fi

    # Calculate age
    TIMESTAMP=$(cat "$HEARTBEAT_FILE" | jq -r '.timestamp' 2>/dev/null)
    CURRENT_EPOCH=$(date +%s)
    HEARTBEAT_EPOCH=$(date -d "$TIMESTAMP" +%s 2>/dev/null || echo 0)
    AGE=$((CURRENT_EPOCH - HEARTBEAT_EPOCH))

    # Check age
    if [ "$AGE" -gt "$MAX_AGE" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ❌ Heartbeat old (${AGE}s), restarting..."

        # Kill old process
        PID=$(cat "$HEARTBEAT_FILE" | jq -r '.pid' 2>/dev/null)
        if [ "$PID" != "null" ] && [ -n "$PID" ]; then
            kill -9 "$PID" 2>/dev/null
            echo "   Killed old process (PID: $PID)"
        fi

        # Start new process
        cd "$PROJECT_DIR"
        nohup python3 unified_trading_system.py --continuous > unified_trading_system_output.log 2>&1 &
        NEW_PID=$!
        echo "   Started new process (PID: $NEW_PID)"

        sleep 600  # Wait 10 minutes before next check
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ✅ System healthy (age: ${AGE}s)"
    fi

    sleep 60
done
