#!/bin/bash
# Trading Bot Wrapper - Keeps bot running with auto-restart and notifications
# Usage: ./trading_wrapper.sh

set -e

# Configuration
PROJECT_DIR="/home/enderj/.openclaw/workspace/solana-jupiter-bot"
LOG_FILE="/tmp/trading_wrapper.log"
BOT_LOG="/tmp/trading_bot.log"
NOTIFICATION_LOG="$PROJECT_DIR/data/notifications.log"
WRAPPER_PID_FILE="/tmp/trading_wrapper.pid"
MAX_RESTARTS=5
RESTART_WINDOW=300  # 5 minutes in seconds

# State tracking
RESTART_COUNT=0
LAST_RESTART=0
GRACEFUL_SHUTDOWN=false

# Check if wrapper is already running
if [ -f "$WRAPPER_PID_FILE" ]; then
    OLD_PID=$(cat "$WRAPPER_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "✅ Trading bot wrapper is already running (PID: $OLD_PID)"
        echo "   To stop it: kill $OLD_PID"
        echo "   To restart: kill $OLD_PID && ./trading_wrapper.sh"
        exit 0
    else
        echo "⚠️  Stale PID file found, cleaning up..."
        rm -f "$WRAPPER_PID_FILE"
    fi
fi

# Save our PID
echo $$ > "$WRAPPER_PID_FILE"

# Graceful shutdown handler
shutdown_handler() {
    echo "$(date): 🛑 Received shutdown signal, stopping wrapper..." >> "$LOG_FILE"
    GRACEFUL_SHUTDOWN=true

    # Stop the bot if it's running
    if [ -f "$BOT_PID_FILE" ]; then
        BOT_PID=$(cat "$BOT_PID_FILE")
        if ps -p "$BOT_PID" > /dev/null 2>&1; then
            echo "$(date): 📤 Stopping bot PID $BOT_PID..." >> "$LOG_FILE"
            kill -TERM "$BOT_PID" 2>/dev/null || true
            sleep 5

            # Force kill if still running
            if ps -p "$BOT_PID" > /dev/null 2>&1; then
                echo "$(date): 🔨 Force killing bot..." >> "$LOG_FILE"
                kill -KILL "$BOT_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$BOT_PID_FILE"
    fi

    echo "$(date): ✅ Wrapper stopped" >> "$LOG_FILE"
    exit 0
}

# Setup signal handlers
trap shutdown_handler SIGINT SIGTERM

# Send notification via log file
send_notification() {
    local message="$1"
    local priority="${2:-normal}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$priority] $message" >> "$NOTIFICATION_LOG" 2>/dev/null || true
}

# Check restart rate limiting
check_restart_limit() {
    local current_time=$(date +%s)

    if [ $current_time -lt $((LAST_RESTART + RESTART_WINDOW)) ]; then
        RESTART_COUNT=$((RESTART_COUNT + 1))
    else
        RESTART_COUNT=1
    fi
    LAST_RESTART=$current_time

    if [ $RESTART_COUNT -gt $MAX_RESTARTS ]; then
        echo "$(date): ❌ RESTART LIMIT REACHED ($RESTART_COUNT restarts in $RESTART_WINDOW seconds)" >> "$LOG_FILE"
        send_notification "🚨 TRADING BOT ALERT: Too many crashes! Stopped to prevent infinite loop." "high"
        return 1
    fi

    return 0
}

# Main loop
cd "$PROJECT_DIR"
BOT_PID_FILE="/tmp/trading_bot.pid"

echo "$(date): ==========================================" >> "$LOG_FILE"
echo "$(date): 🚀 TRADING BOT WRAPPER STARTED" >> "$LOG_FILE"
echo "$(date): ==========================================" >> "$LOG_FILE"
send_notification "🟢 Trading bot wrapper started"

while [ "$GRACEFUL_SHUTDOWN" = false ]; do
    echo "$(date): ▶️ Starting trading bot..." >> "$LOG_FILE"

    # Start the bot in background
    python3 -u unified_trading_system.py --continuous >> "$BOT_LOG" 2>&1 &
    BOT_PID=$!
    echo "$BOT_PID" > "$BOT_PID_FILE"

    echo "$(date): 🆕 Bot started with PID $BOT_PID" >> "$LOG_FILE"

    # Wait for bot to exit
    wait $BOT_PID
    EXIT_CODE=$?

    # Clean up PID file
    rm -f "$BOT_PID_FILE"

    echo "$(date): 🛑 Bot exited with code $EXIT_CODE" >> "$LOG_FILE"

    # Check if this was a graceful shutdown
    if [ "$GRACEFUL_SHUTDOWN" = true ]; then
        echo "$(date): ✅ Graceful shutdown, not restarting" >> "$LOG_FILE"
        break
    fi

    # Check restart limit
    if ! check_restart_limit; then
        echo "$(date): ❌ Restart limit reached, stopping wrapper" >> "$LOG_FILE"
        break
    fi

    # Send restart notification
    if [ $EXIT_CODE -ne 0 ]; then
        send_notification "⚠️ Trading bot crashed (exit code: $EXIT_CODE). Restart #$RESTART_COUNT..." "high"
        echo "$(date): 📡 Restart notification sent" >> "$LOG_FILE"
    fi

    # Wait before restarting
    WAIT_TIME=10
    echo "$(date): ⏳ Waiting ${WAIT_TIME}s before restart..." >> "$LOG_FILE"
    sleep $WAIT_TIME
done

echo "$(date): ==========================================" >> "$LOG_FILE"
echo "$(date): 🏁 TRADING BOT WRAPPER STOPPED" >> "$LOG_FILE"
echo "$(date): ==========================================" >> "$LOG_FILE"
send_notification "🔴 Trading bot wrapper stopped"

exit 0
