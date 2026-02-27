#!/bin/bash
# Start Trading System with Watchdog
# This script starts the trading system with automatic restart capability

set -e

cd "$(dirname "$0")"

# Create logs directory
mkdir -p logs

# Kill any existing trading processes
echo "🛑 Stopping any existing trading processes..."
pkill -f "unified_trading_system.py" 2>/dev/null || true
pkill -f "watchdog.py" 2>/dev/null || true
sleep 2

# Start watchdog in background
echo "🚀 Starting trading watchdog..."
nohup python3 watchdog.py > logs/watchdog.log 2>&1 &
WATCHDOG_PID=$!

echo "✅ Trading system started with watchdog"
echo "   Watchdog PID: $WATCHDOG_PID"
echo ""
echo "📊 Logs:"
echo "   - Trading: logs/trading.log"
echo "   - Watchdog: logs/watchdog.log"
echo ""
echo "🔍 To stop: pkill -f watchdog.py"
echo "🔍 To monitor: tail -f logs/watchdog.log"
