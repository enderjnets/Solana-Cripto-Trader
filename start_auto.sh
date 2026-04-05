#!/bin/bash
# Auto-Learning System Start Script

cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader

echo "========================================="
echo "🚀 AUTO-LEARNING SYSTEM INITIALIZATION"
echo "========================================="
echo ""

# Check if already running
if pgrep -f "python3.*run_auto_learning" > /dev/null; then
    echo "⚠️  Auto-learning already running"
    echo "Stopping previous instance..."
    pkill -f "python3.*run_auto_learning"
    sleep 2
fi

# Start the system
echo "📊 Starting auto-learning system..."
nohup python3 -u run_auto_learning.py > /tmp/auto_learning.log 2>&1 &
PID=$!

echo "✅ System started with PID: $PID"
echo ""

# Wait for initialization
echo "⏳ Waiting for initialization..."
sleep 5

# Check if running
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Process is running"
    echo ""
    echo "📋 Recent logs:"
    tail -30 /tmp/auto_learning.log
    echo ""
    echo "========================================="
    echo "🎯 AUTO-LEARNING SYSTEM ACTIVE"
    echo "========================================="
    echo ""
    echo "📊 Monitor commands:"
    echo "  tail -f /tmp/auto_learning.log"
    echo "  python3 start_auto_learning.py --status"
    echo "  python3 monitor_auto_learning.py"
    echo ""
    echo "🛑 Stop command:"
    echo "  kill $PID"
    echo ""
else
    echo "❌ Failed to start"
    echo ""
    echo "📋 Error logs:"
    cat /tmp/auto_learning.log
fi
