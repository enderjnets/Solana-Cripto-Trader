#!/bin/bash
# Multi-Worker Launcher
# ======================
# Launch multiple worker instances on a single machine

# Configuration
NUM_WORKERS=${1:-3}  # Default: 3 workers
COORDINATOR_URL=${COORDINATOR_URL:-"http://localhost:5001"}

echo "============================================"
echo "🚀 Launching $NUM_WORKERS workers"
echo "📡 Coordinator: $COORDINATOR_URL"
echo "============================================"

# Kill existing workers
pkill -f "crypto_worker.py" 2>/dev/null || true
sleep 2

# Launch workers
for i in $(seq 1 $NUM_WORKERS); do
    echo "🚀 Starting worker $i/$NUM_WORKERS..."
    
    COORDINATOR_URL="$COORDINATOR_URL" \
    WORKER_INSTANCE="$i" \
    NUM_WORKERS="$NUM_WORKERS" \
    INTERVAL="5" \
    nohup python3 crypto_worker.py > /tmp/worker_$i.log 2>&1 &
    
    sleep 3
done

echo ""
echo "✅ $NUM_WORKERS workers launched!"
echo "📝 Logs: /tmp/worker_{1..$NUM_WORKERS}.log"
echo ""
echo "To stop: pkill -f crypto_worker.py"
