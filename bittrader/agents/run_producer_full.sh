#!/bin/bash
# Run BitTrader Producer fully detached
# Usage: bash run_producer_full.sh [limit] [--no-ai-video]

cd /home/enderj/.openclaw/workspace/bittrader/agents
LIMIT=${1:-5}
FLAGS=${2:-""}
LOG="/tmp/producer_$(date +%Y%m%d_%H%M%S).log"

echo "=== BitTrader Producer ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Limit: $LIMIT" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# Run producer
python3 -u producer.py --limit "$LIMIT" $FLAGS >> "$LOG" 2>&1
EXIT=$?

echo "" >> "$LOG"
echo "=== Finished: $(date) | Exit: $EXIT ===" >> "$LOG"

# Copy log
cp "$LOG" data/last_producer.log 2>/dev/null

# Notify via telegram (using openclaw CLI or curl)
RESULT=$(tail -5 "$LOG" | head -3)
echo "Producer finished. Exit: $EXIT. Log: $LOG"
