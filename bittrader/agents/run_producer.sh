#!/bin/bash
cd /home/enderj/.openclaw/workspace/bittrader/agents
LOG="/tmp/producer_$(date +%Y%m%d_%H%M%S).log"
echo "=== Producer Start: $(date) ===" | tee "$LOG"
python3 -u producer.py --limit 10 >> "$LOG" 2>&1
echo "=== Producer End: $(date) ===" | tee -a "$LOG"
cp "$LOG" data/last_producer.log
