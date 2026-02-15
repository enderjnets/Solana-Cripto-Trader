#!/bin/bash
# Persistent Dashboard Runner
# Restarts automatically if killed

cd /home/enderj/.openclaw/workspace/solana-jupiter-bot

while true; do
    echo "$(date): Starting dashboard..."
    python3 dashboard/simple_dashboard.py 8502
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "$(date): Dashboard crashed with exit code $EXIT_CODE"
    else
        echo "$(date): Dashboard stopped"
    fi
    sleep 2
done
