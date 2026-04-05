#!/bin/bash
# Auto-Learning Monitor - Enviar actualizaciones cada 5 minutos

LOG_FILE="/tmp/auto_learning.log"
STATE_FILE="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/data/learner_state.json"

while true; do
    # Check if auto-learning is running
    if ! pgrep -f "run_auto_learning" > /dev/null; then
        echo "⚠️ Auto-learning not running!" >&2
        sleep 300
        continue
    fi
    
    # Get current state
    if [ -f "$STATE_FILE" ]; then
        GENERATION=$(jq -r '.current_generation' "$STATE_FILE")
        TRADES=$(jq -r '.total_trades' "$STATE_FILE")
        DAILY_PNL=$(jq -r '.daily_pnl' "$STATE_FILE")
        TOTAL_PNL=$(jq -r '.total_pnl' "$STATE_FILE")
        EXPLORATION=$(jq -r '.exploration_rate' "$STATE_FILE")
        
        # Format PnL as percentage
        DAILY_PNL_PCT=$(echo "scale=2; $DAILY_PNL * 100" | bc)
        
        # Log status
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Gen: $GENERATION | Trades: $TRADES | Daily PnL: ${DAILY_PNL_PCT}% | Exploration: $EXPLORATION" >> /tmp/auto_learning_monitor.log
        
        # Check for targets
        if (( $(echo "$DAILY_PNL >= 0.05" | bc -l) )); then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 DAILY TARGET REACHED! PnL: ${DAILY_PNL_PCT}%" >> /tmp/auto_learning_monitor.log
        fi
        
        if (( $(echo "$DAILY_PNL <= -0.10" | bc -l) )); then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ MAX DRAWDOWN WARNING! PnL: ${DAILY_PNL_PCT}%" >> /tmp/auto_learning_monitor.log
        fi
    fi
    
    sleep 300  # Check every 5 minutes
done
