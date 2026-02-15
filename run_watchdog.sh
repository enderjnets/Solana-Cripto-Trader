#!/bin/bash
# Enhanced Watchdog for agent_brain_paper - Auto-restart with full session isolation

LOG_FILE="/home/enderj/.openclaw/workspace/solana-jupiter-bot/watchdog.log"
AGENT_PID_FILE="/home/enderj/.openclaw/workspace/solana-jupiter-bot/agent.pid"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log_msg "🚀 Enhanced Watchdog started"

restart_agent() {
    log_msg "🔄 Restarting agent..."
    
    # Kill any existing agent processes
    pkill -9 -f "agent_brain_paper.py" 2>/dev/null
    sleep 1
    
    # Start agent in a completely new session (survives terminal close)
    cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
    setsid nohup python3 agent_brain_paper.py --fast > /tmp/agent_brain.log 2>&1 &
    AGENT_PID=$!
    echo $AGENT_PID > $AGENT_PID_FILE
    
    log_msg "✅ Agent restarted (PID: $AGENT_PID)"
}

# Main loop
while true; do
    # Check if agent is running
    if pgrep -f "agent_brain_paper.py --fast" > /dev/null 2>&1; then
        log_msg "✅ Agent running" 2>/dev/null
    else
        log_msg "⚠️ Agent not running, restarting..."
        restart_agent
    fi
    
    # Check every 2 minutes
    sleep 120
done
