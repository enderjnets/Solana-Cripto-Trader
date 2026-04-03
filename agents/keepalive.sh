#!/bin/bash
# Keepalive para Solana Bot
AGENTS_DIR="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents"
WD_SCRIPT="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/run_watchdog_safe.sh"
LOG_DIR="/home/enderj/.config/solana-jupiter-bot"

# Solo inicia si NO hay watchdog corriendo
RUNNING=$(pgrep -f "run_watchdog_safe" | wc -l)

if [ "$RUNNING" -eq 0 ]; then
    cd "$AGENTS_DIR"
    nohup bash "$WD_SCRIPT" >> "$LOG_DIR/watchdog.log" 2>&1 &
    echo "[$(date)] Keepalive: Started run_watchdog_safe PID $!" >> "$LOG_DIR/keepalive.log"
else
    echo "[$(date)] Keepalive: run_watchdog_safe already running ($RUNNING)" >> "$LOG_DIR/keepalive.log"
fi
