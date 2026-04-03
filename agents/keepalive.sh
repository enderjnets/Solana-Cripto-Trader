#!/bin/bash
# Keepalive para Solana Bot - Usa run_watchdog_safe.sh
# Solo inicia si NO hay ningún watchdog corriendo

AGENTS_DIR="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader"
LOG_DIR="/home/enderj/.config/solana-jupiter-bot"
WD_SCRIPT="$AGENTS_DIR/run_watchdog_safe.sh"

RUNNING=$(pgrep -f "run_watchdog_safe" | wc -l)

if [ "$RUNNING" -eq 0 ]; then
    cd "$AGENTS_DIR"
    nohup bash "$WD_SCRIPT" >> "$LOG_DIR/watchdog.log" 2>&1 &
    echo "[$(date)] Keepalive: Started run_watchdog_safe PID $!" >> "$LOG_DIR/keepalive.log"
else
    echo "[$(date)] Keepalive: run_watchdog_safe already running ($RUNNING)" >> "$LOG_DIR/keepalive.log"
fi
