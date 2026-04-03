#!/bin/bash
#
# 🐕 SOLANA WATCHDOG — Restart seguro del orchestrator
#
LOCKFILE="/tmp/solana_brain_paper.lock"
LOGFILE="${SOLANA_LOG_DIR:-/home/enderj/.config/solana-jupiter-bot}/watchdog.log"
AGENTS_DIR="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents"
DATA_DIR="$AGENTS_DIR/data"

mkdir -p "$(dirname "$LOGFILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"
}

# Prevenir multiples instancias
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        log "WATCHDOG: Ya corriendo (PID $OLD_PID), saliendo."
        exit 0
    else
        log "WATCHDOG: Lock stale, removiendo."
        rm -f "$LOCKFILE"
    fi
fi

echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE; log 'WATCHDOG: Saliendo.'" EXIT

emergency_backup() {
    log "WATCHDOG: ⚠️  Emergency backup..."
    [ -f "$DATA_DIR/trade_history.json" ] && cp "$DATA_DIR/trade_history.json" "$DATA_DIR/trade_history.json.emergency_bak" 2>/dev/null
}

log_memory() {
    local pid=$1
    [ -n "$pid" ] && [ -d "/proc/$pid" ] && log "WATCHDOG: Orchestrator PID $pid — memoria: $(grep VmRSS /proc/$pid/status 2>/dev/null | awk '{print $2/1024"MB"}')"
}

RESTART_COUNT=0
MAX_FAST_RESTARTS=5
BASE_DELAY=5

while true; do
    log "WATCHDOG: ============================================"
    log "WATCHDOG: Iniciando agent_brain_paper.py (intento #$((RESTART_COUNT+1)))"
    log "WATCHDOG: Memoria libre: $(free -m | awk '/^Mem:/{print $7"MB"}')"

    emergency_backup

    # EJECUTAR DESDE agents/ directory — crítico para imports
    cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
    python3 -u agent_brain_paper.py --fast
    EXIT_CODE=$?

    DEAD_PID=$(cat "$LOCKFILE" 2>/dev/null)
    log "WATCHDOG: Brain Paper murio — exit code=$EXIT_CODE (PID: $DEAD_PID)"

    if [ "$EXIT_CODE" -eq 137 ]; then
        log "WATCHDOG: 🚨 Exit 137 (OOM)"
    elif [ "$EXIT_CODE" -eq 143 ]; then
        log "WATCHDOG: ⚠️ Exit 143 (SIGTERM)"
    fi

    if [ "$RESTART_COUNT" -ge "$MAX_FAST_RESTARTS" ]; then
        DELAY=60
        log "WATCHDOG: ⏱️  Demasiados restarts — delay ${DELAY}s"
        RESTART_COUNT=0
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        DELAY=$((BASE_DELAY * RESTART_COUNT))
        [ "$DELAY" -gt 30 ] && DELAY=30
        log "WATCHDOG: ⏱️  Esperando ${DELAY}s..."
    fi

    sleep "$DELAY"
done
