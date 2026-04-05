#!/bin/bash
# Watchdog para mantener el sistema modular de Solana corriendo
# Optimizado 2026-03-08: Ciclo continuo con --live
# Fix 2026-03-28: Logs unificados, códigos de salida claros, rotación automática

AGENTS_DIR="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents"
LOG_DIR="$HOME/.config/solana-jupiter-bot"
LOG_FILE="$LOG_DIR/modular.log"
PID_FILE="$LOG_DIR/modular.pid"

# Crear directorio de logs
mkdir -p "$LOG_DIR"

# ── Función de log unificado ───────────────────────────────────────────────
log() {
    local level="$1"
    shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] [${level}] ${msg}" | tee -a "$LOG_FILE"
}

# ── Rotación de logs (si > 50MB) ──────────────────────────────────────────
rotate_log_if_needed() {
    if [ -f "$LOG_FILE" ]; then
        local size_bytes
        size_bytes=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        local limit_bytes=$((50 * 1024 * 1024))   # 50 MB
        if [ "$size_bytes" -ge "$limit_bytes" ]; then
            mv "$LOG_FILE" "${LOG_FILE}.old"
            log "INFO" "Log rotado — tamaño anterior: $((size_bytes / 1024 / 1024))MB → ${LOG_FILE}.old"
        fi
    fi
}

# ── Arranque ──────────────────────────────────────────────────────────────
log "INFO" "Watchdog Modular v3 - Iniciando..."
log "INFO" "Directorio: $AGENTS_DIR"
log "INFO" "Log unificado: $LOG_FILE"

while true; do
    # Rotar log si es necesario
    rotate_log_if_needed

    # Verificar si el orchestrator modular está corriendo
    if ! pgrep -f "orchestrator.py.*--live" > /dev/null; then
        log "WARN" "Orchestrator modular no está corriendo. Iniciando..."

        cd "$AGENTS_DIR"

        # Modo --live: ciclo continuo cada 120 segundos
        # Redirigir stdout+stderr al log unificado
        nohup python3 -u orchestrator.py --live --interval 120 >> "$LOG_FILE" 2>&1 &
        ORCH_PID=$!
        echo "$ORCH_PID" > "$PID_FILE"

        log "INFO" "Orchestrator modular iniciado (PID: $ORCH_PID)"
        sleep 5

        # Verificar que realmente arrancó
        if kill -0 "$ORCH_PID" 2>/dev/null; then
            log "INFO" "PID $ORCH_PID confirmado vivo"
        else
            # Leer exit code si terminó rápido
            wait "$ORCH_PID" 2>/dev/null
            EXIT_CODE=$?
            log "ERROR" "Orchestrator terminó prematuramente — exit_code=${EXIT_CODE}"
            log "HEALTH" "Ciclo completado — exit_code=${EXIT_CODE} — equity=N/A — posiciones=N/A"
        fi
    else
        # Orchestrator corriendo — verificar si el proceso PID del archivo sigue vivo
        if [ -f "$PID_FILE" ]; then
            STORED_PID=$(cat "$PID_FILE")
            if ! kill -0 "$STORED_PID" 2>/dev/null; then
                # Proceso del PID registrado ya no existe — esperar exit code
                wait "$STORED_PID" 2>/dev/null
                EXIT_CODE=$?
                log "WARN" "PID $STORED_PID ya no existe — exit_code=${EXIT_CODE}"
                if [ "$EXIT_CODE" -eq 0 ]; then
                    log "INFO" "Ciclo anterior terminó con éxito (exit_code=0)"
                else
                    log "ERROR" "Ciclo anterior terminó con error (exit_code=${EXIT_CODE})"
                fi
            fi
        fi
    fi

    # Check cada 120 segundos
    sleep 120
done
