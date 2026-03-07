#!/bin/bash
# Watchdog para mantener el sistema modular de Solana con Agentes IA corriendo

AGENTS_DIR="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents"
LOG_DIR="$HOME/.config/solana-jupiter-bot"
LOG_FILE="$LOG_DIR/modular_ai.log"
PID_FILE="$LOG_DIR/modular_ai.pid"

# Flags para el orchestrator
LEVERAGE_FLAG=""
WATCHDOG_TIMEOUT=180  # Segundos entre checks (aumentado a 180s para dar tiempo a LLM)

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --leverage)
            LEVERAGE_FLAG="--leverage"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Crear directorio de logs
mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🤖 Watchdog Modular AI - Iniciando..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📁 Directorio: $AGENTS_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📝 Log: $LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏱️  Timeout: ${WATCHDOG_TIMEOUT}s (para LLM)"
if [ -n "$LEVERAGE_FLAG" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔗 Leverage Mode: ACTIVO"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🤖 Agentes IA: ACTIVOS"

while true; do
    # Verificar si el proceso del orchestrator_ai está corriendo
    if ! pgrep -f "orchestrator_ai.py" > /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 El sistema modular AI se detuvo. Reiniciando..."

        # Cambiar al directorio de agentes
        cd "$AGENTS_DIR"

        # Iniciar el orchestrator con IA en modo background
        nohup python3 orchestrator_ai.py --once $LEVERAGE_FLAG >> "$LOG_FILE" 2>&1 &
        ORCH_PID=$!

        # Guardar PID
        echo "$ORCH_PID" > "$PID_FILE"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Sistema modular AI reiniciado (PID: $ORCH_PID)"

        # Esperar 2 segundos para asegurar que arranque
        sleep 2
    fi

    # Esperar X segundos antes del siguiente check
    sleep $WATCHDOG_TIMEOUT
done
