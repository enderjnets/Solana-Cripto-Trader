#!/bin/bash
# Watchdog para mantener el sistema modular de Solana corriendo

AGENTS_DIR="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents"
LOG_DIR="$HOME/.config/solana-jupiter-bot"
LOG_FILE="$LOG_DIR/modular.log"
PID_FILE="$LOG_DIR/modular.pid"

# Crear directorio de logs
mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🤖 Watchdog Modular - Iniciando..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📁 Directorio: $AGENTS_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📝 Log: $LOG_FILE"

while true; do
    # Verificar si el proceso del orchestrator está corriendo
    if ! pgrep -f "agents/orchestrator.py" > /dev/null; then
        echo "[$(date '+%H:%M:%S')] 🔄 El sistema modular se detuvo. Reiniciando..."
        
        # Cambiar al directorio de agentes
        cd "$AGENTS_DIR"
        
        # Iniciar el orchestrator en modo background
        nohup python3 orchestrator.py --once >> "$LOG_FILE" 2>&1 &
        ORCH_PID=$!
        
        # Guardar PID
        echo "$ORCH_PID" > "$PID_FILE"
        
        echo "[$(date '+%H:%M:%S')] ✅ Sistema modular reiniciado (PID: $ORCH_PID)"
        
        # Esperar 2 segundos para asegurar que arranque
        sleep 2
    fi
    
    # Esperar 60 segundos antes del siguiente check
    sleep 60
done
