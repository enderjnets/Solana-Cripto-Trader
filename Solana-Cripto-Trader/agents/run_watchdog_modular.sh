#!/bin/bash
# Watchdog para mantener el sistema modular de Solana corriendo
# Optimizado 2026-03-08: Ciclo continuo con --live

AGENTS_DIR="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents"
LOG_DIR="$HOME/.config/solana-jupiter-bot"
LOG_FILE="$LOG_DIR/modular.log"
PID_FILE="$LOG_DIR/modular.pid"

# Crear directorio de logs
mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🤖 Watchdog Modular v2 - Iniciando..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📁 Directorio: $AGENTS_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📝 Log: $LOG_FILE"

while true; do
    # Verificar si el orchestrator modular está corriendo
    if ! pgrep -f "orchestrator.py.*--live" > /dev/null; then
        echo "[$(date '+%H:%M:%S')] 🔄 Orchestrator modular no está corriendo. Iniciando..."
        
        cd "$AGENTS_DIR"
        
        # Modo --live: ciclo continuo cada 120 segundos
        nohup python3 -u orchestrator.py --live --interval 120 >> "$LOG_FILE" 2>&1 &
        ORCH_PID=$!
        echo "$ORCH_PID" > "$PID_FILE"
        
        echo "[$(date '+%H:%M:%S')] ✅ Orchestrator modular iniciado (PID: $ORCH_PID)"
        sleep 5
    fi
    
    # Check cada 120 segundos
    sleep 120
done
