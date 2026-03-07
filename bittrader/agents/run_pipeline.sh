#!/bin/bash
# BitTrader Pipeline v2 - Cada guion como proceso independiente
cd /home/enderj/.openclaw/workspace/bittrader/agents

LOG="/tmp/bittrader_v2_$(date +%Y%m%d_%H%M%S).log"
echo "=== BitTrader Pipeline v2 ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

# Limpiar WIP anterior
rm -f data/guiones_wip.json

# PASO 1: Scout
echo "PASO 1: Scout..." | tee -a "$LOG"
python3 orchestrator.py --scout-only --no-telegram >> "$LOG" 2>&1
echo "Scout exit: $?" | tee -a "$LOG"

# PASO 2: Creator - un guion por vez (10 guiones)
echo "PASO 2: Creator (10 guiones uno a uno)..." | tee -a "$LOG"
for i in 0 1 2 3 4 5 6 7 8 9; do
    echo "  Guion $((i+1))/10..." | tee -a "$LOG"
    timeout 75 python3 run_creator_single.py $i >> "$LOG" 2>&1
    EXIT=$?
    if [ $EXIT -eq 124 ]; then
        echo "  ⏱️ Guion $((i+1)) timeout — continuando" | tee -a "$LOG"
    elif [ $EXIT -ne 0 ]; then
        echo "  ⚠️ Guion $((i+1)) exit $EXIT — continuando" | tee -a "$LOG"
    fi
    sleep 2
done
echo "Creator completo" | tee -a "$LOG"

# Verificar cuántos guiones se generaron
TOTAL=$(python3 -c "import json; d=json.load(open('data/guiones_wip.json')); print(d['stats']['total'])" 2>/dev/null || echo "0")
echo "Guiones generados: $TOTAL" | tee -a "$LOG"

# Copiar WIP a latest si hay contenido
if [ "$TOTAL" -gt "0" ]; then
    cp data/guiones_wip.json data/guiones_latest.json
    echo "guiones_latest.json actualizado con $TOTAL scripts" | tee -a "$LOG"
fi

# PASO 3: Producer
echo "PASO 3: Producer..." | tee -a "$LOG"
python3 orchestrator.py --produce-only --no-telegram >> "$LOG" 2>&1
echo "Producer exit: $?" | tee -a "$LOG"

# PASO 4: Publisher
echo "PASO 4: Publisher..." | tee -a "$LOG"
python3 orchestrator.py --publish-only --no-telegram >> "$LOG" 2>&1
echo "Publisher exit: $?" | tee -a "$LOG"

echo "=== Pipeline COMPLETO: $(date) ===" | tee -a "$LOG"
cp "$LOG" data/last_pipeline.log
echo "Log: $LOG"
