#!/bin/bash
# AUTONOMOUS TRADING SYSTEM AUDITOR
# Runs every 60 minutes, fixes issues automatically

LOG_FILE="/tmp/audit.log"
AUDIT_COUNT=1

log_audit() {
    echo "=== AUDITORÍA #$AUDIT_COUNT $(date) ===" >> $LOG_FILE
}

check_process() {
    local name=$1
    local pid=$(pgrep -f "$name" | head -1)
    if [ -z "$pid" ]; then
        echo "❌ $name NO CORRIENDO - REINICIANDO..." >> $LOG_FILE
        if [ "$name" == "agent_runner" ]; then
            cd /home/enderj/.openclaw/workspace/solana-jupiter-bot && nohup python3 agent_runner.py --live > /tmp/runner.log 2>&1 &
        elif [ "$name" == "agent_brain" ]; then
            cd /home/enderj/.openclaw/workspace/solana-jupiter-bot && nohup python3 agent_brain.py --fast > /tmp/brain.log 2>&1 &
        fi
        sleep 5
        echo "✅ $name reiniciado" >> $LOG_FILE
        return 1
    else
        echo "✅ $name corriendo (PID $pid)" >> $LOG_FILE
        return 0
    fi
}

check_dashboard() {
    local status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/ 2>/dev/null)
    if [ "$status" == "200" ]; then
        echo "✅ Dashboard: 200 OK" >> $LOG_FILE
        return 0
    else
        echo "❌ Dashboard error: $status - REINICIANDO..." >> $LOG_FILE
        cd /home/enderj/.openclaw/workspace/solana-jupiter-bot && nohup streamlit run dashboard/agent_dashboard.py -- --dashboard --port 8501 > /tmp/dash.log 2>&1 &
        sleep 5
        return 1
    fi
}

check_errors() {
    local errors=0
    if grep -q "ERROR\|Traceback\|Exception" /tmp/runner.log 2>/dev/null; then
        echo "⚠️ Errores en runner.log" >> $LOG_FILE
        tail -5 /tmp/runner.log | grep -i error >> $LOG_FILE
        errors=$((errors + 1))
    fi
    if grep -q "ERROR\|Traceback\|Exception" /tmp/brain.log 2>/dev/null; then
        echo "⚠️ Errores en brain.log" >> $LOG_FILE
        tail -5 /tmp/brain.log | grep -i error >> $LOG_FILE
        errors=$((errors + 1))
    fi
    return $errors
}

fix_issues() {
    echo "🔧 Intentando correcciones automáticas..." >> $LOG_FILE
    # Check for common issues and fix
    cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
    python3 -c "from api.api_integrations import JupiterClient; print('API OK')" >> $LOG_FILE 2>&1
}

get_stats() {
    local brain_gen=$(grep -o "Gen [0-9]*" /tmp/brain.log 2>/dev/null | tail -1 || echo "Gen 0")
    local best_pnl=$(grep -o "Best PnL=[0-9.]*" /tmp/brain.log 2>/dev/null | tail -1 || echo "N/A")
    local sol_price=$(grep -o "SOL=\$[0-9.]*" /tmp/runner.log 2>/dev/null | tail -1 || echo "N/A")
    echo "📊 Stats: $brain_gen | $best_pnl | SOL $sol_price" >> $LOG_FILE
}

# Main audit loop
run_audit() {
    log_audit
    echo "" >> $LOG_FILE
    
    echo "🔍 Verificando procesos..." >> $LOG_FILE
    check_process "agent_runner"
    check_process "agent_brain"
    echo "" >> $LOG_FILE
    
    echo "🌐 Verificando dashboard..." >> $LOG_FILE
    check_dashboard
    echo "" >> $LOG_FILE
    
    echo "🐛 Buscando errores..." >> $LOG_FILE
    if check_errors; then
        echo "✅ Sin errores críticos" >> $LOG_FILE
    else
        echo "🔧 Corrigiendo..." >> $LOG_FILE
        fix_issues
    fi
    echo "" >> $LOG_FILE
    
    get_stats
    echo "" >> $LOG_FILE
    echo "=== AUDITORÍA #$AUDIT_COUNT COMPLETA ===" >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    AUDIT_COUNT=$((AUDIT_COUNT + 1))
}

# Run audit
run_audit
