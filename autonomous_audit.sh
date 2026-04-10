#!/bin/bash
# AUTONOMOUS TRADING SYSTEM AUDITOR — Solana Cripto Trader
# Verifica los procesos reales de este deployment: orchestrator + dashboard

LOG_FILE="/tmp/audit.log"
REPO="/home/enderj/.openclaw/workspace/Solana-Cripto-Trader"
AUDIT_COUNT=1
ERRORS=0

log() {
    echo "$1" >> $LOG_FILE
}

log_audit() {
    log "=== AUDITORÍA #$AUDIT_COUNT $(date) ==="
}

# ── 1. Procesos críticos ──────────────────────────────────────────────────

check_orchestrator() {
    local pid=$(pgrep -f "agents/orchestrator.py" | head -1)
    if [ -z "$pid" ]; then
        log "❌ ORQUESTADOR no corre — watchdog debería reiniciarlo"
        ERRORS=$((ERRORS + 1))
        # Intentar reiniciar via watchdog
        local wdog=$(pgrep -f "run_watchdog.sh" | head -1)
        if [ -z "$wdog" ]; then
            log "❌ WATCHDOG tampoco corre — reiniciando watchdog..."
            cd $REPO && nohup bash run_watchdog.sh > /tmp/solana_watchdog.log 2>&1 &
            sleep 5
            log "   → watchdog reiniciado (PID $!)"
        else
            log "   → watchdog activo (PID $wdog), esperando reinicio..."
        fi
        return 1
    else
        log "✅ Orquestador corriendo (PID $pid)"
        return 0
    fi
}

check_dashboard() {
    local status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/api/stats 2>/dev/null)
    if [ "$status" == "200" ]; then
        log "✅ Dashboard: 200 OK"
        return 0
    else
        log "❌ Dashboard error: status=$status — reiniciando..."
        ERRORS=$((ERRORS + 1))
        pkill -f "dashboard/app.py" 2>/dev/null
        sleep 2
        cd $REPO && nohup python3 dashboard/app.py 8081 > /tmp/solana_dashboard.log 2>&1 &
        sleep 4
        local new_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/api/stats 2>/dev/null)
        if [ "$new_status" == "200" ]; then
            log "   ✅ Dashboard reiniciado OK"
        else
            log "   ❌ Dashboard sigue caído ($new_status)"
        fi
        return 1
    fi
}

# ── 2. Salud del portafolio ───────────────────────────────────────────────

check_portfolio() {
    local pf="$REPO/agents/data/portfolio.json"
    if [ ! -f "$pf" ]; then
        log "❌ portfolio.json no existe"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
    local equity=$(python3 -c "
import json
from pathlib import Path
p = json.loads(Path('$pf').read_text())
positions = [x for x in p.get('positions',[]) if x.get('status')=='open']
cap = p.get('capital_usd', 0)
invested = sum(x.get('margin_usd',0) for x in positions)
unrealized = sum(x.get('pnl_usd',0) for x in positions)
equity = cap + invested + unrealized
initial = p.get('initial_capital', 100)
dd = (initial - equity) / initial * 100 if initial > 0 else 0
print(f'equity={equity:.2f} initial={initial:.2f} dd={dd:.2f}% positions={len(positions)}')
" 2>/dev/null)
    log "✅ Portfolio: $equity"

    # Alerta si drawdown > 20%
    local dd=$(echo "$equity" | grep -o 'dd=[0-9.]*' | cut -d= -f2)
    if [ ! -z "$dd" ] && (( $(echo "$dd > 20" | bc -l 2>/dev/null || echo 0) )); then
        log "⚠️ ALERTA: Drawdown $dd% supera 20%"
        ERRORS=$((ERRORS + 1))
    fi
    return 0
}

# ── 3. Integridad de archivos de estado ──────────────────────────────────

check_state_files() {
    local ok=1
    for f in portfolio.json market_latest.json; do
        local fp="$REPO/agents/data/$f"
        if [ ! -f "$fp" ]; then
            log "⚠️ Falta $f"
            ok=0
        elif ! python3 -c "import json; json.loads(open('$fp').read())" 2>/dev/null; then
            log "❌ JSON corrupto: $f"
            ERRORS=$((ERRORS + 1))
            ok=0
        fi
    done
    [ $ok -eq 1 ] && log "✅ Archivos de estado OK"
}

# ── 4. Verificar compilación ──────────────────────────────────────────────

check_compile() {
    if python3 -m py_compile \
        $REPO/agents/orchestrator.py \
        $REPO/agents/executor.py \
        $REPO/agents/martingale_engine.py \
        $REPO/dashboard/app.py 2>/dev/null; then
        log "✅ Compilación OK"
    else
        log "❌ Error de compilación en archivos principales"
        ERRORS=$((ERRORS + 1))
    fi
}

# ── 5. Tests ──────────────────────────────────────────────────────────────

check_tests() {
    local result=$(cd $REPO && timeout 30 python3 test_system.py 2>&1 | tail -3)
    if echo "$result" | grep -q "passed.*0 failed\|7 passed"; then
        log "✅ Tests: $result"
    else
        log "⚠️ Tests con fallos: $result"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────

run_audit() {
    ERRORS=0
    log_audit
    log ""

    log "🔍 Procesos..."
    check_orchestrator
    check_dashboard
    log ""

    log "💰 Portfolio..."
    check_portfolio
    log ""

    log "📁 Estado de archivos..."
    check_state_files
    log ""

    log "🔧 Compilación..."
    check_compile
    log ""

    log "🧪 Tests..."
    check_tests
    log ""

    if [ $ERRORS -eq 0 ]; then
        log "🎉 AUDITORÍA #$AUDIT_COUNT LIMPIA — 0 errores"
    else
        log "⚠️ AUDITORÍA #$AUDIT_COUNT: $ERRORS error(es) encontrado(s)"
    fi
    log ""
    AUDIT_COUNT=$((AUDIT_COUNT + 1))
}

run_audit
