#!/bin/bash
# check_macs.sh — Verifica que ComfyUI esté corriendo en ambas Macs
# Uso: bash check_macs.sh
# Ejecutar antes de producción para confirmar que los servidores están listos.

MAC_M3="10.0.0.232:8188"
MAC_AIR="100.118.215.73:8188"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BitTrader — ComfyUI Health Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Verificando Mac M3 Pro ($MAC_M3)..."
if curl -s --connect-timeout 5 "http://$MAC_M3/system_stats" > /dev/null; then
    echo "  ✅ Mac M3 Pro OK — ComfyUI respondiendo"
    # Mostrar queue status
    QUEUE=$(curl -s --connect-timeout 5 "http://$MAC_M3/queue" 2>/dev/null)
    if [ -n "$QUEUE" ]; then
        echo "  📋 Queue: $QUEUE" | head -c 200
        echo ""
    fi
else
    echo "  ❌ Mac M3 Pro NO responde — iniciar ComfyUI en 10.0.0.232"
fi

echo ""
echo "Verificando Mac Air M4 ($MAC_AIR)..."
if curl -s --connect-timeout 5 "http://$MAC_AIR/system_stats" > /dev/null; then
    echo "  ✅ Mac Air M4 OK — ComfyUI respondiendo"
    QUEUE=$(curl -s --connect-timeout 5 "http://$MAC_AIR/queue" 2>/dev/null)
    if [ -n "$QUEUE" ]; then
        echo "  📋 Queue: $QUEUE" | head -c 200
        echo ""
    fi
else
    echo "  ❌ Mac Air M4 NO responde — iniciar ComfyUI en 100.118.215.73"
fi

echo ""
echo "Verificando PC RTX3070 (localhost:8188)..."
if curl -s --connect-timeout 3 "http://localhost:8188/system_stats" > /dev/null; then
    echo "  ✅ PC RTX3070 OK — ComfyUI respondiendo"
else
    echo "  ℹ️  PC RTX3070 no disponible (fallback secundario, opcional)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Prioridad de producción:"
echo "    1. Mac M3 Pro + Mac Air M4 (paralelo, \$0)"
echo "    2. PC RTX3070 (fallback local, \$0)"
echo "    3. Replicate API (último recurso, costo \$)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
