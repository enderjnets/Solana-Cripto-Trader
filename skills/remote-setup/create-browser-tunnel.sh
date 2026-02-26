#!/bin/bash
# Crear tunnel SSH para navegador remoto
# Uso: bash create-browser-tunnel.sh <IP> <USER> <LOCAL_PORT> <REMOTE_PORT>

IP="$1"
USER="${2:-enderj}"
LOCAL_PORT="${3:-9223}"
REMOTE_PORT="${4:-9223}"

echo "╔════════════════════════════════════════════════╗"
echo "║       CREANDO TUNNEL SSH PARA NAVEGADOR    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Máquina remota: $USER@$IP"
echo "Puerto local: $LOCAL_PORT"
echo "Puerto remoto: $REMOTE_PORT"

# Verificar argumentos
if [ -z "$IP" ]; then
    echo "Uso: $0 <IP> [USER] [LOCAL_PORT] [REMOTE_PORT]"
    echo "Ejemplo: $0 10.0.0.56 enderj 9223 9223"
    exit 1
fi

echo ""
echo "=== CREANDO TUNNEL ==="
echo "Ejecutando:"
echo "  ssh -N -L $LOCAL_PORT:localhost:$REMOTE_PORT $USER@$IP"
echo ""
echo "Presiona Ctrl+C para detener el tunnel"
echo ""

# Crear tunnel
ssh -N -L "$LOCAL_PORT:localhost:$REMOTE_PORT" "$USER@$IP"

echo ""
echo "Tunnel cerrado."
