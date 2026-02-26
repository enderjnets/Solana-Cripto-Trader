#!/bin/bash
# Script para configurar Chromium + Puppeteer en máquina remota
# Uso: bash setup-remote-browser.sh <IP> <PASSWORD> <USER> <PORT>

IP="$1"
PASSWORD="$2"
USER="${3:-enderj}"
PORT="${4:-9223}"

echo "╔════════════════════════════════════════════════╗"
echo "║    CONFIGURANDO NAVEGADOR REMOTO           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "IP: $IP"
echo "Usuario: $USER"
echo "Puerto: $PORT"

# Verificar argumentos
if [ -z "$IP" ] || [ -z "$PASSWORD" ]; then
    echo "Uso: $0 <IP> <PASSWORD> [USER] [PORT]"
    echo "Ejemplo: $0 10.0.0.56 5747 enderj 9223"
    exit 1
fi

echo ""
echo "=== 1. INSTALANDO CHROMIUM ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    sudo apt update
    sudo apt install -y chromium-browser
    which chromium-browser
    chromium-browser --version
"

echo ""
echo "=== 2. CONFIGURANDO XVFB ==="
# Verificar si Xvfb está instalado
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    which Xvfb || echo 'Xvfb no instalado'
    sudo apt install -y xvfb 2>/dev/null || echo 'Xvfb ya instalado'
"

echo ""
echo "=== 3. INICIANDO CHROMIUM ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    # Matar procesos previos
    pkill -9 chromium 2>/dev/null
    sleep 2
    
    # Crear perfil
    mkdir -p /tmp/remote-chromium
    
    # Iniciar Xvfb si no hay display
    if [ -z \"\$DISPLAY\" ]; then
        Xvfb :99 -ac -screen 0 1920x1080x24 &
        export DISPLAY=:99
        sleep 2
    fi
    
    # Iniciar Chromium
    nohup chromium-browser \
        --remote-debugging-port=$PORT \
        --no-first-run \
        --no-default-browser-check \
        --disable-gpu \
        --disable-dev-shm-usage \
        --no-sandbox \
        --user-data-dir=/tmp/remote-chromium \
        about:blank > /tmp/chromium.log 2>&1 &
    
    sleep 5
    echo '✅ Chromium iniciado'
"

echo ""
echo "=== 4. VERIFICANDO ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    ps aux | grep chromium | grep -v grep | head -1
    curl -s http://localhost:$PORT/json/version | head -5
"

echo ""
echo "=== 5. CREANDO TUNNEL SSH (OPCIONAL) ==="
echo "Para acceder desde tu máquina local:"
echo "  ssh -N -L $PORT:localhost:$PORT $USER@$IP"
echo ""
echo "O ejecuta en tu máquina local:"
echo "  bash create-browser-tunnel.sh $IP $USER $PORT"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║      CONFIGURACIÓN DE NAVEGADOR COMPLETA     ║"
echo "╚══════════════════════════════════════════════╝"
