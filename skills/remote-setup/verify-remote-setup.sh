#!/bin/bash
# Verificación completa de máquina remota con OpenClaw
# Uso: bash verify-remote-setup.sh <IP> <PASSWORD> <USER>

IP="$1"
PASSWORD="$2"
USER="${3:-enderj}"

echo "╔════════════════════════════════════════════════╗"
echo "║     VERIFICACIÓN COMPLETA - OPENCLAW       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Máquina: $USER@$IP"
echo ""

# Verificar argumentos
if [ -z "$IP" ] || [ -z "$PASSWORD" ]; then
    echo "Uso: $0 <IP> <PASSWORD> [USER]"
    echo "Ejemplo: $0 10.0.0.56 5747 enderj"
    exit 1
fi

echo "=== 1. CONECTIVIDAD ==="
if ping -c 2 "$IP" > /dev/null 2>&1; then
    echo "✅ Máquina accesible"
else
    echo "❌ No se puede alcanzar $IP"
fi

echo ""
echo "=== 2. SISTEMA ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    hostname
    cat /etc/os-release | grep PRETTY_NAME
    uptime | head -1
"

echo ""
echo "=== 3. NODE.JS ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    echo 'Node version:'
    node --version 2>/dev/null || echo '❌ No instalado'
    
    echo 'NPM version:'
    npm --version 2>/dev/null || echo '❌ No instalado'
"

echo ""
echo "=== 4. OPENCLAW ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    echo 'Binary:'
    which openclaw 2>/dev/null || echo '❌ No instalado'
    
    echo 'Versión:'
    openclaw --version 2>/dev/null || echo '❌ No funciona'
    
    echo 'Skills:'
    ls /usr/lib/node_modules/openclaw/skills/ 2>/dev/null | wc -l | xargs echo 'disponibles'
"

echo ""
echo "=== 5. GATEWAY ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    ps aux | grep openclaw-gateway | grep -v grep | head -1 || echo '❌ Gateway no corriendo'
    
    # Verificar puerto
    netstat -tlnp 2>/dev/null | grep 18789 || ss -tlnp | grep 18789 || echo 'Puerto 18789 no abierto'
"

echo ""
echo "=== 6. CHROMIUM ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    ps aux | grep chromium | grep -v grep | head -1 || echo '❌ Chromium no corriendo'
    
    # Verificar puertos de debugging
    netstat -tlnp 2>/dev/null | grep -E '9222|9223|9224' || ss -tlnp | grep -E '9222|9223|9224' || echo 'Puertos de debugging no abiertos'
"

echo ""
echo "=== 7. DOCKER ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    docker --version 2>/dev/null || echo '❌ No instalado'
    docker ps 2>/dev/null | head -3 || echo 'Docker no accesible'
"

echo ""
echo "=== 8. RECURSOS ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    free -h | grep Mem
    df -h / | tail -1
"

echo ""
echo "=== 9. RED ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    hostname -I | head -1
"

echo ""
echo "=== 10. SSH ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    ls -la ~/.ssh/ 2>/dev/null | head -5 || echo 'SSH no configurado'
"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║          VERIFICACIÓN COMPLETA              ║"
echo "╚══════════════════════════════════════════════╝"
