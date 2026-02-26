#!/bin/bash
# Script de instalación automática para OpenClaw en máquina remota
# Uso: bash install-openclaw-remote.sh <IP> <PASSWORD> <USER>

IP="$1"
PASSWORD="$2"
USER="${3:-enderj}"

echo "╔════════════════════════════════════════════════╗"
echo "║   INSTALACIÓN OPENCLAW - MÁQUINA REMOTA     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "IP: $IP"
echo "Usuario: $USER"

# Verificar argumentos
if [ -z "$IP" ] || [ -z "$PASSWORD" ]; then
    echo "Uso: $0 <IP> <PASSWORD> [USER]"
    echo "Ejemplo: $0 10.0.0.56 5747 enderj"
    exit 1
fi

echo ""
echo "=== 1. VERIFICANDO CONEXIÓN ==="
if ping -c 2 "$IP" > /dev/null 2>&1; then
    echo "✅ Máquina accesible"
else
    echo "❌ No se puede alcanzar $IP"
    exit 1
fi

echo ""
echo "=== 2. INSTALANDO DEPENDENCIAS ==="

# Instalar nvm y Node.js 22
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    # Instalar nvm
    export NVM_DIR=\"\$HOME/.nvm\"
    if [ ! -s \"\$NVM_DIR/nvm.sh\" ]; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    fi
    
    # Cargar nvm
    source \$NVM_DIR/nvm.sh
    
    # Instalar Node 22
    nvm install 22
    nvm use 22
    echo 'Node version:'
    node --version
"

echo ""
echo "=== 3. CLONANDO OPENCLAW ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    sudo mkdir -p /usr/lib/node_modules
    cd /usr/lib/node_modules
    sudo git clone https://github.com/openclaw/openclaw.git
    echo '✅ OpenClaw clonado'
"

echo ""
echo "=== 4. INSTALANDO OPENCLAW ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    cd /usr/lib/node_modules/openclaw
    sudo pnpm install
    echo '✅ Dependencias instaladas'
"

echo ""
echo "=== 5. CONFIGURANDO PNPM GLOBAL ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    cd /usr/lib/node_modules/openclaw
    sudo pnpm setup
    sudo pnpm config set global-bin-dir /usr/local/bin
    sudo pnpm add -g /usr/lib/node_modules/openclaw
    echo '✅ Binary instalado'
"

echo ""
echo "=== 6. VERIFICANDO INSTALACIÓN ==="
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$IP" "
    echo '=== OPENCLAW ==='
    which openclaw
    openclaw --version
    
    echo ''
    echo '=== SKILLS ==='
    ls /usr/lib/node_modules/openclaw/skills/ | wc -l
    
    echo ''
    echo '=== GATEWAY ==='
    ps aux | grep openclaw-gateway | grep -v grep | head -1
"

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║           INSTALACIÓN COMPLETA               ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Para conectar:"
echo "  ssh $USER@$IP"
echo ""
echo "Para verificar:"
echo "  ssh $USER@$IP 'openclaw --version'"
