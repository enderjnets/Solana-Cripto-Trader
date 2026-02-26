# Remote Setup Skill

Configura cualquier computadora remota con OpenClaw usando SSH.

## Description

Esta skill permite configurar una computadora remota con OpenClaw, instalando todas las dependencias necesarias, configurando el gateway, y habilitando capacidades de navegación automatizada.

## Prerequisites

- Acceso SSH a la máquina remota
- Contraseña sudo o clave SSH configurada
- Conexión de red activa

## Setup Steps

### 1. Conexión SSH Inicial

```bash
# Conectar a la máquina remota
ssh enderj@<IP_REMOTA>

# Verificar sistema
hostname
cat /etc/os-release | grep PRETTY_NAME
```

### 2. Instalar Node.js (requiere >=22.12.0)

```bash
# Instalar nvm si no existe
export NVM_DIR="$HOME/.nvm"
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# Instalar Node.js 22
source $NVM_DIR/nvm.sh
nvm install 22
nvm use 22

# Verificar
node --version  # Debe ser v22.x.x
```

### 3. Clonar e Instalar OpenClaw

```bash
# Crear directorio
sudo mkdir -p /usr/lib/node_modules
cd /usr/lib/node/modules

# Clonar repositorio
sudo git clone https://github.com/openclaw/openclaw.git
cd openclaw

# Instalar dependencias
sudo pnpm install

# Configurar pnpm global
sudo pnpm setup
sudo pnpm config set global-bin-dir /usr/local/bin

# Instalar binary
sudo pnpm add -g /usr/lib/node_modules/openclaw
```

### 4. Verificar OpenClaw

```bash
# Verificar instalación
which openclaw
openclaw --version

# Verificar skills
ls /usr/lib/node_modules/openclaw/skills/ | wc -l
```

### 5. Configurar Gateway

```bash
# El gateway se ejecuta automáticamente
# Verificar estado
ps aux | grep openclaw-gateway

# El gateway corre en puerto 18789
```

### 6. Instalar Chromium para Navegación

```bash
# Instalar Chromium (Ubuntu/Debian)
sudo apt update
sudo apt install -y chromium-browser

# O usar snap si está disponible
which chromium-browser
chromium-browser --version
```

### 7. Instalar Puppeteer (para control de navegador)

```bash
# En la máquina LOCAL
cd /home/enderj/.openclaw/workspace
npm install puppeteer
```

### 8. Configurar SSH para Acceso Sin Contraseña

```bash
# Generar clave SSH si no existe
ssh-keygen -t ed25519

# Copiar clave a la máquina remota
ssh-copy-id enderj@<IP_REMOTA>

# O manualmente
cat ~/.ssh/id_ed25519.pub | ssh enderj@<IP_REMOTA> 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
```

## Comandos Útiles de Conexión

### Conexión SSH Básica

```bash
# Conectar
ssh enderj@<IP>

# Con sshpass (para scripts)
sshpass -p "CONTRASEÑA" ssh enderj@<IP> "comando"

# Con clave SSH
ssh -i ~/.ssh/id_ed25519 enderj@<IP>
```

### Tunnel SSH para Navegador Remoto

```bash
# Crear tunnel para acceder al navegador de otra máquina
ssh -N -L 9223:localhost:9223 enderj@<IP_REMOTA>

# Puerto local -> Puerto remoto
ssh -N -L PUERTO_LOCAL:localhost:PUERTO_REMOTO enderj@<IP>
```

### Ejemplo Completo de Tunnel

```bash
# En laptop local
ssh -N -L 9223:localhost:9223 enderj@10.0.0.56

# Ahora desde local podemos acceder al navegador remoto
curl http://localhost:9223/json/version
```

## Scripts de Automatización

### Script de Control de Navegador (ekoo-browser.js)

```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.connect({
    browserURL: 'http://127.0.0.1:9223',
    defaultViewport: { width: 1280, height: 800 }
  });

  const page = await browser.newPage();
  
  // Comandos disponibles
  // open <url> - Abrir página
  // screenshot [archivo] - Tomar screenshot
  // title - Mostrar título
  // eval <código> - Ejecutar JS
  // list - Listar páginas
  // close - Cerrar navegador
  
  await browser.disconnect();
})();
```

### Script de Servicio Chromium

```bash
#!/bin/bash
# start-ekoo-chromium.sh

CHROMIUM_PORT=9224
CHROMIUM_PROFILE=/tmp/ekoo-chromium

mkdir -p $CHROMIUM_PROFILE

# Matar procesos previos
pkill -9 chromium 2>/dev/null
sleep 2

# Verificar/crear Xvfb
if [ -z "$DISPLAY" ]; then
    Xvfb :99 -ac -screen 0 1920x1080x24 &
    export DISPLAY=:99
    sleep 2
fi

# Iniciar Chromium
chromium-browser \
    --remote-debugging-port=$CHROMIUM_PORT \
    --no-first-run \
    --no-default-browser-check \
    --disable-gpu \
    --disable-dev-shm-usage \
    --no-sandbox \
    --user-data-dir=$CHROMIUM_PROFILE \
    about:blank &
```

## Problemas Comunes y Soluciones

### Error: Node version incompatible

```
Error: openclaw requires Node >=22.12.0
Solución: Instalar Node.js 22 usando nvm
```

### Error: sudo requiere contraseña

```
Problema: La contraseña sudo no funciona en scripts
Solución: Verificar contraseña correcta o configurar sudo sin contraseña
```

### Error: Puerto ya en uso

```
Problema: El puerto de debugging está ocupado
Solución: Usar otro puerto (9222, 9223, 9224, etc.)
```

### Error: No hay display para Chromium

```
Problema: chromium requiere X server
Solución: Usar Xvfb para crear display virtual
```

### Error: Connection timeout

```
Problema: La máquina remota no responde
Solución: Verificar que la máquina esté encendida y conectada a red
```

## Verificación Final

```bash
# Verificar todos los componentes
echo "=== VERIFICACIÓN FINAL ==="

# 1. Sistema
hostname
cat /etc/os-release | grep PRETTY_NAME

# 2. Node.js
node --version

# 3. OpenClaw
which openclaw
openclaw --version

# 4. Gateway
ps aux | grep openclaw-gateway | grep -v grep

# 5. Chromium
ps aux | grep chromium | grep -v grep

# 6. Recursos
free -h | grep Mem
df -h / | tail -1

# 7. Red
hostname -I | head -1
```

## Configuraciones por Defecto

| Componente | Valor |
|------------|-------|
| **Puerto Gateway** | 18789 |
| **Puerto Chromium** | 9222/9223/9224 |
| **Usuario SSH** | enderj |
| **Directorio OpenClaw** | /usr/lib/node_modules/openclaw |
| **Directorio NVM** | ~/.nvm |

## Referencias

- OpenClaw Docs: https://docs.openclaw.ai
- NVM: https://github.com/nvm-sh/nvm
- Puppeteer: https://pptr.dev
- SSH: https://www.openssh.com/

## Author

Creado: 2026-02-14
Basado en configuración de ASUS dorada (10.0.0.56)
