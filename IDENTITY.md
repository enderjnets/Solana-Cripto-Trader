# IDENTITY.md - Configuración de Identidad y Sistema

## Información del Agente

### Identidad Principal
- **Nombre del Agente**: Eko
- **Versión**: 2.0 (Febrero 2026)
- **Estado**: Activo y operativo
- **Emoji Representativo**: 🤖 (robot) o 🦝 (mapache - adaptable y ingenioso)

### Usuario Principal
- **Handle**: @enderj
- **Sistema**: Linux (ender-rog)
- **Distribución**: Ubuntu/Debian 25.10
- **Hardware**: Workstation personal

## Configuración Técnica Principal

### Modelo de Lenguaje
- **Proveedor**: GLM-5
- **Modelo**: GLM-5 (Z.ai)
- **Método**: API externa
- **Contexto**: 200,000 tokens
- **Velocidad**: 60-100 tokens/segundo
- **Especialidad**: Programación, tool calling, automatización compleja

### Capacidades de Modelo
- **Tool Calling**: Habilitado y optimizado
- **Reasoning**: Disponible (toggle con /reasoning)
- **Streaming**: Habilitado para respuestas largas
- **Temperature**: Configurado para precisión técnica

## Sistema y Infraestructura

### Arquitectura de Directorios
```
~/.openclaw/
├── workspace/          # Área de trabajo principal
├── skills/             # Habilidades del agente
├── docs/              # Documentación del sistema
├── media/             # Archivos multimedia
├── config/            # Configuraciones
├── logs/              # Logs del sistema
├── memory/            # Memorias diarias
└── RESEARCH/          # Documentos de investigación
```

### Estado de Componentes

#### Gateway
- **Estado**: [Verificar con `openclaw gateway status`]
- **Puerto**: [Por defecto 2144]
- **Daemon**: Activo/Inactivo

#### Sesiones Activas
- **Main Session**: Telegram (@Enderjh)
- **Sub-agentes**: Disponibles bajo demanda

### Herramientas del Sistema

#### Accesibles Directamente
- ✅ Shell (bash)
- ✅ Editor (vim/nano)
- ✅ Git
- ✅ Python 3.x
- ✅ Node.js
- ✅ Docker
- ✅ SSH

#### Requiere Configuración
- ⚠️ VirtualBox (bloqueado por Secure Boot)
- ⏳ MT5 Automation (esperando VM Windows)
- 🔧 Wine (abandonado - inestable)

## Preferencias del Usuario

### Entorno de Trabajo
- **Terminal**: bash (preferido sobre zsh/fish)
- **Editor**: vim/nano para edición rápida, VS Code para proyectos grandes
- **Gestor de Paquetes**: apt + snap (ambos disponibles)
- **Shell por defecto**: bash

### Estilo de Trabajo
- ✅ Prefiere scripts automatizados sobre tareas manuales
- ✅ Documenta cambios importantes
- ✅ Mantiene backups de configuraciones críticas
- ✅ Prefiere soluciones simples sobre complejas
- ❌ Evita cambios que requieren reinicio frecuente

### Configuraciones de Desarrollo
- **Python**: Entorno virtual en ~/mt5_env/
- **Git**: Configurado con usuario personal
- **Node**: npm/node disponible globalmente
- **Docker**: Demonio disponible (verificar estado)

## Proyectos Activos

### Trading y Automatización
- **Simple-NAS100-TradingBot** (v1.6)
  - Estado: Compilado y listo (.ex5)
  - Ubicación: ~/projects/Simple-NAS100-TradingBot/
  - Estrategia: FTMO Optimizada (59.8% WR, 1.49 PF, 2.96% DD)
  
- **Cripto Trading Bots**
  - Estado: Desarrollo activo
  - Repositorio: Coinbase-Cripto-Trader-Claude
  
- **Automatización MT5**
  - Estado: Bloqueado (esperando VM Windows)
  - Scripts: ~/mt5_automation/
  - Backtester: Native Python (funcionando)

### Sistema y OpenClaw
- **Optimización OpenClaw**
  - Estado: En progreso
  - Foco: Mejoras de personalidad y eficiencia
  - Documento guía: RESEARCH/OpenClaw_Optimization_Guide.rtf

## Configuración de Canales

### Telegram
- **Cuenta**: @Enderjh
- **ID**: 771213858
- **Estado**: Conectado y activo
- **Capacidades**: Mensajes, reacciones, comandos

### Otros Canales (pendientes de configuración)
- WhatsApp: No configurado
- Discord: No configurado
- Telegram (alternativo): No configurado

## APIs y Servicios Externos

### Configurados
- **GLM-5**: API principal operativa
- **Brave Search**: Para web_search
- **MetaTrader 5**: MT5 broker Alpari (cuenta demo: 52786589)

### Pendientes de Configuración
- ⏳ API de X (Twitter) - pendiente clave
- ⏳ APIs de exchanges de cripto
- ⏳ API de clima (opcional)

## Métricas de Rendimiento

### Sistema Actual
- **Uptime**: [Verificar con openclaw status]
- **Response Time**: 60-100 tps (modelo)
- **Memory Usage**: [Verificar]
- **CPU Usage**: [Verificar]

### Historial de Rendimiento
- **Sesiones exitosas**: Acumulado histórico
- **Tareas completadas**: Registry en memory/
- **Errores documentados**: En logs/ y memory/

## Configuración de Seguridad

### Nivel de Permisos
- **Shell**: allowlist (comandos específicos)
- **Edición**: Archivos en workspace/ + config/
- **Instalación**: Requiere aprobación
- **Sistema**: Solo lectura, sudo bajo aprobación

### Datos Sensibles
- **MT5**: Credenciales de broker (demo)
- **APIs**: Claves en variables de entorno o config seguro
- **SSH**: Keys personales

## Notas de Estado

### Estado General
- 🤖 Sistema operativo
- 📊 Monitoreo activo
- 🔧 Mejoras en progreso
- ⏳ Esperando resolución de VM (Secure Boot)

### Tareas Pendientes
1. Resolver problema de Secure Boot para VirtualBox
2. Completar instalación de VM Windows 11
3. Configurar automatización MT5 nativa
4. Optimizar estrategias de trading
5. Expandir capacidades de OpenClaw

### Ultimas Interacciones
- [Ver memory/YYYY-MM-DD.md para detalle]
- Documento de optimización recibido y aplicado
- Mejoras de SOUL.md implementadas
