# TOOLS.md - Configuración Local y Herramientas Específicas

## Configuración Personal del Sistema

### Información del Hardware
- **Hostname**: ender-rog
- **OS**: Ubuntu/Debian 25.10
- **Kernel**: Linux 6.17.0-14-generic
- **Arquitectura**: x64

### Recursos del Sistema
- **CPU**: [Verificar con `lscpu`]
- **RAM**: [Verificar con `free -h`]
- **Almacenamiento**: [Verificar con `df -h`]

## Ollama / Qwen Local

### Configuración
- **Servicio**: `systemctl status ollama` (auto-start)
- **Puerto**: 11434
- **Modelo**: qwen2.5:14b (9GB, GPU RTX 3070)
- **Provider OpenClaw**: `local` (NO "ollama")
- **API**: openai-completions + /v1 + injectNumCtx=false

### Comandos útiles
```bash
# Estado
systemctl status ollama
ollama list

# Parar (liberar VRAM para gaming/video)
sudo systemctl stop ollama

# Iniciar
sudo systemctl start ollama

# Test rápido
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:14b","messages":[{"role":"user","content":"hola"}],"stream":false}'
```

### ⚠️ Notas importantes
- Qwen ocupa 7.3GB de 8GB VRAM → Whisper debe correr en CPU
- Provider DEBE llamarse `local` (no `ollama`) para evitar detección automática
- `injectNumCtxForOpenAICompat: false` es obligatorio (evita error de RAM)
- contextWindow: 32768 en config (mínimo OpenClaw es 16K)

## Coding Agent — Configuración de Modelos

### Orden de preferencia (orden de Ender, 2026-03-30):
1. **Opus 4.6** (`claude/claude-opus-4-6`) — Primera opción (CEO + coding)
2. **MiniMax M2.7** (`minimax/MiniMax-M2.7`) — Fallback

### Uso:
- **Claude Code CLI**: `claude --model claude-opus-4-6 --permission-mode bypassPermissions --print 'tarea'`
- **Sub-agente OpenClaw**: `sessions_spawn(task="...", model="claude/claude-opus-4-6")`
- **Si Opus falla (rate limit/error)**: reintentar con `model="minimax/MiniMax-M2.7"`

## Herramientas de Desarrollo

### Python
- **Versión**: Python 3.x
- **Entornos Virtuales**:
  - `~/mt5_env/` - Para automatización MT5
  - Ubicación de scripts: `~/mt5_automation/`
- **Paquetes Principales**: [Ver requirements.txt]

### Node.js
- **Versión**: v22.22.0
- **NPM**: Disponible globalmente
- **OpenClaw**: Instalado en `/usr/lib/node_modules/openclaw/`

### Git
- **Usuario**: enderjnets
- **Repositorios**: GitHub configurado
- **Flujo**: Pull → Work → Commit → Push

### Docker
- **Estado**: Disponible
- **Comandos**: `docker`, `docker-compose`
- **Uso**: Contenedores cuando sea necesario

## Configuración de Red

### SSH
- **Servicio**: Habilitado
- **Configuración**: ~/.ssh/
- **Hosts Conocidos**: [Agregar según necesidad]

### Conexiones Activas
- **API MiniMax**: Conectada
- **Brave Search**: Configurada
- **MT5 (Alpari)**: Demo account 52786589

## Herramientas de Trading

### MetaTrader 5 (MT5)
- **Instalación**: Wine → Xvfb (headless)
- **Estado**: Limitado (Wine inestable)
- **Cuenta Demo**: 52786589
- **Broker**: Alpari
- **Símbolos**: NAS100, NY200, cripto
- **Automatización**: Esperando VM Windows

### Scripts de Automatización
- **Backtester Nativo**: Python (funcionando)
- **Optimizer**: `ftmo_optimizer_v2.py`
- **Download Scripts**: En RESEARCH/

## Virtualización

### VirtualBox
- **Versión**: 7.2.2
- **Estado**: Instalado pero bloqueado
- **Problema**: Secure Boot impide cargar vboxdrv
- **ISO**: `~/Downloads/Win11_25H2_English_x64.iso` (7.7 GB)

### Estado de VM
- **Windows 11**: Creada pero no operativa
- **Próximo Paso**: Firmar módulo o deshabilitar Secure Boot

## Preferencias de Terminal

### Shell
- **Default**: bash
- **Prompt**: Personalizado según sistema
- **History**: Habilitado

### Editores
- **Edición Rápida**: vim/nano
- **Proyectos Grandes**: VS Code
- **Configuración**: ~/.vimrc, ~/.nanorc

### Utilidades Útiles
- **ls, cd, cp, mv, rm**: Comandos básicos
- **grep, find, awk, sed**: Procesamiento de texto
- **jq, yq**: JSON/YAML parsing
- **curl, wget**: Descargas HTTP
- **rsync**: Sincronización

## Integraciones de Mensajería

### Telegram
- **Cuenta**: @Enderjh
- **ID**: 771213858
- **Capacidades**: 
  - Mensajes de texto
  - Reacciones
  - Comandos
  - Media (imágenes, documentos)

### Canales Potenciales
- **WhatsApp**: Pendiente configuración
- **Discord**: Pendiente configuración
- **Signal**: Pendiente configuración

## Configuración de APIs

### APIs Configuradas
- **MiniMax**: API_KEY en config
- **Brave Search**: API_KEY en config
- **MT5**: Credenciales en ~/.mt5/

### APIs Pendientes
- ⏳ X (Twitter) API
- ⏳ Exchanges de criptomonedas
- ⏳ APIs de clima (opcional)

## Habilidades y Skills

### Skills Instaladas
- **coding-agent**: Control de agentes de código
- **healthcheck**: Auditoría de seguridad
- **skill-creator**: Creación de nuevas skills
- **video-frames**: Extracción de video
- **weather**: Información meteorológica

### Skills Personalizadas
- [Agregar según desarrollo]

## Monitoreo y Logs

### Logs del Sistema
- **OpenClaw**: ~/.openclaw/logs/
- **MT5**: ~/.mt5/logs/
- **Docker**: `docker logs` si aplicable

### Comandos de Monitoreo
```bash
# Sistema
htop          # CPU y RAM
iotop         # I/O de disco
nethtop       # Uso de red

# OpenClaw
openclaw status    # Estado general
openclaw gateway   # Gestión de gateway

# Docker
docker ps          # Contenedores activos
docker stats       # Métricas
```

## Audio y Multimedia

### TTS (Text-to-Speech)
- **Habilitado**: Sí
- **Proveedor**: ElevenLabs (sag)
- **Voz Preferida**: Nova (cálida, ligeramente británica)
- **Speaker Default**: [Por determinar]

### Reproducción
- **Comando**: afplay (macOS) o mplayer (Linux)
- **Ubicación**: ~/.openclaw/media/

## Accesos Directos y Alias

### Alias de Bash Útiles
```bash
# OpenClaw
alias ocs='openclaw status'
alias ocg='openclaw gateway'

# Sistema
alias ll='ls -la'
alias grep='grep --color=auto'

# Python
alias ve='source ~/mt5_env/bin/activate'

# Git
alias gs='git status'
alias gl='git log --oneline'
```

## Notas de Seguridad

### Archivos Sensibles
- ~/.mt5/ - Credenciales MT5
- ~/.ssh/ - Keys SSH
- ~/.openclaw/config/ - Configuraciones con API keys

### Permisos
- Scripts ejecutables: Verificar antes de ejecutar
- Downloads: Escanear antes de abrir
- Ejecutables: Solo de fuentes confiables

## Gmail y Google Calendar (gog CLI)

### Estado
- **Cuenta**: enderjnets@gmail.com
- **Comando**: `gog` (wrapper automático, no necesita variables de entorno)
- **Acceso**: Gmail (leer, buscar, enviar) + Google Calendar (ver, crear, editar eventos)

### Gmail — Comandos principales
```bash
# Buscar emails no leídos
gog gmail search 'is:unread' --max 10

# Buscar emails de alguien
gog gmail search 'from:persona@ejemplo.com' --max 5

# Ver contenido de un mensaje (usa el ID del search)
gog gmail get <messageId>

# Enviar email
gog gmail send --to destinatario@gmail.com --subject "Asunto" --body "Mensaje"

# Responder a un hilo
gog gmail send --to a@b.com --subject "Re: algo" --body "respuesta" --reply-to-message-id <msgId>
```

### Google Calendar — Comandos principales
```bash
# Ver próximos eventos (calendario principal)
gog calendar events primary --max 10

# Ver eventos en rango de fechas
gog calendar events primary --from 2026-02-24 --to 2026-03-01

# Crear evento
gog calendar create primary --summary "Reunión" --from 2026-02-27T14:00:00 --to 2026-02-27T15:00:00

# Listar todos los calendarios
gog calendar calendars

# Calendarios disponibles: primary, Familia, NBA, Timberwolves Schedule
```

### Notas
- El comando `gog` está en `/home/enderj/.local/bin/gog` (wrapper)
- No necesita `--account` ni variables de entorno (ya están configuradas)
- Para JSON usa `--json`, para texto plano usa `--plain`

## Calendario y Recordatorios

### Eventos Regulares
- [Por configurar]

### Días Festivos (Denver)
- [Por agregar al calendario]

## Contactos de Emergencia

### Auto-diagnóstico
- **Comando**: `openclaw status`
- **Logs**: ~/.openclaw/logs/

### Reinicio de Servicios
- **Gateway**: `openclaw gateway restart`
- **Sistema**: `systemctl restart openclaw` (si instalado)

## Changelog de Configuración

### Actualizaciones Recientes
- **2026-02-12**: Documento de optimización aplicado
- **2026-02-12**: SOUL.md actualizado con personalidad expandida
- **2026-02-12**: IDENTITY.md actualizado con información completa
- **2026-02-12**: HEARTBEAT.md implementado con tareas proactivas
- **2026-02-12**: TOOLS.md creado con configuración detallada

### Próximas Actualizaciones
- [Agregar según necesidad]

## MiniMax TTS — Text-to-Speech (2026-03-30)

### Plan Standard ($30/mes)
- **300,000 credits/mes**
- **RPM**: 50 requests/min
- **Characters/request**: hasta 10,000
- **Script**: `~/.openclaw/workspace/skills/minimax-tts/tts.py`

### Voz BitTrader — DEFAULT para TODOS los videos (orden Ender 2026-03-30)
- **Voice ID**: `Spanish_ThoughtfulMan` — Spanish thoughtful male voice
- Configurada en `producer.py` TTS_VOICE (primary)
- Fallback: Edge-TTS `es-MX-JorgeNeural`

### Modelos
| Modelo | Descripción |
|--------|-------------|
| `speech-2.8-hd` | ⭐ Latest HD — default |
| `speech-2.8-turbo` | Latest Turbo — rápido |
| `speech-2.6-hd` | HD con prosodia |
| `speech-02-hd` | Estabilidad superior |

### Voces preset
| ID | Descripción |
|----|-------------|
| `male_1` | Masculino claro (default) |
| `male_2` | Masculino alternativo |
| `female_1` | Femenino claro |
| `female_2` | Femenino alternativo |
| `Spanish_ThoughtfulMan` | ⭐ Voz default BitTrader (2026-03-30) |

**40 idiomas**: Español, Inglés, Português, 中文, etc.

### Uso
```bash
python3 skills/minimax-tts/tts.py "Hola Ender" /tmp/audio.mp3
python3 skills/minimax-tts/tts.py --voice female_1 "Hola" /tmp/out.mp3
python3 skills/minimax-tts/tts.py --speed 0.9 --format wav "Hola" /tmp/out.wav
```

---

## VAPI — Asistente de Voz Telefónico

### Credenciales
- **Private Key**: f361bb66-8274-403a-8c0c-b984d7dd1cee
- **Public Key (Phone ID)**: 64fcd5de-ab68-4ae0-93f6-846ce1209cce
- **Assistant ID (Eko)**: 225a9f9f-5d58-412a-b8df-81b72c799a4a
- **Número de teléfono**: +1 (720) 824-9313

### Configuración
- **Voz**: Fernando (ElevenLabs dlGxemPxFMTY7iXagmOj)
- **Modelo**: Claude Sonnet 4.6
- **STT**: Deepgram Nova-3 (español)
- **Estado**: ✅ ACTIVO

### Uso
- Llamadas entrantes: Cualquiera puede llamar al +17208249313
- Llamadas salientes: Via API con curl o skill de OpenClaw
