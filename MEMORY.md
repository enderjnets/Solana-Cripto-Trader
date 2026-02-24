# Memoria Persistente de Eko

## Hechos Clave sobre @enderj
- Usuario avanzado de tecnología (Linux Ubuntu 25.10)
- YouTube: @bittrader9259 - Trading de criptomonedas/NAS100
- Broker: Alpari (cuenta demo: 52786589)
- Sistema híbrido (no confirmado si aún tiene macOS)

## Proyectos Activos
- **Simple-NAS100-TradingBot** - Bot de trading para NAS100 (v1.6, FTMO_v5)
- Cripto trading bots
- Automatización de sistemas
- Integración IA local

## Configuración Técnica
- **GitHub**: enderjnets
- **MT5**: Instalado en `/home/enderj/.mt5/`
- **Python**: 3.13 con entorno virtual `~/mt5_env/`
- **Wine**: 10.0 para ejecutar MT5 en Linux
- **Xvfb**: Display virtual para headless operation
- **Automatización**: Scripts en `~/mt5_automation/`

## Repositorios
- ✅ Coinbase-Cripto-Trader-Claude
- ✅ Simple-NAS100-TradingBot (PRINCIPAL)
- ❌ Button-EE3 (BORRADO)

## Preferencias de Trabajo
- Prefiere eficiencia sobre complicación
- Documenta cambios importantes
- Mantiene backups de configuraciones
- Trabaja remotamente desde terminal

## Historial de Decisiones
- 2026-02-10: Configurado MT5 en Linux con Wine + Xvfb
- 2026-02-10: Creado sistema de automatización Python para MT5
- 2026-02-10: Bot SimpleNY200_v1.6 compilado exitosamente (.ex5 creado)
- 2026-02-10: Datos históricos NAS100 descargados en MT5
- 2026-02-10: Iniciado desarrollo de Nueva Estrategia v2.0 (TrendMomentum)
- 2026-02-10: **Wine abandonado para automatización** (too slow/unstable)
- 2026-02-10: Creado backtester nativo Python (funciona sin Wine)
- 2026-02-10: **FTMO Strategy OPTIMIZADA y PASSABLE** 🎉

## Estado Actual
- 🤖 Bot v1.6: ✅ Compilado
- 📊 MT5: ✅ Corriendo con datos NAS100
- 🔧 Scripts automatización: ✅ Native Python backtester listo
- 🆕 Nueva estrategia v2.0: ✅ OPTIMIZADA (RR 3.0, Risk 0.5%)
- 📈 FTMO Challenge: **PASSABLE** - 59.8% WR, 1.49 PF, 2.96% DD
- 💻 VirtualBox: ⚠️ **BLOQUEADO por Secure Boot**

## Progreso VM Windows (2026-02-12)
- ✅ VirtualBox 7.2.2 instalado
- ✅ VM "Windows 11" ya creada
- ✅ ISO encontrado: `Win11_25H2_English_x64.iso` (7.7 GB)
- ❌ Módulo vboxdrv bloqueado por Secure Boot

## Problema Secure Boot
- Secure Boot está habilitado
- mokutil disponible pero sin clave generada
- Solución pendiente: Firmar módulo o deshabilitar Secure Boot

## Tareas Pendientes
1. ~~Validar estrategia con datos reales MT5 (HCC format issue)~~
2. ~~Ajustar backtester para leer formato HCC nativo~~ (Archivo LFS, no datos reales)
3. ~~Compilar FTMO_v1.0.mq5 en MetaEditor (requiere usuario manual)~~ (Wine inestable)
4. ~~Configurar VirtualBox~~ - Resuelto (Secure Boot deshabilitado)
5. ~~Instalar Windows 11~~ - VM operativa
6. ~~Configurar MT5 nativo~~ - Instalado en VM Windows
7. **Compilar FTMO_v5.mq5** - Copiado a VM, pendiente compilación

## Investigación MQL5 (2026-02-15)
- **Sintaxis corregida basada en investigación del usuario:**
  - `iRSI(symbol, tf, period, 0)` - 4 params, 0=PRICE_CLOSE
  - `iMA(symbol, tf, period, 0, 1, 0)` - 6 params, 1=MODE_EMA  
  - `iATR(symbol, tf, period)` - 3 params sin shift
- **Archivo creado:** FTMO_v5.mq5 (sintaxis corregida)

## Investigación Actual (2026-02-11)
- **Wine en Linux:** ❌ ABANDONADO - Too slow, timeouts, inestable
- **Solución propuesta:** Windows Virtualizado con VirtualBox
- **Documento:** `RESEARCH/MT5_Windows_VM_Automation_Plan.md`
- **Costo estimado:** $15-200 (licencia Windows)
- **Tiempo implementación:** 6-8 horas

## Plan de Automatización VM Windows
1. Instalar VirtualBox
2. Crear VM Windows 10/11
3. Instalar MT5 nativo
4. Configurar API REST + SSH
5. Automatización completa desde Linux

## Veredicto
> Wine no funciona para MT5. VM Windows es la solución confiable.

## Mejoras de Sistema OpenClaw (2026-02-12)

### Documento de Optimización Recibido
- **Archivo**: RESEARCH/OpenClaw_Optimization_Guide.rtf
- **Título**: "Análisis Arquitectónico Integral y Marco de Optimización Avanzada"
- **Fecha**: Febrero 2026
- **Contenido**: Guía detallada para optimizar OpenClaw

### Cambios Implementados

#### SOUL.md - Personalidad Expandida
- ✅ Principios operativos más detallados (Proactividad, Precisión, Privacidad, Transparencia)
- ✅ Marco ético formalizado
- ✅ Estilo de comunicación estructurado
- ✅ Personalidad expandida con características específicas
- ✅ Comportamiento en situaciones específicas codificado
- ✅ Configuración avanzada de proactividad y reportes
- ✅ Protocolo de evolución y aprendizaje

#### IDENTITY.md - Configuración Completa
- ✅ Información del agente detallada (nombre, versión, emoji)
- ✅ Arquitectura de directorios documentada
- ✅ Estado de componentes del sistema
- ✅ Herramientas del sistema clasificadas
- ✅ Preferencias de desarrollo y trabajo
- ✅ Proyectos activos con estado actual
- ✅ Configuración de canales de comunicación
- ✅ APIs y servicios externos
- ✅ Métricas de rendimiento
- ✅ Configuración de seguridad

#### HEARTBEAT.md - Sistema Proactivo
- ✅ Frecuencia de chequeos definida (30min-12h)
- ✅ Horario de actividad configurado
- ✅ 5 categorías de tareas proactivas:
  1. Sistema y Recursos
  2. Trading y Datos Financieros
  3. Comunicaciones y Notificaciones
  4. Calendario y Agenda
  5. Salud del Sistema OpenClaw
- ✅ Protocolo de alertas con 4 niveles de severidad
- ✅ Tareas automáticas vs. requiere aprobación
- ✅ Heartbeat state tracking en JSON

#### TOOLS.md - Configuración Local
- ✅ Hardware y recursos documentados
- ✅ Herramientas de desarrollo (Python, Node, Git, Docker)
- ✅ Configuración de red y APIs
- ✅ Herramientas de trading (MT5, scripts)
- ✅ Estado de Virtualización (bloqueado por Secure Boot)
- ✅ Preferencias de terminal y aliases
- ✅ Integraciones de mensajería
- ✅ Notas de seguridad
- ✅ Accesos directos y comandos útiles

#### USER.md - Perfil de Usuario
- ✅ Información personal estructurada
- ✅ Contexto de proyectos principales
- ✅ Preferencias de comunicación
- ✅ Configuración técnica conocida
- ✅ Notas de interacción
- ✅ Patrones y mejores prácticas

### Impacto de las Mejoras
- **Eficiencia**: Heartbeats estructurados reducen llamadas API
- **Personalidad**: SOUL.md más completo mejora consistencia de respuestas
- **Memoria**: USER.md mejora comprensión del contexto
- **Automatización**: HEARTBEAT.md habilita tareas proactivas
- **Mantenimiento**: TOOLS.md centraliza configuración local

### Métricas de Rendimiento Esperadas
- Tiempo de respuesta: Mantener 60-100 tps
- Tasa de sugerencias aceptadas: Objetivo >80%
- Alertas críticas: Respuesta inmediata
- Tareas automatizadas: Incremento significativo

### Próximos Pasos de Optimización
1. ✅ Archivos base mejorados
2. 🔄 Implementar monitoring de métricas
3. ⏳ Ajustar based on feedback del usuario
4. ⏳ Expandir capacidades según necesidad

> Nota: Las mejoras siguen el marco arquitectónico de OpenClaw v2.0 con integración Grok y visualización de Agent Teams.

## Sistema de Voz Bidireccional (2026-02-12)

### Whisper - Speech to Text
- **Instalado**: Entorno `~/.openclaw/voice_env/`
- **Modelo**: small (461MB)
- **Estado**: ✅ Funcionando
- **Primera prueba**: Transcripción exitosa de nota de voz en español

### gTTS - Text to Speech (Gratuito)
- **Instalado**: `gTTS 2.5.4`
- **Voz**: Español automática (gratis)
- **Estado**: ✅ Primera nota de voz enviada al usuario
- **Costo**: $0

### Intentos Previamente
- **Piper TTS**: Modelos españoles no disponibles, problemas de configuración
- **Coqui TTS**: Incompatible con Python 3.13
- **ElevenLabs Nova**: Sin API key (pendiente configurar)

### Flujo Funcional
```
Nota de voz → Whisper (transcribe) → Eko procesa → gTTS (audio) → Usuario
```

### Archivos del Sistema de Voz
- `~/.openclaw/workspace/voice_system.py` - Motor Whisper
- `~/.openclaw/workspace/telegram_voice_handler.py` - Handler Telegram
- `~/.openclaw/workspace/setup_elevenlabs.sh` - Configuración API ElevenLabs

## Actualización Sistema de Voz (2026-02-13)

### Python 3.11 con SSL
- ✅ `libssl-dev` instalado con sudo
- ✅ Python 3.11 recompilado desde fuentes (`~/.local/python311/`)
- ✅ Soporte SSL habilitado

### Chatterbox TTS
- ✅ **Instalado exitosamente** con Python 3.11
- Entorno: `~/.openclaw/chatterbox_env/`
- Paquetes clave:
  - chatterbox-tts 0.1.6
  - torch 2.6.0
  - torchaudio 2.6.0
  - transformers 4.46.3

### Voz Catalina
- Usuario quiere clonar voz "Catalina"
- Sin sample de audio disponible temporalmente
- gTTS como fallback

### SSH
- Clave ED25519 generada (`~/.ssh/id_ed25519`)
- Pendiente agregar al servidor remoto

### Red
- IP principal: 10.0.0.240
- IP VirtualBox: 192.168.122.1

## Sistema de Voz - Actualización 13 Feb 2026

### ElevenLabs Configurado
- API key recibida del usuario
- Guardada en `~/.config/elevenlabs.json`
- Voces disponibles en cuenta: Solo inglés (Roger, Sarah, Laura, Charlie, George, etc.)
- **Sin voces españolas** - usuario necesita agregar desde elevenlabs.io/voice-library

### Preferencia de Voz Establecida
- Usuario envía notas de voz en español → responder en español
- Usuario envía notas de voz en inglés → responder en inglés
- Usuario requiere **acento español** - gTTS no lo tiene (suena robótico/gringo)
- Solución: Agregar voces españolas a cuenta ElevenLabs

## Problema Ekobit Telegram (2026-02-14)

### Síntoma Reportado
- @EkoBit_Bot no responde a mensajes

### Investigación Realizada
1. Gateway encontrado activo pero sin procesar mensajes
2. Logs mostraban: "Removed orphaned user message to prevent consecutive user turns"
3. Offset obsoleto identificado en `~/.openclaw/telegram/update-offset-default.json`

### Acciones Tomadas
1. Reiniciado gateway múltiples veces (último PID: 123863)
2. Reseteado offset a 0: `{"version":1,"lastUpdateId":0}`
3. Gateway confirmado activo en ASUS dorada (10.0.0.56)

### Estado Actual
- Gateway funcionando pero sin respuestas confirmadas
- Offset reseteado
- Esperando mensaje nuevo del usuario para probar

## Skill Taxes Denver (Febrero 2026)
- Creada skill `taxes_denver/` basada en investigación Gemini
- Archivos: SKILL.md, REFERENCE.md, QUICK_REF.md
- Contiene info sobre: LLC, 1099-K/NEC, deducciones Uber, Ley OBBBA 2025
- Dirigida a conductors de Uber/Lyft en Denver, Colorado

## Solana Jupiter Bot (Feb 2026)
- **Problema:** Precios erróneos en BTC/ETH (Jupiter API daba $424 en vez de $67k)
- **Solución:** Modificado para usar CoinGecko API para precios reales
- **Bugfix:** Ciclo infinito en master_orchestrator.py (bloque try fuera del while)
- **Estado:** Bot corriendo con watchdog auto-reinicio, ~80 estrategias por ciclo

## Investigación Topstep (Feb 2026)
- **Resultado:** NO tiene API pública para automatización
- Broker de futuros manual (Trading Combine → Funded Account)
- Alternativas con API: MT5, Tradovate, CQG, Interactive Brokers
