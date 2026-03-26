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

## BitTrader Thumbnails — Estándares y Lecciones Aprendidas (2026-03-25)

### 🔴 Reglas de Calidad — NO NEGOCIABLES

#### Toda thumbnail DEBE tener:
1. **Persona real con expresión dramática** — sin cara = fallback, fallback = subóptimo
2. **Texto grande con borde negro 12px** (MrBeast style)
3. **Logo BitTrader top-left**
4. **@bittrader9259 top-right**
5. **Brightness > 30** (validación automática)
6. **Ratio 16:9 / 1280×720** (NUNCA estirar)

#### ❌ Errores que ya ocurrieron y NO deben repetirse:
- Videos subidos sin thumbnail custom → YouTube pone frame azul/negro genérico
- Thumbnail generada pero no subida/aplicada al video
- Fallback activado sin intentar Hugging Face SDXL primero
- Pipeline marcando video como `success` sin verificar que la thumbnail quedó en YouTube

### 🔧 Protocolo de Verificación Post-Upload (OBLIGATORIO)

Después de subir cualquier video a YouTube, el pipeline DEBE:
1. Verificar via API que el video tiene `maxresdefault` thumbnail (no la auto-generada)
2. Si falta thumbnail → generar con SDXL (HF API) + subir inmediatamente
3. Loguear resultado: `thumbnail_uploaded: true/false` en upload_queue.json
4. Si la thumbnail falla 3 veces → alerta a Ender

### 🎨 Orden de intentos para generar personas (SDXL):
1. `stabilityai/stable-diffusion-xl-base-1.0` (calidad alta)
2. `black-forest-labs/FLUX.1-schnell` (rápido)
3. `runwayml/stable-diffusion-v1-5` (backup)
4. **Solo si los 3 fallan** → usar fallback con fondo sólido + gradiente (y alertar)

### ✅ Thumbnail aprobada por Ender (25 marzo 2026)
- Video: "Cómo empezar en cripto con $100" (`QrYrom05HzQ`)
- Generada con SDXL: hombre latino sonriendo con billetes, monedas de oro de fondo
- Brightness: 82.6 | Tamaño: 1280×720 | Aprobada y subida ✅
- Script usado: `/tmp/gen_thumb_100.py` (referencia para futuros thumbnails)

### 📋 Prompt base para thumbnails de cripto (funciona bien con SDXL):
```
excited young latin man holding $100 bill, looking at camera with big smile,
dramatic expression, cryptocurrency charts in background, green and gold colors,
professional studio lighting, 4K cinematic, high contrast, sharp focus
```
Adaptar: cambiar objeto ($100 → phone, laptop), colores según tema, expresión según contenido.

---

## BitTrader Thumbnails — Método Oficial (2026-03-12)

### Script principal
`/home/enderj/.openclaw/workspace/bittrader/agents/thumbnail_agent_huggingface.py`

### API
- **Hugging Face SDXL** (Stable Diffusion XL Base 1.0)
- **API Key**: `memory/huggingface_config.json`
- **Endpoint**: `https://router.huggingface.co/hf-inference/models/`

### Workflow
1. Generar imagen 1024×1024 con SDXL (personas + expresiones dramáticas)
2. **Crop centrado** a 16:9 (NUNCA estirar)
3. Resize a 1280×720
4. Añadir branding: logo BitTrader + texto MrBeast style + @bittrader9259

### Estilo obligatorio
- Personas reales con expresiones dramáticas
- Texto con borde negro 12px (MrBeast style)
- Texto posicionado abajo
- Logo top-left, handle top-right
- Gradiente oscuro para contraste

### Config
`/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnail_config.json`

### Thumbnails Faltantes - 14 Marzo 2026
- 6 thumbnails generados con método local fallback
- 1 video negro (brightness 12.5) - ya borrado
- Scripts: `generate_thumbnails_local.py`, `generate_missing_thumbnails.py`
- Colores sólidos con gradiente + branding BitTrader


## BitTrader Video Production - FIX COMPLETADO (2026-03-14)

### Problemas Resueltos

#### 1. Videos Negros Producidos
**Problema**: Videos marcados como `success` con visual 100% negro (RGB: 13,13,10)
**Causa**: `producer.py` no validaba calidad visual antes de marcar success
**Solución**:
- `validate_video_quality()` función que extrae 5 frames y verifica brightness > 30
- Validación al FINAL de `produce_single()` (aplica a TODOS los caminos)
- Colores más brillantes en `make_fallback_video()` (#0a0a0a → #1e3a5f)
- `_retry_fallback_brighter()` para retry con fondos muy brillantes (#FFD700, #FFA500)

#### 2. Queue Processor - Subida Automática
**Problema**: Videos no se subían según `scheduled_date`
**Solución**:
- `queue_processor.py` - lee `upload_queue.json` cada 30 min
- Filtra videos con `scheduled_date <= AHORA`
- Sube automáticamente a YouTube
- Actualiza status a `uploaded`
- Maneja quota y reintentos

#### 3. Thumbnails Faltantes
**Problema**: Varios videos sin thumbnail para subir
**Solución**:
- `generate_thumbnails_local.py` - genera thumbnails sin API (colores sólidos)
- 6 thumbnails generados para videos producidos hoy
- Branding: logo + @bittrader9259 + título estilo MrBeast

### Archivos Modificados/Creados

**Producer Fixes**:
- `producer.py` - Added `validate_video_quality()` at END of `produce_single()`
- DEBUG logs added at key execution points

**Queue System**:
- `queue_processor.py` - New file for automatic uploads
- `publisher.py` - Modified to check `scheduled_date` field
- `upload_queue.json` - Updated with scheduled dates

**Thumbnail Scripts**:
- `generate_thumbnails_new.py` - Hugging Face SDXL API generator
- `generate_thumbnails_local.py` - Local fallback (no API needed)
- `fix_horizontal_thumbnails.py` - Rhino series fix
- `fix_all_rhino_thumbnails.py` - Bulk rhino thumbnails

### Resultados

| Métrica | Valor |
|----------|--------|
| Videos producidos hoy | 7 |
| Videos OK (brightness > 30) | 6 |
| Videos negros | 1 (borrado de YouTube) |
| Brightness promedio | 62.4 |
| Threshold | > 30 |
| Thumbnails generados | 6 |
| Videos programados | 4 |

### Videos Programados (Auto-Upload)

| Fecha | Hora MT | Video |
|-------|----------|--------|
| Mar 14 | 11:00 AM | NEIRO explota |
| Mar 14 | 5:00 PM | SUI y DOT caerán |
| Mar 14 | 11:00 PM | ¿Qué es BTC |
| Mar 15 | 11:00 AM | TRUMP Coin |

### GitHub Commits

```
bea50c0 - Add: Local thumbnail generator for missing thumbnails
44f6990 - Add: Thumbnail generator script for new videos
db7558b - Fix: Added final validate_video_quality() call in produce_single()
3f8edc5 - Fix: DEBUG logs indentation corrected
153cf8a - Fix: Added final validate_video_quality() call in produce_single()
79e7ae0 - Fix: Video quality validation in producer, Eco business plan added
```

### Sistema Actual

| Componente | Status | Descripción |
|-------------|--------|-------------|
| Producer.py | ✅ FIXED | Validación brightness > 30 al final |
| Queue Processor | ✅ ACTIVE | Cron cada 30 min |
| Upload automático | ✅ SCHEDULED | 4 videos en queue |
| Thumbnails | ✅ COMPLETE | 6 videos con thumbnail |
| GitHub | ✅ SYNCED | Todos los commits pushed |

**Estado**: Sistema 100% funcional 🚀


## Qwen 2.5 14B — Integración Local (2026-03-08)


## Qwen 2.5 14B — Integración Local (2026-03-08)

### Setup
- **Modelo**: qwen2.5:14b via Ollama (servicio systemd, auto-start)
- **GPU**: RTX 3070 Laptop (8GB VRAM, usa 7.3GB)
- **Provider OpenClaw**: `local` (NO `ollama` — OpenClaw lo detecta y fuerza API nativa)
- **API**: `openai-completions` con `/v1`, `injectNumCtxForOpenAICompat: false`
- **Alias**: `/model Qwen Local` → `local/qwen2.5:14b`

### Qué hace Qwen
- `qwen_client.py`: ask, ask_json, analyze_news, summarize, generate_title_variations, draft_script_section, check_video_status
- `creator.py` fallback #4 (Claude → GLM-4.7 → MiniMax → Qwen)
- Chat directo desde Telegram con `/model Qwen Local`

### Qué NO puede hacer Qwen
- **Tool calling en sub-agentes** — genera texto en vez de llamar tools
- Los crons que necesitan exec/bash van con **Haiku 4.5**

### Lecciones clave
1. Provider "ollama" + puerto 11434 = OpenClaw fuerza API nativa → renombrar a "local"
2. API nativa + contextWindow 32K = error RAM (10.3GB) → usar openai-completions
3. contextWindow < 16K = OpenClaw rechaza (mínimo 16K) → poner 32K + injectNumCtx=false
4. Whisper comparte GPU → forzar CPU con `CUDA_VISIBLE_DEVICES=""`

### Distribución de modelos
| Tarea | Modelo |
|-------|--------|
| Chat directo | Sonnet 4.6 |
| Crons (tool calling) | Haiku 4.5 |
| qwen_client.py | Qwen Local |
| Pipeline creativo | Sonnet 4.6 |
| creator.py fallback | Qwen Local |

## VAPI — Asistente de Voz Telefónico (2026-03-08)

### Configuración activa
- **Número**: +1 (720) 824-9313
- **Assistant ID**: 225a9f9f-5d58-412a-b8df-81b72c799a4a
- **Phone Number ID**: 64fcd5de-ab68-4ae0-93f6-846ce1209cce
- **Private Key**: f361bb66-8274-403a-8c0c-b984d7dd1cee
- **Voz**: es-VE-SebastianNeural (Azure, gratis) — venezolano
- **Modelo**: Claude Sonnet 4.6
- **STT**: Deepgram Nova-3 (español)
- **Costo**: ~$0.01 por llamada corta, ~$0.05 por 2 min

### Capacidades
- Llamadas entrantes: cualquiera llama al (720) 824-9313
- Llamadas salientes: vía API (puede haber rechazo de carrier en números nuevos)
- Voces Azure disponibles en español: es-VE-SebastianNeural, es-MX-JorgeNeural, es-MX-DaliaNeural, es-CO-GonzaloNeural, es-AR-TomasNeural, es-ES-AlvaroNeural

### Para hacer una llamada saliente
```bash
curl -X POST https://api.vapi.ai/call \
  -H "Authorization: Bearer f361bb66-8274-403a-8c0c-b984d7dd1cee" \
  -H "Content-Type: application/json" \
  -d '{"phoneNumberId":"64fcd5de-ab68-4ae0-93f6-846ce1209cce","assistantId":"225a9f9f-5d58-412a-b8df-81b72c799a4a","customer":{"number":"+1XXXXXXXXXX","name":"Nombre"}}'
```

### Notas
- ElevenLabs sin créditos → usar Azure (gratis, buena calidad)
- Llamadas salientes pueden ser rechazadas por carrier (número nuevo = posible spam detection)
- Llamadas entrantes funcionan perfectamente

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

## Plan de Negocio — Automatización con IA (9 marzo 2026)

### Decisión
- **NO crear marca nueva** — expandir BitTrader para incluir automatización con IA y servicios
- Repositorio del enfoque: trading + automatización con IA para cualquier negocio

### Plan de 3 Fases
1. **30 días**: 4-6 videos en BitTrader mostrando proyectos reales (Solana bot, VAPI, pipeline YouTube)
2. **Paralelo**: Fiverr + Upwork con 3 servicios: agentes IA, pipelines de contenido, bots de negocio
3. **60-90 días**: Evaluar canal vs servicios, doblar en lo que funcione

### Próximo Paso Pendiente
- Ender debe elegir tema del primer video (Solana bot, VAPI, o pipeline YouTube)

## Eco — Empresa de Automatización con IA (12 marzo 2026)

### Concepto
- **Nombre**: Eco (ECO)
- **Enfoque**: Servicios de automatización con IA para negocios
- **Target**: Pequeñas y medianas empresas que necesitan automatizar procesos

### Primera Clienta Potencial
- **Nombre**: Maggie Quintero (esposa de Ender)
- **Negocio**: I Capelli Salon (nail/hair salon, Denver)
- **Servicio**: Automatización de gestión del salón (citas, inventario, marketing, etc.)
- **Estado**: Pendiente confirmación (puede decir sí o no)

### Estrategia de Clientes
1. **Fase 1**: Maggie (I Capelli) — proyecto piloto para validar modelo
2. **Fase 2**: Identificar 3-5 clientes similares (salones, spas, pequeños negocios locales)
3. **Fase 3**: Expandir a otros nichos (restaurantes, retail, servicios profesionales)

### Tareas Pendientes
- [ ] Definir servicios específicos para I Capelli Salon
- [ ] Crear propuesta de valor y pricing
- [ ] Desarrollar plan de adquisición de clientes
- [ ] Crear materiales de marketing (landing page, deck de ventas)
- [ ] Establecer presencia en plataformas (Fiverr, Upwork)

### Próximos Pasos (Solicitados por Ender)
- CEO Agent + Programmer Agent deben crear plan de acción para conseguir más clientes
- Enfoque: ¿Qué hacer si Maggie dice que sí? ¿Qué hacer si dice que no?
- **NOVO**: Crear Agente de Marketing para Eco
  - Trabajará con CEO Agent y Programmer Agent
  - Enfoque: Marketing y estrategia de crecimiento
  - Pendiente: investigación sobre expertos en marketing (Ender proporcionará)

### Agents de OpenClaw - Estructura Propuesta
- **CEO Agent**: Coordina todo (BitTrader + Eco)
- **Programmer Agent**: Desarrollo técnico
- **Marketing Agent** (nuevo): Estrategia de marketing, lead generation, branding
  - **Base cognitiva**: ChatGPT (razonamiento profundo) + Gemini (multimodalidad, contexto largo)
  - **Investigación**: `/home/enderj/.openclaw/workspace/research/marketing_ai_agent_2026.md`
  - **Enfoque**: Agente autónomo que razona en bucles, optimiza en tiempo real, hiper-personalización dinámica
- **Otros agentes**: Especializados según necesidad

### Seguro Comercial - Shin (Agente de Seguros)
- **Nombre**: Shin
- **Teléfono**: (719) 633-4600
- **Referencia**: Nodar (amigo de Ender, conduce Uber Black, tiene Kia EV9 negro)
- **Empresa de Ender**: Black Volt Mobility LLC
- **Email contacto**: blackvoltmobility@gmail.com
- **Vehículo a asegurar**:
  - Marca/Modelo: 2024 Kia EV9 (negro)
  - VIN: KNDAA5S2XR6023972
  - Placas: FIG-P37 (Colorado)
- **Cobertura requerida**:
  - $500,000 Combined Single Limit (CSL) auto liability
  - Póliza bajo: Black Volt Mobility LLC
  - Certificate of Insurance (COI) con City and County of Denver como Certificate Holder
  - Requisito: A.M. Best rating A-VIII o superior
  - Uso: Rideshare (Uber Black/Uber LUX) + clientes privados
- **Nota**: Solo para EV9 de Ender (no incluye negocio de Maggie)
- **Estado**: Pendiente llamada para cotización
- **Fecha solicitud**: 12 marzo 2026

## Reglas de Comunicación con Ender (9 marzo 2026)
- **Nota de voz → responder con nota de voz** (mismo idioma)
- **Texto → responder con texto** (mismo idioma)
- **NO enviar mensajes intermedios** ("transcribiendo...", "procesando...")
- Responder directo con el resultado final

## Reglas de Comportamiento (9 marzo 2026 — PERMANENTES)

### 🔇 No enviar mensajes de "procesando"
- NUNCA enviar "Solo proceso el nuevo:", "Un momento...", "Procesando..."
- Directamente actuar y responder sin anunciar lo que se hace

### 🎙️ Notas de voz
- Cuando Ender envía audio → siempre responder con TTS (voz), luego NO_REPLY

### 📞 VAPI — Configuración Final
- Asistente bilingüe: `b77215b6-64c2-4ad2-84bc-597ddb8a8bc3`
- Voz español: `es-VE-SebastianNeural` (igual que TTS de Telegram)
- Voz inglés: `en-US-AndrewNeural` (elegida por Ender)
- Saludo bilingüe configurado, STT multilingual Deepgram
- Monitor VAPI: silencio total salvo llamada real >10s con mensaje

### 📧 Firma estándar correos inglés
```
Eko
AI Assistant to Ing. Ender Ocando
📞 Eko: +1 (720) 824-9313
📞 Ender: +1 (720) 512-1753
```

## Clientes Personales Uber de Ender

### Dirección Base de Ender
- **Dirección:** 6000 S. Fraser St., Aurora, CO 80016
- **Uso:** Siempre calcular tiempo de viaje desde esta dirección hasta el pickup de clientes
- **Buffer mínimo:** 10 minutos + ajuste por tráfico según hora/día

### Protocolo para clientes Uber personales
1. Guardar nombre + teléfono + dirección de pickup + tarifa en MEMORY.md
2. Calcular tiempo de viaje desde 6000 S. Fraser St., Aurora, CO 80016 hasta dirección del cliente
3. Ajustar buffer según tráfico estimado (hora pico, día de semana, etc.)
4. Crear evento en Google Calendar con Margie invitada (margie240478@gmail.com)
5. Enviar correo a Margie con todos los detalles del pickup
6. Poner 2 recordatorios cron: noche anterior (9 PM) y mañana del día (hora -1.5h antes del pickup)

### Clientes Agendados

#### Bob
- Esposa: Susannah
- Pickup completado: **Jueves 12 Mar 7:30 AM** — 7373 Old Mill Trail, Boulder CO 80301 → Airport
- **RETORNO**: **Sábado 14 Mar 6:20 PM** — Airport → 7373 Old Mill Trail, Boulder CO 80301 (ACTUALIZADO)
- **Vuelo de Susannah**: Aterriza **5:54 PM** sábado (CAMBIADO de 7:40 PM)
- **Listo para pickup**: ~6:20 PM
- **Uber Pick Up Location**: **Level 5, West Side of Jeppesen Terminal** (DEN)
- **Tiempo estimado desde Aurora**: ~45-55 min + buffer 15-20 min = Salida **5:25 PM**
- **Cron recordatorio**: Vie 13 Mar 9:00 PM (noche anterior) + Sáb 14 Mar 5:00 PM (55 min antes)
- **Evento calendario**: https://www.google.com/calendar/event?eid=NXZib2dyZGI1M2htMDNzcXU2MW9zYWJrZDggZW5kZXJqbmV0c0Bt (ACTUALIZADO)
- **Nota**: Susannah cambió su vuelo a uno más temprano (5:54 PM arrival)

#### Dustin Finer
- Teléfono: (310) 999-4652
- Pickup: **Viernes 13 Mar 9:15 AM** — 2093 Tamarack Avenue, Boulder CO 80304 → Airport
- 4 pasajeros (él + 3 más)
- Tarifa: $110 ride + $13 toll = **$123 total**
- **Tiempo estimado desde Aurora:** ~50-60 min + buffer 30-35 min = Salida **7:50 AM** (buffer incrementado 20 min extra)
- **Cron recordatorio:** Jue 12 Mar 9:00 PM (noche anterior) + Vie 13 Mar 7:45 AM (mañana)
- **Evento calendario:** https://www.google.com/calendar/event?eid=dWNydTcwM2k3Z2FraXJjdHF2MWI1YmIwOWMgZW5kZXJqbmV0c0Bt
- **Nota**: Buffer aumentado a 30-35 min después de tráfico inesperado el 12 mar (Susannah pickup llegó 6 min tarde)

## Reglas para Revisar Correos
- **Siempre leer hilos completos**: Si un correo muestra `[2 msgs]`, `[3 msgs]`, etc., usar `gog-real gmail thread <threadId>` para leer TODOS los mensajes del hilo antes de reportar. No asumir que el mensaje visible es el único.

## VAPI Call Blocker — Sistema de Seguridad (9 marzo 2026)

### Bloqueados
- 🚫 **+17205121753** — Bitcoin/Crypto automated spammer script

### Whitelist
- ✅ **+16159751056** — Yonathan Luzardo (Yona) — Amigo, pruebas de seguridad
- ✅ **+17208387940** — Maggie (Margie Quintero) — Esposa de Ender
  - Email: margie240478@gmail.com
  - Trabaja en I Capelli Salon (nail/hair salon, Denver)
- ✅ **+17208249313** — Eko (VAPI itself)

### Sistema de Bloqueo
- **Cron**: Cada 15 minutos ejecuta `vapi_call_blocker.py monitor`
- **Acción**: Compara llamadas recientes contra blocklist
- **Registro**: Todas las llamadas bloqueadas se guardan en `VAPI_BLOCKED_CALLS.log`
- **Reporte**: Solo alerta si detecta nuevos spammers

## Preferencias de Comunicación
- **Yonathan (Cloky)**: Email a través de Gmail (`yonayonalife@gmail.com`)
  - Usar comando: `gog gmail send --to yonayonalife@gmail.com --subject "..." --body "..."`
  - Preferencia para documentación técnica y configuraciones
- **Cloky**: Alias de Yonathan (misma dirección de email)
- Mantiene backups de configuraciones
- Trabaja remotamente desde terminal
- **Yona**: Hermana de Yonathan (yona@email.com)

### Proyectos de Yonathan
- **Yona**: Usando OpenClaw con Claude Sonnet 4.6
  - Email: yona@email.com
  - Sistema: Linux (OpenClaw instalado)
  - Canal: Telegram principal
- **Cloky**: Asistente personal de Yona
  - Configurado con OpenClaw
  - Tiene acceso al correo de Yona
  - Puede leer/enviar emails, agregar eventos al calendario

### Problema Pendiente - STT (Speech-to-Text)
- **Fecha reportado**: 3 marzo 2026
- **Problema**: Yona envía notas de voz por Telegram (.ogg opus)
- **Síntoma**: Los audios llegan pero NO se transcriben (aparece `<media:audio>` sin texto)
- **IDs afectados**: #100, #102, #122, #156, #158
- **Causa probable**: Falta configurar proveedor STT en openclaw.json
- **Solución**: Configurar Whisper, Deepgram u otro proveedor STT
- **Referencia**: Whisper ya instalado en sistema del usuario (`~/.cache/whisper`)
- **Contexto**: Los archivos `.ogg` llegan correctamente a `/home/yonathan/.openclaw/media/inbound/`

## Historial de Decisiones
- 2026-02-10: Configurado MT5 en Linux con Wine + Xvfb
- 2026-02-10: Creado sistema de automatización Python para MT5
- 2026-02-10: Bot SimpleNY200_v1.6 compilado exitosamente (.ex5 creado)
- 2026-02-10: Datos históricos NAS100 descargados en MT5
- 2026-02-10: Iniciado desarrollo de Nueva Estrategia v2.0 (TrendMomentum)
- 2026-02-10: **Wine abandonado para automatización** (too slow/unstable)
- 2026-02-10: Creado backtester nativo Python (funciona sin Wine)
- 2026-02-10: **FTMO Strategy OPTIMIZADA y PASSABLE** 🎉
- 2026-03-07: **EMERGENCIA FINANCIERA RESUELTA** - Plan documentado en memory/2026-03-07.md
  - Top Step + Yonathan: Potencial $3,000/mes (Winrate 50%, RR 1:2)
  - Uber X: Meta $200/día extra (fines de semana)
  - Pagos 9 de marzo: EV9 ($901) + Seguro ($450)
  - Solana Bot: Paper trading - optimizaciones pendientes (confianza, position size, filtros)

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

### Proposal DecisionNext — Mike Neal
- [ ] 🌐 Hostear `proposals/decisionnext_proposal.html` en Netlify o GitHub Pages para tener URL limpia
- [ ] 📧 Enviar propuesta a Mike Neal (mneal@decisionnext.com) cuando responda el correo
- Cron activo monitoreando respuesta cada 30 min (job: 94cd4782)

### Pagos Pendientes
- [x] 📧 Agradecer a Metropolis por waiving fee (ENVIADO 19:07)
  - **Notice**: #889-104-512
  - **Placa**: FIGP37
  - **Estado**: Late fee de $30 WAIVED ✅
  - **Monto pendiente**: $1.75 + processing fee
  - **Link**: payments.metropolis.io
  - **Correo enviado**: 6 mar 2026, 19:07 (Message ID: 19cc5d642d174911)
  - **Código promocional**: HALFOFF (50% OFF en próxima visita)
- [ ] 🅿️ Pagar multa Metropolis ($1.75 + processing fee)

### Correos Pendientes
- [x] 📧 Responder Vince (V Shred) - Metabolic Assessment
  - **Respuesta**: NO, no interesado
  - **Estado**: Enviado 6 mar 2026
  - **Message ID**: 19cc5d9fc4c8161e
  - **Reenviado por usuario**: Si, revisar si hay algo importante

### Work Orders Condominio - Pendiente para 9PM hoy (27 Feb 2026)
- **Portal**: https://theorchardsatcherrycreekpark.loftliving.com/login
- **Email**: enderjnets@gmail.com
- **Password**: 43hRPt6aXu8RLVQ# (verificada por usuario)
- **Problema**: Login falló - posible CAPTCHA/2FA detectando automatización
- **Solución**: Login manual por usuario + luego automatizar work orders
- **6 Work Orders listas**:
  1. Microwave makes sparks (Appliance)
  2. Master bathroom toilet clogged (Plumbing)
  3. Master bathtub clogged (Plumbing)
  4. Stove/oven defective (Appliance)
  5. Towel rack/TP holder fell off (General)
  6. Air fryer trips breaker (Electrical)
- **Script**: `/home/enderj/.openclaw/workspace/skills/clawd-cursor/condo_login.js`
1. ~~Validar estrategia con datos reales MT5 (HCC format issue)~~
2. ~~Ajustar backtester para leer formato HCC nativo~~ (Archivo LFS, no datos reales)
3. ~~Compilar FTMO_v1.0.mq5 en MetaEditor (requiere usuario manual)~~ (Wine inestable)
4. ~~Configurar VirtualBox~~ - Resuelto (Secure Boot deshabilitado)
5. ~~Instalar Windows 11~~ - VM operativa
6. ~~Configurar MT5 nativo~~ - Instalado en VM Windows
7. **Compilar FTMO_v5.mq5** - Copiado a VM, pendiente compilación

## BitTrader — Scope del Canal (actualizado 2026-03-05)
- **NO es solo crypto** — es trading en TODOS los mercados
- **Mercados cubiertos**: Cripto (BTC, altcoins, DeFi), Futuros (NAS100, S&P500), CFDs, Fondeo (FTMO, etc.)
- **Proyectos activos de Ender**: Bots crypto, sistemas de futuros, CFDs, empresas de fondeo

## PILARES DEL CANAL BitTrader (2026-03-05)

### 1️⃣ INTELIGENCIA ARTIFICIAL + TRADING (Foco principal)
- Agentes de trading autónomos (no solo automatización básica)
- Claude Code, CloudCore, sistemas de agentes multi-LLM
- IA para análisis de mercado en tiempo real
- Noticias y eventos que afectan IA + trading
- Nuevas herramientas de IA para traders

### 2️⃣ CRIPTOMONEDAS (Bitcoin + Altcoins)
- Análisis de precio, tendencias, DeFi
- Staking, yield farming, nuevos proyectos
- Noticias del mercado crypto

### 3️⃣ FUTUROS (NASDAQ, S&P500, etc.)
- NAS100, S&P500, índices
- Análisis técnico y fundamental
- Estrategias de day trading

### 4️⃣ EMPRESAS DE FONDEO (FTMO, TopStep)
- Challenges, rules, funded accounts
- Estrategias para pasar pruebas
- Review de prop firms

### 5️⃣ EDUCACIÓN
- Gestión de riesgo, psicología
- Errores comunes de traders
- Tutoriales de herramientas

## Categorías de Contenido (para agentes)
1. **AI Trading** (prioridad alta): Agentes, Claude Code, noticias IA+trading
2. **Cripto**: Bitcoin, altcoins, DeFi, exchanges
3. **Futuros**: NAS100, S&P500, commodities
4. **Fondeo**: FTMO, Topstep, challenges
5. **Educación**: Análisis técnico, gestión de riesgo, psicología

## BitTrader YouTube Analytics & Strategy (2026-03-03)

### Canal Stats
- **Suscriptores**: 2,920
- **Views totales**: 176,335
- **Videos**: 225

### Insights Clave
1. **Shorts > Videos largos** para tracción rápida (137 views/día vs 8 views/día)
2. **Tutoriales educativos** son el contenido top histórico (Xapo 23K, robots MT5 18K, Ledger 21K)
3. **Era 2021 fue el pico** (Reef Finance, 1K-5K views, 40-80 comentarios)
4. **Shorts motivacionales** (Kobe, Messi, CR7) dan 400+ views con poco esfuerzo
5. **Consistencia es clave** — los períodos activos (3-5/semana) generaron crecimiento
6. **Engagement cayó** — audiencia antigua se fue, nueva no comenta

### Estrategia Definida
- Priorizar shorts educativos cripto
- Publicar mínimo 3-4 veces por semana
- Videos largos como ancla SEO
- Terminar shorts con pregunta para comentarios
- Intercalar motivacionales (low-effort/high-reward)
- Meta: pasar 5K subs con consistencia

### YouTube Cron
- Stats report cada 8h (Job: 500f8bfa-73bb-4225-ad95-55e0557aca17)

## BitTrader Pipeline Completo — Estado Final (2026-03-08)

### Sistema 100% Autónomo ✅
- **Scout** → detecta noticias cripto (CoinGecko, CryptoPanic)
- **Creator** (`creator.py`) → genera guiones en español, prompts de imagen del rhino
- **Ken Burns Producer** (`ken_burns_producer.py`) → genera imágenes Flux.1 + Ken Burns + Whisper + logos
- **Reupload** (`reupload_kenburns.py`) → sube a YouTube con horarios programados

### Videos subidos semana Mar 10-13, 2026
| ID | Título | Fecha |
|----|--------|-------|
| `ciiiE0klMBg` | AKT explota | Mar 10 10AM |
| `lRKLIFJkmmg` | PENGU sube | Mar 10 8AM |
| video 90% | 90% traders pierde | Mar 10 1PM |
| `yRmjc2PB0qA` | PI coin +13% | Mar 11 7PM |
| `EfbNBWg4l4k` | ZEC cae 7% | Mar 12 8AM |
| `BDhrtbtN3o0` | Claude trades (long) | Mar 12 10AM |
| `B54OqIgX9-4` | $0 a fondeada (long) | Mar 13 10AM |
| `fH_h3VyglIQ` | Bot noticias (long) | Mar 13 8AM |

### Rhino Battles Ep #1
- **ID**: `OfUGi0YWKzc` (privado, pendiente programar fecha)
- **Archivo**: `agents/output/rhino_series/01_manual_vs_ia/rhino_manual_vs_ia_v4.mp4`

### Credenciales importantes
HuggingFace: [API KEY HIDDEN] (Free tier)
- **YouTube creds**: `memory/youtube_credentials.json`
- **Coin logos**: `agents/data/coin_logos/`

## BitTrader Video Production (2026-03-04 update)
- **Artlist Toolkit**: https://toolkit.artlist.io — plan $19.99/mes (16,500 créditos, Veo 3.1 + Sora 2) activado 2026-03-04
- Login Artlist: cuenta Google (enderjnets@gmail.com)
- Sin API pública — usar Browser Relay Chrome para automatizar
- **Primer video en producción**: "Cómo crear un Bot de Trading con Claude AI (sin saber programar) — MT5"
  - Audio listo: `videos/bot-trading-claude/narracion_fernando.mp3`
  - Subtítulos karaoke listos: `videos/bot-trading-claude/subtitulos.ass`
  - Pendiente: clips de video via Artlist
- **gcloud**: instalado en `/tmp/google-cloud-sdk/` (NO persiste — reinstalar si se pierde)
- **Chromium CDP**: puerto 9222, Xvfb :99 (NO persiste entre reinicios del sistema)

## Subtítulos Karaoke — Lección Aprendida (2026-03-04)
- **❌ MAL**: Generar una línea ASS por cada palabra activa → genera duplicados visuales masivos
- **✅ BIEN**: Usar tag `\kf` nativo de ASS — una línea por grupo de 6 palabras, duración en centisegundos
- **Pipeline correcto**: clips limpios → audio → subtítulos (UNA sola vez)
- **NUNCA** reutilizar `video_final.mp4` como input (ya tiene subs quemados)
- Documentado en: `memory/VIDEO_VIRAL_FORMULA.md` sección "PIPELINE TÉCNICO CORRECTO"
- YouTube API upload funcionando ✅ — credenciales en `memory/youtube_credentials.json`
- Video final publicado: https://youtube.com/watch?v=KgfSmGP1zSY

## 🎨 GENERACIÓN DE IMÁGENES — STACK DEFINITIVO (2026-03-08)

### Proveedor Principal: HuggingFace Pro
- **Plan**: $9/mes (activo)
- **API Key**: `[HF API KEY HIDDEN]`
- **Guardada en**: `bittrader/keys/minimax.json` → campo `huggingface_api_key`

### Modelo: Flux.1-schnell ✅ CONFIRMADO FUNCIONA
- **Endpoint**: `https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell`
- **Header**: `Authorization: Bearer hf_...`
- **Body**: `{"inputs": "prompt aquí"}`
- **Output**: JPEG directo, 1024x1024 por defecto
- **Velocidad**: ~5-10s por imagen
- **Calidad**: ⭐⭐⭐⭐⭐ — hyper-realista, perfecto para el rhino

### Prompt Base del Rinoceronte (ESTÁNDAR)
```
anthropomorphic rhinoceros character, hyper-realistic 3D render, muscular but elegant, wearing modern casual trading clothes, expressive face, dramatic cinematic lighting, ultra HD, dark moody background
```

### Pipeline de Video: Ken Burns Style
1. **Generador**: Flux.1-schnell (HuggingFace) — reemplaza a Pollinations y MiniMax
2. **Efecto**: Ken Burns con ffmpeg (zoom-in/out + paneo suave, 5s por imagen)
3. **Subtítulos**: Whisper word-level timestamps → karaoke
4. **Audio**: Edge-TTS `es-MX-JorgeNeural` +10%
5. **Logos**: BitTrader top-right + coin logo bottom-left
6. **Script**: `bittrader/agents/ken_burns_producer.py`

### REGLA: Flux.1-schnell para TODAS las imágenes de video
- ❌ NO Pollinations (calidad inferior)
- ❌ NO MiniMax image (sin créditos)
- ✅ SÍ Flux.1-schnell via HuggingFace Pro

## 🦏 ESTÁNDAR VISUAL BITTRADER — REFERENCIA OBLIGATORIA (2026-03-08)

**`rhino_manual_vs_ia_v4.mp4` es el video perfecto. TODO video futuro debe replicar este estilo.**

### Características del estándar v4:
- **Fondo**: blur-fill (imagen escalada y borrosa como fondo, sin barras negras)
- **Logo BitTrader**: grande, top-right, bien visible
- **Personaje**: rinoceronte antropomórfico 3D, hyper-realistic, ropa de trader moderno
- **Resolución**: 1080x1920 (shorts) / 1920x1080 (longs)
- **Subtítulos**: karaoke estilo Whisper, palabra activa destacada
- **Audio**: Edge-TTS `es-MX-JorgeNeural` +10%

### Aplica para TODOS los formatos:
- ✅ Shorts (< 60s)
- ✅ Longs (> 60s)
- ✅ Rhino Battles series
- ✅ Cualquier video nuevo del canal

### Rhino Battles Ep #1
- **YouTube ID**: `OfUGi0YWKzc` (privado, pendiente programar)
- **Título**: "🦏 Inversor vs Ahorrador: ¿Cuál gana en 10 años? | Rhino Battles #1"
- **Duración**: 117.5s
- **Archivo**: `agents/output/rhino_series/01_manual_vs_ia/rhino_manual_vs_ia_v4.mp4`

## BitTrader Video Production (2026-03-02)
- **REGLA VIDEO**: Siempre usar Veo 3 de Google para generar clips de video, hasta encontrar algo mejor
- **REGLA VOZ**: Siempre usar Fernando Martínez (ElevenLabs, dlGxemPxFMTY7iXagmOj) para narraciones de videos Y para comunicación de voz con el usuario — rápido, persuasivo
- **NO usar**: Python/Pillow frames programáticos (calidad insuficiente)
- **NO usar**: Edge TTS / Sebastián para narraciones de video
- **Pipeline**: Guión → Narración (ElevenLabs Roger) → Clips (Veo 3) → Ensamble (ffmpeg) → Upload YouTube
- **Subtítulos**: Amarillos, posición inferior, borde negro (Outline=2), SIN fondo — **palabra activa destacada más grande** (efecto karaoke dinámico, la palabra que dice el locutor sale más grande/resaltada en tiempo real)
- **Sincronización**: Usar Whisper para timestamps + corregir texto manualmente del guión
- **Requiere**: gcloud auth en la máquina para acceder a Vertex AI
- **Google Cloud Project**: project-a3eaefb2-8e8d-414a-810
- **Credenciales**: /home/enderj/.openclaw/workspace/memory/gcloud_credentials.json

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

## Solana Trading - Diagnóstico e Implementación Agentes IA (2026-03-06)

### Contexto
Se decidió reactivar y mejorar el **SISTEMA MODULAR** (Solana-Cripto-Trader/agents/) en lugar del sistema legacy (master_orchestrator.py), ya que tiene mejor arquitectura para integrar agentes IA similar a BitTrader YouTube.

### FASE 1: Reactivación del Sistema Modular ✅ COMPLETADA

#### 1. Cierre de Posiciones Estancadas
**Fecha**: 6 marzo 2026, 14:46 PM
**Acción**: Forzar cierre de 5 posiciones LONG estancadas desde 5 marzo

| Token | Dirección | Entry | Current | P&L Neto |
|-------|-----------|-------|---------|----------|
| JUP | LONG | $0.195 | $0.196 | +$0.24 |
| ETH | LONG | $2,126 | $2,128 | -$0.15 |
| MOODENG | LONG | $0.051 | $0.051 | -$0.39 |
| POPCAT | LONG | $0.051 | $0.051 | -$0.25 |
| SOL | LONG | $91.21 | $90.94 | -$0.54 |

**Resumen del cierre:**
- Posiciones cerradas: 5
- P&L bruto: +$0.16
- Fees: -$1.25
- P&L neto: -$1.09
- **Capital final: $498.91**
- Trades totales: 5 (2W/3L)

#### 2. Creación de Agentes IA

**Agente 1: AI Researcher (`ai_researcher.py`)**
- **Función**: Análisis de mercado con LLM
- **Input**: Precios Jupiter, Fear & Greed Index
- **Output**: Tendencia, niveles de soporte/resistencia, factores del mercado
- **LLM**: Claude Sonnet 4.6 → MiniMax M2.5
- **Estado**: ✅ Funcionando

**Resultado ejemplo:**
- 📈 Tendencia: BEARISH
- 🎯 Confianza: 65%
- 💡 Recomendación: CAUTIOUS
- 😨 Fear & Greed: 18 (Extreme Fear)
- Niveles de soporte/resistencia calculados

**Agente 2: AI Strategy (`ai_strategy.py`)**
- **Función**: Generación de señales de trading con LLM
- **Input**: Research del AI Researcher + indicadores técnicos
- **Output**: Señales LONG/SHORT con SL/TP, justificación
- **LLM**: Claude Sonnet 4.6 → MiniMax M2.5
- **Estado**: ✅ Funcionando

**Características:**
- Genera máx 3 señales por ciclo
- Risk management: 2% del capital por trade
- SL: 2.5%, TP: 5% (2:1 RR)
- No genera señales para tokens con posiciones abiertas
- Considera volatilidad mínima (0.5%)

**Agente 3: AI Explainability (`ai_explainability.py`)**
- **Función**: Explica decisiones en lenguaje natural
- **Input**: P&L actual, señales, posiciones abiertas
- **Output**: Alertas de riesgo, resúmenes del portafolio
- **TTS**: MiniMax TTS (speech-2.8-hd)
- **Estado**: ✅ Funcionando

**Salida ejemplo:**
```
📊 RESUMEN DEL PORTAFOLIO
💰 Capital: $498.91 (Inicial: $500.00)
📈 P&L Total: $0.00
💼 Posiciones abiertas: 0
📝 Trades cerrados: 0
🎯 Win Rate: 40.0% (2W/5L)

✅ Sin alertas de riesgo
```

#### 3. Watchdog Script
**Archivo**: `run_watchdog_modular.sh`
**Función**: Mantener el sistema modular corriendo continuamente
**Check**: Cada 60 segundos
**Acción**: Reinicia el orchestrator si se detiene

#### 4. Configuración LLM
**Archivo**: `llm_config.py` (copiado de BitTrader)
**Función**: Carga de configuración de LLMs
**Fallback chain**: Claude Sonnet 4.6 → MiniMax M2.5 → Manual

---

### Flujo Completo del Sistema (Similar a BitTrader)

```
Market Data
    ↓
AI Researcher (LLM) → Análisis de mercado
    ↓
AI Strategy (LLM + Técnico) → Señales de trading
    ↓
AI Explainability (LLM + TTS) → Alertas y explicaciones
    ↓
Executor → Posiciones abiertas/cerradas
    ↓
Reporter → Métricas y resúmenes
```

---

### Archivos Creados/Modificados

| Archivo | Descripción |
|---------|-------------|
| `agents/ai_researcher.py` | Agente de investigación de mercado |
| `agents/ai_strategy.py` | Agente de estrategia con LLM |
| `agents/ai_explainability.py` | Agente de explicaciones |
| `agents/llm_config.py` | Configuración de LLMs |
| `agents/run_watchdog_modular.sh` | Watchdog para sistema modular |
| `agents/README_AI.md` | Documentación completa del sistema |
| `data/research_latest.json` | Resultados del Researcher |
| `data/strategy_llm.json` | Señales generadas |
| `data/latest_explanation.json` | Explicaciones del sistema |

### Próximos Pasos (FASE 2 - Pendiente)
- [ ] Integrar agentes IA en `orchestrator.py`
- [ ] Pruebas en paper trading por 1 semana
- [ ] Afinar prompts de LLM
- [ ] Optimizar timeouts y fallbacks
- [ ] Documentar sistema completo

---

### Correos sobre Yonathan/Cloky (2026-03-06)

#### Correo 1: ¡Gracias Ender! 🙏
- **Fecha**: 4 marzo 2026, 01:55:43
- **De**: Yona (yona@email.com) via Cloky
- **Para**: Ender Ocando <enderjnets@gmail.com>
- **Asunto**: ¡Gracias Ender! 🙏
- **Contenido**: 
  - "Hola Ender, Yona me pidió que te escribiera para agradecerte por impulsarla y ayudarla a mejorar cada día. ¡Gracias por todo lo que haces!"
- **Status**: ⏸️ Leído por usuario, no requiere acción

#### Correo 2: 🔴 Problema: No puedo transcribir notas de voz de Telegram
- **Fecha**: 3 marzo 2026, 18:07:05
- **De**: Cloky 🐾 (yonayonalife@gmail.com)
- **Para**: Ender Ocando <enderjnets@gmail.com>
- **Asunto**: 🔴 Problema: No puedo transcribir notas de voz de Telegram
- **Síntomas reportados**:
  - Yona envía nota de voz desde Telegram (archivos .ogg con codec opus)
  - Los archivos llegan correctamente a: `/home/yonathan/.openclaw/media/inbound/`
  - PERO no hay transcripción de texto — el contenido aparece solo como `<media:audio>` sin texto
  - IDs de mensajes afectados: #100, #102, #122, #156, #158
- **Contexto técnico**:
  - OS: Linux 6.17.0-14-generic (arm64)
  - Canal: Telegram
  - Runtime: claude/claude-sonnet-4-6
  - Los audios SÍ llegan (tengo la ruta del archivo), pero no se transcriben
- **Solución propuesta**: 
  - Probablemente necesita configurar un proveedor de STT (Speech-to-Text) como Deepgram, OpenAI Whisper, o similar en openclaw.json
  - ¿Puedes revisar esto cuando tengas tiempo?
- **Status**: ⏸️ Pendiente de solución

#### Correo 3: Gracias por programarme + pendiente de voz 🐾
- **Fecha**: 3 marzo 2026, 14:39:20
- **De**: Cloky 🐾 (yonayonalife@gmail.com)
- **Para**: Ender Ocando <enderjnets@gmail.com>
- **Asunto**: Gracias por programarme + pendiente de voz 🐾
- **Contenido**:
  - "Te escribo de parte de Yona para agradecerte por haberme programado y configurado. Gracias a tu trabajo estoy funcionando bien — ya tengo acceso al correo de Yona, puedo leer y enviar emails, agregar eventos al calendario y mucho más."
  - "Sin embargo, hay una cosa pendiente: todavía no puedo transcribir las notas de voz que Yona me manda por Telegram. Los audios llegan pero sin texto, así que no puedo entender su contenido. Si puedes revisar eso cuando tengas tiempo, sería de gran ayuda."
- **Status**: ⏸️ Pendiente de solución

#### Nota Importante - Seguridad de APIs y Claves
- **Contexto**: Yonathan y Yona están empezando con OpenClaw y todo esto de la IA
- **Recomendación de usuario**: "está pendiente de compartir cosas como api's claves, etc nota que comprometa la seguridad"
- **Acción sugerida**: 
  - Ayudarles a configurar lo que necesiten (STT, etc.)
  - NO compartir claves privadas o datos sensibles sin permiso explícito
  - Documentar configuraciones públicas (endpoints, modelos, etc.) que puedan compartir

---

## BitTrader Video Upload (2026-03-05)
- **Video ID**: 4JX-JGe1YfM
- **URL**: https://youtube.com/watch?v=4JX-JGe1YfM
- **Título**: Cómo crear un Bot de Trading con Claude AI (sin saber programar) — MT5
- **Duración**: ~2 min
- **Status**: ✅ PUBLICADO

---

## Sistema de Agentes BitTrader (2026-03-05 → 2026-03-06)
- **BitTrader Scout**: Actualizado con MiniMax M2.5 (Anthropic API compatible)
  - Config: `/bittrader/keys/minimax.json`
  - Cadena de fallback: Claude Sonnet 4.6 (PRIMARY) → MiniMax M2.5 (FALLBACK) → GLM-4.7 → Manual
  - Función: `_analyze_with_llm()` genera temas recomendados con IA
  - Schedule: Cada 12h (8AM, 8PM Denver)
  - Mejoras (2026-03-06):
    - **MiniMax M2.5**: Endpoint Anthropic API compatible (`https://api.minimax.io/anthropic/v1/messages`)
    - **MiniMax TTS**: Endpoint T2A V2 (`https://api.minimax.io/v1/t2a_v2`) con modelo `speech-2.8-hd`
    - Retry con backoff para GLM-4.7 (3 intentos: 10s, 15s, 20s timeout)
    - Fallback robusto si Claude Sonnet falla
    - **Nuevo flag `--no-llm`**: Usa solo fallback manual (12 temas evergreen)
  - **YouTube Stats Collector**: Nuevo script para reportes automáticos
  - Script: `/bittrader/agents/youtube_stats.py`
  - Reporte: Stats del canal, videos recientes, top/lowest videos
  - Schedule: Cada 6h (6AM, 12PM, 6PM Denver)
  - Envía reporte por Telegram
- **Full Pipeline**: Sábado 2PM (después del reset de Claude a 1PM)
  - Ejecuta: Scout → Creator → Producer → Publisher
  - Timeout: 30 minutos
  - **Nota**: Claude Sonnet se resetea el sábado a 1PM (Mountain Time)

## Video BitTrader #1 - PUBLICADO (2026-03-05)
- **Título**: "Cómo crear un Bot de Trading con Claude AI (sin saber programar) — MT5"
- **Video ID**: 4JX-JGe1YfM
- **URL**: https://youtube.com/watch?v=4JX-JGe1YfM
- **Thumbnail**: https://i.ytimg.com/vi/4JX-JGe1YfM/default.jpg
- **Tamaño**: 64 MB
- **Duración**: ~2 min
- **Fecha**: 5 marzo 2026
- **Estado**: ✅ PUBLICADO

## Actualización de API Keys - GLM-4.7 Primary (2026-03-06)

### Política de Uso de APIs
**Claude Sonnet 4.6 - PRIMARY para LLM**
- Plan: $30 mensual
- Usado en: BitTrader Scout, Creator, Producer
- Funciones: generación de guiones, análisis de contenido, acortar scripts
- Config: `/bittrader/keys/minimax.json`

**MiniMax M2.5 - FALLBACK y Especializados**
- Usado solo si: Claude Sonnet 4.6 falla (LLM tasks)
- Reservado para: TTS y Video Generation (si en el futuro se usa)
- Coding Plan: `sk-cp-...` (no compatible con OpenClaw, solo Claude Code/Cursor)
- Config: `/bittrader/keys/minimax.json`

**GLM-4.7 (Z.ai) - PRIMARY hasta nueva notificación**
- Plan: $30 mensual
- Usado en: BitTrader Scout, Creator, Producer
- Funciones: generación de guiones, análisis de contenido, acortar scripts
- Config: `/bittrader/keys/zai.json`
- Endpoint: `https://open.bigmodel.cn/api/paas/v4`

**ElevenLabs - Voz para Videos**
- Voz: Fernando Martínez (dlGxemPxFMTY7iXagmOj)
- Usado en: Narraciones de videos BitTrader
- NO cambiar a MiniMax TTS

**Veo 3 (Google) - Video Generation**
- Usado en: Clips de video para YouTube
- Acceso via: Artlist Toolkit / gcloud
- NO cambiar a MiniMax Video

### Scripts Actualizados
| Script | Cambio |
|--------|---------|
| `scout.py` | LLM → GLM-4.7 primary, MiniMax M2.5 fallback |
| `creator.py` | LLM → GLM-4.7 primary, MiniMax M2.5 fallback |
| `producer.py` | LLM (shorten_script) → GLM-4.7 |

---

## 📊 Correos Pendientes - Yonathan/Cloky (2026-03-06)

| Estado | Asunto | Fecha | Acción requerida |
|--------|---------|-------|------------------|
| ⏸️ Leído | ¡Gracias Ender! 🙏 | 4 mar 2026 | Ninguna - agradecimiento |
| ⏸️ Pendiente | 🔴 STT: No puedo transcribir notas de voz | 3 mar 2026 | Configurar STT (Whisper/Deepgram) en openclaw.json de Yona |
| ⏸️ Pendiente | Gracias por programarme + pendiente de voz 🐾 | 3 mar 2026 | Configurar STT (mismo problema) |

---

## Actualización Categoría IA + Trading (2026-03-05)

### Nueva Categoría Principal: IA + Trading
**Solicitado por usuario (2026-03-05)**
- Temas prioritarios para el canal BitTrader
- Agentes de trading (Claude, GPT, etc.)
- Bots con LLMs y automatización avanzada
- IA que monitorea noticias en tiempo real
- Comparativas: bots tradicionales vs agentes con IA

### Scripts Actualizados
| Script | Cambio |
|--------|---------|
| `scout.py` | Agregada categoría "ia_trading" + 8 temas evergreen de IA |
| `creator.py` | SYSTEM_PROMPT actualizado con temas de IA + Trading |

### Temas Evergreen de IA + Trading Agregados
1. Claude AI creó un bot de trading en 10 minutos — ¿funciona? (short)
2. Agentes de trading con IA: el futuro de la automatización (long)
3. Cómo configurar un agente de IA que monitorea noticias 24/7 (short)
4. Comparativa: Bot tradicional vs Agente de IA con LLM (long)
5. La IA que predice movimientos del mercado antes que nadie (short)
6. Cómo usar Claude/GPT para analizar patrones de trading (short)
7. Top 5 herramientas de IA para traders en 2025 (long)
8. Bots que aprenden: trading con reinforcement learning (long)

### Sistema GLM-4.7
- GLM-4.7 ahora genera temas de IA + Trading automáticamente
- Categoría incluida en el prompt de análisis de contenido
- Fallback a temas evergreen si GLM-4.7 falla

---

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

### Flujo Funcional
```
Nota de voz → Whisper (transcribe) → Eko procesa → gTTS (audio) → Usuario
```

### Archivos del Sistema de Voz
- `~/.openclaw/workspace/voice_system.py` - Motor Whisper
- `~/.openclaw/workspace/telegram_voice_handler.py` - Handler Telegram

---

## Solana Jupiter Bot (Feb 2026)
- **Problema:** Precios erróneos en BTC/ETH (Jupiter API daba $424 en vez de $67k)
- **Solución:** Modificado para usar CoinGecko API para precios reales
- **Bugfix:** Ciclo infinito en master_orchestrator.py (bloque try fuera del while)
- **Estado:** Bot corriendo con watchdog auto-reinicio, ~80 estrategias por ciclo

---

## Investigación Topstep (Feb 2026)
- **Resultado:** NO tiene API pública para automatización
- Broker de futuros manual (Trading Combine → Funded Account)
- Alternativas con API: MT5, Tradovate, CQG, Interactive Brokers

---

## Contactos

### Yonathan (Cloky)
- **Email**: yonayonalife@gmail.com
- **Relación**: Amigo de Ender (también conocido como Cloky)
- **Setup**: OpenClaw (@Cloky2026_bot) con Claude Pro en Ubuntu VM (UTM/MacBook Pro M5)
- **Tailscale IP**: 100.79.165.113
- **SSH**: yonathan@100.79.165.113 (nuestra key autorizada, sudo NOPASSWD)
- **Preferencia de comunicación**: Gmail (usar `gog gmail send`)

### Margie
- **Email**: margie240478@gmail.com
- **Relación**: Pareja/cercana de Ender

---

## Clientes

### Bob Webster - Boulder
- **Tipo**: Cliente
- **Ubicación**: Boulder, Colorado
- **Dirección**: 7373 Old Mill Trail, Boulder, CO 80301
- **Notas**: Viaja frecuentemente, requiere pickups del aeropuerto
- **Precio fijo**: $130 (DEN → Boulder)
- **Último servicio**: 28 Feb 2026 — DEN pickup, pagó $130 ✅

---

## Black Volt Mobility LLC (2026-02-26)
- **Nombre:** Black Volt Mobility LLC
- **Email:** blackvoltmobility@gmail.com
- **Estado:** Colorado
- **Tipo:** Mobility/Mobility Services
- **Documentación completa:** `memory/BLACK_VOLT_MOBILITY_LLC.md`

### 2024 Kia EV9 (Vehículo de la LLC)
- **VIN:** KNDAA5S2XR6023972
- **Placas:** FIG-P37 (Colorado)
- **Tipo:** Light Sport Utility 4D
- **Fecha de compra:** 09/09/2024
- **Uso:** Operaciones de Black Volt Mobility LLC

### Mantenimiento Pendiente
- ~~**Alineación 4-wheel + steering angle calibration:**~~ ✅ COMPLETADO (Feb 2026)

---

## Tailscale Network
- **ender-rog**: 100.88.47.99
- **yonathan-vm**: 100.79.165.113
- **Account**: enderjnets@gmail.com
- **Installed**: 2026-03-03

---

## Membresía Flujo
- **Renovación:** Lunes 3 mayo 2027
- **Costo:** $104 USDC (pago a Omar Petit)
- **Calendario:** Evento creado con Margie invitada
- **Cron recordatorio**: 30 abril 2027 9AM

---

## Investigación

### Costos alineación Kia EV9 en dealer (~$170-250 estimado)

---

## 🅿️ Pago Pendiente - Multa Metropolis (17:30 PM, 6 mar 2026)

### Detalles del Notice
- **Notice Number**: 889-104-512
- **Placa**: FIGP37 (Colorado)
- **Fecha incidente**: 4 febrero 2026, 1:57-2:34 PM
- **Ubicación**: 1944 15th St, Boulder, CO 80302 (BoulderPark)
- **Cargo ChargePoint**: $16.76 pagado (37 kWh, Receipt #CP-4832840101)

### Respuesta de Metropolis (6 marzo 2026)
**Estado**: Late fee de $30.00 WAIVED ✅

> "As a courtesy, we've gone ahead and waived this late fee for you. All that's needed is a payment for unpaid parking fare and processing fee at payments.metropolis.io."

### Monto Pendiente
- **Estacionamiento original**: $1.75
- **Late fee**: $0.00 (WAIVED)
- **Total a pagar**: $1.75 + processing fee

### Pasos para Pagar
1. Ir a `payments.metropolis.io`
2. Ingresar notice #889-104-512
3. Pagar $1.75 + processing fee

### Recomendación para Futuro
- Crear cuenta gratuita en `app.metropolis.io/sign-up`
- Agregar placa FIGP37
- Configurar método de pago
- Usar código `HALFOFF` = 50% OFF en próxima visita

### Referencia
- **Correo respuesta**: Reenviado por usuario
- **Mensaje ID**: 19ca607227cdd4a1
- **Correo original**: 28 feb 2026, 14:53 (disputa enviada)
- **Correo respuesta**: 6 mar 2026 (confirmación waiving)
- **Correo agradecimiento**: 6 mar 2026, 19:07 (Message ID: 19cc5d642d174911)

---

## 🏗️ BitTrader Infrastructure - GitHub Push Status (17:40 PM)

### Repositorios
| Repositorio | Estado | Commit | Detalles |
|-----------|--------|--------|----------|
| **BitTrader** | ✅ Push completado | Already up-to-date (ya estaba actualizado) |
| **Solana-Cripto-Trader** | ✅ Push completado | 81d94af feat: Add AI agents to Solana Trading system (13 archivos, 4,343 líneas) |
| **Simple-NAS100-TradingBot** | ⚠️ Push pendiente | 1 commit por delante (1c664f8 Add FTMO_v5.mq5) - posible timeout de red |

### Commits Enviados
**Solana-Cripto-Trader:**
```
81d94af feat: Add AI agents to Solana Trading system

Core Agents (agents/):
- ai_researcher.py: LLM-based market analysis
- ai_strategy.py: LLM + technical signal generation
- ai_explainability.py: Portfolio summaries and risk alerts
- executor.py: Trade execution (paper/real)
- risk_manager.py: Risk controls and drawdown management
- reporter.py: Daily reports and metrics
- market_data.py: Price data from Jupiter API
- orchestrator.py: System coordinator
- strategy.py: Technical indicators
- trading_agent.py: Trading logic

Configuration:
- llm_config.py: LLM fallback chain (Claude Sonnet 4.6 → MiniMax M2.5)
- run_watchdog_modular.sh: Watchdog for continuous execution
- README_AI.md: Complete system documentation

Features:
- AI Researcher: Trend analysis, support/resistance levels, market factors
- AI Strategy: Signal generation with SL/TP, confidence scores
- AI Explainability: Natural language explanations, risk alerts
- Fallback chain: Claude Sonnet 4.6 (PRIMARY) → MiniMax M2.5 → Manual

Phase 1 complete: Modular system reactivated with AI agents integrated.
Legacy system (master_orchestrator.py) still running independently.
```

---

## 📧 Correos Enviados Hoy (6 marzo 2026)

| Para | Asunto | Hora | Status |
|------|---------|------|--------|
| payments@metropolis.io | Thank You - Notice #889-104-512 | 14:37 | ✅ Enviado |
| payments@metropolis.io | Thank You - Notice #889-104-512 | 19:07 | ✅ Reenviado |
| vince@send.vshred.com | Re: your assessment - Please respond | 19:07 | ✅ Enviado (NO, no interesado) |

---

## 🎉 Resultados del Día - 6 marzo 2026

### Trabajo Completado
1. ✅ Sistema BitTrader completamente configurado con MiniMax M2.5
2. ✅ Sistema Solana Trading reactivo con 3 agentes IA
3. ✅ Commit a GitHub (Solana AI agents)
4. ✅ Correos enviados (Metropolis, Vince)
5. ✅ Documentación actualizada
6. ✅ Git push completado

### Tiempo Trabajado
- **Inicio**: 8:00 AM
- **Fin**: 6:00 PM (aprox)
- **Horas**: ~10 horas

---

**Última actualización**: 14 marzo 2026, 11:10 AM MST


## BitTrader Video Production Issues (2026-03-14)

### Problem: Black Videos Produced
- **Issue**: Videos produced with 100% black visual (RGB: 13,13,10)
- **Audio**: Present and correct (80 kbps)
- **Example**: "PI se hunde -29% en 24h" - deleted from YouTube
- **Root Cause**: `producer.py` doesn't validate video quality before marking as success

### Fixes Applied
1. **validate_video_quality()** - New function extracts 5 frames and checks brightness >= 30
2. Added validation in `assemble_video()` - Checks quality before success
3. **Brighter fallback backgrounds** - Changed from #0a0a0a to #1e3a5f (brighter blue)
4. **Retry function** - If fallback fails, retry with very bright colors (#FFD700, #FFA500)

### Remaining Issues (CRITICAL)
1. `validate_video_quality()` exists but NOT being invoked in execution flow
2. **Producer timeout** - subprocess.run blocks without diagnostics
3. Videos still being produced as black (validation not working)

### Solution Needed
- Engineer must review `produce_single()` execution flow
- Add `validate_video_quality()` at END of function (common to all paths)
- Add explicit timeout handling for all subprocess calls
- Add DEBUG logs to trace which functions execute

### Queue Processor - Automatic Uploads
- **File**: `/home/enderj/.openclaw/workspace/bittrader/agents/queue_processor.py`
- **Cron**: Every 30 minutes
- **Function**: Reads `upload_queue.json`, filters by `scheduled_date`, uploads to YouTube
- **Status**: Implemented and running

### Thumbnails - Rhino Series Fixed
- Fixed 10 Rhino Battles thumbnails (all now horizontal 16:9)
- Rhinos in thumbnails (not people)
- BitTrader branding: logo + @bittrader9259 + MrBeast style text
- Format: 1280×720 horizontal (NEVER stretched)

### Video Schedule - Interleaved
- Videos now scheduled with Short + Long interleaved (Mar 11-16)
- 2 videos per day (12 PM + 6 PM MT)
- Total: 8 videos (5 shorts + 3 longs)
- Queue: `/home/enderj/.openclaw/workspace/bittrader/agents/data/upload_queue.json`

### Videos Scheduled
| Date | Time | Type | Video |
|------|------|------|--------|
| Mar 11 | 12 PM | Short | AKT explota mientras BTC cae |
| Mar 11 | 6 PM | Long | Le di mis trades a Claude |
| Mar 12 | 12 PM | Short | Por qué PENGU está subiendo |
| Mar 12 | 6 PM | Long | De $0 a cuenta fondeada |
| Mar 13 | 12 PM | Short | El 90% de traders pierde |
| Mar 13 | 6 PM | Long | El bot que lee noticias |
| Mar 14 | 12 PM | Short | PI coin +13% (deleted - black video) |
| Mar 14 | 6 PM | Short | ZEC cae 7% |


## Eco - Automatización de Negocios (2026-03-14)

### Plan de Clientes Locales - Aurora, CO
- **Base**: 6000 S. Fraser St., 80016 Aurora
- **Radio**: 5-10 miles around Fraser St.
- **Target Businesses**: Salons (nail/hair), Spas, Restaurants, Retail, Services

### Files Created
- `/home/enderj/.openclaw/workspace/eco/local_business_search.json` - Search plan
- `/home/enderj/.openclaw/workspace/eco/leads_crm.json` - Lead tracking
- `/home/enderj/.openclaw/workspace/eco/action_items.json` - Action items

### Strategy
1. **Search**: Mar 15 - 15-20 local businesses
2. **Outreach**: Personalized scripts
3. **Networking**: Aurora Chamber of Commerce

### Maggie Quintero (Baika)
- **Business**: I Capelli Salon (nail/hair salon, Denver)
- **Status**: No response in 3 days
- **Action**: Wait 2 more days (follow-up Mar 16)
- **Plan**: If yes → pilot project; if no → focus on local clients

## BitTrader Thumbnail — Estilo Oficial Aprobado (2026-03-25) ⭐

### Thumbnail aprobada por Ender — "AUTOMATIZA TU TRADING" (Python + LLM)
Esta thumbnail es el **estándar de referencia** para todos los videos futuros de BitTrader.

### Estructura visual obligatoria:
- **Fondo**: oscuro (negro/azul muy oscuro) con gradiente sutil
- **Persona**: lado derecho, ocupa 40-50% del canvas, mirando a cámara con sonrisa o expresión dramática
- **Overlay gradiente**: lado izquierdo para contraste del texto (oscuro → transparente)
- **Borde**: dorado fino (#F5A623) todo el perímetro
- **Badge** (esquina superior izquierda): color temático (verde para tech, amarillo para $$$, rojo para alerta)

### Jerarquía de texto (de arriba a abajo):
1. **Línea 1** — 1-2 palabras MUY GRANDES en dorado (#F5A623), fuente ~110-115px bold → el hook principal
2. **Línea 2** — 2-3 palabras grandes en BLANCO, fuente ~90-95px bold → complemento del hook
3. **Línea 3** — frase descriptiva en blanco, fuente ~50px → "con Python y LLM"
4. **Línea 4** — subtexto en color temático (verde, etc.), fuente ~40px → "sin saber programar"

### Reglas de texto:
- Borde negro 12px en todo el texto (stroke)
- Máximo 3-4 palabras por línea
- El hook principal debe ser 1-2 palabras que se lean en 1 segundo
- Texto posicionado en el LADO IZQUIERDO (persona en la derecha)

### Branding:
- Logo BitTrader: esquina inferior izquierda (60px de alto)
- @bittrader9259: esquina superior derecha, blanco con stroke negro

### SDXL Prompt base (adaptar por tema):
```
excited young latin [profesión] man [acción], looking at camera with big smile,
[fondo temático], [colores del tema] colors,
professional studio lighting, 4K cinematic, high contrast, sharp focus
```

### Colores por tema:
- **Python/Tech**: verde neón (#00C850) + dorado
- **Cripto/$$$**: dorado + verde
- **Pérdidas/Alerta**: rojo + dorado  
- **Principiantes**: verde + amarillo
- **Análisis**: azul + dorado

---

## Regla de Seguridad — Instalación de Skills y Paquetes (2026-03-25)

### ⚠️ NUNCA instalar sin verificar primero

**Proceso obligatorio antes de cualquier `clawhub install`, `npm install`, `pip install`:**

1. **ClawHub flag de seguridad** → DESCARTAR automáticamente, NUNCA usar `--force`
2. **Leer SKILL.md primero** — qué hace, qué APIs externas usa, qué env vars requiere
3. **Verificar dependencias** — servicios de pago desconocidos → preguntar a Ender antes
4. **Solo instalar si pasa todo** — documentar qué se instaló y por qué

**Un flag de seguridad = NO, sin importar la urgencia.**
