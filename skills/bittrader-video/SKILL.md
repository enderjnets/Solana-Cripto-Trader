# BitTrader Video Production Skill

## Descripción
Skill para producir videos de YouTube para el canal BitTrader (@bittrader9259). Cubre el pipeline completo: guión → narración → clips IA → subtítulos → ensamble → upload.

## Cuándo Usar
- Cuando Ender pida crear videos para BitTrader/YouTube
- Cuando se necesite producir shorts o videos largos sobre cripto/trading
- Cuando se necesite re-editar o ajustar videos existentes

---

## Pipeline de Producción

### Paso 1: Guión
1. Escribir guión en español
2. Definir metadata: título, descripción, tags, hashtags
3. Guardar en `PLAN.md` dentro del directorio del proyecto

### Paso 2: Narración (ElevenLabs Roger)
**REGLA ABSOLUTA: Siempre usar Roger de ElevenLabs para narración.**
- **Voice ID**: `CwhRBWXzGAHq8TQ4Fs17`
- **Model**: `eleven_multilingual_v2`
- **Settings**: `stability: 0.5, similarity_boost: 0.75`
- **NO usar**: Edge TTS, gTTS, ni ninguna otra voz

```bash
curl -s "https://api.elevenlabs.io/v1/text-to-speech/CwhRBWXzGAHq8TQ4Fs17" \
  -H "xi-api-key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<GUIÓN>",
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
  }' --output narration_roger.mp3
```

API Key en: `~/.openclaw/workspace/memory/` (buscar en archivos de config)

### Paso 3: Clips de Video (Veo 3)
**REGLA ABSOLUTA: Siempre usar Veo 3 de Google para generar clips.**
- **NO usar**: Python/Pillow frames programáticos (calidad insuficiente)
- **Endpoint**: `https://us-central1-aiplatform.googleapis.com/v1/projects/project-a3eaefb2-8e8d-414a-810/locations/us-central1/publishers/google/models/veo-3.0-generate-preview:predictLongRunning`
- **Credenciales**: `/home/enderj/.openclaw/workspace/memory/gcloud_credentials.json`
- **Project**: `project-a3eaefb2-8e8d-414a-810`

#### Prompts para Veo 3:
- Describir escena visual que acompañe la narración
- Estilo: profesional, tech, trading, cripto
- Duración por clip: 5-8 segundos
- Resolución: mínimo 720p (escalar a 1080p si necesario)

#### Flujo de generación:
1. Dividir guión en segmentos (uno por clip)
2. Crear prompt visual para cada segmento
3. Generar clips en paralelo con Veo 3
4. Descargar cuando estén listos (poll LRO)
5. Escalar a resolución target con `lanczos`

### Paso 4: Subtítulos
**Estilo aprobado — NO cambiar sin autorización:**

#### Videos Verticales (Shorts, 1080x1920):
```
FontSize=12
FontName=DejaVu Sans Bold
PrimaryColour=&H00FFFF&    (amarillo en BGR)
OutlineColour=&H000000&    (negro)
BorderStyle=1               (outline, SIN fondo/caja)
Outline=2
Shadow=1
Alignment=2                 (bottom center)
MarginV=25
```

#### Videos Horizontales (Principal, 1920x1080):
```
FontSize=7
FontName=DejaVu Sans Bold
PrimaryColour=&H00FFFF&    (amarillo en BGR)
OutlineColour=&H000000&    (negro)
BorderStyle=1               (outline, SIN fondo/caja)
Outline=2
Shadow=1
Alignment=2                 (bottom center)
MarginV=20
```

#### Proceso de sincronización:
1. Generar SRT con Whisper: `whisper narration_roger.mp3 --model small --language es --output_format srt`
2. **IMPORTANTE**: Conservar timestamps de Whisper pero corregir texto contra guión original
3. Errores comunes de Whisper: "Bote"→"bot", "Boeing"→"BOINC", "tradin"→"trading", "Read Chaden"→"Grid Trading", "3B las verdes"→"3 velas verdes", "Bactes"→"backtests"
4. Guardar SRT corregido como `subs_roger.srt` en el directorio del video

### Paso 5: Ensamble (ffmpeg)

#### Short (vertical):
```bash
ffmpeg -y -i video_veo3.mp4 -i narration_roger.mp3 \
  -vf "subtitles=subs_roger.srt:force_style='FontSize=12,FontName=DejaVu Sans Bold,PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=25'" \
  -map 0:v -map 1:a -c:v libx264 -crf 23 -preset fast -c:a aac -shortest \
  output_final.mp4
```

#### Video principal (horizontal):
```bash
ffmpeg -y -i video_clean.mp4 -i narration_roger.mp3 \
  -vf "subtitles=subs_roger.srt:force_style='FontSize=7,FontName=DejaVu Sans Bold,PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=20'" \
  -map 0:v -map 1:a -c:v libx264 -crf 23 -preset fast -c:a aac -shortest \
  output_final.mp4
```

**⚠️ IMPORTANTE**: Siempre usar el video base SIN subtítulos quemados. Nunca aplicar subtítulos sobre un video que ya tiene subtítulos.

### Paso 6: Preview para Telegram
Comprimir a <16MB para enviar preview:
```bash
# Horizontal (720p)
ffmpeg -y -i output_final.mp4 -c:v libx264 -crf 32 -preset fast -vf "scale=1280:720" -c:a aac -b:a 96k preview_tg.mp4

# Vertical (mantener resolución, solo comprimir)
ffmpeg -y -i output_final.mp4 -c:v libx264 -crf 30 -preset fast -c:a aac -b:a 96k preview_tg.mp4
```

### Paso 7: Upload a YouTube
- **Credenciales**: `/home/enderj/.openclaw/workspace/memory/veo_credentials.json`
- **Canal**: @bittrader9259
- Usar YouTube Data API v3 con resumable upload
- Siempre subir versión FULL quality (no comprimida)
- Idioma: `es` (español)
- Categoría: `22` (People & Blogs)
- `selfDeclaredMadeForKids: false`

#### Mejores horas para publicar (Mountain Time):
- **Shorts**: 10:00 AM y 2:00 PM
- **Videos largos**: 10:00 AM
- **NO publicar** todos al mismo tiempo — escalonar con 4+ horas entre cada uno

---

## Estructura de Archivos

```
bittrader/
└── [fecha_proyecto]/
    ├── PLAN.md                     # Guiones y metadata
    ├── video_principal/
    │   ├── narration_roger.mp3     # Narración ElevenLabs
    │   ├── clips_veo3/             # Clips individuales de Veo 3
    │   │   ├── clip_01.mp4
    │   │   ├── clip_02.mp4
    │   │   └── concatenated.mp4    # Clips concatenados (SIN subs)
    │   ├── subs_roger.srt          # Subtítulos sincronizados
    │   └── video_principal_final.mp4
    ├── short1_[nombre]/
    │   ├── narration_roger.mp3
    │   ├── short1_veo3.mp4         # Clip Veo 3 (SIN subs)
    │   ├── subs_roger.srt
    │   └── short1_final.mp4
    ├── short2_[nombre]/
    └── short3_[nombre]/
```

---

## Lecciones Aprendidas

### Subtítulos
- ASS FontSize es relativo a PlayResY (default 288). FontSize=24 en un video 1920 de alto = ~160px efectivos. Usar FontSize=8-12 para vertical, 4-7 para horizontal.
- **NUNCA** poner fondo negro en subtítulos (`BorderStyle=1` no `BorderStyle=3`)
- Siempre verificar visualmente con screenshot antes de enviar al usuario
- Si se regeneran subtítulos sobre un video que ya tenía, usar el video BASE limpio

### Voz
- ElevenLabs Roger es la voz oficial de BitTrader — para todo
- Roger suena natural en español con `eleven_multilingual_v2`
- ~1000 caracteres ≈ 6-8 segundos de audio

### Veo 3
- Clips salen en 720p — escalar a 1080p con `lanczos`
- Generación toma 2-5 minutos por clip
- Generar en paralelo para ahorrar tiempo
- Usar `setpts` para ajustar duración de clips a segmentos de narración

### YouTube
- Escalonar publicaciones (no todos de golpe)
- Shorts necesitan `#shorts` en título o descripción
- Videos se pueden subir como privados y programar publicación con cron
