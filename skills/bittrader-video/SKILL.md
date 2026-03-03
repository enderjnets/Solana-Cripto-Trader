# BitTrader Video Production Skill

## Descripción
Skill para producir videos de YouTube para el canal BitTrader (@bittrader9259). Cubre el pipeline completo: guión → narración → clips IA → subtítulos dinámicos → marca de agua → ensamble → upload.

## Cuándo Usar
- Cuando Ender pida crear videos para BitTrader/YouTube
- Cuando se necesite producir shorts o videos largos sobre cripto/trading
- Cuando se necesite re-editar o ajustar videos existentes

---

## Pipeline de Producción (Orden Exacto)

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
- **⚠️ CRÍTICO**: No usar `\n` dentro del JSON — reemplaza saltos de línea con espacio o genera el JSON con Python

```bash
curl -s "https://api.elevenlabs.io/v1/text-to-speech/CwhRBWXzGAHq8TQ4Fs17" \
  -H "xi-api-key: 98b4670ec5f8658b81fbcf0d9a685b3f3348ac6420a6be261ee7d5ceaf656a8f" \
  -H "Content-Type: application/json" \
  -d '{"text":"<GUIÓN SIN SALTOS DE LINEA>","model_id":"eleven_multilingual_v2","voice_settings":{"stability":0.5,"similarity_boost":0.75}}' \
  --output narration_roger.mp3
```

### Paso 3: Clips de Video (Veo 3)
**REGLA ABSOLUTA: Siempre usar Veo 3 de Google para generar clips.**
- **NO usar**: Python/Pillow frames programáticos (calidad insuficiente)
- **Endpoint**: `https://us-central1-aiplatform.googleapis.com/v1/projects/project-a3eaefb2-8e8d-414a-810/locations/us-central1/publishers/google/models/veo-3.0-generate-preview:predictLongRunning`
- **Credenciales**: `/home/enderj/.openclaw/workspace/memory/gcloud_credentials.json`
- **Project**: `project-a3eaefb2-8e8d-414a-810`

#### ⚠️ CRÍTICO: Orientación de clips
- **Shorts (vertical)**: SIEMPRE especificar `"aspectRatio": "9:16"` en el request
- **Video principal (horizontal)**: SIEMPRE especificar `"aspectRatio": "16:9"`
- **NUNCA rotar un clip horizontal para hacerlo vertical** — se ve mal. Usar crop inteligente:
  ```bash
  # Convertir clip horizontal a vertical con crop (no rotación)
  ffmpeg -y -i clip_horizontal.mp4 -vf "crop=ih*9/16:ih,scale=1080:1920" clip_vertical.mp4
  ```
- Verificar siempre la orientación antes de usar: `ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 clip.mp4`

#### Flujo de generación:
1. Dividir guión en segmentos (uno por clip)
2. Crear prompt visual para cada segmento con aspectRatio correcto
3. Generar clips en paralelo con Veo 3
4. Descargar cuando estén listos (poll LRO)
5. Escalar a resolución target con `lanczos`
6. Guardar clips limpios (SIN subs, SIN watermark) antes de procesar

### Paso 4: Subtítulos Dinámicos (Word-by-Word)
**Estilo aprobado: resaltado de palabra activa, efecto karaoke estilo TikTok.**

#### Generar timestamps por palabra con Whisper:
```bash
source ~/.openclaw/voice_env/bin/activate
whisper narration_roger.mp3 --model small --language es --output_format json --word_timestamps True --output_dir /tmp/whisper_words/
```

#### Generar ASS con subtítulos dinámicos:
Usar el script `/tmp/dynamic_subs.py` (o recrearlo desde SKILL):

```python
#!/usr/bin/env python3
"""Generate dynamic karaoke-style ASS subtitles from Whisper word timestamps."""
import json, sys

def time_to_ass(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def generate_dynamic_ass(whisper_json_path, output_path, is_vertical=True, words_per_group=4):
    with open(whisper_json_path) as f:
        data = json.load(f)
    
    all_words = []
    for seg in data['segments']:
        for w in seg.get('words', []):
            word = w['word'].strip()
            if word:
                all_words.append({'word': word, 'start': w['start'], 'end': w['end']})
    
    if is_vertical:
        play_res_x, play_res_y = 1080, 1920
        normal_size, highlight_size, margin_v = 55, 70, 350
    else:
        play_res_x, play_res_y = 1920, 1080
        normal_size, highlight_size, margin_v = 45, 58, 80
    
    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans Bold,{normal_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for i, word in enumerate(all_words):
        group_start = max(0, i - 1)
        group_end = min(len(all_words), i + words_per_group)
        parts = []
        for j in range(group_start, group_end):
            w = all_words[j]['word']
            if j == i:
                parts.append(f"{{\\c&H00FFFF&\\fs{highlight_size}\\b1}}{w}{{\\c&HFFFFFF&\\fs{normal_size}\\b1}}")
            else:
                parts.append(w)
        text = " ".join(parts)
        events.append(f"Dialogue: 0,{time_to_ass(word['start'])},{time_to_ass(word['end'])},Default,,0,0,0,,{text}")
    
    with open(output_path, 'w') as f:
        f.write(ass_content + "\n".join(events) + "\n")
    print(f"Generated {len(events)} lines → {output_path}")

if __name__ == "__main__":
    is_v = sys.argv[3] == "vertical" if len(sys.argv) > 3 else True
    wpg = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    generate_dynamic_ass(sys.argv[1], sys.argv[2], is_v, wpg)
```

```bash
python3 dynamic_subs.py whisper_output.json subs_dynamic.ass vertical 4
```

### Paso 5: Ensamble (ffmpeg)
**⚠️ SIEMPRE usar el video base limpio (SIN subtítulos quemados) como fuente.**

#### Short (vertical) con subs dinámicos:
```bash
ffmpeg -y -i short_veo3.mp4 -i narration_roger.mp3 \
  -vf "ass=subs_dynamic.ass" \
  -map 0:v -map 1:a -c:v libx264 -crf 23 -preset fast -c:a aac -shortest \
  short_no_wm.mp4
```

#### Video principal (horizontal) con subs dinámicos:
```bash
ffmpeg -y -i video_clean.mp4 -i narration_roger.mp3 \
  -vf "ass=subs_dynamic.ass" \
  -map 0:v -map 1:a -c:v libx264 -crf 23 -preset fast -c:a aac -shortest \
  video_no_wm.mp4
```

### Paso 6: Marca de Agua (Watermark)
**OBLIGATORIO en TODOS los videos de BitTrader.**
- **Logo**: `/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png`
- **Posición**: Esquina superior derecha
- **Opacidad**: 75% (`aa=0.75`)
- **Tamaño**: 180px horizontal / 120px vertical
- **Filtro**: `lanczos` (mejor calidad, evita pixelado)

```bash
# Horizontal (1920x1080)
ffmpeg -y -i video_no_wm.mp4 \
  -i /home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png \
  -filter_complex "[1:v]scale=180:-1:flags=lanczos,format=rgba,colorchannelmixer=aa=0.75[logo];[0:v][logo]overlay=W-w-25:20" \
  -c:v libx264 -crf 23 -preset fast -c:a copy video_final.mp4

# Vertical (1080x1920)
ffmpeg -y -i short_no_wm.mp4 \
  -i /home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png \
  -filter_complex "[1:v]scale=120:-1:flags=lanczos,format=rgba,colorchannelmixer=aa=0.75[logo];[0:v][logo]overlay=W-w-20:20" \
  -c:v libx264 -crf 23 -preset fast -c:a copy short_final.mp4
```

### Paso 7: Preview para Telegram
```bash
# Horizontal (comprimir a 720p)
ffmpeg -y -i video_final.mp4 -c:v libx264 -crf 32 -preset fast -vf "scale=1280:720" -c:a aac -b:a 96k preview_tg.mp4

# Vertical (solo comprimir, mantener resolución)
ffmpeg -y -i short_final.mp4 -c:v libx264 -crf 30 -preset fast -c:a aac -b:a 96k preview_tg.mp4
```
Target: <16MB para enviar por Telegram.

### Paso 8: Upload a YouTube
- **Credenciales**: `/home/enderj/.openclaw/workspace/memory/veo_credentials.json`
- **Canal**: @bittrader9259 | ID: `UCA6fx-l4lwysKEmtq39XUdw`
- Subir SIEMPRE en full quality (no comprimida)
- Idioma: `es` | Categoría: `22` | `selfDeclaredMadeForKids: false`
- **Mejores horas (Mountain Time)**: Shorts a las 10AM y 2PM | Videos largos a las 10AM
- **NO publicar todos juntos** — escalonar con 4h+ entre cada uno
- Se puede subir como `private` y luego cambiar a `public` con cron

---

## Estructura de Archivos

```
bittrader/
├── assets/
│   └── bittrader_logo_transparent.png   # Logo PNG RGBA oficial
└── [fecha_DDMMM]/
    ├── PLAN.md                           # Guiones y metadata
    ├── video_principal/
    │   ├── narration_roger.mp3
    │   ├── clips_veo3/
    │   │   ├── clip_01.mp4              # Clips limpios (16:9)
    │   │   └── concatenated.mp4         # Concatenado SIN subs
    │   ├── subs_dynamic.ass             # Subtítulos dinámicos
    │   └── video_principal_final.mp4    # CON subs + watermark
    ├── short1_[nombre]/
    │   ├── narration_roger.mp3
    │   ├── short1_veo3.mp4              # Clip limpio (9:16)
    │   ├── subs_dynamic.ass
    │   └── short1_final.mp4             # CON subs + watermark
    ├── short2_[nombre]/
    └── short3_[nombre]/
```

---

## Lecciones Clave

### Subtítulos
- ASS FontSize escala según PlayResY. En 1920px alto, usar normal=55 highlight=70. En 1080px alto, usar normal=45 highlight=58.
- **NUNCA** poner fondo negro (`BorderStyle=1`, no `3`)
- **NUNCA** aplicar subtítulos sobre video que ya tiene subtítulos quemados — usar base limpia
- Verificar siempre con screenshot antes de enviar

### Voz Roger
- La misma voz para CHAT y para VIDEOS — es la voz oficial de BitTrader
- JSON de ElevenLabs no acepta `\n` literal — usar espacio en su lugar
- ~100 chars/seg de audio

### Veo 3
- Especificar `aspectRatio` desde el inicio (9:16 o 16:9)
- Un clip horizontal usado en short vertical debe hacerse con CROP, no rotación
- Generación ~2-5 min/clip, generar en paralelo
- Escalar con `lanczos` para mejor calidad

### Watermark
- Logo: `/home/enderj/.openclaw/workspace/bittrader/assets/bittrader_logo_transparent.png`
- Tamaño real: 500x500. Escalar a 180px/120px con lanczos para evitar pixelado
- Opacidad 75% — visible pero no invasivo

### YouTube
- Escalonar publicaciones
- Shorts privados → público con cron jobs programados
- Analytics: revisar vistas por video con YouTube Data API v3
