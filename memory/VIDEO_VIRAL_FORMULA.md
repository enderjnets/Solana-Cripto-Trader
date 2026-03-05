# Fórmula de Video Viral — Aprendizaje del Video Cochinilla (70M views)

---

## 🎬 PIPELINE TÉCNICO CORRECTO — Subtítulos Karaoke ASS

**Fecha aprendizaje:** 2026-03-04
**Problema resuelto:** Subtítulos duplicados / desincronizados

### ❌ NUNCA hacer esto (genera duplicados):
```python
# MAL: Una línea por cada palabra activa — crea 252 líneas simultáneas
for j, word in enumerate(group):
    parts = []
    for k, w in enumerate(group):
        if k == j:
            parts.append(f"{{\\rActive}}{w['word']}{{\\rNormal}}")
        else:
            parts.append(w["word"])
    # Genera una línea completa para cada palabra → SUPERPOSICIÓN VISUAL
```

### ✅ SIEMPRE usar karaoke nativo `\kf` de ASS:
```python
# BIEN: Una sola línea por grupo, cada palabra con su duración
for group in groups:
    line_start = group[0]["start"]
    line_end = group[-1]["end"]
    text = ""
    for w in group:
        duration_cs = int((w["end"] - w["start"]) * 100)  # centisegundos
        text += f"{{\\kf{duration_cs}}}{w['word']} "
    # UNA sola línea de diálogo — sin duplicados
    ass.append(f"Dialogue: 0,{ts(line_start)},{ts(line_end)},Karaoke,,0,0,0,,{text.strip()}")
```

### Style correcto para karaoke:
```
Style: Karaoke,Arial,52,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,40,1
```
- `PrimaryColour`: Color del texto inactivo = Amarillo (`&H0000FFFF`)
- `SecondaryColour`: Color de la palabra activa al resaltarse = Blanco (`&H00FFFFFF`)
- `\kf` = karaoke fill (gradual, más suave que `\k`)
- Grupos de 6 palabras máximo por línea

### Pipeline completo (orden correcto):
1. **Transcribir** audio con Whisper (`word_timestamps=True`) → `whisper_words.json`
2. **Generar ASS** con `\kf` nativo → `subtitulos.ass`
3. **Concatenar clips** limpios (sin audio) → `/tmp/clips_v.mp4`
4. **Añadir audio** de narración → `/tmp/video_audio.mp4`
5. **Quemar subtítulos** (UNA sola vez) → `video_final_720p.mp4`
6. **Upload YouTube** con API (credenciales en `memory/youtube_credentials.json`)

### ⚠️ Error crítico a evitar:
- NO usar `video_final.mp4` como input del paso de subtítulos (ya tiene subs quemados)
- SIEMPRE partir desde los **clips originales limpios** del directorio `clips/`

---

*Extraído: 2026-03-04 | Análisis por Claude Code*

---

## 🎬 REGLA #1 — Hook en los primeros 3 segundos

- Abrir con imagen visual misteriosa SIN explicar qué es
- Silencio total o sonido ambiente al inicio (NO música de entrada inmediata)
- Crear un **PUZZLE VISUAL** — algo que no se entiende a primera vista
- Mostrar el resultado final PRIMERO antes de explicar cómo llegaste ahí
- Manos sobre objeto/pantalla = anticipación de acción inminente

**Aplicado al trading:**
- Abrir con bot ejecutando trades en vivo, números verdes, gráfico subiendo
- Sin texto, sin explicación — solo la imagen
- El espectador piensa: "¿Qué está pasando aquí?" → no puede irse

---

## 📖 ESTRUCTURA NARRATIVA — Arco Viral

**Misterio → Revelación → Proceso → Drama → Resultado/Legado**

| Segmento | Contenido | Duración |
|---|---|---|
| Misterio | Imagen impactante sin contexto | 0:00 - 0:05 |
| Revelación | "Esto lo hizo una IA, no un humano" | 0:05 - 0:20 |
| Proceso | Paso a paso (respeto por el oficio) | 0:20 - 2:30 |
| Drama | "Casi falla aquí, pero..." | 2:30 - 4:00 |
| Resultado | Resultado + CTA suscripción | 4:00 - 5:00 |

---

## 🎨 TÉCNICAS VISUALES

- **Close-ups** del tema central = íntimo, conecta con el espectador
- **Mix de tomas**: estáticas para impacto + dinámicas para acción
- **Profundidad de campo reducida** — foco en lo importante, fondo borroso
- **Color grading**: colores que refuercen el tema (trading = verde/negro)
- **Texto en pantalla**: posición inferior, refuerza términos clave
- **Transiciones temáticas**: el objeto del video conecta una escena con la siguiente

---

## 🎵 DISEÑO DE AUDIO

- Música ambient/sutil al inicio → crece a medida que avanza la narrativa
- Efectos de sonido sutiles pero precisos (clicks, teclado, notificaciones)
- Narración: clara, calmada, autoritaria — **estilo documental**
- Sincronización perfecta: narración + visual + efectos al mismo tiempo
- **Voz**: Fernando Martínez (ElevenLabs, dlGxemPxFMTY7iXagmOj) — rápido, persuasivo

## 🖼️ THUMBNAIL — Plantilla Oficial BitTrader (APROBADA 2026-03-04)
- **Script**: `videos/bot-trading-claude/crear_thumbnail.py` — usar como base para todos los videos
- **Estilo**: Fondo negro, diagonal dorada sutil, texto izquierda, personaje derecha
- **Colores**: Dorado `#F5A623`, Blanco, Negro `#0F0F0F`
- **Textos**: Líneas 1-2 dorado (100px), Líneas 3-4 blanco (78px), subtítulo dorado (36px)
- **Logo**: BitTrader arriba izquierda (`BIBLIOTECA/bittrader_logo.png`)
- **Mascota**: Rinoceronte corriendo con tablet (`BIBLIOTECA/bittrader_rhino_running.jpg`)
- **Panel stats**: Arriba derecha con métricas del video (opcional pero efectivo)
- **Botón**: WATCH NOW rojo+dorado abajo izquierda
- **Resolución**: 1280x720 JPEG quality 95

---

## 🧠 TRIGGERS EMOCIONALES (los 5 que hacen viral)

1. **Curiosidad** — pregunta sin respuesta inmediata en el hook
2. **Asombro** — mostrar algo que el espectador no sabía que existía
3. **Respeto** — proceso laborioso, conocimiento especializado
4. **Drama/Intriga** — un conflicto o problema en el camino
5. **Orgullo/Identidad** — que la audiencia se sienta parte de algo

**Aplicado al trading:**
- Curiosidad: "¿Una IA puede realmente ganar dinero sola?"
- Asombro: Ver el bot ejecutar trades en tiempo real
- Respeto: Mostrar el proceso técnico real (no magia)
- Drama: "En este trade casi pierde todo..."
- Identidad: "Tú también puedes hacer esto"

---

## ✅ CHECKLIST PRE-PRODUCCIÓN (aplicar a cada video)

- [ ] ¿Los primeros 3 segundos crean un puzzle visual?
- [ ] ¿Empezamos sin música (silencio o ambient)?
- [ ] ¿Mostramos el resultado ANTES de explicar el proceso?
- [ ] ¿Hay un momento de drama o conflicto en el video?
- [ ] ¿El audio (música + narración + efectos) están sincronizados?
- [ ] ¿Los subtítulos destacan la palabra activa más grande?
- [ ] ¿El CTA final es específico y tiene urgencia?

---

## 📌 CONFIGURACIÓN TÉCNICA DEL PIPELINE

- **Voz**: Alberto Rodríguez (ElevenLabs, l1zE9xgNpUTaQCZzpNJa)
- **Clips de video**: Veo 3 (Google Vertex AI)
- **Subtítulos**: Amarillos, posición inferior, borde negro (Outline=2), SIN fondo
- **Efecto karaoke**: Palabra activa más grande/destacada en tiempo real
- **Ensamble**: ffmpeg
- **Formato**: Faceless (screen recording + voz en off + clips Veo 3)

---

*Fuente: Análisis del video viral "Cochinilla/Tinte Rojo" (70M+ views en Facebook)*
