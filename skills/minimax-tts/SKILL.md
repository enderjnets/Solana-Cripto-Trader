# MiniMax TTS Skill

Text-to-Speech usando MiniMax Speech API.

**Docs**: https://platform.minimax.io/docs/api-reference/speech-t2a-intro

---

## Plan Standard ($30/mes) — Activado 2026-03-30
- **300,000 credits/mes**
- **RPM**: 50 requests/min
- **Characters/request**: hasta 10,000

---

## Modelos Disponibles

| Modelo | Descripción | Recomendado |
|--------|-------------|-------------|
| `speech-2.8-hd` | **Latest HD** — máxima calidad | ✅ Default |
| `speech-2.8-turbo` | Latest Turbo — rápido | |
| `speech-2.6-hd` | HD con prosodia excelente | |
| `speech-2.6-turbo` | Turbo, 40 idiomas | |
| `speech-02-hd` | Ritmo y estabilidad superior | |
| `speech-02-turbo` | Multilingual mejorado | |

---

## Voces Disponibles

| ID | Descripción |
|----|-------------|
| `male_1` | ⭐ Masculino claro (default) |
| `male_2` | Masculino alternativo |
| `female_1` | Femenino claro |
| `female_2` | Femenino alternativo |

**40 idiomas soportados**: Español, Inglés, Português, Français, Deutsch, 中文, 日本語, 한국어, etc.

---

## Formatos de Audio
`mp3` (default), `pcm`, `flac`, `wav` (solo non-streaming)

---

## Uso

```bash
# Básico
python3 tts.py "Hola Ender, soy Eko." /tmp/audio.mp3

# Con voz específica
python3 tts.py --voice female_1 "Buenos días" /tmp/audio.mp3

# Modelo y velocidad
python3 tts.py --model speech-02-turbo --speed 0.9 "Texto rápido" /tmp/audio.mp3

# Formato WAV (no-streaming)
python3 tts.py --format wav "Hola" /tmp/audio.wav

# Ajustes
python3 tts.py --speed 1.2 --volume 1.5 "Más rápido y fuerte" /tmp/out.mp3
```

**Parámetros:**
- `--speed, -s`: Velocidad 0.5–2.0 (default: 1.0)
- `--volume`: Volumen 0.0–2.0 (default: 1.0)
- `--pitch`: Tono -500 a 500 (default: 0)

---

## Integración con OpenClaw

Para usar con el comando `[[tts:...]]` de OpenClaw, el audio debe generarse y enviarse manualmente.
