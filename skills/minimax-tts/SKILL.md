# MiniMax TTS Skill

Use MiniMax Speech 2.0 API for text-to-speech.

## API
- **Endpoint**: `https://api.minimax.io/v1/t2a_v2`
- **Models**: `speech-02-hd` (high quality), `speech-02-turbo` (fast)
- **Voices**: `male_1`, `male_2`, `female_1`, `female_2`
- **Format**: MP3, 32kHz, mono, 128kbps
- **API Key**: Configurado en `tts.py`

## Voces Disponibles
| ID | Descripción |
|----|-------------|
| `male_1` | Masculino, voz clara |
| `male_2` | Masculino, alternativo |
| `female_1` | Femenino, voz clara |
| `female_2` | Femenino, alternativo |

## Usage

```bash
# Básica
python3 /home/enderj/.openclaw/workspace/skills/minimax-tts/tts.py "Hola Ender" /tmp/audio.mp3

# Con voz específica
python3 /home/enderj/.openclaw/workspace/skills/minimax-tts/tts.py --voice female_1 "Hola" /tmp/audio.mp3

# Velocidad
python3 /home/enderj/.openclaw/workspace/skills/minimax-tts/tts.py --speed 0.9 "Hola" /tmp/audio.mp3
```

## Plan Suscrito (2026-03-30)
- **Plan**: Standard $30/mes
- **Credits**: 300,000 por mes
- **RPM**: 50 requests/min
- **Voice slots**: 100
- **Modelos**: T2A v2, T2A large v2

## Configurar como TTS de OpenClaw
Para usar esta voz con el comando `[[tts:...]]`, configurar en OpenClaw como skill de TTS o usar el script directamente.
