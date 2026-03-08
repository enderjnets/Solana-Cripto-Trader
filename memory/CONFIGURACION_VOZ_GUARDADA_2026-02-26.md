# Configuración de Voz - Guardada para Futuro Uso

## Voz Actual Configurada

### Español
- **Nombre:** Roger
- **ID:** CwhRBWXzGAHq8TQ4Fs17
- **Género:** Masculina
- **Descripción:** Laid-Back, Casual, Resonant
- **Modelo:** eleven_multilingual_v2
- **Proveedor:** ElevenLabs

### Inglés
- **Nombre:** Sarah
- **ID:** EXAVITQu4vr4xnSDxMaL
- **Género:** Femenina
- **Modelo:** eleven_flash_v2
- **Proveedor:** ElevenLabs

## Sistema Funcionando

**Archivo:** `voice_system_api.py`
- **Ubicación:** `/home/enderj/.openclaw/workspace/voice_system_api.py`
- **Estado:** ✅ Funcionando
- **API:** ElevenLabs API directa (via requests)

**Comando para generar notas de voz:**
```bash
python3 /home/enderj/.openclaw/workspace/voice_system_api.py
```

**API Endpoint:**
```
https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
```

## Configuración de Archivo

```python
VOICES = {
    "en": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah
        "model": "eleven_flash_v2",
        "name": "Sarah"
    },
    "es": {
        "voice_id": "CwhRBWXzGAHq8TQ4Fs17",  # Roger - CONFIGURADO POR USUARIO
        "model": "eleven_multilingual_v2",
        "name": "Roger"
    },
}
```

## Uso Futuro

### Para generar notas de voz en español:
```bash
cd /home/enderj/.openclaw/workspace
python3 -c "
from voice_system_api import text_to_speech
text = 'Tu texto aquí'
audio = text_to_speech(text, lang='es')
print(audio)
"
```

### Para generar notas de voz en inglés:
```bash
python3 -c "
from voice_system_api import text_to_speech
text = 'Your text here'
audio = text_to_speech(text, lang='en')
print(audio)
"
```

## Características de la Voz

**Roger (Español - Masculina):**
- Tono: Relajado y casual
- Acento: Español neutro/americano
- Uso recomendado: Conversaciones, explicaciones, notas de voz
- Calidad: Excelente (ElevenLabs API)

## API Key

**Guardada en:** `voice_system_api.py`
**No cambiar a menos de ser necesario:**
```
98b4670ec5f8658b81fbcf0d9a685b3f3348ac6420a6be261ee7d5ceaf656a8f
```

## Limitación Importante

**Nota:** El sistema integrado de OpenClaw (`tts` tool) puede NO usar esta configuración.

**Para asegurar notas de voz con Roger:**
1. Usar `voice_system_api.py` directamente
2. O especificar en OpenClaw que use Roger (CwhRBWXzGAHq8TQ4Fs17)

## Próximos Pasos

- [x] Configurar Roger como voz masculina en español
- [x] Probar generación de audio
- [x] Enviar nota de voz corregida
- [x] Usuario confirmó: "Siempre hazla con Roger"
- [x] Documentar preferencia permanente
- [ ] Actualizar sistema integrado de OpenClaw (si es necesario)
- [ ] Documentar para futuros desarrolladores

## Preferencia Permanente del Usuario

🔔 **REGLA DE ORO:**

**El usuario ha confirmado explícitamente:**
> "Siempre hazla con Roger"

**Esto significa:**
- ✅ ROGER SIEMPRE para notas de voz en español
- ✅ No confirmar con el usuario antes de usar Roger
- ✅ Usar automáticamente Roger cada vez
- ✅ No preguntar "¿Qué voz prefieres?"

**Documentación:**
- `memory/PREFERENCIA_VOZ_ROGER_SIEMPRE.md` - Preferencia permanente guardada

---

**Fecha de configuración:** 2026-02-26
**Voz preferida por usuario:** Roger (masculina, español)
**Estado:** ✅ Guardada y probada
