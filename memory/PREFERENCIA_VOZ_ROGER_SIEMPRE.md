# Preferencia de Voz del Usuario - Sebastián (Español Venezolano)

## Configuración Permanente

**Fecha de configuración:** 2026-03-01 (actualizado)
**Usuario:** Ender (@enderj)

### Voz para Español: ROGER (ElevenLabs) — SIEMPRE

**Voz preferida:**
- **Nombre:** Roger
- **ID:** CwhRBWXzGAHq8TQ4Fs17
- **Modelo:** eleven_multilingual_v2
- **Género:** Masculina
- **Proveedor:** ElevenLabs (API)
- **Créditos renuevan:** 13 de marzo 2026
- **Nota:** Usuario pidió explícitamente usar ElevenLabs Roger para TODAS las notas de voz (2026-03-02)
- **Fallback:** Sebastián (es-VE-SebastianNeural, Edge TTS) solo si se acaban los créditos

### Voz para Inglés: Sarah (ElevenLabs, cuando haya créditos)

- **Nombre:** Sarah
- **ID:** EXAVITQu4vr4xnSDxMaL
- **Género:** Femenina

### Voz anterior (backup): Roger (ElevenLabs)
- **Nota:** ElevenLabs créditos se renuevan el 13 de marzo 2026
- Usar solo cuando haya créditos disponibles

## Regla de Oro

🔔 **REGLA IMPORTANTE:**

**Cuando el usuario pida una nota de voz en español:**
1. SIEMPRE usar Sebastián (es-VE-SebastianNeural) via Edge TTS
2. Comando: `source ~/.openclaw/voice_env/bin/activate && edge-tts --voice "es-VE-SebastianNeural" --text "TEXTO" --write-media /tmp/voice_output.mp3`
3. Enviar con `message` tool (filePath + asVoice=true)
4. NO usar gTTS (voz robótica)
5. NO usar ElevenLabs hasta que haya créditos
6. **PRONUNCIACIÓN:** Escribir "Énder" (con tilde) en el texto TTS para que pronuncie la R correctamente

### Proceso para Generar Notas de Voz

1. **Detectar idioma del texto**
2. **Si es español:**
   - Usar `voice_system_api.py` con Roger
   - Confirmar en el mensaje: "Voz: Roger (masculina, español)"
3. **Si es inglés:**
   - Usar Sarah
   - Confirmar en el mensaje: "Voz: Sarah (femenina, inglés)"

### Implementación

**Sistema a usar:** `/home/enderj/.openclaw/workspace/voice_system_api.py`

**Comando:**
```bash
python3 -c "
from voice_system_api import text_to_speech
text = 'Tu texto aquí'
audio = text_to_speech(text, lang='es')
print(audio)
"
```

### Configuración del Sistema

```python
VOICES = {
    "en": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah
        "model": "eleven_flash_v2",
        "name": "Sarah"
    },
    "es": {
        "voice_id": "CwhRBWXzGAHq8TQ4Fs17",  # Roger - PREFERIDO DEL USUARIO
        "model": "eleven_multilingual_v2",
        "name": "Roger"
    },
}
```

## Ejemplos de Confirmación

**Cuando envíe una nota de voz:**
```
🎧 Nota de voz: [Tema de la nota]
(Voz: Roger - masculina en español con ElevenLabs)
```

## Referencias

- **Archivo de configuración:** `/home/enderj/.openclaw/workspace/voice_system_api.py`
- **Documentación:** `memory/CONFIGURACION_VOZ_GUARDADA_2026-02-26.md`
- **Prueba exitosa:** 2026-02-26 19:46 MST (Mensaje ID: 4079)

## Historial de Ajustes

- **2026-02-26:** Usuario confirmó "Siempre hazla con Roger"
- **2026-02-26:** Configuración creada y probada
- **2026-02-26:** Primera nota de voz con Roger enviada (Mensaje ID: 4079)

---

**ESTA ES LA CONFIGURACIÓN PERMANENTE.**
**CAMBIAR SOLO SI EL USUARIO LO SOLICITA EXPLÍCITAMENTE.**
