# Plan de Integración Qwen2.5 14B + OpenClaw/Eko Rog

## Estado: ✅ COMPLETADO (2026-03-08)

## Arquitectura Final

```
Usuario → Telegram → Eko Rog (OpenClaw)
                          │
                    ┌─────┼──────────┐
                    │     │          │
              Sonnet 4.6  Haiku 4.5  Qwen 2.5 14B
              (chat/       (crons/    (análisis/
               creativo)   tools)     resúmenes)
```

---

## Configuración OpenClaw (openclaw.json)

```json
{
  "local": {
    "baseUrl": "http://localhost:11434/v1",
    "apiKey": "ollama-local",
    "api": "openai-completions",
    "injectNumCtxForOpenAICompat": false,
    "models": [{
      "id": "qwen2.5:14b",
      "name": "Qwen 2.5 14B (Local)",
      "contextWindow": 32768,
      "maxTokens": 4096
    }]
  }
}
```

### ⚠️ Reglas críticas de configuración
1. Provider DEBE llamarse `local` (NO `ollama`) — OpenClaw detecta "ollama" + puerto 11434 y fuerza API nativa
2. `injectNumCtxForOpenAICompat: false` es OBLIGATORIO — sin esto, inyecta num_ctx que pide 10.3GB RAM
3. `contextWindow: 32768` mínimo — OpenClaw requiere ≥16K tokens
4. Modelo NO debe tener `api:` override — hereda del provider

---

## Distribución de Modelos (FINAL)

| Tarea | Modelo | Razón | Costo |
|-------|--------|-------|-------|
| Chat directo | Sonnet 4.6 | Calidad, personalidad | $0 (proxy) |
| Full Pipeline (Sáb 2PM) | Sonnet 4.6 | Creatividad para videos | $0 (proxy) |
| Crons con tools | **Haiku 4.5** | Tool calling confiable | $0 (proxy) |
| qwen_client.py | **Qwen Local** | Análisis, resúmenes | $0 (local) |
| creator.py fallback | **Qwen Local** | Último recurso | $0 (local) |

### Por qué Qwen NO puede hacer crons
- Qwen 14B no ejecuta tools de OpenClaw correctamente
- En vez de llamar `exec` para correr scripts, genera texto explicando qué haría
- Haiku 4.5 sí usa tools confiablemente

---

## qwen_client.py — Funciones Disponibles

| Función | Uso | Tiempo | Calidad |
|---------|-----|--------|---------|
| `is_available()` | Verificar si Ollama responde | <1s | ✅ |
| `ask(prompt)` | Preguntas generales | 2-4s | ✅ Buena |
| `ask_json(prompt)` | Respuestas JSON estructuradas | 3-4s | ✅ JSON válido |
| `analyze_news(headline, body)` | Sentimiento + impacto de noticias | 6-8s | ✅ Comparable a Sonnet |
| `summarize(text, max_words)` | Resúmenes concisos | 2-3s | ✅ Buena |
| `generate_title_variations(title)` | 5 variaciones de títulos | 10s | ✅ Creativos |
| `draft_script_section(topic, section, duration)` | Guiones de video | 6s | ✅ Natural |
| `check_video_status(video_info)` | Verificar estado de videos | 9s | ✅ Detecta problemas |

---

## Crons Actuales

| Cron | Modelo | Frecuencia |
|------|--------|-----------|
| YouTube Stats | Haiku 4.5 | 6AM/12PM/6PM |
| YouTube Monitor | Haiku 4.5 | Cada 6h |
| BitTrader Scout | Haiku 4.5 | 8AM/8PM |
| Verificar Pipeline | Haiku 4.5 | Sáb 2:30PM |
| Full Pipeline | Sonnet 4.6 | Sáb 2PM |
| Backup Reminder | systemEvent | 3AM |

---

## Hardware

- **GPU**: NVIDIA RTX 3070 Laptop (8GB VRAM)
- **Qwen usa**: 7.3GB VRAM (de 9GB modelo Q4)
- **Ollama**: servicio systemd (auto-start al boot)
- **Whisper STT**: forzado a CPU (`CUDA_VISIBLE_DEVICES=""`)

---

## Lecciones Aprendidas

1. **Provider naming**: OpenClaw auto-detecta "ollama" → renombrar a otro nombre
2. **RAM vs VRAM**: API nativa inyecta num_ctx → error RAM. openai-completions + injectNumCtx=false → usa VRAM
3. **Mínimo context**: OpenClaw requiere ≥16K tokens → contextWindow debe ser ≥16384
4. **Tool calling**: Modelos <30B no son confiables para tool calling en OpenClaw
5. **GPU sharing**: Qwen + Whisper no caben juntos → Whisper en CPU

---

## Checklist

- [x] Ollama instalado como servicio systemd
- [x] Qwen2.5 14B descargado y funcionando
- [x] Provider `local` configurado en OpenClaw
- [x] `qwen_client.py` con 8 funciones operativas
- [x] `creator.py` con Qwen como fallback #4
- [x] Crons migrados a Haiku 4.5 (tool calling)
- [x] `/model Qwen Local` disponible en Telegram
- [x] Auditoría completa — todo verde
- [x] Documentación actualizada (MEMORY.md, TOOLS.md, memory/2026-03-08.md)
