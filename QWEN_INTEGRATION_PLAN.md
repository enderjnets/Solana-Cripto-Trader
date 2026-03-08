# Plan de Integración Qwen2.5 14B + OpenClaw/Eko Rog

## Arquitectura: Claude + Qwen en paralelo

```
Usuario → Telegram → Eko Rog (OpenClaw)
                          │
                    ┌─────┴──────┐
                    │            │
              Claude Sonnet   Qwen2.5 14B
              (complejo/       (repetitivo/
               creativo)       monitoreo)
```

---

## Fase 1 — Base (LISTO ✅)
- Ollama instalado con detección RTX 3070
- Qwen2.5 14B descargado (9GB, Q4 cuantizado)
- API local corriendo: http://localhost:11434

---

## Fase 2 — Integración con Scripts de BitTrader

### 2a. creator.py — Guiones con Qwen
Modificar para usar Qwen como opción de modelo:
```python
def call_llm(prompt, use_local=False):
    if use_local:
        # Ollama local (gratis, ilimitado)
        r = requests.post("http://localhost:11434/api/generate",
            json={"model": "qwen2.5:14b", "prompt": prompt, "stream": False})
        return r.json()["response"]
    else:
        # Claude Sonnet (complejo/creativo)
        return call_claude(prompt)
```

**Cuándo usar Qwen en creator.py:**
- Investigación de noticias (scout)
- Variaciones de títulos/tags
- Primeros borradores de guiones

**Cuándo usar Claude en creator.py:**
- Guión final (creatividad, humor, ritmo)
- Rhino Battles (contenido premium)

### 2b. ken_burns_producer.py — Sin cambios
(No usa LLM, solo Flux.1 + ffmpeg)

---

## Fase 3 — Integración con OpenClaw

### 3a. Qwen como modelo de Cron Jobs
Configurar los cron jobs de monitoreo para usar Qwen en vez de Claude:

```json
{
  "payload": {
    "kind": "agentTurn",
    "model": "ollama/qwen2.5:14b",
    "message": "Verifica estado de videos YouTube..."
  }
}
```

**Jobs que van a Qwen (monitoreo repetitivo):**
- YouTube Monitor (cada 6h)
- Email check (cada 4h)
- Solana bot alerts (cada 30min)
- System health check

**Jobs que van a Claude (decisiones importantes):**
- Análisis de métricas de canal
- Generación de nuevo contenido
- Respuestas complejas al usuario

### 3b. Configuración OpenClaw (openclaw.json)
Agregar Ollama como proveedor:
```json
{
  "providers": {
    "ollama": {
      "baseUrl": "http://localhost:11434",
      "models": ["qwen2.5:14b"]
    }
  }
}
```

---

## Fase 4 — Fallback Automático

Cuando Claude llega al límite de rate o tokens:
1. OpenClaw detecta el error
2. Auto-switch a `ollama/qwen2.5:14b`
3. Notifica al usuario: "Usando Qwen local (Claude en límite)"
4. Cuando Claude vuelve, regresa automáticamente

---

## Distribución de Tareas (Regla de Oro)

| Tarea | Modelo | Razón |
|-------|--------|-------|
| Chat directo con usuario | Claude | Calidad, personalidad |
| Rhino Battles / videos premium | Claude | Creatividad máxima |
| Guiones simples / noticias | Qwen | Repetitivo, barato |
| Monitoreo YouTube | Qwen | Solo verificar datos |
| Revisar emails | Qwen | Filtrar, resumir |
| Alertas Solana bot | Qwen | Solo formatear datos |
| Análisis técnico trading | Qwen | Datos estructurados |
| Decisiones importantes | Claude | Juicio crítico |
| Debugging código complejo | Claude | Razonamiento profundo |
| Documentar funciones simples | Qwen | Tarea mecánica |

---

## Ahorro estimado de tokens Claude

| Tarea actual | Frecuencia | Tokens Claude/día |
|-------------|-----------|-------------------|
| YouTube monitor (cron) | 4x/día | ~2,000 |
| Email check (heartbeat) | 4x/día | ~1,500 |
| Solana alerts | 48x/día | ~4,800 |
| Guiones noticias | 5x/semana | ~10,000/sem |
| **TOTAL** | | **~8,300/día → $0** |

Con Qwen manejando esas tareas = **~$5-8/día ahorro** en tokens Claude.

---

## Siguiente Paso Inmediato

1. Agregar `use_local=True` flag en creator.py para scouts/investigación
2. Actualizar cron de YouTube Monitor para usar Qwen
3. Crear wrapper `qwen_client.py` en `bittrader/agents/`

---

## Estado
- [x] Ollama instalado
- [x] Qwen2.5 14B descargado y funcionando
- [x] API local verificada (localhost:11434)
- [ ] Integración creator.py
- [ ] Wrapper qwen_client.py
- [ ] Crons actualizados a Qwen
- [ ] Fallback automático OpenClaw
