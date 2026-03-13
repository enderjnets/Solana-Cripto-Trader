# Plan de Estrategia — BitTrader (Actualizado 2026-03-13)

## Decisión: Idioma del Canal

### Estrategia Seleccionada
**Opción 1: Público hispanohablante**

- **Títulos**: Español
- **Descripciones**: Español  
- **Guiones**: Español (TTS)
- **SEO**: Hashtags en español (#bitcoin #trading #criptomonedas)
- **Público objetivo**: Hisponohablantes (LatAm + España + hispanos en US)

### Razón de la Decisión
- El contenido está 100% en español (guión + TTS)
- YouTube detecta el idioma del AUDIO, no del título
- Título en inglés + video en español crea desconexión para audiencia
- No mezclar idiomas entre título y contenido

---

## Análisis de Problema: Crecimiento Lento (30 views/video promedio)

### Estadísticas Actuales
- **Subs**: 2,920
- **Total Views**: 178,000
- **Promedio últimos 30 días**: 30 views/video
- **Comentarios**: 0 (cero interacción)

### Problemas Identificados

#### 1. Sin CTAs (Call to Actions)
- Los videos NO piden comentarios, likes, subscripciones
- Sin engagement → algoritmo no recomienda
- Videos con 0 comentarios se perciben como de baja calidad

#### 2. Títulos sin optimización SEO
- Los títulos actuales son informativos pero no "clickbait"
- Falta de curiosidad, controversia, urgencia
- Sin palabras clave estratégicas (#bitcoin #trading #altcoins)

#### 3. Horarios de publicación
- Videos subidos en horarios no optimizados
- Falta de análisis de cuándo está activa la audiencia hispanohablante

#### 4. Sin consistencia
- Publicación irregular
- Sin calendario editorial definido

---

## Plan de Acción 5 Prioridades (Todas en español)

### ✅ 1. Títulos Optimizados en Español (Prioridad ALTA)

**Qué hacer:**
- Generar títulos con:
  - Curiosidad: "¿Por qué el 90% de traders pierde?"
  - Urgencia: "OJO: Este token puede explotar HOY"
  - Controversia: "La verdad de los bots de trading"
  - Números concretos: "3 estrategias que multiplican tu capital"

**Reglas:**
- Máx 60 caracteres (shorts) / 70 caracteres (longs)
- Incluir palabras clave: #bitcoin #trading #crypto
- NO emojis en el título (YouTube ya pone automáticamente)
- Evitar palabras genéricas ("buen video", "información importante")

**Cambios en creator.py:**
```python
# Prompt SHORT en español con optimización de títulos
TEMAS DE IA + TRADING (prioridad alta):
- Agentes de trading con Claude/GPT
- Bots que monitorean noticias en tiempo real
- Comparativas: bots tradicionales vs agentes con IA
- Cómo usar LLMs para analizar patrones de mercado

PUNTOS CLAVE:
- Usa hooks poderosos en títulos (¿Por qué? ¿Cómo? Los 3 secretos...)
- Genera CTA con pregunta para comentarios
- Escribe en español neutro/latino

EJEMPLOS DE TÍTULOS QUE FUNCIONAN:
- "¿Por qué el 90% de traders pierde dinero?"
- "3 cosas que nadie te dice del trading"
- "Este bot me ganó $500 en 1 semana (VERDAD)"
```

### ✅ 2. CTAs en Todos los Videos (Prioridad ALTA)

**Qué hacer:**
- Agregar CTA con pregunta al final de CADA guión
- Ejemplos:
  - "¿Tú qué opinas? Comenta abajo"
  - "¿Has probado esto? Cuéntame tu experiencia"
  - "¿Qué trading usas? Me interesa saber"

**Cambios en creator.py:**
```python
# SYSTEM_PROMPT actualizado
SIEMPRE termina con CTA con una pregunta para los comentarios.
Ejemplos:
- "¿Tú qué opinas? Comenta abajo"
- "¿Has probado esto? Cuéntame tu experiencia"
- "¿Qué trading usas? Me interesa saber"
```

### ✅ 3. Programar Horarios Óptimos (Prioridad MEDIA)

**Qué hacer:**
- Analizar horarios de mayor actividad del público hispanohablante
- Horarios sugeridos:
  - 9:00 AM - 12:00 PM MT (morning)
  - 7:00 PM - 9:00 PM MT (evening)

**Implementar:**
- Agregar programación automática en `publisher.py`
- Opción: `--schedule "09:00,19:00"` para publicar en esos horarios

### ✅ 4. Análisis de Rendimiento (Prioridad MEDIA)

**Qué hacer:**
- Monitorear métricas semanales:
  - Views por video
  - Watch time (retención)
  - CTR (Click-through rate de thumbnails)
  - Engagement (comentarios, likes)

**Herramientas:**
- `youtube_stats.py` ya existe
- Agregar análisis semanal automático
- Identificar patrones de videos exitosos

### ✅ 5. Thumbnails Mejorados (Prioridad BAJA)

**Qué hacer:**
- Crear thumbnails con:
  - Rostros expresivos (rhino emocionado/preocupado)
  - Texto corto y legible
  - Colores de alto contraste
  - Números y porcentajes visuales

**Herramientas:**
- Thumbnail Agent ya existe
- Mejorar prompts para más impacto visual

---

## Implementación: Cambios en creator.py

### Cambios Realizados

1. **SYSTEM_PROMPT** — Añadidas reglas de título y CTA
2. **Prompt SHORT** — Simplificado, enfocado en formato
3. **Prompt LONG** — Simplificado, enfocado en formato
4. **Parser** — Acepta tanto "TITULO" como "TITLE" variants

### Cambios Pendientes

1. Agregar ejemplos de títulos optimizados en español
2. Añadir instrucción explícita de CTA con pregunta
3. Probar sistema con GLM-4.7

---

## Estado Actual

- ✅ Git backup completado
- ✅ Creator.py restaurado a versión original
- ✅ Estrategia de idioma definida (TODO en español)
- ⏳ Ajustar creator.py para títulos optimizados en español
- ⏳ Agregar CTAs automáticos en todos los guiones
- ⏳ Implementar programación de horarios

---

## Próximo Paso

Actualizar creator.py con:
1. Ejemplos de títulos optimizados en español (hooks, curiosidad, números)
2. Instrucción de CTA con pregunta al final de cada guión
3. Test con GLM-4.7 para verificar formato

---

*Fecha: 13 marzo 2026*
*Decisión: Mantener TODO en español (títulos, descripciones, guiones)*
