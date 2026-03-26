# 🤖 Eco AI Automation — Sistema de Prospección Autónomo

Sistema completo para prospección outbound y demos personalizadas del AI Voice Agent service.

**Stack:** Python + VAPI API  
**Última actualización:** 2026-03-24  
**Versión:** 1.0

---

## 📋 ARCHIVOS DEL SISTEMA

| Archivo | Función |
|---------|---------|
| `create_demo_assistant.py` | Crea assistants VAPI personalizados por negocio |
| `prospecting_assistant.py` | Crea el assistant de prospección outbound "Eko" |
| `outreach_caller.py` | Ejecuta llamadas outbound automatizadas |
| `crm_tracker.py` | CRM para tracking de prospectos |
| `crm_prospects.md` | Base de datos de prospectos (editable a mano) |
| `voice_agent_sales_plan.md` | Plan completo de ventas + scripts |
| `calls_log.json` | Log automático de llamadas (generado) |
| `crm_data.json` | CRM en formato JSON (generado) |

---

## 🚀 SETUP INICIAL (Hacer solo una vez)

### 1. Verificar dependencias

```bash
cd /home/enderj/.openclaw/workspace/eco
python3 -c "import requests; print('✅ requests OK')"
```

Si falla: `pip3 install requests`

### 2. Crear el Prospecting Assistant en VAPI

```bash
python3 prospecting_assistant.py
```

Guarda el **Assistant ID** que devuelve. Luego edita `outreach_caller.py`:

```python
# Línea ~50 en outreach_caller.py
PROSPECTING_ASSISTANT_ID = "PEGA_EL_ID_AQUI"
```

### 3. Probar la demo existente

Llama al número de Eko directamente:
```
+1 (720) 824-9313
```

---

## 📞 FLUJO DE USO DIARIO

### Paso 1 — Agregar teléfonos a los prospectos

Busca los teléfonos en Google Maps y agrégatlos:

```bash
# Opción A: editar crm_prospects.md directamente
# Opción B: usar el CRM tracker
python3 crm_tracker.py --add-phone 1 +13035551234
python3 crm_tracker.py --add-phone 4 +17205559876
```

Ver quién tiene teléfono y quién no:
```bash
python3 crm_tracker.py --ready
```

### Paso 2 — Ver estado del CRM

```bash
# Resumen ejecutivo
python3 crm_tracker.py

# Lista completa
python3 crm_tracker.py --list

# Solo los PENDIENTES
python3 crm_tracker.py --list PENDIENTE
```

### Paso 3 — Hacer llamadas de prospección

**Primero prueba con dry-run (sin llamadas reales):**
```bash
python3 outreach_caller.py --dry-run
```

**Luego ejecuta campaña real (5 llamadas):**
```bash
python3 outreach_caller.py --mode prospect --max-calls 5
```

**Llamada individual de prueba:**
```bash
python3 outreach_caller.py --single \
  --phone +13035551234 \
  --business "Salon Sorella" \
  --industry salon
```

### Paso 4 — Crear demo personalizada para un prospecto interesado

Cuando alguien dice "sí, quiero escuchar la demo":

```bash
# Crear assistant personalizado
python3 create_demo_assistant.py --name "Salon Sorella" --type salon

# Hacer la llamada de demo directamente
python3 outreach_caller.py --mode demo \
  --phone +13035551234 \
  --business "Salon Sorella" \
  --industry salon
```

### Paso 5 — Actualizar el CRM después de cada contacto

```bash
# Marcar como contactado
python3 crm_tracker.py --update 1 CONTACTADO --note "Llamé, no contestó"

# Marcar que respondió
python3 crm_tracker.py --update 1 RESPONDIÓ --note "Interesada, quiere demo"

# Marcar demo realizada
python3 crm_tracker.py --update 1 DEMO --note "Demo exitosa, enviando propuesta"

# Marcar cerrado
python3 crm_tracker.py --update 1 CERRADO --note "Setup $500 pagado. Go-live 2026-04-01"
```

### Paso 6 — Ver métricas

```bash
python3 crm_tracker.py --stats
python3 outreach_caller.py --status
```

---

## 🎯 COMANDOS DE REFERENCIA RÁPIDA

### create_demo_assistant.py

```bash
# Crear demo para salón (español)
python3 create_demo_assistant.py --name "Salon Sorella" --type salon

# Crear demo para dental (inglés)
python3 create_demo_assistant.py --name "Denver Smiles Dental" --type dental --lang en

# Ver tipos disponibles
python3 create_demo_assistant.py --list-types

# Preview del prompt sin crear
python3 create_demo_assistant.py --name "Mi Negocio" --type spa --preview

# Listar todos los assistants en VAPI
python3 create_demo_assistant.py --list

# Eliminar assistant
python3 create_demo_assistant.py --delete ASSISTANT_ID
```

### prospecting_assistant.py

```bash
# Crear assistant de prospección
python3 prospecting_assistant.py

# Ver el system prompt
python3 prospecting_assistant.py --show-prompt

# Actualizar assistant existente
python3 prospecting_assistant.py --update ASSISTANT_ID
```

### outreach_caller.py

```bash
# Campaña de prospección (5 llamadas, 60s entre cada una)
python3 outreach_caller.py --mode prospect

# Campaña con más llamadas
python3 outreach_caller.py --mode prospect --max-calls 10 --interval 30

# Demo personalizada
python3 outreach_caller.py --mode demo --phone +1XXX --business "Nombre" --industry salon

# Llamada individual
python3 outreach_caller.py --single --phone +1XXX --business "Nombre" --industry dental

# Dry run (sin llamadas reales)
python3 outreach_caller.py --dry-run

# Ver estado de llamadas VAPI
python3 outreach_caller.py --status

# Ver prospectos con/sin teléfono
python3 outreach_caller.py --prospects
```

### crm_tracker.py

```bash
# Resumen
python3 crm_tracker.py

# Lista todos
python3 crm_tracker.py --list

# Filtrar por estado
python3 crm_tracker.py --list RESPONDIÓ

# Actualizar estado
python3 crm_tracker.py --update 5 CONTACTADO

# Actualizar con nota y teléfono
python3 crm_tracker.py --update 5 CONTACTADO --note "Dejé mensaje" --phone +13035551234

# Agregar teléfono
python3 crm_tracker.py --add-phone 5 +13035551234

# Estadísticas de conversión
python3 crm_tracker.py --stats

# Ver quién está listo para llamar
python3 crm_tracker.py --ready

# Exportar a JSON
python3 crm_tracker.py --export
```

---

## 🏭 INDUSTRIAS DISPONIBLES

| Código | Industria |
|--------|-----------|
| `salon` | Salón de belleza |
| `spa` | Spa / Salón de spa |
| `dental` | Clínica dental |
| `chiropractic` | Quiropráctica |
| `insurance` | Agencia de seguros |
| `medspa` | Med spa / Estética médica |
| `massage` | Masajes / Wellness |
| `lash` | Extensiones de pestañas |
| `esthetics` | Faciales / Estética |
| `dermatology` | Dermatología |
| `veterinary` | Veterinaria |
| `auto` | Auto detailing |
| `tax` | Impuestos / Contabilidad |

---

## 🔑 CREDENCIALES VAPI

| Variable | Valor |
|----------|-------|
| VAPI Private Key | `f361bb66-8274-403a-8c0c-b984d7dd1cee` |
| Phone Number ID | `64fcd5de-ab68-4ae0-93f6-846ce1209cce` |
| Eko Assistant ID | `225a9f9f-5d58-412a-b8df-81b72c799a4a` |
| Número de teléfono | `+1 (720) 824-9313` |

---

## 📊 FLUJO COMPLETO DEL SISTEMA

```
┌─────────────────────────────────────────────────────────┐
│                    ECO AI AUTOMATION                     │
│                  Sistema de Prospección                  │
└─────────────────────────────────────────────────────────┘

crm_prospects.md           ──► Lista de 20 prospectos
       │
       ▼
crm_tracker.py             ──► Agrega teléfonos, trackea estados
       │
       ▼
prospecting_assistant.py   ──► Crea "Eko" en VAPI (solo 1 vez)
       │
       ▼
outreach_caller.py         ──► Llama a prospectos automáticamente
       │
       ├── Interesado ──► create_demo_assistant.py
       │                   └── Assistant personalizado por negocio
       │                   └── outreach_caller.py --mode demo
       │
       └── No interesado ──► crm_tracker.py --update X DESCARTADO
                              └── Follow-up programado (manual)

Estado final:
  CERRADO → Onboarding manual → Cliente activo ($400/mes)
```

---

## ⚡ PRÓXIMOS PASOS INMEDIATOS

1. **[ ] Ejecutar setup inicial** (ver arriba)
2. **[ ] Buscar teléfonos** de los 20 prospectos en Google Maps
3. **[ ] Agregar teléfonos** con `crm_tracker.py --add-phone`
4. **[ ] Hacer dry-run** para verificar configuración
5. **[ ] Lanzar primera tanda** de 5 llamadas

### Timing recomendado para llamadas:
- **Mejor horario:** Martes-Jueves, 10am-12pm o 2pm-4pm Mountain Time
- **Evitar:** Lunes AM, Viernes PM, fines de semana

---

## 🐛 TROUBLESHOOTING

### Error: "REEMPLAZAR_CON_ID_DE_prospecting_assistant.py"
Solución: Ejecutar `python3 prospecting_assistant.py` y actualizar el ID en `outreach_caller.py`

### Error al crear assistant: 401 Unauthorized
Solución: Verificar que VAPI_API_KEY es correcta en los scripts

### Error al hacer llamada: "Phone number not found"
Solución: Verificar PHONE_NUMBER_ID en `outreach_caller.py`

### El CRM no parsea el archivo
Solución: No modificar la estructura de la tabla en `crm_prospects.md`

### Las llamadas no salen
Verificar:
1. Que el número de VAPI tiene llamadas salientes habilitadas
2. Que el formato del teléfono es +1XXXXXXXXXX
3. El saldo de VAPI en https://dashboard.vapi.ai

---

## 💡 TIPS Y MEJORES PRÁCTICAS

1. **Empieza con dry-run** siempre antes de llamadas reales
2. **Máximo 5 llamadas** en la primera tanda para probar el sistema
3. **Actualiza el CRM inmediatamente** después de cada llamada
4. **Las demos personalizadas** tienen mucho más impacto que el pitch genérico
5. **Fernando (voz de Eko)** en español convierte mejor con prospectos hispanos
6. Los assistants de demo **se pueden eliminar** después de 30 días (`--delete`)

---

*Eco AI Automation LLC — Sistema construido el 2026-03-24*  
*Para soporte técnico: Eko AI Agent en Telegram*
