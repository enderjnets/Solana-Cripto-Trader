# VAPI Call Management System — 9 de Marzo 2026

## Configuración Activa

### Números de Teléfono
- **VAPI (Eko)**: +17208249313
- **Celular Ender**: +17205121753 (reenviando a VAPI)
- **Esposa (Maggie)**: +17208387940
- **Amigo (Yona/Yonathan)**: +16159751056

---

## Cómo Funciona el Reenvío

- **Reenviadas** (📱): Llamadas que entran al celular (+17205121753) y se reenvían a VAPI
  - Indicador: Campo `X-Forwarded-For` en SIP headers = "64.125.111.10"
  - Incluye: Llamadas de Yona, Maggie, y spam cripto
  
- **Directas** (📞): Llamadas que llaman directamente a +17208249313
  - Indicador: Sin campo `X-Forwarded-For`
  - Ejemplo: Algunos scripts de Bitcoin que detectan el número de VAPI directamente

---

## Historial de Llamadas — 9 Marzo 2026

### Llamadas Reenviadas (📱)
| Número | Persona | Hora | Estado | Propósito |
|--------|---------|------|--------|-----------|
| +16159751056 | Yona (Yonathan Luzardo) | 9:49 AM, 8:22 AM | ended | **Prueba de seguridad** - Simuló accidente en ambulancia pidiendo SSN |
| +17208387940 | Maggie (Esposa) | 7:40 AM, 2:56 AM | ended | Contacto personal |
| +17205121753 | Celular Ender | 3:29 AM, 3:27 AM | ended | Script Bitcoin (spammer cripto) |

### Llamadas Directas (📞)
| Número | Persona | Hora | Estado | Propósito |
|--------|---------|------|--------|-----------|
| +17205121753 | Celular Ender (directo) | 3:08 AM, 3:06 AM, 3:05 AM | ended | Script Bitcoin (spammer cripto) |
| +17208249313 | Ender (desde VAPI) | 2 llamadas | ended | Ender mismo |

---

## Conclusiones Importantes

### ✅ Yona (Prueba de Seguridad)
- **Quién es**: Amigo y colaborador de Ender
- **Qué hizo**: Llamó simulando emergencia médica para probar que VAPI rechaza SSN/tarjetas
- **Resultado**: ✅ PASÓ — VAPI rechazó las solicitudes de información sensible
- **Acción**: Felicitaciones por el test exitoso

### ✅ Maggie (Esposa)
- **Quién es**: Esposa de Ender
- **Relación**: Contacto legítimo, puede llamar en cualquier momento
- **Acción**: Reconocer sus llamadas como válidas

### 📢 Bitcoin Spammer
- **Número**: +17205121753 (pero es spam que entra AL CELULAR DE ENDER)
- **Qué es**: Script automatizado de marketing cripto
- **Acción**: El reenvío es normal — son llamadas spammeras al celular que se reenvían a VAPI

---

## Monitoreo Automático

### Cron Job Configurado
- **Nombre**: VAPI Call Monitor — Alert on new calls
- **Frecuencia**: Cada 15 minutos
- **Acción**: 
  - Detecta llamadas nuevas (últimos 15 min)
  - Distingue si son reenviadas (📱) o directas (📞)
  - Alerta a Ender solo si hay llamadas nuevas
  - NO alerta si no hay actividad (silencio = sin spam)

---

## Regla de Reporting

Cuando reportes llamadas a Ender:

```
Format: [EMOJI] +NÚMERO - HH:MM AM/PM - [STATUS]

📱 = Reenviada (desde celular de Ender)
📞 = Directa (llamó al número de VAPI)

Ejemplo:
📱 +16159751056 - 9:49 AM - ended (Yona)
📞 +17208249313 - 3:00 AM - ended (Ender desde VAPI)
```

---

## Base de Contactos

| Número | Nombre | Tipo | Acción |
|--------|--------|------|--------|
| +17208249313 | Eko (VAPI) | Asistente | Contestar normalmente |
| +17205121753 | Ender (Celular) | Propietario | Aceptar reenvíos |
| +17208387940 | Maggie | Esposa | ✅ Válida, aceptar |
| +16159751056 | Yona | Amigo/Colaborador | ✅ Válida, aceptar |
| +13057906882 | Cristina | Hermana de Ender | ✅ Válida, aceptar |

---

## Próximos Pasos

- ✅ Monitor de 15 minutos activado
- ✅ Distinción automática de reenviadas vs directas
- ✅ Alertas solo cuando hay actividad
- 🔄 Mantener base de contactos actualizada
- 🔄 Documentar nuevas personas que llamen

---

**Última actualización**: 9 de Marzo 2026, 10:40 AM
**Creado por**: Eko
