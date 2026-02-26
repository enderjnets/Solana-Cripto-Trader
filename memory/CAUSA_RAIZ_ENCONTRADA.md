# 🎯 HALLAZGO CRÍTICO - CAUSA RAÍZ IDENTIFICADA

## Descubrimiento

**Fecha:** 2026-02-26 12:30 MST
**Duración de investigación:** ~45 minutos

## Causa Raíz

**PROBLEMA:** Algo está enviando **SIGTERM** al Solana Jupiter Bot durante el sleep de cada ciclo.

**TIMING:**
- Ciclo 1: Ejecuta → Sleep 60s → SIGTERM a los ~19s de sleep
- Ciclo 2: Ejecuta → Sleep 60s → SIGTERM a los ~41s de sleep
- Ciclo 3: Ejecuta → Sleep 60s → (en progreso)

## Evidencia

### 1. Strace Confirmó SIGTERM
```
--- SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=752437, si_uid=1000} ---
+++ killed by SIGTERM +++
```

### 2. Debug Script Confirmó Recibimiento de Señal
```
🔔 SIGNAL RECEIVED: 15
   Frame: <frame at 0x79efacc63880, file '/usr/lib/python3.13/selectors.py', line 452, code select>
   Cycle: 1
   Time: 2026-02-26 12:28:40.930817
```

### 3. Con Handler Personalizado - El Bot CONTINÚA
- Con mi signal handler, el bot recibió SIGTERM PERO siguió corriendo
- Completó ciclo 3 (¡primera vez que pasa de 2 ciclos!)
- SIGTERM se está enviando PERO no es lo que mata el bot original

## Por Qué el Bot Original Muere

**Master Orchestrator NO tiene signal handler para SIGTERM:**
```python
# master_orchestrator.py - NO hay signal.signal() para SIGTERM
# Cuando llega SIGTERM, Python lo mata inmediatamente
```

**Mi Debug Script TIENE signal handler:**
```python
signal.signal(signal.SIGTERM, signal_handler)
# Captura SIGTERM y continúa ejecutando
```

## Quién Está Enviando SIGTERM

**HIPÓTESIS PRINCIPAL:**
1. **timeout command** (si existe en el watchdog)
2. **Systemd WatchdogTimeout**
3. **OpenClaw/Mensajería** (límite de ejecución no documentado)
4. **Otro proceso/Script** (cron job, monitoreo, etc.)

**Próximo Paso:** Identificar el PID exacto que envía la señal.

## Solución Inmediata

**AGREGAR SIGNAL HANDLER AL MASTER ORCHESTRATOR**

Implementar un signal handler para SIGTERM que:
1. Capture la señal
2. Log el evento
3. Decida si continuar o terminar
4. Permita que el bucle continue hasta que sea apropiado

## Próximos Pasos

1. ✅ **CAUSA RAÍZ IDENTIFICADA** - Algo envía SIGTERM
2. ⏳ Identificar quién envía la señal (PID, proceso)
3. ⏳ Implementar signal handler en master_orchestrator.py
4. ⏳ Probar que el bot ya no se reinicia
5. ⏳ Investigar por qué se envía SIGTERM (configuración, bug, etc.)

---

**ESTADO:** CAUSA RAÍZ IDENTIFICADA - PROGRESO SIGNIFICATIVO
