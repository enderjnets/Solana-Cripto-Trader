# 🚨 BUG CRÍTICO: Exceso de Posiciones Abiertas

**Fecha:** 2026-02-24 09:45 AM MST
**Estado:** 🔴 CRÍTICO - Límite de 5 posiciones violado

---

## 🐛 Problema Detectado

**Límite Configurado:** 5 posiciones concurrentes
**Posiciones Abiertas:** 22 posiciones
**Exceso:** +340% sobre el límite

### Distribución por Symbol

```
BONK: 6 posiciones
JTO: 5 posiciones
PUMP: 2 posiciones
WEN: 2 posiciones
BOME: 2 posiciones
MEW: 2 posiciones
ORCA: 1 posición
WIF: 1 posición
SOL: 1 posición
```

**TOTAL: 22 posiciones abiertas**

---

## 💰 Impacto en Capital

```
Balance disponible: $477.91
Capital en posiciones: $807.90
Total portfolio: $1,285.81

Riesgo real: ~63% del portfolio en posiciones (debería ser max 75%)
```

---

## 🔍 Causa Raíz (Posible)

El límite `max_concurrent_positions` no está funcionando correctamente en:

**Archivo:** `unified_trading_system.py`
**Función:** `execute_trade()`
**Línea problemática:** ~1188

```python
# Check concurrent position limit
open_trades = self.paper_engine.get_open_trades()
max_concurrent = profile.get("max_concurrent_positions", profile.get("max_concurrent", 5))
if len(open_trades) >= max_concurrent:
    logger.warning(f"⚠️ Max concurrent trades reached ({max_concurrent})")
    return False
```

**Hipótesis:**
1. ❌ Las posiciones no se están cerrando correctamente
2. ❌ El check de límite se está haciendo antes de abrir nuevos trades
3. ❌ Los trades se abren pero el check no los ve
4. ❌ Hay múltiples instancias del bot abriendo trades

---

## ⚠️ Evidencia en los Logs

```bash
# Límites detectados:
2026-02-24 07:25:20: ⚠️ Max concurrent trades reached: 5
2026-02-24 07:58:20: ⚠️ Max concurrent trades reached: 5
```

**PERO** las posiciones siguen aumentando después de estos mensajes.

---

## 🔧 Solución Inmediata (Recomendada)

### Opción 1: Cerrar Posiciones Manualmente

```python
# Crear script para cerrar posiciones duplicadas
python3 close_excess_positions.py
```

### Opción 2: Resetear Paper Trading

```python
# En unified_trading_system.py:
self.paper_engine.reset()
```

### Opción 3: Investigar el Bug (Recomendado a largo plazo)

1. Verificar que `paper_engine.get_open_trades()` devuelve las posiciones correctas
2. Verificar que el límite se aplica ANTES de abrir un trade
3. Verificar que no hay múltiples instancias del bot
4. Verificar que las posiciones se cierran correctamente en SL/TP

---

## 📊 Estado del Bot

```
✅ Wrapper corriendo (PID: 336860)
✅ Bot corriendo (PID: 336866)
⚠️ 22 posiciones abiertas (deberían ser 5)
⚠️ Límite de concurrentes violado
```

---

## 🎯 Acciones Necesarias

### Urgente (HOY)
- [ ] Investigar por qué el límite no funciona
- [ ] Cerrar posiciones excedentes manualmente
- [ ] Verificar que no hay múltiples bots corriendo

### A mediano plazo (Esta semana)
- [ ] Añadir validación de estado en el paper_engine
- [ ] Añadir logs de debugging para posiciones
- [ ] Añadir cleanup automático de posiciones huérfanas

### A largo plazo
- [ ] Revisar toda la lógica de gestión de posiciones
- [ ] Añadir tests unitarios para límites de trading
- [ ] Implementar sistema de alertas por exceso de posiciones

---

## 🚨 Riesgos

### Si se mantiene esta situación:

1. **Over-leverage:**
   - Demasiado capital en riesgo simultáneo
   - Pérdidas excesivas si el mercado se mueve en contra

2. **Liquidez:**
   - No hay balance suficiente para cubrir SL
   - Riesgo de liquidación en trading real

3. **Control perdido:**
   - El sistema no respeta sus propios límites
   - Riesgo de comportamiento impredecible

---

**Reportado por:** Eko (EkoBit)
**Fecha:** 2026-02-24
**Prioridad:** 🚨 CRÍTICA

🦞
