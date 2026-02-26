# 🔍 INVESTIGACIÓN SISTEMÁTICA - Reinicios Constantes
Fecha: 2026-02-26 12:23 MST
Objetivo: Identificar y resolver causa raíz de reinicios del Solana Jupiter Bot

## Estado Inicial

**Síntoma:** Bot se reinicia cada 1-2 minutos automáticamente
**Tiempo:** ~6+ horas con problema activo
**Total reinicios:** 325+ desde las 10:00 MST
**Recursos del sistema:** Normales (RAM, disco, CPU OK)

## Hipótesis a Probar

1. **Timeout de Python/Ejecución** - Límite de tiempo no documentado
2. **Network/Connection Issues** - Conexiones que se cierran
3. **Memory Leak** - Acumulación progresiva
4. **Configuración OpenClaw** - Restricciones de entorno
5. **Error en Master Orchestrator** - Bug que causa terminación
6. **Signal Handler** - Señal no capturada correctamente
7. **Infinite Loop/Deadlock** - Ciclo infinito detectado por kernel
8. **Resource Limit** - Límite de recursos no configurado

## Plan de Investigación

### FASE 1: Análisis de Logs (INMEDIATO)
- Buscar patrones antes/después de cada reinicio
- Identificar última acción antes de terminación
- Buscar errores, excepciones, warnings
- Verificar timestamps y correlación

### FASE 2: Análisis de Código
- Revisar master_orchestrator.py
- Buscar sys.exit(), raise, return prematuro
- Verificar signal handlers
- Analizar loops infinitos/recursión

### FASE 3: Pruebas de Aislamiento
- Ejecutar sin OpenClaw/mensajería
- Ejecutar en aislamiento total
- Ejecutar con logging máximo
- Ejecutar con strace para ver syscalls

### FASE 4: Verificación de Sistema
- Límites de recursos (ulimit)
- Configuración systemd
- Límites de kernel
- Configuración OpenClaw

---

**Iniciando FASE 1: Análisis de Logs**
