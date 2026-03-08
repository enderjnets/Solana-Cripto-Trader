# SOLUCIÓN V5.0 - Arquitectura Async Pura
Fecha: 2026-02-26 12:00 MST

## Problema Identificado

El sistema con threading tiene problemas fundamentales:
1. Thread del master termina prematuramente (15-36 segundos)
2. No hay excepciones visibles en logs
3. Cuando thread termina, main loop detecta y sale
4. Signal handler llama sys.exit(0) causando terminación del watchdog

## Arquitectura Nueva - V5.0

### Principios
- **CERO threading** - Usar solo asyncio
- **Single event loop** - Todo corre en un loop
- **Async tasks** - Concurrency mediante asyncio
- **Graceful shutdown** - Cancelación de tareas, no sys.exit()

### Diseño

```
Main Process
├── Event Loop
│   ├── Task: Master Orchestrator
│   │   └── while running: research + backtest + audit + trade
│   └── Task: Auto-Learning
│       └── while running: adapt parameters + evolve strategies
└── Signal Handler
    └── Cancel tasks + graceful shutdown
```

### Ventajas
- ✅ Elimina problemas de threading
- ✅ Elimina race conditions
- ✅ Mejor manejo de señales
- ✅ Más simple y mantenible
- ✅ Mejor performance

### Código Estructura

```python
async def run_master():
    """Master orchestrator task"""
    while running:
        # Research, Backtest, Audit, Trade
        await asyncio.sleep(60)

async def run_auto_learning():
    """Auto-learning task"""
    while running:
        # Adapt parameters
        await asyncio.sleep(60)

async def main():
    """Main async function"""
    global running
    running = True

    # Create tasks
    master_task = asyncio.create_task(run_master())
    auto_learning_task = asyncio.create_task(run_auto_learning())

    try:
        # Wait for both tasks
        await asyncio.gather(master_task, auto_learning_task)
    except asyncio.CancelledError:
        logger.info("Tasks cancelled, shutting down...")
    finally:
        logger.info("Shutdown complete")

# Run with proper signal handling
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: loop.stop())

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
```

### Signal Handling Mejorado

**ANTES (Problemático):**
```python
def signal_handler(signum, frame):
    running = False
    wrapper.stop()
    sys.exit(0)  # ❌ Termina inmediatamente
```

**DESPUÉS (Correcto):**
```python
def signal_handler():
    global running
    running = False  # Solo marca para terminar
    # Loop detectará y cerrará graceful
```

### Watchdog Script Mejorado

**ANTES:**
```bash
if [ $EXIT_CODE -ne 0 ]; then
    # Reinicia solo en error
    break  # Sale en exit 0
fi
```

**DESPUÉS:**
```bash
# Siempre reinicia
# El servicio systemd maneja la terminación
```

## Plan de Implementación

1. ✅ Diseñar arquitectura
2. ⏳ Implementar start_master_v5.py
3. ⏳ Pruebas manuales (15 min)
4. ⏳ Despliegue systemd
5. ⏳ Monitoreo extendido (30 min)
6. ⏳ Validar estabilidad completa

## Esperado

- ✅ Sin reinicios constantes
- ✅ Uptime continuo >1 hora
- ✅ Auto-learning funcional
- ✅ Trading activo
- ✅ Graceful shutdown

---

**Estado**: Implementación en curso
