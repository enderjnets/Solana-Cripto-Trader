# 🔍 DIAGNÓSTICO CRÍTICO - Señal SIGTERM Automática
Fecha: 2026-02-26 12:09 MST

## Hallazgo

El sistema se detiene automáticamente después de **3 segundos** porque recibe **SIGTERM** sin causa aparente.

### Evidencia

```
12:09:06 - Sistema inicia
12:09:06 - Primer ciclo completo ejecutado
12:09:09 - "🛑 Signal handler called - cancelling all tasks"
12:09:09 - Todas las tareas canceladas
12:09:09 - Sistema detenido
```

**Tiempo activo: ~3 segundos**

## Análisis

### Posibles Causas

1. **Timeout command** - `timeout 180` puede estar enviando SIGTERM prematuro
   - ❌ NO es esto - timeout solo debería enviar SIGTERM después de 180s

2. **Proceso padre terminando** - Si el shell termina, el proceso hijo recibe SIGHUP
   - ❌ NO es esto - El proceso no está recibiendo SIGHUP

3. **Systemd watchdog** - Systemd puede estar matando el proceso
   - ✅ POSIBLE - Puede haber un timeout de systemd no documentado

4. **Interactive shell** - El shell puede estar enviando SIGTERM cuando detecta EOF
   - ✅ POSIBLE - Cuando no hay más input, el shell puede enviar SIGTERM

5. **Proceso en background** - Cuando un proceso está en background, puede recibir señales
   - ✅ POSIBLE - El comando está siendo ejecutado con `&`

## Siguiente Paso

Ejecutar proceso sin timeout ni background para verificar si el problema es causado por el entorno de ejecución.

### Comando
```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
nohup python3 -u start_master_v5.py > /tmp/v5_test.log 2>&1 &
```

### Monitoreo
```bash
# Verificar que el proceso se mantenga vivo
watch -n 5 'ps aux | grep start_master_v5 | grep -v grep'

# Ver logs
tail -f /tmp/v5_test.log
```

---

**Estado**: Investigando causa de SIGTERM
**Hipótesis actual**: Comando en background con pipe (tee) causando terminación prematura
