# Cron Jobs Monitoring - 2026-03-06

## Cron Jobs Activos

| Job ID | Nombre | Schedule | Próxima ejecución | Estado |
|--------|--------|----------|-------------------|--------|
| 65e60b83 | YouTube Stats | 6AM, 12PM, 6PM | Hoy 6:00 PM | ⚠️ 4 errores |
| cf5b9cad | BitTrader Scout | 8AM, 8PM | Hoy 8:00 PM | ⚠️ 4 errores |
| 1045ce58 | System Backup | 3AM diario | Sáb 3:00 AM | ✅ OK |
| 3d10567f | Full Pipeline | Sáb 2PM | Sáb 2:00 PM | ✅ Pendiente |
| 36286552 | **Recordatorio Verificar** | Sáb 2:30PM | Sáb 2:30 PM | ✅ Nuevo |

---

## Monitoreo - 6 Marzo 2026

### Test del Sistema (14:12 PM)
**Resultado**: ✅ TODOS LOS SISTEMAS OPERATIVOS

| Componente | Estado |
|-----------|--------|
| MiniMax Coding Key | ✅ Configurada |
| scout.py import | ✅ Funcionando |
| creator.py import | ✅ Funcionando |
| producer.py import | ✅ Funcionando |
| youtube_stats.py import | ✅ Funcionando |
| scout_latest.json | ✅ Existe |
| youtube_stats_latest.json | ✅ Existe |
| YouTube credentials | ✅ Configuradas |
| gcloud credentials | ✅ Configuradas |

### Ejecución 6PM (Hoy)
- **YouTube Stats**: ⚠️ Pendiente monitoreo
- **Expected**: Reporte de stats del canal
- **Script**: `python3 youtube_stats.py`
- **Location**: `/home/enderj/.openclaw/workspace/bittrader/agents/`

### Ejecución 8PM (Hoy)
- **Scout**: ⚠️ Pendiente monitoreo
- **Expected**: Precio BTC, trending coins, temas recomendados
- **Script**: `python3 scout.py --crypto-only`
- **Fallback**: MiniMax M2.5 (Coding Plan key)

---

## Monitoreo - 7 Marzo 2026 (Sábado)

### Ejecución 3AM
- **System Backup**: ✅ Automático
- **Script**: `BACKUP_REMINDER`

### Ejecución 2PM - Full Pipeline ⭐ CRÍTICO
- **Full Pipeline**: ⚠️ PENDIENTE VERIFICACIÓN
- **Script**: `python3 orchestrator.py --full`
- **Timeout**: 30 minutos (1800 segundos)
- **Expected**:
  - Scout: Generar temas
  - Creator: Crear guiones
  - Producer: Acortar guiones
  - Publisher: Subir videos a YouTube
- **Reporte requerido**:
  - Cuántos videos se produjeron
  - Qué temas
  - URLs de YouTube si se subieron
  - Cualquier error

### Ejecución 2:30PM - Recordatorio de Verificación ⭐ NUEVO
- **Recordatorio**: ⚠️ PENDIENTE
- **Job ID**: 36286552
- **Script**: Agente verifica resultados del Full Pipeline
- **Timeout**: 2 minutos (120 segundos)
- **Acción**: Verifica y reporta al usuario con resumen
- **Archivos a revisar**:
  - `data/youtube_stats_latest.json`
  - `data/scout_latest.json`

---

## Checklist de Verificación

### Antes del Full Pipeline (Sábado 2PM)
- [ ] Claude Sonnet rate limit reset (esperado 1PM MST)
- [ ] MiniMax M2.5 funcionando como fallback
- [ ] Scripts testeados: scout.py, creator.py, producer.py, publisher.py
- [ ] YouTube credentials válidas
- [ ] gcloud credentials válidas
- [ ] Espacio en disco disponible

### Durante el Full Pipeline
- [ ] Scout ejecuta sin errores
- [ ] Creator genera guiones correctamente
- [ ] Producer acorta guiones para shorts
- [ ] Publisher sube videos a YouTube
- [ ] Timeout no se agota (30 min)

### Después del Full Pipeline
- [ ] Videos subidos exitosamente
- [ ] URLs de YouTube reportadas
- [ ] Sin errores críticos
- [ ] Logs revisados
- [ ] Memory actualizado

---

## Errores Previos (Resueltos)

### YouTube Stats - 4 Errores Consecutivos
- **Causa**: API rate limit (CoinGecko o YouTube)
- **Solución**: Esperar reset de rate limit
- **Fallback**: Reintentar en próxima ejecución

### Scout - 4 Errores Consecutivos
- **Causa**: API rate limit (Claude Sonnet)
- **Solución**: MiniMax M2.5 configurado como fallback
- **Fallback**: Temas manuales si MiniMax falla

---

## Notas de Configuración

### MiniMax Coding Plan
- **Key**: `minimax_coding_key` en minimax.json
- **Endpoint**: `https://api.minimax.io/anthropic/v1/messages`
- **Model**: `MiniMax-M2.5`
- **Status**: ✅ Funcionando

### Claude Sonnet 4.6
- **Endpoint**: `http://127.0.0.1:8443/v1/messages`
- **Rate limit reset**: Sábado 1PM MST
- **Status**: ⚠️ Rate limit activo (hasta 1PM)

---

## Comandos de Verificación

### Ver logs del sistema
```bash
# Ver logs de YouTube Stats
cat /home/enderj/.openclaw/workspace/bittrader/agents/data/youtube_stats_20260306_*.txt

# Ver logs de Scout
cat /home/enderj/.openclaw/workspace/bittrader/agents/data/scout_report_2026-03-06.json

# Ver últimos datos
cat /home/enderj/.openclaw/workspace/bittrader/agents/data/youtube_stats_latest.json
cat /home/enderj/.openclaw/workspace/bittrader/agents/data/scout_latest.json
```

### Ejecutar manualmente para testing
```bash
cd /home/enderj/.openclaw/workspace/bittrader/agents

# Test YouTube Stats
python3 youtube_stats.py

# Test Scout
python3 scout.py --crypto-only

# Test Full Pipeline
python3 orchestrator.py --full
```

---

## Contacto de Emergencia
Si hay errores críticos durante el Full Pipeline:
- Verificar logs: `~/.openclaw/logs/`
- Ejecutar manualmente para debugging
- Contactar a @enderj en Telegram
