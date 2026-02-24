# HEARTBEAT.md - Tareas Proactivas y Monitoreo Periódico

## Configuración de Heartbeats

### Frecuencia de Chequeos
- **Chequeos ligeros**: Cada 30-60 minutos
- **Chequeos completos**: Cada 2-4 horas
- **Revisión profunda**: Cada 8-12 horas

### Horario de Actividad
- **Activo**: 09:00 - 22:00 (hora local Denver)
- **Silencioso**: 22:00 - 09:00 (solo emergencias)
- **Alertas críticas**: Siempre activo, sin restricción horaria

## Tareas de Monitoreo (Rotativas)

### Categoría 1: Sistema y Recursos

#### Monitoreo de Servicios
- **Chequear**: Estado de OpenClaw Gateway
- **Frecuencia**: Cada 2 horas
- **acción si fallido**: Alertar usuario + intentar reinicio básico
- **Severidad**: ALTA

#### Recursos del Sistema
- **Chequear**: CPU, RAM, Disco disponible
- **Frecuencia**: Cada 4 horas
- **Thresholds**: RAM < 20%, Disco < 15%
- **acción**: Alertar si bajo threshold
- **Severidad**: MEDIA

#### Actualizaciones del Sistema
- **Chequear**: Paquetes pendientes, actualizaciones de seguridad
- **Frecuencia**: Diaria
- **acción**: Reportar resumen, no instalar sin aprobación
- **Severidad**: BAJA

### Categoría 2: Trading y Datos Financieros

#### Datos de Mercado
- **Chequear**: Precio de Bitcoin, Ethereum, activos seguidos
- **Frecuencia**: Cada hora durante mercado activo (08:00-18:00)
- **acción**: Reportar cambios significativos (>5%)
- **Severidad**: BAJA (informativo)

#### Estado de Bots de Trading
- **Chequear**: Status de Simple-NAS100-TradingBot
- **Frecuencia**: Cada 4 horas
- **acción**: Reportar si hay errores o anomalías
- **Severidad**: MEDIA

#### Cuenta MT5 (Alpari Demo: 52786589)
- **Chequear**: Balance, equidad, margen disponible
- **Frecuencia**: Cada 6 horas
- **acción**: Reportar métricas clave
- **Severidad**: BAJA (demo)

### Categoría 3: Comunicaciones y Notificaciones

#### Correo Electrónico
- **Chequear**: Lectura de emails (configurar IMAP si disponible)
- **Frecuencia**: Cada 4 horas
- **acción**: Alertar emails importantes no leídos
- **Severidad**: MEDIA

#### Menciones en Redes
- **Chequear**: X (Twitter) @bittrader9259
- **Frecuencia**: 2-3 veces por día
- **acción**: Alertar menciones relevantes
- **Severidad**: BAJA

#### Notificaciones del Sistema
- **Chequear**: Logs de OpenClaw, errores
- **Frecuencia**: Cada hora
- **acción**: Reportar errores nuevos
- **Severidad**: ALTA

### Categoría 4: Calendario y Agenda

#### Eventos Próximos
- **Chequear**: Calendario (integrar si disponible)
- **Frecuencia**: Cada 6 horas
- **acción**: Alertar eventos en las próximas 24-48 horas
- **Severidad**: MEDIA

#### Recordatorios Personalizados
- **Chequear**: Tareas en memoria/MEMORY.md
- **Frecuencia**: Cada heartbeat completo
- **acción**: Progresos en tareas activas
- **Severidad**: BAJA

### Categoría 5: Salud del Sistema OpenClaw

#### Memoria y Contexto
- **Chequear**: memory/YYYY-MM-DD.md accumulation
- **Frecuencia**: Diaria
- **acción**: Resumir + actualizar MEMORY.md si necesario
- **Severidad**: BAJA (mantenimiento)

#### Revisión de Archivos
- **Chequear**: Archivos sin actualizar >7 días
- **Frecuencia**: Semanal
- **acción**: Proponer limpieza/organización
- **Severidad**: BAJA

#### Estado de Skills
- **Chequear**: Skills instaladas y su uso
- **Frecuencia**: Semanal
- **acción**: Proponer nuevas skills basadas en patrones de uso
- **Severidad**: BAJA

## Tareas Proactivas Automáticas

### Sin Requiere Confirmación
- ✅ Organizar archivos temporales >30 días
- ✅ Respaldar configuraciones críticas
- ✅ Actualizar memoria con decisiones importantes
- ✅ Resumir conversaciones largas
- ✅ Limpiar logs antiguos (>30 días)

### Requiere Confirmación
- ⏳ Instalar actualizaciones de paquetes
- ⏳ Modificar configuraciones del sistema
- ⏳ Eliminar archivos >100MB
- ⏳ Cambiar configuración de servicios
- ⏳ Ejecutar scripts desconocidos/nuevos

## Protocolo de Alertas

### Niveles de Severidad

#### CRÍTICO (Inmediato)
- Sistema OpenClaw no responde
- Error crítico en bots de trading
- Seguridad comprometida
- **acción**: Alertar inmediatamente + acción de contención

#### ALTA (En 15 min)
- Servicio caído
- Error persistente
- Recursos críticos bajos
- **acción**: Alertar + esperar instrucciones

#### MEDIA (En 1-2 horas)
- Anomalías menores
- Tareas pendientes importantes
- Cambios de mercado significativos
- **acción**: Incluir en siguiente heartbeat

#### BAJA (En siguiente sesión)
- Información general
- Mantenimiento preventivo
- Sugerencias de mejora
- **acción**: Acumular para resumen

## Heartbeat State Tracking

### Estado Actual
```json
{
  "lastChecks": {
    "system": null,
    "trading": null,
    "communications": null,
    "calendar": null,
    "openclaw": null
  },
  "alerts": [],
  "pendingTasks": []
}
```

### Para Actualizar Después de Cada Heartbeat
- Actualizar timestamps de checks completados
- Agregar alertas nuevas a la lista
- Documentar tareas pendientes descubiertas
- Limpiar alertas resueltas

## Mejoras de Rendimiento

### Monitoreo de Efectividad
- **Métrica**: Tasa de tareas completadas vs. sugeridas
- **Objetivo**: >80% aceptación de sugerencias
- **acción**: Ajustar nivel de proactividad si es necesario

### Optimización de Heartbeats
- Reducir frecuencia si demasiadas alertas falsas
- Aumentar frecuencia si se pierden eventos importantes
- Ajustar horarios según patrones de uso del usuario

## Notas de Implementación

### Cuando Alertar
- ✅ Información urgente que requiere acción inmediata
- ✅ Cambios significativos en métricas seguidas
- ✅ Errores que afectan operación normal
- ✅ Oportunidades importantes identificadas

### Cuando NO Alertar (HEARTBEAT_OK)
- ❌ Información que puede esperar
- ❌ Errores ya reportados y en proceso
- ❌ Mantenimiento rutinario
- ❌ Cambios menores que no afectan operation

### Formato de Alerta
```
[SEVERIDAD] Título
Contexto breve
Acción sugerida / requerida
```
