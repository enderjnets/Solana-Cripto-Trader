# 🚀 AUTO-LEARNING SYSTEM - ACTIVO

## ✅ Estado Actual
**PID:** 600650
**Status:** 🟢 RUNNING
**Iniciado:** 2026-02-25 16:33:44 MST

---

## 📊 Sistema Configurado

### Capital y Targets
- **Capital Inicial:** $500.00
- **Target Diario:** 5%
- **Max Drawdown:** 10%

### Auto-Learning
- **Generation:** 0
- **Total Trades:** 0
- **Daily PnL:** 0.00%
- **Exploration Rate:** 0.300

### Parámetros Actuales
- Stop Loss: 2.5%
- Take Profit: 5%
- Position Size: 10%
- Leverage: 5x
- Confidence Threshold: 0.6
- Max Trades/Day: 10
- Risk per Trade: 5%

---

## 🔧 Comandos de Monitoreo

### Ver logs en tiempo real
```bash
tail -f /tmp/auto_learning.log
```

### Ver estado actual
```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
python3 start_auto_learning.py --status
```

### Dashboard de monitoreo
```bash
python3 monitor_auto_learning.py
```

### Detener sistema
```bash
kill 600650
```

---

## 📈 Sistema de Actualizaciones

El sistema enviará actualizaciones automáticas cada 5 minutos con:
- Número de trades
- Daily PnL
- Generation actual
- Exploration rate
- Estado de targets

---

## 🎯 Flujo de Aprendizaje

1. **Cada Trade:**
   - Sistema registra resultado
   - RL aprende del trade
   - Parámetros se adaptan

2. **Cada 5 minutos:**
   - Parámetros actualizados
   - Status report generado

3. **Cada Hora (si 50+ trades):**
   - Evolución genética ejecutada
   - Nuevas estrategias generadas
   - Mejor estrategia aplicada

---

## 📊 Comparación con Sistema Anterior

| Métrica | Sistema Anterior | Auto-Learning (Target 7 días) |
|---------|------------------|------------------------------|
| Capital Final | $498.13 | **$675** |
| Daily PnL | -0.37% | **+5%** |
| Win Rate | 50% | **60%+** |
| Total Trades | 12 | **70+** |

---

## 🔔 Próximas Actualizaciones

El sistema te mantendrá informado sobre:
- ✅ Primer trade completado
- ✅ Targets diarios alcanzados
- ✅ Evoluciones de estrategias
- ✅ Mejoras en win rate
- ⚠️ Alertas de drawdown

---

**Sistema activo y aprendiendo hacia el objetivo de 5% diario.** 🚀

**Logs:** `/tmp/auto_learning.log`
**PID:** 600650
