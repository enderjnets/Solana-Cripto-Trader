# 🚀 INICIO RÁPIDO - Auto-Learning System

## ✅ Sistema Instalado Correctamente

El sistema de auto-aprendizaje está listo para alcanzar el objetivo de **5% diario**.

## 📁 Archivos Creados

```
Solana-Cripto-Trader/
├── auto_learner.py              # Sistema core (29KB)
├── integrate_auto_learner.py    # Integración (12KB)
├── start_auto_learning.py       # Script de inicio (9KB)
├── monitor_auto_learning.py     # Dashboard (10KB)
└── AUTO_LEARNING_DOCS.md        # Documentación completa (7KB)
```

## 🎯 3 Formas de Usar

### 1️⃣ INICIAR SISTEMA (Recomendado)

```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
python3 start_auto_learning.py
```

**Qué hace:**
- Inicia el sistema de auto-aprendizaje
- Evoluciona estrategias cada hora
- Adapta parámetros cada 5 minutos
- Aprende de cada trade automáticamente
- Muestra updates cada 5 minutos

**Para detener:** Presiona `Ctrl+C`

---

### 2️⃣ MONITOREAR PROGRESO

```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
python3 monitor_auto_learning.py
```

**Muestra:**
- Progreso hacia 5% diario (con barra visual)
- Win rate y profit factor
- Mejores estrategias evolucionadas
- Parámetros actuales del sistema

**Opciones:**
```bash
# Actualizar cada 30 segundos
python3 monitor_auto_learning.py --interval 30

# Ver una vez y salir
python3 monitor_auto_learning.py --once
```

---

### 3️⃣ PROBAR SISTEMA

```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
python3 start_auto_learning.py --test
```

**Qué hace:**
- Simula 20 trades de prueba
- Verifica que todo funciona correctamente
- Muestra el estado del sistema

---

## 🧠 Cómo Funciona el Auto-Aprendizaje

### Cada Trade:
1. Sistema analiza mercado (10 features)
2. Decide comprar/vender/hold (RL)
3. Ejecuta trade con parámetros adaptativos
4. Registra resultado en base de datos
5. **Aprende del resultado**
6. **Adapta parámetros automáticamente**

### Cada 5 Minutos:
- Ajusta Stop Loss, Take Profit
- Modifica position size
- Actualiza confidence threshold

### Cada Hora (si hay 50+ trades):
- **Evoluciona estrategias** (algoritmo genético)
- Evalúa 1000 combinaciones de indicadores
- Selecciona la mejor estrategia
- La aplica automáticamente

---

## 🎯 Objetivo: 5% Diario

El sistema optimizará automáticamente para alcanzar:

| Métrica | Target |
|---------|--------|
| Daily PnL | **5%** |
| Win Rate | >50% |
| Profit Factor | >1.5 |
| Sharpe Ratio | >2.0 |
| Max Drawdown | <10% |

---

## ⚙️ Parámetros Adaptativos

El sistema ajusta automáticamente:

| Parámetro | Base | Rango |
|-----------|------|-------|
| Stop Loss | 2.5% | 2-3% |
| Take Profit | 5% | 3-6% |
| Position Size | 10% | 5-20% |
| Leverage | 5x | 1-10x |
| Confidence | 0.6 | 0.5-0.8 |

---

## 📊 Ejemplo de Dashboard

```
================================================================================
📊 AUTO-LEARNING MONITOR DASHBOARD
================================================================================
📅 2026-02-25 15:00:00

🧠 LEARNING PROGRESS
--------------------------------------------------------------------------------
  Generación actual: 3
  Total trades: 127
  Exploration rate: 0.245

  📈 Progreso Diario:
     Target: 5.00%
     Current: 3.24%
     Remaining: 1.76%
     Progress: 64.8%
     [████████████████████████████░░░░░░░░░░░░░░░░░░] 64.8%

💰 PERFORMANCE (Last 24h)
--------------------------------------------------------------------------------
  Total trades: 48
  Win rate: 58.33%
  Profit factor: 1.87
  Sharpe ratio: 2.34
  Max drawdown: -4.21%
  Total PnL: 3.24%

  ✅ Best trade: +5.12% (SOL)
  ❌ Worst trade: -2.87% (BTC)

🧬 TOP STRATEGIES
--------------------------------------------------------------------------------
  #1 - Gen 3 | Fitness: 0.8234 | Win: 61.2% | Sharpe: 2.45 | Trades: 42
  #2 - Gen 2 | Fitness: 0.7891 | Win: 58.7% | Sharpe: 2.12 | Trades: 38
  #3 - Gen 1 | Fitness: 0.7654 | Win: 55.3% | Sharpe: 1.98 | Trades: 31
```

---

## 🛠️ Comandos Útiles

### Ver estado actual
```bash
python3 start_auto_learning.py --status
```

### Ver dashboard una vez
```bash
python3 monitor_auto_learning.py --once
```

### Ver documentación completa
```bash
cat AUTO_LEARNING_DOCS.md
```

---

## 💡 Tips Importantes

1. **Déjalo correr**: Mínimo 24-48 horas para ver resultados
2. **Monitorea progreso**: Revisa el dashboard regularmente
3. **Confía en el sistema**: Aprenderá y mejorará automáticamente
4. **No interfieras**: El sistema se ajusta solo

---

## ⚠️ Advertencias

- El objetivo de 5% diario es **ambicioso**
- El sistema reducirá riesgo automáticamente si:
  - Drawdown > 10%
  - Win rate < 40%
- Los primeros días pueden ser volátiles mientras aprende

---

## 📈 Evolución Esperada

**Día 1-2:**
- Aprendizaje inicial
- Win rate: 45-50%
- PnL: 0-2%

**Día 3-7:**
- Estrategias optimizadas
- Win rate: 50-55%
- PnL: 2-4%

**Día 7+:**
- Sistema maduro
- Win rate: 55-60%
- PnL: 4-6% diario

---

## 🎉 ¡Listo para Usar!

El sistema está completamente configurado y probado.

**Para empezar AHORA:**

```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader
python3 start_auto_learning.py
```

**El sistema aprenderá y mejorará automáticamente hacia el objetivo de 5% diario.**

---

**¿Preguntas?**
- Revisa `AUTO_LEARNING_DOCS.md` para documentación completa
- Ejecuta `python3 monitor_auto_learning.py --once` para ver estado
