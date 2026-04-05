# Auto-Learning Trading System v1.0

## 🎯 Objetivo

Sistema de auto-aprendizaje completo para alcanzar **5% diario** con máximo **10% drawdown**.

## 🧠 Características

### 1. Genetic Algorithm Evolution
- Evoluciona estrategias de trading automáticamente
- Cada hora (si hay suficientes trades)
- Descubre combinaciones óptimas de indicadores (RSI, EMA, Bollinger)
- 50 poblaciones × 20 generaciones = 1000 estrategias evaluadas

### 2. Reinforcement Learning
- Aprende de cada trade en tiempo real
- Sistema Q-Learning con experience replay
- Adapta decisiones de compra/venta basándose en resultados
- Exploración vs explotación balanceada

### 3. Adaptive Parameters
- Ajusta Stop Loss y Take Profit dinámicamente
- Adapta position size según rendimiento
- Modifica leverage según win rate
- Actualiza confidence threshold

### 4. Performance Feedback Loop
- Cada trade alimenta el sistema de aprendizaje
- Base de datos SQLite persistente
- Análisis de últimos 100 trades para optimización
- Tracking de métricas: Win Rate, Sharpe, Max DD

### 5. Risk Management Adaptativo
- Stop Loss dinámico: 2-3%
- Take Profit dinámico: 3-6%
- Position sizing: 5-20% del capital
- Leverage adaptativo: 1-10x
- Max 10% drawdown

## 🚀 Inicio Rápido

### Opción 1: Sistema Standalone

```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader

# Iniciar sistema de auto-aprendizaje
python3 start_auto_learning.py
```

### Opción 2: Integrar con Sistema Existente

```python
# En tu script de trading actual
from integrate_auto_learner import AutoLearningIntegration
from master_orchestrator import MasterOrchestrator

master = MasterOrchestrator()
integration = AutoLearningIntegration(master)
integration.start()
```

### Opción 3: Monitorear Sistema

```bash
# Ver dashboard de progreso
python3 monitor_auto_learning.py

# Actualización cada 30 segundos
python3 monitor_auto_learning.py --interval 30

# Ver una vez y salir
python3 monitor_auto_learning.py --once
```

## 📊 Monitoreo

### Dashboard en Tiempo Real

El monitor muestra:
- **Learning Progress**: Generación actual, trades totales, exploración
- **Daily Progress**: Barra de progreso hacia 5% diario
- **Performance**: Win rate, profit factor, Sharpe, max DD
- **Top Strategies**: Mejores 5 estrategias evolucionadas
- **Current Parameters**: Parámetros activos del sistema

### Base de Datos

Ubicación: `data/auto_learning.db`

Tablas:
- `performance`: Historial de trades
- `strategies`: Estrategias evolucionadas
- `learning_state`: Estado del aprendizaje
- `market_states`: Estados para RL

## 🧬 Algoritmo Genético

### Genoma de Estrategia

Cada estrategia tiene:
- **Entry Rules**: Hasta 3 reglas de entrada
  - Indicador (RSI, EMA, SMA, Bollinger)
  - Operador (>, <)
  - Threshold
- **Exit Rules**: Reglas de salida
- **Parameters**: SL, TP, position size

### Evolución

1. **Población inicial**: 50 estrategias aleatorias
2. **Evaluación**: Backtest con datos históricos
3. **Selección**: Mejores 25% sobreviven
4. **Crossover**: Combinación de padres
5. **Mutación**: 15% de probabilidad
6. **Repetir**: 20 generaciones

### Fitness Function

```python
fitness = (win_rate * 0.3) + (profit_factor * 0.3) + (sharpe * 0.2) - (max_dd * 0.2)
```

## 🤖 Reinforcement Learning

### Estado del Mercado

10 features:
1. RSI (0-100)
2. EMA diff (fast - slow)
3. Volatilidad
4. Volume ratio
5. Price momentum
6. Bollinger Bands position
7. Trend strength
8. Recent PnL
9. Trade count
10. Time of day

### Acciones

3 acciones posibles:
- 0: Buy (Long)
- 1: Sell (Short)
- 2: Hold (No trade)

### Recompensa

```
reward = pnl_pct * 100
```

### Hiperparámetros

- Learning rate: 0.001
- Discount factor: 0.95
- Exploration rate: 0.3 → 0.01 (decay)
- Experience replay: 10,000 memories
- Batch size: 32

## ⚙️ Parámetros Adaptativos

### Stop Loss
- Base: 2.5%
- Rango: 2-3%
- Adaptación: ±0.5% según win rate

### Take Profit
- Base: 5%
- Rango: 3-6%
- Adaptación: ±0.5% según win rate

### Position Size
- Base: 10%
- Rango: 5-20%
- Adaptación: Según PnL total

### Leverage
- Base: 5x
- Rango: 1-10x
- Adaptación: Según win rate

### Confidence Threshold
- Base: 0.6
- Rango: 0.5-0.8
- Adaptación: Según win rate

## 📈 Targets

### Objetivo Principal
- **5% diario**
- **10% max drawdown**

### Métricas de Éxito
- Win Rate: >50%
- Profit Factor: >1.5
- Sharpe Ratio: >2.0
- Max DD: <10%

## 🔄 Flujo de Trabajo

```
1. Trade ejecutado
   ↓
2. Resultado grabado en DB
   ↓
3. RL aprende del trade
   ↓
4. Parámetros adaptados
   ↓
5. Cada hora: ¿50+ trades?
   ↓ Sí
6. Evolución genética
   ↓
7. Mejor estrategia aplicada
   ↓
8. Loop continúa
```

## 🛠️ Configuración

### Archivos Principales

- `auto_learner.py`: Sistema core
- `integrate_auto_learner.py`: Integración con master
- `start_auto_learning.py`: Script de inicio
- `monitor_auto_learning.py`: Dashboard

### Ajustar Targets

En `auto_learner.py`:
```python
DAILY_TARGET = 0.05  # 5%
MAX_DRAWDOWN = 0.10  # 10%
```

### Ajustar Frecuencia

```python
EVOLUTION_INTERVAL = 3600  # 1 hora
ADAPTATION_INTERVAL = 300  # 5 minutos
```

### Ajustar Algoritmo Genético

```python
POPULATION_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.15
```

## 🐛 Troubleshooting

### Error: "Not enough trades for evolution"
- Necesitas mínimo 50 trades
- El sistema acumulará trades automáticamente

### Error: "Genetic miner not initialized"
- El sistema se inicializará automáticamente
- O provee datos históricos manualmente

### Performance lenta
- Reduce `POPULATION_SIZE` a 30
- Reduce `GENERATIONS` a 10
- Aumenta `EVOLUTION_INTERVAL` a 7200

### No hay estrategias evolucionadas
- Verifica que hay suficientes trades
- Revisa logs en `data/auto_learning.db`
- Aumenta `MIN_TRADES_FOR_EVOLUTION`

## 📝 Logs

Los logs se guardan en:
- `data/learner_state.json`: Estado actual
- `data/q_table.pkl`: Tabla Q del RL
- `data/auto_learning.db`: Base de datos SQLite

## 🎓 Mejores Prácticas

1. **Deja correr el sistema**: Mínimo 24-48 horas para ver resultados
2. **Monitorea daily progress**: Verifica que se acerca al 5%
3. **Revisa strategies**: Mira qué estrategias están ganando
4. **Ajusta targets**: Si 5% es muy agresivo, reduce a 3%
5. **Backup regular**: Respalda `auto_learning.db`

## 🔮 Próximas Mejoras

1. **Deep Q-Network (DQN)**: En lugar de Q-table
2. **Multi-agent system**: Múltiples learners especializados
3. **Sentiment analysis**: Incorporar noticias
4. **Cross-asset learning**: Aprender de múltiples pares
5. **Real-time dashboard**: Dashboard web en vivo

## 📞 Soporte

Para problemas o preguntas:
1. Revisa logs en consola
2. Ejecuta `monitor_auto_learning.py --once`
3. Verifica `data/auto_learning.db`

---

**¡El sistema aprenderá y mejorará automáticamente hacia el objetivo de 5% diario!** 🚀
