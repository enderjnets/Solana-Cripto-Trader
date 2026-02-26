# Portafolio de Trading Automatizado

---

## 🎯 Visión General

Desarrollador de sistemas de trading automatizado con experiencia probada en MT5, Solana/Web3, y bots de criptomonedas. Especialista en estrategias que han pasado backtests de FTMO con 59.8% de win rate y 1.49 de profit factor.

---

## 📊 Métricas de Rendimiento

| Proyecto | Win Rate | Profit Factor | Max DD | Estrategia |
| -------- | --------- | ------------- | ------- | ---------- |
| **FTMO Challenge** | 59.8% | 1.49 | 2.96% | Trend Following |
| **Solana Bot v3.1** | 16.7%* | TBD | 10% | Bidirectional (LONG+SHORT) |

*Solana Bot en early testing, esperando WR 40%+

---

## 🤖 Proyectos de Trading

---

## 1. Simple-NAS100-TradingBot (v1.6)

### 📋 Descripción
Expert Advisor (EA) para MetaTrader 5 optimizado para FTMO Challenge.

### 🎯 Características
- **Plataforma:** MetaTrader 5 (MT5)
- **Lenguaje:** MQL5
- **Símbolo:** NAS100
- **Estrategia:** Trend Following
- **Tipo:** Cuentas Prop Firms (FTMO, TopstepTrader, etc.)

### 📊 Rendimiento
| Métrica | Valor |
| ------- | ----- |
| **Win Rate** | 59.8% |
| **Profit Factor** | 1.49 |
| **Max Drawdown** | 2.96% |
| **Risk per Trade** | 0.5% |
| **Daily Risk** | 2.0% |
| **Reward/Risk** | 2.5:1 (SL 1%, TP 2.5%) |
| **Backtest Period** | 2023-2024 |

### 🔧 Funcionalidades
- Detección de tendencias (EMA crossovers)
- Stop-loss dinámico basado en ATR
- Take-profit ajustado según volatilidad
- Gestión de riesgo por trade
- Límite diario de pérdidas
- Filtrado de sesiones de trading

### 📁 Repositorio
https://github.com/enderjnets/Simple-NAS100-TradingBot

### 🎓 Resultados FTMO
✅ **Passed** - La estrategia cumple con los requisitos de FTMO Challenge:
- Win Rate > 40% ✅
- Profit Factor > 1.4 ✅
- Max Drawdown < 10% ✅

---

## 2. Solana-Cripto-Trader (v3.1 - Drift Protocol)

### 📋 Descripción
Bot de trading automatizado para mercados perpetuales de Solana con simulación completa de Drift Protocol.

### 🎯 Características
- **Plataforma:** Solana Blockchain
- **DEX:** Jupiter (swaps), Drift Protocol (perps)
- **Lenguaje:** Python 3.13
- **Símbolos:** BTC, ETH, SOL, ADA, XRP, DOT, LINK
- **Estrategia:** Bidirectional (LONG + SHORT)

### 🏗️ Arquitectura
Sistema multi-agente con 4 componentes:

1. **Researcher Agent**
   - Analiza mercado en tiempo real
   - Detecta tendencias (EMA, RSI)
   - Genera señales de trading

2. **Backtester Agent**
   - Valida estrategias históricamente
   - Calcula métricas (Sharpe, Win Rate, Max DD)
   - Aprueba/rechaza estrategias

3. **Auditor Agent**
   - Evalúa riesgo de trades
   - Aprueba solo trades con alta confianza
   - Gestiona exposición

4. **Trading Agent**
   - Ejecuta trades en modo paper trading
   - Simula Drift Protocol (leverage, fees, liquidation)
   - Calcula P&L neto ajustado por fees

### 📊 Rendimiento (Paper Trading)
| Métrica | Valor |
| ------- | ----- |
| **Capital** | $498.13 |
| **P&L Diario** | -0.37% |
| **Posiciones** | 1 abierta (DOGE LONG) |
| **Win Rate** | 16.7%* |
| **Target WR** | 40%+ |
| **Daily Target** | 5% |
| **Max Drawdown** | 10% |

*Early testing - esperando WR 40%+ para rentabilidad con RR 1.5:1

### 🔧 Simulación Drift Protocol
| Característica | Valor |
| ------------- | ----- |
| **Leverage** | 5.0x (configurable) |
| **Trading Fee** | 0.05% por trade |
| **Liquidation Threshold** | 80% collateral |
| **Borrowing Fee** | 0.01% por hora (SHORTs) |
| **Funding Rate** | 0.01% cada 8 horas |
| **Stop Loss** | -2% |
| **Take Profit** | +3% |
| **RR Ratio** | 1.5:1 |

### 📈 Estrategias de Trading

#### **Mercado BULLISH**
- 📈 **LONG** en tendencias alcistas
- 🔥 **PUMP** en movimientos fuertes >3%

#### **Mercado BEARISH**
- 📉 **SHORT** en tendencias bajistas
- ❌ No comprar en dips (evitar caídas)

#### **Mercado NEUTRAL**
- 📊 **RANGE TRADING**
  - LONG en soporte
  - SHORT en resistencia

### 📁 Repositorio
https://github.com/enderjnets/Solana-Cripto-Trader

### 🎯 Métricas de Sistema
| Métrica | Valor |
| ------- | ----- |
| **Ciclos ejecutados** | 88+ |
| **Estrategias generadas** | 0-3 por ciclo |
| **Posiciones máximas** | 5 simultáneas |
| **Intervalo de check** | 60 segundos |
| **Precios** | CoinGecko API (real-time) |

---

## 3. MT5 Automation System

### 📋 Descripción
Sistema completo de automatización para MetaTrader 5 con backtester nativo en Python.

### 🎯 Características
- **Plataforma:** MetaTrader 5
- **Lenguajes:** MQL5, Python 3.13
- **Backtester:** Nativo en Python (sin MT5)

### 🔧 Funcionalidades
- **Backtester Nativo Python**
  - Simula trades sin abrir MT5
  - Calcula métricas de estrategia
  - Optimiza parámetros automáticamente

- **Optimizer v2**
  - Optimiza estrategias para FTMO
  - Busca mejores parámetros (SL, TP, RR)
  - Evalúa miles de combinaciones

- **Download Scripts**
  - Descarga datos históricos de MT5
  - Exporta a formato nativo Python
  - Soporta múltiples símbolos

### 📁 Repositorio
https://github.com/enderjnets/Solana-Cripto-Trader (scripts de MT5)

---

## 4. OpenClaw Trading Integration

### 📋 Descripción
Sistema de agentes autónomos de IA integrado con plataformas de trading.

### 🎯 Características
- **Framework:** OpenClaw (Node.js + Python)
- **Modelo:** MiniMax M2.1
- **Capacidades:** Multi-agent orchestration

### 🔧 Funcionalidades
- **Agentes de Trading**
  - Generación de señales de trading
  - Análisis de mercado en tiempo real
  - Ejecución automatizada

- **Sistema de Heartbeat**
  - Monitoreo proactivo 24/7
  - Alertas automáticas
  - Gestión de eventos

- **Skills de Superpowers**
  - Brainstorming (creación de estrategias)
  - Systematic Debugging
  - Test-Driven Development
  - Writing Plans (roadmaps)

### 🎯 Integraciones
- **Telegram:** Notificaciones y control
- **YouTube:** @bittrader9259
- **Telegram:** @Enderjh

---

## 📊 Comparación de Estrategias

| Estrategia | Win Rate | Profit Factor | RR | Risk/Trade | Risk/Diario |
| ---------- | --------- | ------------- | -- | ---------- | ----------- |
| **FTMO Challenge** | 59.8% | 1.49 | 2.5:1 | 0.5% | 2.0% |
| **Solana v3.1** | 16.7%* | TBD | 1.5:1 | 10% | 10% |
| **FTMO v2** | TBD | TBD | TBD | TBD | TBD |

*Early testing

---

## 🎯 Metas Futuras

1. **Pass FTMO Challenge** con cuenta real
2. **Expandir a más prop firms** (TopstepTrader, MyForexFunds)
3. **Implementar trading real en Solana** con Drift Protocol
4. **Crear bots para exchanges centralizados** (Binance, Coinbase)
5. **Desarrollar estrategias de arbitraje** entre DEXes

---

## 📧 Contacto

- **Email:** [Tu email]
- **Telegram:** @Enderjh
- **GitHub:** @enderjnets
- **YouTube:** @bittrader9259

---

*Última actualización: Febrero 24, 2026*
