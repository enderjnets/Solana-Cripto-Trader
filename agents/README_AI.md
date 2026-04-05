# 🤖 Solana Trading Bot - Sistema con Agentes IA

Sistema modular de trading para Solana potenciado con **Inteligencia Artificial**.

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (Coordinador)                             │
└─────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────┬───────────────┬─────────────────┐
    ↓               ↓               ↓                 ↓
┌────────┐     ┌───────────┐  ┌──────────┐     ┌───────────┐
│ Market │     │  AI       │  │  AI      │     │ Executor  │
│  Data  │     │ Researcher│  │ Strategy │     │           │
│(Jupiter)│     │   (LLM)   │  │  (LLM)   │     │ (Paper)   │
└────────┘     └───────────┘  └──────────┘     └───────────┘
    ↓               ↓               ↓                 ↓
Precios        Tendencia      Señales          Posiciones
reales         fundamental   LONG/SHORT       abiertas
               + técnica
```

## 🧠 Agentes IA

### 1. AI Researcher Agent
**Archivo**: `ai_researcher.py`

**Función**: Análisis de mercado con LLM

**Input:**
- Precios actuales (Jupiter API)
- Fear & Greed Index
- Cambios de 24h

**Output**: `data/research_latest.json`
- Tendencia (BULLISH/BEARISH/NEUTRAL)
- Confianza (0.0 - 1.0)
- Niveles de soporte/resistencia
- Factores del mercado
- Recomendación (CAUTIOUS/AGGRESSIVE/NEUTRAL)

**LLM**: Claude Sonnet 4.6 → MiniMax M2.5

---

### 2. AI Strategy Agent (Enhanced)
**Archivo**: `ai_strategy.py`

**Función**: Generación de señales de trading

**Input:**
- Análisis del AI Researcher
- Indicadores técnicos (RSI, volatilidad)
- Estado del portafolio

**Output**: `data/strategy_llm.json`
- Señales LONG/SHORT (máx 3 por ciclo)
- Entry price, SL, TP
- Tamaño de posición (2% del capital)
- Justificación de cada señal
- Confianza (0.0 - 1.0)

**LLM**: Claude Sonnet 4.6 → MiniMax M2.5

**Reglas de Trading:**
- Riesgo por trade: 2% del capital ($10 USD)
- Stop Loss: 2.5% del entry
- Take Profit: 5% del entry (2:1 RR)
- Máximo 3 señales por ciclo
- Máximo 5 posiciones abiertas simultáneas

**Criterios de Entrada:**
- LONG: RSI < 30 (sobreventa) + cambio 24h > -3% + volatilidad > 0.5%
- SHORT: RSI > 70 (sobrecompra) + cambio 24h > 3% + volatilidad > 0.5%
- MOMENTUM: RSI 50-70 + cambio 24h > 2% (LONG) o < -2% (SHORT)

---

### 3. AI Explainability Agent
**Archivo**: `ai_explainability.py`

**Función**: Explicación de decisiones en lenguaje natural

**Input:**
- Señales del Strategy Agent
- P&L actual
- Posiciones abiertas

**Output**: `data/latest_explanation.json`
- Resumen del portafolio
- Explicación de señales
- Alertas de riesgo (si aplica)
- Lista de posiciones con P&L

**TTS**: MiniMax TTS (speech-2.8-hd) para alertas de voz

**Alertas de Riesgo:**
- Drawdown > 5%: ⚠️ Drawdown alto
- Drawdown > 10%: ⚠️ Drawdown crítico
- 5 posiciones abiertas: ⚠️ Máximo alcanzado
- P&L reciente negativo: ⚠️ Pérdidas acumuladas

---

## 🚀 Quickstart

### Ejecutar un ciclo completo

```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents

# 1. Obtener precios
python3 market_data.py

# 2. Análisis con IA
python3 ai_researcher.py

# 3. Generar señales
python3 ai_strategy.py

# 4. Ejecutar trades
python3 executor.py

# 5. Generar reporte
python3 reporter.py
```

### Ejecutar todo el flujo

```bash
# Un ciclo completo
python3 orchestrator.py --once

# Ejecución continua (cada 60s)
python3 orchestrator.py --live

# Con output detallado
python3 orchestrator.py --once --debug
```

### Watchdog (ejecución automática)

```bash
# Iniciar watchdog
./run_watchdog_modular.sh

# O usar el cron job (ver abajo)
```

---

## 📊 Datos de Mercado

### Tokens Monitoreados

SOL · BTC · ETH · JUP · BONK · RAY · PENGU · FARTCOIN · MOODENG · GOAT · WIF · POPCAT

### APIs Usadas

| API | URL | Auth |
|-----|-----|------|
| Jupiter Price v2 | `api.jup.ag/price/v2` | Gratis |
| CoinGecko Markets | `api.coingecko.com/api/v3` | Gratis |
| Fear & Greed | `api.alternative.me/fng` | Gratis |

---

## 🤖 Configuración de LLM

### Fallback Chain

```
Claude Sonnet 4.6 (PRIMARY)
  ↓ [Rate limit]
MiniMax M2.5 (FALLBACK)
  ↓ [Si falla]
Manual fallback
```

### Modelos

| LLM | Uso | Estado |
|-----|-----|--------|
| Claude Sonnet 4.6 | Análisis complejo, investigación | PRIMARY |
| MiniMax M2.5 | Fallback principal | Configurado |
| MiniMax TTS (speech-2.8-hd) | Voz de alertas | Configurado |

### Configuración

**Archivo**: `llm_config.py` (compartido con BitTrader)

```python
# Fallback chain
CLAUDE_API_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL = "claude/claude-sonnet-4-6"

MINIMAX_API_URL = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"
MINIMAX_KEY = "sk-cp-..."  # Coding Plan key
```

---

## 📁 Archivos de Datos

Todos en `data/` (en `.gitignore`):

| Archivo | Contenido |
|---------|-----------|
| `market_latest.json` | Precios actuales de todos los tokens |
| `research_latest.json` | Análisis del AI Researcher |
| `strategy_llm.json` | Señales del AI Strategy |
| `latest_explanation.json` | Explicaciones del AI Explainability |
| `portfolio.json` | Capital, posiciones abiertas, P&L |
| `trade_history.json` | Historial completo de trades |
| `daily_report.json` | Último reporte generado |
| `price_history.json` | Historial de precios para RSI |

---

## 📈 Parámetros de Trading

### Risk Management

- **Riesgo por trade**: 2% del capital
- **Stop Loss**: 2.5% fijo
- **Take Profit**: 5% (2x SL)
- **Max posiciones**: 5 simultáneas
- **Pause automático**: drawdown > 8%
- **Stop automático**: drawdown > 10%

### Filtros de Entrada

- **Volatilidad mínima**: 0.5% (para evitar tokens estancados)
- **Market cap mínimo**: $50M (evitar meme coins microcap)
- **RSI sobrecompra**: > 70 (para SHORT)
- **RSI sobreventa**: < 30 (para LONG)

---

## 🔧 Modos de Operación

### Paper Trading (Default)

```bash
# Ejecutar en modo paper (simulación con precios reales)
python3 orchestrator.py --once
```

**Características:**
- ✅ Precios reales de Jupiter
- ✅ Sin riesgo financiero
- ✅ Backtesting en tiempo real
- ✅ Todos los features activos

---

### Real Trading (Opcional)

⚠️ **Solo cuando estés listo:**

```bash
# Configurar wallet en .env
HOT_WALLET_PRIVATE_KEY=[...]

# Ejecutar con --real
python3 orchestrator.py --once --real
```

**Advertencia:**
- 🔴 Trades reales con Solana
- 🔴 Pérdidas financieras posibles
- 🔴 Requiere configuración segura

---

## 📝 Reportes y Alertas

### Reporte Diario

```bash
python3 reporter.py --daily
```

Genera:
- Resumen del portafolio
- P&L del día
- Win rate
- Top/peor trades

### Alertas Automáticas

El sistema envía alertas por Telegram cuando:
- Una posición toca SL o TP
- Drawdown > 5%
- Sistema se detiene (via watchdog)

---

## 🔍 Debugging

### Ver logs del sistema

```bash
# Logs del sistema modular
cat ~/.config/solana-jupiter-bot/modular.log

# Logs del sistema legacy
cat ~/.config/solana-jupiter-bot/master.log
```

### Ejecutar agentes individualmente

```bash
# Test cada agente por separado
python3 market_data.py --debug
python3 ai_researcher.py --debug
python3 ai_strategy.py --debug
python3 executor.py --debug
python3 reporter.py --debug
```

### Ver estado actual

```bash
# Portfolio
cat data/portfolio.json | python3 -m json.tool

# Research
cat data/research_latest.json | python3 -m json.tool

# Señales
cat data/strategy_llm.json | python3 -m json.tool
```

---

## 🤖 Integración con BitTrader

Este sistema reutiliza la infraestructura de **BitTrader YouTube**:

### Archivos compartidos

| Archivo | Propósito |
|---------|-----------|
| `llm_config.py` | Configuración de LLMs (mismo archivo) |
| Fallback chain | Claude → MiniMax → Manual |
| API keys | Mismas credenciales de MiniMax |

### Similitudes de arquitectura

```
BitTrader YouTube:
  Scout (LLM) → Creator (LLM) → Producer (LLM) → Publisher

Solana Trading:
  Researcher (LLM) → Strategy (LLM) → Executor → Reporter
```

---

## 📊 Comparación con Sistema Legacy

| Característica | Legacy | Modular (IA) |
|--------------|--------|-------------|
| Arquitectura | Monolítico | Modular + IA |
| Análisis | Técnico solo | Técnico + Fundamental (LLM) |
| Señales | Reglas fijas | Reglas + IA (LLM) |
| Transparencia | Baja | Alta (explicaciones en lenguaje natural) |
| Escalabilidad | Difícil | Fácil (agregar agentes) |
| Ajustes | Requiere código | Prompts de LLM |

---

## 📅 Roadmap

### Semana 1: Base ✅
- [x] Reactivar sistema modular
- [x] Crear estructura de agentes
- [x] Implementar AI Researcher
- [x] Implementar AI Strategy
- [x] Implementar AI Explainability

### Semana 2: Integración (Pendiente)
- [ ] Actualizar orchestrator.py para flujo completo
- [ ] Pruebas en paper trading
- [ ] Afinar prompts de LLM
- [ ] Optimizar timeouts

### Semana 3: Optimización (Pendiente)
- [ ] Backtesting de estrategias IA
- [ ] Comparar vs sistema legacy
- [ ] Ajustar parámetros de riesgo
- [ ] Documentar resultados

### Semana 4: Producción (Pendiente)
- [ ] Monitoreo continuo
- [ ] Alertas de riesgo
- [ ] Reportes diarios automáticos
- [ ] Decidir migración definitiva

---

## 🆘 Solución de Problemas

### El sistema no genera señales

**Causa probable**: El LLM está en modo "CAUTIOUS" debido al Fear & Greed Index

**Solución**:
1. Revisar `data/research_latest.json`
2. Ver recomendación del Researcher
3. Ajustar prompts de Strategy para ser más agresivos

### Capital no se actualiza

**Causa probable**: El sistema se detuvo antes de cerrar posiciones

**Solución**:
1. Cerrar posiciones manualmente
2. Ejecutar: `python3 executor.py`
3. Reiniciar watchdog

### LLM rate limit

**Causa**: Claude Sonnet alcanzó el límite de uso

**Solución**: El sistema usa automáticamente MiniMax M2.5 como fallback

---

## 📝 Notas

- **Modo default**: Paper Trading (sin riesgo financiero)
- **LLMs**: Configurados con fallback automático
- **Watchdog**: Mantiene el sistema corriendo continuamente
- **Logs**: Guardados en `~/.config/solana-jupiter-bot/`
- **Git**: No commitear archivos en `data/` (en `.gitignore`)

---

## 📞 Contacto

- **GitHub**: https://github.com/enderjnets/Solana-Cripto-Trader
- **Telegram**: @Enderjh
- **Email**: enderjnets@gmail.com

---

**Última actualización**: 6 marzo 2026
**Versión**: v2.0 (con Agentes IA)
