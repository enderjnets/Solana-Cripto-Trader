# Reporte de Revisión Completa — Solana Trading Bot
**Fecha:** 2026-03-16
**Revisado por:** BitTrader Team (Claude Opus 4.6)
**Score General:** 45/100 — CRÍTICO
**Estado:** Bot IDLE — sin trades desde 2026-03-14

---

## Resumen Ejecutivo

El bot de trading de Solana tiene una arquitectura modular bien diseñada con 11+ indicadores técnicos, 5 estrategias, y gestión de riesgo basada en Kelly Criterion. Sin embargo, presenta **bugs críticos** que impiden su correcto funcionamiento: el trailing stop es código muerto, las decisiones de rotación y cierre nunca se ejecutan, hay inconsistencias en nombres de campos entre módulos, y el bot lleva **2 días idle** sin abrir nuevas posiciones.

**P&L actual:** +$20.24 (+4.05%) sobre $500 de capital paper en 3 trades.

---

## 1. Arquitectura General

### 1.1 Estructura del Proyecto

```
Solana-Cripto-Trader/
├── master_orchestrator.py      # Pipeline legacy: Research → Backtest → Audit → PaperTrade
├── agents/
│   ├── orchestrator.py         # Pipeline modular ACTIVO: Market → Risk → Strategy → Executor
│   ├── executor.py             # Ejecución de trades (paper_drift mode)
│   ├── risk_manager.py         # Gestión de riesgo + Kelly Criterion + LLM decisions
│   ├── strategy.py             # 5 estrategias + 11 indicadores técnicos
│   ├── market_data.py          # Jupiter Price API + CoinGecko + Fear&Greed
│   ├── compound_engine.py      # Position sizing compuesto
│   ├── daily_target.py         # Evaluación de target diario (ACTIVO)
│   ├── daily_profit_target.py  # CÓDIGO MUERTO — nunca importado
│   ├── ai_strategy.py          # Estrategia con LLM
│   ├── ai_researcher.py        # Investigación con AI
│   ├── reporter.py             # Reportes
│   ├── daily_reporter.py       # Reporter diario (cron activo)
│   ├── run_watchdog_modular.sh # Watchdog del orchestrator
│   └── data/                   # JSONs de estado
└── (legacy bots: v3-v7, no activos)
```

### 1.2 Pipeline Activo (agents/orchestrator.py)

```
Market Data → Risk Manager → Strategy → Executor → Smart Rotation → Daily Target → LLM Decisions → Reporter
```

- Ciclo cada 60 segundos (hardcoded)
- Watchdog `run_watchdog_modular.sh` reinicia si cae (cada 120s)
- Datos de precio via Jupiter Price API v3 + CoinGecko enrichment
- 12 tokens monitoreados: SOL, BTC, ETH, JUP, BONK, RAY, PENGU, FARTCOIN, MOODENG, GOAT, WIF, POPCAT

### 1.3 Pipeline Legacy (master_orchestrator.py)

```
Researcher → Backtester → Auditor → PaperTrader
```

- 7 tokens: BTC, ETH, SOL, ADA, XRP, DOT, LINK
- **Backtester es FAKE** — genera Sharpe ratios con `hash(token) % 100` y siempre aprueba
- Estado guardado en `~/.config/solana-jupiter-bot/master_state.json`

---

## 2. Lógica de Trading

### 2.1 Señales de Entrada (strategy.py)

**5 estrategias evaluadas en orden de prioridad:**

| # | Estrategia | Condiciones clave |
|---|-----------|-------------------|
| 1 | `trend_momentum` | EMA20 > EMA50, RSI 40-70, MACD > signal, precio > EMA20 |
| 2 | `breakout` | Precio > Bollinger superior, volumen > 1.5x promedio |
| 3 | `oversold_bounce` | RSI < 30, precio < BB inferior, divergencia alcista |
| 4 | `golden_cross` | EMA20 cruza EMA50 al alza |
| 5 | `macd_cross` | MACD cruza signal al alza |

- Confianza mínima: 0.60 (strategy) → 0.70 (orchestrator)
- Mínimo 2 indicadores alineados
- SL/TP dinámico basado en ATR: SL = 2x ATR, TP = 4x ATR

**Problema:** El `break` después del primer match de estrategia crea una jerarquía oculta — `trend_momentum` siempre tiene prioridad.

### 2.2 Señales de Salida (executor.py)

| Mecanismo | LONG | SHORT |
|-----------|------|-------|
| Stop Loss | -2.5% | -5.0% (BUG: invertido) |
| Take Profit | +5.0% | +2.5% (BUG: invertido) |
| Liquidación | -4% move @ 5x leverage | -4% move @ 5x leverage |
| Timeout | 8 horas | 8 horas |
| Trailing Stop | 0.5% trail | **CÓDIGO MUERTO** |

### 2.3 Scoring de Señales

- **Long scoring:** base 0.0, suma puntos por EMA alignment, RSI position, MACD, volumen, Bollinger
- **Short scoring:** base 0.30 (sesgo permanente hacia shorts), mismos indicadores invertidos
- El sesgo de +0.30 en shorts puede generar señales SHORT con menos evidencia técnica

---

## 3. Gestión de Riesgo

### 3.1 Parámetros de Riesgo (INCONSISTENTES entre módulos)

| Parámetro | executor.py | risk_manager.py | compound_engine.py |
|-----------|-------------|-----------------|-------------------|
| Risk per trade | 2% fallback | 1.5% | 1.5% |
| Stop Loss | 2.5% | 3.0% | 3.0% |
| Take Profit | 5.0% | 7.5% | 7.5% |
| Max Drawdown | N/A | 10% | **15%** |
| Leverage | 3x default | Tiered 2-10x | N/A |
| Max Positions | 3 | 3 | N/A |
| Min Position | N/A | $8.0 | $8.0 |
| Max Exposure | N/A | 25% margin | 30% margin |

**Problema crítico:** Los valores por defecto del executor (2.5%/5%) difieren de risk_manager y compound_engine (3%/7.5%). Cuál "gana" depende de si la señal incluye sl_price/tp_price.

### 3.2 Position Sizing

- Kelly Criterion (half-Kelly) basado en las últimas 20 trades
- Rango: 0.5% — 3% del capital
- Compound sizing crece con el capital
- Leverage tiers: {score 1: 2x, 2: 3x, 3: 5x, 4: 7x, 5: 10x}

### 3.3 Drawdown — NO ENFORCED

El `MAX_DRAWDOWN = 0.10` está definido en risk_manager.py pero:
- El master_orchestrator.py **NUNCA** verifica drawdown en runtime
- El bot seguirá operando aunque pierda 50%+ del capital
- Solo el risk_manager evalúa drawdown y puede pausar, pero la función de emergency close requiere que haya 3 posiciones abiertas (BUG en línea 170)

---

## 4. Bugs Críticos Encontrados

### SEVERIDAD: CRÍTICA

| # | Bug | Archivo | Línea | Impacto |
|---|-----|---------|-------|---------|
| 1 | **Trailing stop es código muerto** — detecta trigger pero nunca cierra la posición | master_orchestrator.py | 1148-1153 | Posiciones no se cierran cuando trailing stop se activa |
| 2 | **Decisiones nunca se ejecutan** — Smart Rotation, Daily Target, Position Decisions generan recomendaciones pero no actúan | orchestrator.py | 119, 151, 184 | Rotación inteligente, cierre por target, y decisiones LLM son puramente decorativas |
| 3 | **Campo `opened_at` vs `open_time`** — risk_manager busca `opened_at`, executor guarda `open_time` | risk_manager.py / executor.py | 393, 498 / 328 | Detección de posiciones stale y scoring temporal completamente rotos |
| 4 | **`_save_portfolio()` typo** — función se llama `save_portfolio` sin underscore | executor.py | 628, 655 | NameError crash cuando kill switch o daily loss limit se activan |
| 5 | **SHORT SL/TP invertido** — TP a 2.5%, SL a 5.0% (ratio 0.5:1 vs 2:1 esperado) | master_orchestrator.py | 1186-1199 | Shorts toman ganancias pequeñas y pérdidas grandes |
| 6 | **`daily_loss` calcula siempre 0** — busca `closed_at` pero trades tienen `close_time` | executor.py | 648 | Límite de pérdida diaria nunca se activa |

### SEVERIDAD: ALTA

| # | Bug | Archivo | Línea | Impacto |
|---|-----|---------|-------|---------|
| 7 | **Backtester es fake** — siempre aprueba con stats sintéticos | master_orchestrator.py | 859-863 | No hay validación real de estrategias |
| 8 | **Watchdog pasa flags inexistentes** (`--live`, `--interval`) | run_watchdog_modular.sh | 25 | Orchestrator ignora los flags, corre con defaults |
| 9 | **Emergency close requiere 3 posiciones** — nunca activa con 1-2 | risk_manager.py | 170-171 | Emergencias no se manejan con pocas posiciones |
| 10 | **State file write no atómico** — crash durante save corrompe JSON | master_orchestrator.py | 174 | Pérdida de estado en caso de crash |
| 11 | **Funding rate simulation determinista** — `0.5 > 0.5` siempre False | master_orchestrator.py | 650-651 | Siempre recibe funding, nunca paga |
| 12 | **`total_pnl` se sobreescribe cada ciclo** — solo suma el ciclo actual | master_orchestrator.py | 1232 | Historial de P&L se pierde |
| 13 | **`daily_pnl` nunca se resetea** — calcula lifetime PnL, no diario | master_orchestrator.py | 1256 | Métrica diaria es incorrecta |

### SEVERIDAD: MEDIA

| # | Bug | Archivo | Línea | Impacto |
|---|-----|---------|-------|---------|
| 14 | **`daily_profit_target.py` es código muerto** — nunca importado, directorio incorrecto | daily_profit_target.py | Todo | 650 líneas de código inútil |
| 15 | **`volume_24h` siempre 0 si CoinGecko falla** | market_data.py | 165 | OBV y VWAP calculan sobre 0 volumen |
| 16 | **Short scoring tiene sesgo +0.30** — favorece shorts artificialmente | strategy.py | 635 | Asimetría en generación de señales |
| 17 | **ATR fallback 0.5%** cuando no hay datos high/low | strategy.py | 269-272 | SL/TP artificialmente ajustados en crypto volátil |
| 18 | **`MIN_POSITION_USD = $8` puede exceder capital** | compound_engine.py | 169 | Posición forzada mayor al capital disponible |
| 19 | **`detect_market_trend` sin else branch** | master_orchestrator.py | 396-401 | Posible UnboundLocalError en trend |
| 20 | **Momentum "5min" es realmente ~2min** (ciclo de 120s) | market_data.py | 155 | Etiqueta engañosa en datos |

---

## 5. Estado Actual del Bot

### 5.1 Portfolio

```
Capital:          $520.24
Capital Inicial:  $500.00
Retorno Total:    +4.05% (+$20.24)
Drawdown:         0.0%
Posiciones:       0 abiertas
Modo:             paper_drift
```

### 5.2 Historial de Trades (Sistema Modular)

| # | Token | Dirección | Estrategia | P&L | Razón de cierre |
|---|-------|-----------|-----------|-----|-----------------|
| 1 | GOAT | Long | trend_momentum | +$12.38 (+9.91%) | Risk Agent — near TP |
| 2 | SOL | Long | macd_cross | +$14.57 (+11.66%) | Risk Agent — near TP |
| 3 | BTC | Short | macd_cross | -$6.71 (-5.37%) | Manual close (audit) |

**Win Rate:** 66.7% (2/3)
**Profit Factor:** 3.98 ($26.95 / $6.71)

### 5.3 Mercado Actual (datos stale — 3 días)

- **Todos los tokens RSI > 77** (deeply overbought)
- **Fear & Greed Index: 15** (Extreme Fear)
- Divergencia significativa: precios alcistas + sentimiento bearish
- **Datos de mercado no se actualizan desde 2026-03-13**

### 5.4 Estado de Componentes

| Componente | Estado | Último Cambio |
|-----------|--------|---------------|
| Orchestrator Pipeline | Idle (sin señales) | Mar 14 |
| Market Data | **STALE (3 días)** | Mar 13 |
| Strategy Engine | Sin señales (RSI overbought) | Mar 13 |
| Executor | Sin trades | Mar 14 |
| Risk Manager | Healthy | Mar 14 |
| LLM Health | OK (0 fallos) | Mar 14 |
| Daily Reporter | Activo (cron) | Mar 15 |
| Watchdog | Activo | Continuo |

---

## 6. Logs y Comportamiento Reciente

### 6.1 Sistema Legacy (master_orchestrator.py)

- **Offline desde Feb 19-20** (~25 días sin actividad)
- 6 restarts en Feb 18 por crash loop (cada ~30 min)
- CoinGecko 429 rate limiting frecuente
- Señales generadas pero **0 trades ejecutados**
- Evento crítico en ciclo 60: pricing failure causó P&L fantasma de -$95 (-19%)

### 6.2 Sistema Modular (agents/)

- Último trade: Mar 14 (cierre manual de BTC short en auditoría)
- 720+ ciclos sin abrir nuevos trades (reportado en programmer_tasks.json)
- Causa raíz: `paper_drift` mode bloquea capital + RSI overbought global
- Daily reporter (cron) reporta "No activity today" repetidamente

### 6.3 Score de Auditoría

**Última auditoría (2026-03-15): 72/100 — WARNING**
- Actividad de trading: Baja
- Gestión de riesgo: Adecuada
- Datos de mercado: Stale

---

## 7. Mejoras Recomendadas

### Prioridad 1 — Bugs Críticos (Hacer AHORA)

1. **Implementar ejecución de decisiones en orchestrator.py:**
   - Smart Rotation debe llamar a `executor.close_position()` cuando detecta posiciones stale
   - Daily Target debe ejecutar cierre masivo cuando se alcanza el target
   - Position Decisions CLOSE/REDUCE deben pasar al executor

2. **Corregir campo `opened_at` → `open_time`** en risk_manager.py (líneas 393, 498) para que la detección de posiciones stale funcione

3. **Corregir typo `_save_portfolio` → `save_portfolio`** en executor.py (líneas 628, 655)

4. **Implementar trailing stop real** en master_orchestrator.py — agregar `should_close = True` cuando `sl_triggered` es True

5. **Corregir SHORT SL/TP** — invertir la lógica para que shorts tengan ratio 2:1

6. **Corregir cálculo de daily loss** — cambiar `closed_at` → `close_time` en executor.py línea 648

### Prioridad 2 — Mejoras de Robustez

7. **Unificar parámetros SL/TP/drawdown** — crear un solo archivo de configuración (`config.json`) y que todos los módulos lo lean:
   ```json
   {
     "risk_per_trade": 0.015,
     "stop_loss_pct": 0.03,
     "tp_multiplier": 2.5,
     "max_drawdown_pct": 0.10,
     "max_positions": 3,
     "default_leverage": 3
   }
   ```

8. **Implementar escritura atómica de estado** — write to temp file + `os.rename()`

9. **Enforcer MAX_DRAWDOWN en runtime** — agregar check al inicio de cada ciclo que detenga el trading si drawdown > 10%

10. **Implementar retry logic en market_data.py** — 3 reintentos con backoff exponencial para Jupiter y CoinGecko

11. **Corregir watchdog flags** — quitar `--live --interval 120` o implementar estos args en orchestrator.py argparse

12. **Agregar validación de estado al cargar JSON** — try/except en todas las lecturas de archivos de estado

### Prioridad 3 — Mejoras Funcionales

13. **Reemplazar backtester fake** con backtesting real usando price_history.json

14. **Eliminar código muerto:**
    - `daily_profit_target.py` (nunca importado)
    - Pipeline legacy de master_orchestrator.py si ya no se usa

15. **Implementar reset diario de métricas** — daily_pnl debe resetearse a las 00:00 UTC

16. **Diversificar fuentes de datos** — agregar fallback a otro proveedor cuando CoinGecko retorna 429

17. **Agregar alertas de inactividad** — si 0 trades en 24h+ y hay capital disponible, enviar notificación

18. **Corregir sesgo de short scoring** — evaluar si el base 0.30 es intencional o un bug, documentar la decisión

19. **Warm-up period** — la estrategia necesita 52 data points para EMA50. Documentar que el bot es ciego ~1 hora después de restart

20. **Implementar live trading** — `real_open_position()` es un stub que retorna None. Integrar con Jupiter Swap API para ejecución real

---

## 8. Matriz de Riesgo Operacional

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|-----------|
| Bot idle por RSI overbought global | ALTA | MEDIO | Agregar estrategia mean-reversion, ampliar condiciones de entrada |
| Crash durante save corrompe estado | MEDIA | ALTO | Escritura atómica + backups |
| CoinGecko rate limit deja bot ciego | ALTA | ALTO | Múltiples fuentes de datos + cache inteligente |
| Kill switch/daily loss crash por typo | ALTA | CRÍTICO | Corregir `_save_portfolio` → `save_portfolio` |
| Drawdown sin límite en producción | MEDIA | CRÍTICO | Enforcer MAX_DRAWDOWN check cada ciclo |
| Campo `opened_at` vs `open_time` | CONFIRMADO | ALTO | Unificar nombres de campo |

---

## 9. Conclusiones

### Lo Bueno
- Arquitectura modular bien pensada con separación de responsabilidades
- 11 indicadores técnicos + 5 estrategias diversificadas
- Kelly Criterion para position sizing
- Sistema de scoring cuantitativo + LLM para decisiones
- Paper trading permite validar sin riesgo real
- Profit factor de 3.98 en los 3 trades ejecutados

### Lo Malo
- **Múltiples decisiones se evalúan pero nunca se ejecutan** (Smart Rotation, Daily Target, Position Decisions)
- **Inconsistencias de parámetros** entre 3+ módulos (SL, TP, drawdown)
- **Nombres de campo incompatibles** (`opened_at` vs `open_time`, `closed_at` vs `close_time`)
- Bot efectivamente idle sin trades por 2+ días

### Lo Feo
- Trailing stop es puro código muerto
- Backtester fake que siempre aprueba todo
- `_save_portfolio` typo que crasheará en el peor momento posible (emergency close)
- 650 líneas de código muerto en `daily_profit_target.py`
- SHORT positions con ratio riesgo/recompensa invertido (0.5:1)
- Datos de mercado stale de 3 días

### Recomendación Final

**No mover a producción (live trading) hasta resolver los bugs de Prioridad 1.** El bot en su estado actual es funcional para paper trading básico, pero los bugs de ejecución de decisiones, campo names mismatch, y SL/TP invertido en shorts causarán pérdidas reales significativas en un entorno live.

---

*Generado por BitTrader Team — Claude Opus 4.6*
*Fecha de revisión: 2026-03-16*
*Archivos analizados: 15 módulos Python, 3 scripts shell, 15 archivos de estado JSON, 10+ logs*
