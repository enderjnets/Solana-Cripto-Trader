# 🤖 Solana Trading Bot — Sistema de 5 Agentes

Sistema modular de trading para Solana con precios reales de Jupiter.
**Por defecto corre en Paper Trading** (simulación con precios reales).

## Arquitectura

```
orchestrator.py
    ├── 1. market_data.py   → Precios reales via Jupiter API v2
    ├── 2. risk_manager.py  → Kelly Criterion, SL/TP, drawdown limits
    ├── 3. strategy.py      → Señales: Momentum / Breakout / Oversold
    ├── 4. executor.py      → Abre/cierra posiciones (paper o real)
    └── 5. reporter.py      → Métricas, alertas Telegram, voz MiniMax
```

## Quickstart

```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents

# Un ciclo completo (paper trading)
python3 orchestrator.py --once

# Ciclo continuo cada 60s (paper trading)
python3 orchestrator.py --live

# Solo reporte con voz a Telegram
python3 orchestrator.py --report

# Output detallado
python3 orchestrator.py --once --debug
```

## Agentes Individuales

Cada agente puede ejecutarse de forma independiente:

```bash
python3 market_data.py          # Obtiene precios → data/market_latest.json
python3 market_data.py --debug  # Con precios por token

python3 risk_manager.py         # Evalúa riesgo → data/risk_report.json
python3 strategy.py             # Genera señales → data/signals_latest.json
python3 executor.py             # Ejecuta trades → data/portfolio.json
python3 reporter.py             # Reporte → data/daily_report.json
python3 reporter.py --daily     # Fuerza reporte completo con TTS
```

## Tokens Monitoreados

SOL · BTC · ETH · JUP · BONK · RAY · PENGU · FARTCOIN · MOODENG · GOAT · WIF · POPCAT

## Estrategias

| Estrategia | Condición | Dirección |
|-----------|-----------|-----------|
| **Momentum** | >3% en 5min, RSI 50-70 | Long/Short |
| **Breakout** | Rompe máximo 4h, vol alto | Long |
| **Oversold** | RSI <30, -5% en 24h, MC>$50M | Long |

## Parámetros de Riesgo

- **Riesgo por trade**: 2% del capital (Kelly simplificado)
- **Stop Loss**: 2.5% fijo
- **Take Profit**: 5% (2x SL)
- **Max posiciones**: 5 simultáneas
- **Pause automático**: drawdown >8%
- **Stop automático**: drawdown >10%

## Activar Trades Reales

⚠️ Solo cuando estés listo:

```bash
# Configura en .env del proyecto:
HOT_WALLET_PRIVATE_KEY=[...]  # Tu keypair de Solana

# Luego:
python3 orchestrator.py --once --real
# Te pedirá confirmación explícita
```

## Archivos de Estado

Todos en `data/` (en .gitignore, nunca se suben):

| Archivo | Contenido |
|---------|-----------|
| `market_latest.json` | Precios actuales de todos los tokens |
| `risk_report.json` | Evaluaciones de riesgo por token |
| `signals_latest.json` | Señales de trading activas |
| `portfolio.json` | Capital, posiciones abiertas, P&L |
| `trade_history.json` | Historial completo de trades |
| `daily_report.json` | Último reporte generado |
| `price_history.json` | Historial de precios para RSI |
| `alerts_state.json` | Control de alertas enviadas |

## APIs Usadas

| API | URL | Auth |
|-----|-----|------|
| Jupiter Price v2 | `api.jup.ag/price/v2` | Gratis, sin key |
| CoinGecko Markets | `api.coingecko.com/api/v3` | Gratis, sin key |
| Fear & Greed Index | `api.alternative.me/fng` | Gratis, sin key |
| MiniMax TTS | `api.minimax.io/v1/t2a_v2` | Key en bittrader/keys/ |
| Telegram Bot | `api.telegram.org/bot{token}` | Token en .env |

## Diferencias vs master_orchestrator.py

El archivo `master_orchestrator.py` **NO fue modificado** — es el sistema legacy.
Este sistema en `agents/` es completamente independiente con:

- ✅ Estado persistente real en `data/portfolio.json`
- ✅ Win rate calculado correctamente
- ✅ Sin variables paper simuladas (solo paper con precios reales)
- ✅ P&L basado en precios reales de Jupiter
- ✅ Arquitectura modular (cada agente testeable independiente)
- ✅ Modo real disponible (requiere activación explícita)
