# 🚀 Solana Multi-Agent Trading System

Sistema de trading automatizado con múltiples agentes de IA para Solana.

## 🤖 Agentes

| Agente | Función |
|--------|---------|
| 👁️ **Market Scanner** | descubre nuevos tokens |
| 📊 **Anal Escanea mercados,yst** | Analiza oportunidades de trading |
| ⚖️ **Risk Manager** | Aprueba trades, gestiona TP/SL |
| 🤖 **Trader** | Ejecuta trades (paper trading) |
| 👑 **CEO** | Supervisa meta diaria (5%) |
| 🧪 **Strategy Generator** | Genera nuevas estrategias |
| 📈 **Backtester** | Prueba estrategias |
| ⚡ **Optimizer** | Optimiza parámetros |

## 🎯 Características

- **Meta diaria**: 5% profit
- **Trading ultra sensible**: Detecta movimientos desde 0.5%
- **Multi-posición**: Hasta 6 posiciones simultáneas
- **Paper Trading**: $500 capital virtual
- **Descubrimiento de tokens**: DEX Screener, Birdeye, Raydium

## 🚀 Instalación

```bash
# Clonar
git clone https://github.com/enderjnets/Solana-Cripto-Trader.git
cd Solana-Cripto-Trader

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install solana solders requests httpx python-dotenv

# Ejecutar
python multi_agent_trader.py
```

## ⚙️ Configuración

Editar `.env`:
```bash
SOLANA_RPC_DEVNET=https://api.devnet.solana.com
HOT_WALLET_ADDRESS=tu_direccion_wallet
```

## 📊 Parámetros de Trading

| Parámetro | Valor |
|-----------|-------|
| Meta diaria | 5% |
| Take Profit | 1.5% |
| Stop Loss | 1% |
| Tamaño trade | 20% capital |
| Posiciones máx | 6 |
| Ciclo | 20 segundos |

## 📈 Tokens Monitoreados

- SOL, BTC, ETH, USDC, USDT
- BONK, WIF, PEPE (meme coins)

## 💰 Estado Actual

```
Capital: $500 (paper)
Meta: 5% diario
```

## 📁 Estructura

```
├── multi_agent_trader.py   # Sistema principal
├── agents/                 # Agentes del sistema
├── tools/                 # Herramientas (Jupiter, wallet)
├── config/                # Configuración
└── .env                  # Variables de entorno
```

## 🔧 Desarrollo

```bash
# Editar estrategia
nano multi_agent_trader.py

# Ver logs
tail -f /tmp/multi_agent.log

# Estado
cat ~/.config/solana-jupiter-bot/multi_agent_state.json
```

## 📝 Licencia

MIT

---

*Versión: 2.0*
*Actualizado: 2026-02-15*
