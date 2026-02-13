# 🚀 Jupiter Solana Trading Bot 🦞

Bot de trading automatizado para Solana usando Jupiter DEX Aggregator, MiniMax M2.1 y arquitectura distribuida.

## 🎯 Estrategia: SOL + BTC + USDT

**Objetivo:** Acumular SOL y BTC, mantener USDT para oportunidades de mercado.

| Activo | Target | Min | Max |
|--------|--------|-----|-----|
| SOL | 40% | 30% | 60% |
| BTC | 40% | 30% | 60% |
| USDT | 20% | 10% | 40% |

**Reglas:**
- Rebalancear cuando desviación > 5%
- Comprar USDT cuando portafolio baje 10%
- Comprar SOL/BTC en dips > 15%
- Tomar ganancias en +15%

**Archivos USDT:**
- `usdt_config.py` - Configuración de estrategia
- `check_portfolio.py` - Verificar portafolio completo

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    JUPITER SOLANA TRADING BOT                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │  OpenClaw       │    │  MiniMax M2.1   │                  │
│  │  (Skills + Mem) │    │  (Reasoning)    │                  │
│  └────────┬────────┘    └────────┬────────┘                  │
│           │                        │                            │
│           └───────────┬────────────┘                            │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SISTEMA EXISTENTE (Enlace Simbólico)     │   │
│  │  ┌─────────────┐ ┌───────────┐ ┌──────────────────┐     │   │
│  │  │Coordinator  │ │  Workers  │ │ Strategy Miner  │     │   │
│  │  │ (Flask+SQL) │ │  (8x)     │ │ (Genetic Algo)  │     │   │
│  │  └─────────────┘ └───────────┘ └──────────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   JUPITER API V6                        │   │
│  │  /quote → /swap → Priority Fees + Jito Tips           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
/home/enderj/.openclaw/workspace/solana-jupiter-bot/
│
├── PROJECT.md              # Documentación principal
├── SOUL.md                 # Personalidad del agente
├── requirements.txt        # Dependencias Python
├── .env.example            # Variables de entorno (plantilla)
│
├── 📁 skills/
│   ├── trading_skill.md        # Estrategias de trading
│   └── jupiter_api_skill.md    # Integración Jupiter V6
│
├── 📁 tools/
│   ├── jupiter_client.py       # API client para Jupiter
│   └── solana_wallet.py        # Gestión de wallets SOL
│
├── 📁 config/
│   └── config.py                # Configuración centralizada
│
├── 📁 backtesting/
│   └── solana_backtester.py    # Backtester con Numba JIT
│
├── 📁 workers/
│   └── jupiter_worker.py       # Worker distribuido
│
├── 📁 dashboard/
│   └── solana_dashboard.py      # Streamlit dashboard
│
├── 📁 sistema_existente/       # → Proyecto Coinbase (enlace simbólico)
│   ├── coordinator_port5001.py
│   ├── strategy_miner.py
│   ├── numba_backtester.py
│   ├── crypto_worker.py
│   └── ...
│
└── 📁 tests/
    └── (pendiente)
```

## 🚀 Instalación Rápida

### 1. Clonar y Entrar

```bash
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
```

### 2. Crear Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `solana>=0.30.0` - SDK de Solana
- `solders>=0.20.0` - Tipos de Solana
- `httpx>=0.25.0` - Cliente HTTP async
- `numba>=0.58.0` - JIT acceleration (4000x speedup)
- `streamlit>=1.28.0` - Dashboard web
- `python-telegram-bot>=20.0` - Notificaciones Telegram

### 4. Configurar Variables de Entorno

```bash
cp .env.example .env
nano .env
```

Llenar:
```bash
# Red
SOLANA_RPC_DEVNET=https://api.devnet.solana.com

# Wallet
HOT_WALLET_ADDRESS=tu_direccion_aqui

# Telegram (opcional)
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id

# MiniMax M2.1 (opcional)
MINIMAX_API_URL=http://localhost:8090/v1
MINIMAX_API_KEY=tu_api_key
```

### 5. Crear Wallet

```bash
python tools/solana_wallet.py
```

Esto creará:
- Wallet encriptada: `~/.config/solana-jupiter-bot/wallet.enc`
- Información: `~/.config/solana-jupiter-bot/wallet_info.json`

---

## 📖 Uso

### Iniciar Dashboard

```bash
cd dashboard
streamlit run solana_dashboard.py
```

Acceder: http://localhost:8501

### Iniciar Workers

```bash
# Worker individual
python workers/jupiter_worker.py --coordinator http://localhost:5001

# Múltiples workers
for i in 1 2 3; do
    python workers/jupiter_worker.py \
        --coordinator http://localhost:5001 \
        --instance $i \
        --num-workers 3 &
done
```

### Configurar Coordinator

```bash
cd sistema_existente
python coordinator_port5001.py
```

---

## ⚙️ Configuración

### Parámetros de Trading

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `risk_level` | MEDIUM | LOW/MEDIUM/HIGH |
| `max_position_pct` | 10% | Máximo por trade |
| `stop_loss_pct` | 3% | Stop loss |
| `take_profit_pct` | 6% | Take profit |
| `max_daily_loss_pct` | 10% | Daily loss limit |

### Tokens Soportados

| Token | Mint Address |
|-------|--------------|
| SOL | `So11111111111111111111111111111111111111112` |
| USDC | `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` |
| USDT | `Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW` |
| JUP | `JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2` |
| BONK | `DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP` |

---

## 🧪 Testing

### Verificar Portafolio (SOL + BTC + USDT)

```bash
python3 check_portfolio.py
```

Expected output:
```
==================================================
  🦞 SOLANA PORTFOLIO (SOL + BTC + USDT)
==================================================
  Wallet: H9GF6t5hd...
  
  💰 SOL:  5.0000 x $80.00 = $400.00
  ₿ BTC:   0.00001250 x $42000.00 = $525.00
  ₮ USDT:  200.00 x $1.00 = $200.00
  
  📊 TOTAL: $1,125.00
  
  Allocation:
    SOL:  35.6%
    BTC:  46.7%
    USDT: 17.8%
==================================================
```

### Test de Backtester

```bash
python backtesting/solana_backtester.py
```

Expected output:
```
📊 Generating sample data...
🔧 Pre-computing indicators...
🧬 Creating sample genome...
🚀 Running backtest...
   PnL: +5.23% (example)
   Trades: 45
   Win Rate: 68.2%
```

---

## 📊 Benchmarks

| Componente | Valor |
|------------|-------|
| Backtest speed | **0.001s** (Numba JIT) |
| Speedup | **4000x** vs Python puro |
| Workers paralelos | **8** activos |
| Max PnL (backtest) | **$230.50** |

---

## 🛡️ Seguridad

### Medidas Implementadas

1. **Wallet Encriptada** - Fernet encryption
2. **Hot/Cold Separation** - Solo 10% en hot wallet
3. **Límites de Riesgo** - Stop-loss automático
4. **Logs de Auditoría** - Todas las decisiones

### Mejores Prácticas

- ✅ Usar testnet primero
- ✅ Verificar transacciones antes de ejecutar
- ✅ Limitar tamaño de posiciones
- ✅ Mantener funds mínimos en hot wallet
- ✅ Auditoría regular de logs

---

## 🔧 Desarrollo

### Agregar Nueva Estrategia

1. Editar `skills/trading_skill.md`
2. Definir reglas de entrada/salida
3. Testear con backtester
4. Deploy en dashboard

### Modificar Parámetros de API

Editar `config/config.py`:
```python
@dataclass
class JupiterConfig:
    default_slippage_bps: int = 50  # 0.5%
    priority_fee_default: int = 1000  # lamports
```

---

## 📝 APIs de Referencia

- **Jupiter API**: https://dev.jup.ag/api-reference
- **Solana Docs**: https://docs.solana.com
- **MiniMax M2.1**: https://huggingface.co/MiniMaxAI/MiniMax-M2.1
- **vLLM**: https://docs.vllm.ai

---

## 🤝 Contribuir

1. Fork el proyecto
2. Crear branch: `git checkout -b feature/nueva-feature`
3. Commit: `git commit -m "Agrega nueva feature"`
4. Push: `git push origin feature/nueva-feature`
5. Crear Pull Request

---

## 📄 Licencia

MIT License - Ver LICENSE

---

## 👤 Autor

**Ender Ocando** (@enderjh)

---

*Última actualización: 2026-02-09*
*Versión: 1.0.0*
