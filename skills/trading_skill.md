# Skill: Trading Strategy - Jupiter Solana

## Descripción
Esta skill define las estrategias de trading para el bot de Jupiter en Solana, incluyendo reglas de entrada, salida, gestión de riesgo y parámetros de posición.

## Objetivos
- Maximizar PnL mientras se limita drawdown
- Diversificar entre major pairs y altcoins
- Adaptar estrategia según condiciones de mercado

---

## 📊 Indicadores Técnicos

### Indicadores Principales

| Indicador | Períodos | Uso |
|-----------|----------|-----|
| **RSI** | 14, 21 | Sobreventa/sobrecompra |
| **SMA** | 20, 50, 200 | Tendencia |
| **EMA** | 9, 21 | Cruces de precio |
| **VOLSMA** | 20 | Volumen vs promedio |

### Reglas de Entrada

#### Long (Compra)
```
Condición 1: RSI < 35 AND Precio > SMA_50
Condición 2: EMA_9 cruza encima de EMA_21
Condición 3: VOLSMA_20 > 1.2x promedio
→
ENTRADA: Comprar X% del capital
```

#### Short (Venta)
```
Condición 1: RSI > 70 AND Precio < SMA_50
Condición 2: EMA_9 cruza debajo de EMA_21
Condición 3: VOLSMA_20 > 1.2x promedio
→
ENTRADA: Vender X% del capital
```

### Reglas de Salida

#### Take Profit
```
TP_LARGO: +5% desde entrada
TP_CORTO: -5% desde entrada (buy to cover)
```

#### Stop Loss
```
SL_LARGO: -3% desde entrada
SL_CORTO: +3% desde entrada (sell to cover)
```

---

## 🎯 Perfiles de Riesgo

### Conservador
```yaml
risk_level: LOW
max_position_pct: 0.05  # 5% del capital por trade
stop_loss_pct: 0.02    # 2% stop loss
take_profit_pct: 0.04  # 4% take profit
max_daily_trades: 3
max_daily_loss_pct: 0.05  # 5% daily loss limit
```

### Moderado (Default)
```yaml
risk_level: MEDIUM
max_position_pct: 0.10  # 10% del capital por trade
stop_loss_pct: 0.03     # 3% stop loss
take_profit_pct: 0.06   # 6% take profit
max_daily_trades: 5
max_daily_loss_pct: 0.10  # 10% daily loss limit
```

### Agresivo
```yaml
risk_level: HIGH
max_position_pct: 0.15  # 15% del capital por trade
stop_loss_pct: 0.05     # 5% stop loss
take_profit_pct: 0.10   # 10% take profit
max_daily_trades: 8
max_daily_loss_pct: 0.15  # 15% daily loss limit
```

---

## 💰 Gestión de Posición

### Tamaño de Posición
```
Position_Size = (Account_Balance × Risk_Pct) / Stop_Loss_Distance

Ejemplo:
- Account: 10 SOL
- Risk_Pct: 10% → 1 SOL arriesgable
- Stop_Loss: 3%
- Position = 1 SOL / 0.03 = 33.33 SOL max
```

### Diversificación
```
Majors (SOL-USDC): Máximo 40% del capital
Altcoins (JUP-SOL): Máximo 30% del capital
Stablecoins: Mínimo 20% del capital
Reservado: 10% para oportunidades
```

---

## 📈 Configuración por Par

### SOL-USDC (Major)
```yaml
symbol: SOL-USDC
max_position_pct: 0.40
slippage_max: 0.01  # 1%
priority_fee: 1000    # lamports
use_jito: false
```

### JUP-SOL (Altcoin)
```yaml
symbol: JUP-SOL
max_position_pct: 0.15
slippage_max: 0.02   # 2%
priority_fee: 2000
use_jito: true
jito_tip: 500
```

### BONK-USDC (Microcap)
```yaml
symbol: BONK-USDC
max_position_pct: 0.05
slippage_max: 0.03   # 3%
priority_fee: 5000
use_jito: true
jito_tip: 1000
```

---

## 🔄 Condiciones de Mercado

### Bull Market
- Tendencia: Alcista (precio > SMA_200)
- RSI: Sobreventa en 30-40
- Posiciones: Larger, más frecuentes
- Take Profit: Más agresivo (+8-10%)

### Bear Market
- Tendencia: Bajista (precio < SMA_200)
- RSI: Sobrecompra en 60-70
- Posiciones: Más pequeñas, defensivas
- Take Profit: Conservador (+3-5%)

### Sideways
- Tendencia: Lateral
- RSI: Rango 40-60
- Posiciones: Solo en breaks
- Take Profit: Cercano (+4%)

---

## ⚠️ Reglas de Seguridad

### Siempre Verificar
1. Liquidez en Jupiter > $10,000 para el par
2. Slippage estimado < slippage_max
3. Fees totales < 2% del trade
4.余额 suficiente para fees (~0.01 SOL)

### Nunca
1. Trade contra tendencia mayor
2. Duplicate positions en mismo par
3. Ignorar stop-loss
4. Usar más del 30% daily

---

## 📋 Parámetros por Defecto

```yaml
# Configuración Global
default_risk_level: MEDIUM
max_concurrent_positions: 5
min_trade_size_sol: 0.01
max_slippage_pct: 0.02
priority_fee_auto: true
jito_tip_auto: true

# Timeframes
analysis_timeframe: 1h
confirmation_timeframe: 15m

# Rebalance
rebalance_threshold: 0.10  # 10% drift
rebalance_interval: 24h
```

---

## 🔧 Funciones del Agente

### analyze_market(symbol, timeframe)
Analiza condiciones de mercado para un símbolo.

### calculate_position_size(symbol, account_balance, stop_loss_pct)
Calcula tamaño óptimo de posición.

### check_entry_conditions(symbol)
Evalúa si hay condiciones para entrada.

### execute_trade(symbol, side, size, params)
Ejecuta trade según estrategia.

### monitor_position(position)
Monitorea posición abierta y gestiona salida.

---

*Skill Version: 1.0*
*Last Updated: 2026-02-09*
