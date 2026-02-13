# Skill: Trading Puro - Active Trading Strategy v1.0

## Descripción
Estrategia de trading puro para hacer crecer $500 mediante operaciones activas en Jupiter DEX.

## Objetivos
- Convertir capital inicial en más mediante trading
- Buscar oportunidades 24/7
- Reinvertir ganancias automáticamente
- Mantener reserva USDT para oportunidades

---

## 🎯 REGLAS FUNDAMENTALES

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Capital inicial** | $500 | USD equivalente |
| **Riesgo por trade** | 5% | $25 máximo por operación |
| **Stop loss** | -10% | Cerrar posición en -10% |
| **Take profit** | +20% | Cerrar posición en +20% |
| **Límite diario** | -15% | No perder más del 15% diario |
| **Meta mensual** | +50% | Crecimiento objetivo |

---

## 💰 GESTIÓN DE CAPITAL

### Por Trade
```
Position_Size = (Capital × 0.05) / Stop_Loss_Distance

Ejemplo:
- Capital: $500
- Riesgo: 5% = $25
- Stop Loss: 10%
- Position = $25 / 0.10 = $250 max por trade
```

### Reinversión de Ganancias
```
70% → Reinvestir en nuevos trades
30% → Acumular como reserva USDT
```

### Reserva USDT
```
Objetivo: 30% del portafolio en USDT
Trigger de compra: Mercado baja >15%
Trigger de venta: Mercado sube >20%
```

---

## 📊 ASIGNACIÓN DE CAPITAL

| Par | Peso | Riesgo | Descripción |
|-----|------|--------|-------------|
| SOL-USDC | 30% | Bajo | Major pair, alta liquidez |
| cbBTC-USDC | 25% | Bajo | Bitcoin en Solana |
| JUP-SOL | 15% | Medio | DeFi growth |
| RAY-SOL | 10% | Medio | DeFi established |
| BONK-USDC | 10% | Alto | Meme con potencial |
| WIF-SOL | 10% | Alto | Meme trend |

---

## 🔄 FLUJO DE TRADING

```
1. AGENTE SCOUT
   └─ Scanea Jupiter DEX para oportunidades
   └─ Filtra por liquidez > $10,000
   └─ Identifica pares con momentum

2. AGENTE ANALYST
   └─ Analiza RSI, MACD, volumen
   └─ Calcula risk/reward ratio
   └─ Determina tamaño de posición

3. AGENTE TRADER
   └─ Ejecuta entrada con slippage < 2%
   └─ Configura stop loss automático
   └─ Configura take profit automático

4. AGENTE RISK MANAGER
   └─ Monitorea exposición total
   └─ Verifica límites diarios
   └─ Cierra posiciones si necesario

5. AGENTE ACCOUNTANT
   └─ Calcula ganancias/pérdidas
   └─ Reinvierte 70%
   └─ Acumula 30% en USDT
```

---

## 📈 ENTRADA Y SALIDA

### Condiciones de Entrada (LONG)
```
1. RSI < 40 (sobreventa)
2. Precio > SMA_20 (tendencia alcista)
3. Volumen > 1.5x promedio
4. Momentum positivo
→
ENTRADA: Comprar con stop loss -10%, take profit +20%
```

### Condiciones de Entrada (SHORT)
```
1. RSI > 70 (sobrecompra)
2. Precio < SMA_20 (tendencia bajista)
3. Volumen > 1.5x promedio
4. Momentum negativo
→
ENTRADA: Vender con stop loss +10%, take profit -20%
```

### Gestión de Posición
```
Premio/Riesgo mínimo: 2:1
Trailing stop: Activar en +10%
Split take profit: 50% en +15%, 50% en +25%
```

---

## 🛡️ REGLAS DE SEGURIDAD

### Siempre
1. Verificar liquidez Jupiter > $10,000
2. Slippage estimado < 2%
3. Fees totales < 1% del trade
4.余额 suficiente para fees (~0.01 SOL)

### Nunca
1. Trade sin stop loss
2. Exceder 5% riesgo por trade
3. Trade en pares con < $10,000 liquidez
4. Ignorar límites diarios

### Límites Diarios
```
Max trades: 10
Max pérdida diaria: -15%
Max ganancia diaria: +50% (tomar profits)
```

---

## 📋 CONFIGURACIÓN POR DEFECTO

```yaml
# Capital
initial_capital: 500
min_trade_size: 10  # USD

# Riesgo
risk_per_trade: 0.05  # 5%
stop_loss_default: 0.10  # 10%
take_profit_default: 0.20  # 20%
daily_loss_limit: 0.15  # 15%

# Reinversión
reinvest_rate: 0.70  # 70%
reserve_rate: 0.30     # 30%

# USDT Reserve
usdt_target: 0.30
usdt_buy_trigger: -0.15  # Buy dip > 15%
usdt_sell_trigger: 0.20  # Take profit > 20%

# JUPITER
max_slippage: 0.02
priority_fee: 1000  # lamports
use_jito: true
jito_tip: 1000
```

---

## 🔧 FUNCIONES DEL AGENTE

### scout_opportunities()
```
Scan Jupiter DEX for trading opportunities
Return: [{pair, liquidity, volume, signal_strength}]
```

### analyze_entry(pair, side)
```
Technical analysis for entry conditions
Return: {entry_price, stop_loss, take_profit, confidence}
```

### calculate_position_size(pair, risk)
```
Calculate optimal position size based on risk
Return: position_size_in_usd
```

### execute_trade(pair, side, size)
```
Execute trade via Jupiter API
Return: {tx_signature, entry_price, status}
```

### monitor_position(position)
```
Track open position
Close on: stop_loss, take_profit, or signal reversal
Return: {pnl, status}
```

### manage_capital()
```
Track portfolio value
Reinvest 70% of profits
Accumulate 30% in USDT
Return: {total_value, reinvested, reserved}
```

---

## 📊 KPIs DE ÉXITO

| Métrica | Objetivo | Mínimo aceptable |
|---------|----------|------------------|
| Win rate | 60% | 50% |
| Avg PnL per trade | +8% | +5% |
| Monthly growth | +50% | +20% |
| Max drawdown | -15% | -25% |
| Sharpe ratio | >1.5 | >1.0 |

---

*Strategy Version: 1.0*
*Last Updated: 2026-02-13*
*Objective: Grow $500 through active trading*
