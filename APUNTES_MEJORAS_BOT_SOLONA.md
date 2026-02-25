# 📋 APUNTES - Mejoras para Hacer el Bot Solana Rentable

**Fecha:** 24 Febrero 2026 | **Bot:** Solana Trading Bot v3.1 | **Estado:** Paper Trading

---

## 🎯 **OBJETIVO PRINCIPAL**

**Hacer el bot RENTABLE aumentando el Win Rate de 16.7% a 40%+**

---

## 📊 **DIAGNÓSTICO ACTUAL**

### **Estado del Bot (24 Feb 2026)**

| Métrica | Valor Actual | Meta | Gap | Estado |
| ------- | ------------ | ---- | --- | ------ |
| **Win Rate** | 16.7% | 40%+ | -23.3% | ❌ CRÍTICO |
| **Total Trades** | 6 | - | - | - |
| **Wins** | 1 | - | - | - |
| **Losses** | 5 | - | - | - |
| **RR Ratio** | 1.5:1 | 2.5:1 | -1.0 | ❌ BAJO |
| **Risk/Trade** | 10% | 0.5% | +9.5% | ❌ MUY ALTO |
| **SL** | -2% | -1% | +1% | ⚠️ AGRESIVO |
| **TP** | +3% | +2.5% | +0.5% | ⚠️ AGRESIVO |
| **Leverage** | 5.0x | - | - | - |

### **Análisis de Últimos 6 Trades**

| Token | Entry | Exit | P&L | % | Motivo |
| ----- | ----- | ---- | --- | - | ------ |
| ETH | $1,820 | $1,858.59 | +$1.01 | +2.03% | ✅ TP |
| BTC | $63,348 | $62,695 | -$0.51 | -1.03% | ❌ SL |
| SOL | $77.28 | $76.45 | -$0.54 | -1.07% | ❌ SL |
| WBTC | $63,037 | $62,654 | -$0.54 | -1.08% | ❌ SL |
| BTC | $63,445 | $62,802 | -$0.50 | -1.01% | ❌ SL |
| SOL | $77,23 | $76.45 | -$0.50 | -1.01% | ❌ SL |

**Estadísticas Cerrados:**
- **TPs (Take Profits):** 1/6 trades (16.7%)
- **SLs (Stop Losses):** 5/6 trades (83.3%)
- **P&L Neto:** -$1.56
- **Average P&L:** -$0.26/trade

---

## ⚠️ **PROBLEMAS IDENTIFICADOS**

### **Problema 1: Win Rate Críticamente Bajo**

**Síntoma:**
- Solo 1 de 6 trades fue ganador (16.7%)
- 83.3% de trades terminaron en SL

**Causa Raíz:**
- TP muy agresivo (+3%) vs SL conservador (-2%)
- Mercado en consolidación (poca volatilidad)
- Estrategias muy optimistas (entry en dips sin confirmación fuerte)

**Impacto:**
- Con RR 1.5:1, se requiere mínimo 40% Win Rate para ser rentable
- Con 16.7%, el resultado esperado es -25% cada 50 trades

---

### **Problema 2: RR Ratio Desbalanceado**

**Síntoma:**
- SL -2% vs TP +3% = RR 1.5:1
- FTMO Strategy usa 2.5:1 (SL -1% vs TP +2.5%)
- Ratio de riesgo/recompensa es muy bajo

**Causa Raíz:**
- SL demasiado agresivo (solo 2%)
- TP demasiado ambicioso (requiere 3% de movimiento)
- Mercado no tiene la volatilidad necesaria para 3%

**Impacto:**
- Se necesitan más wins para compensar losses
- Un loss cancela 1.5 wins
- Presión psicológica por tener muchos SLs

---

### **Problema 3: Risk/Trade Demasiado Alto**

**Síntoma:**
- Risk/Trade: 10% del capital
- FTMO Strategy: 0.5% del capital
- Diferencia: 20x mayor riesgo

**Causa Raíz:**
- Tamaño de posición demasiado grande (5% del capital con 5x leverage = 25% exposure)
- Sin gestión de riesgo progresiva

**Impacto:**
- 5 losses consecutivos = -50% del capital
- Drawdown máximo alcanzable muy rápido
- Recuperación muy difícil (necesita +100% para recuperar -50%)

---

### **Problema 4: Filtros de Entrada Débiles**

**Síntoma:**
- Entrar en "dips" sin confirmación fuerte
- Múltiples entries en el mismo activo durante corto tiempo
- No hay filtros de volatilidad o tendencia

**Causa Raíz:**
- Estrategias basadas puramente en precio (dip detection)
- Sin análisis de momentum o RSI filters
- Sin filtros de volumen o liquidez

**Impacto:**
- Too many low-quality trades
- Entradas en medio de consolidación
- Muchos SLs por falta de movimiento

---

## 🔧 **SOLUCIONES Y CAMBIOS REQUERIDOS**

### **Cambio 1: Ajustar SL y TP**

#### **Opción A: RR Ratio 2.0:1 (Recomendada)**

**Nuevos Parámetros:**
```python
# Cambiar en workers/jupiter_worker.py y master_orchestrator.py
STOP_LOSS_PERCENT = -2.5%      # Aumentado de -2% a -2.5%
TAKE_PROFIT_PERCENT = +5.0%    # Aumentado de +3% a +5%
RISK_REWARD_RATIO = 2.0:1       # Mejorado de 1.5:1 a 2.0:1
```

**Beneficios:**
- ✅ SL más amplio (más tolerancia al ruido del mercado)
- ✅ TP más alto (mayor ganancia por trade)
- ✅ RR 2.0:1 = más fácil ser rentable (solo necesita 33% Win Rate)
- ✅ Menos trades cerrados por SL prematuro

**Riesgo:**
- ⚠️ Mayor exposure por trade (si no hay trailing stop)
- ⚠️ Posiciones abiertas más tiempo

---

#### **Opción B: RR Ratio 1.2:1 (Conservadora)**

**Nuevos Parámetros:**
```python
STOP_LOSS_PERCENT = -1.5%      # Reducido de -2% a -1.5%
TAKE_PROFIT_PERCENT = +1.8%    # Reducido de +3% a +1.8%
RISK_REWARD_RATIO = 1.2:1       # Reducido de 1.5:1 a 1.2:1
```

**Beneficios:**
- ✅ TP más cercano (más fácil de alcanzar)
- ✅ Más trades cerrados por TP
- ✅ Turnover más rápido (más trades por día)

**Riesgo:**
- ⚠️ Win Rate requerido aumenta a 45%+
- ⚠️ P&L por trade más bajo

---

### **Cambio 2: Implementar Trailing Stop**

**Archivo a Modificar:** `workers/jupiter_worker.py`

**Código a Agregar:**
```python
# Agregar después de entry position
class TrailingStop:
    def __init__(self, initial_sl_percent=2.0, trail_percent=0.5):
        self.initial_sl_percent = initial_sl_percent
        self.trail_percent = trail_percent
        self.highest_price = None
        self.lowest_price = None
        
    def update_trailing_stop(self, current_price, direction):
        if direction == 'LONG':
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
            
            # Calculate trailing stop
            trail_distance = self.highest_price * (self.trail_percent / 100)
            trailing_sl = self.highest_price - trail_distance
            
            # Return updated SL if it's better than initial
            initial_sl = current_price * (1 - self.initial_sl_percent / 100)
            return max(trailing_sl, initial_sl)
        
        elif direction == 'SHORT':
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
            
            # Calculate trailing stop
            trail_distance = self.lowest_price * (self.trail_percent / 100)
            trailing_sl = self.lowest_price + trail_distance
            
            # Return updated SL if it's better than initial
            initial_sl = current_price * (1 + self.initial_sl_percent / 100)
            return min(trailing_sl, initial_sl)
        
        return current_price

# Usar en el loop de monitoreo
trailing_stop = TrailingStop(initial_sl_percent=2.0, trail_percent=0.5)

for price in price_stream:
    # Update trailing stop
    new_sl = trailing_stop.update_trailing_stop(price, direction)
    
    # Check if hit
    if (direction == 'LONG' and price <= new_sl) or \
       (direction == 'SHORT' and price >= new_sl):
        close_position(reason='TRAILING_STOP')
```

**Beneficios:**
- ✅ Protege ganancias mientras el precio se mueve a favor
- ✅ Reduce riesgo cuando el precio corre
- ✅ Evita revertir ganancias a pérdidas
- ✅ SL dinámico adaptado a volatilidad

---

### **Cambio 3: Filtros de Entrada Mejorados**

#### **Filtro 1: Confirmación de Tendencia**

**Archivo a Modificar:** `workers/jupiter_worker.py`

**Código a Agregar:**
```python
# Agregar función de confirmación de tendencia
def confirm_trend(ema_short, ema_long, rsi, current_price):
    """
    Confirmación fuerte de tendencia antes de entrar
    
    Returns:
        True: Entrada permitida
        False: Rechazar entrada
    """
    # Filtro 1: EMA alignment (EMA short > EMA long para LONG)
    emma_aligned = ema_short > ema_long
    
    # Filtro 2: RSI no en sobrecompra/sobreventa
    rsi_ok = 30 < rsi < 70
    
    # Filtro 3: Price acción reciente (no lateral)
    price_action_ok = abs(current_price - ema_long) / ema_long > 0.01
    
    return emma_aligned and rsi_ok and price_action_ok

# Usar antes de entry
if confirm_trend(ema_short, ema_long, rsi, price):
    open_position()
else:
    log("Tendencia no confirmada - Esperar...")
```

---

#### **Filtro 2: Filtro de Volatilidad**

**Código a Agregar:**
```python
# Agregar función de volatilidad
def check_volatility(prices_24h, atr, current_price):
    """
    Solo entrar si hay suficiente volatilidad
    
    Returns:
        True: Volatilidad suficiente
        False: Volatilidad muy baja (evitar consolidación)
    """
    # Volatilidad mínima requerida (2% en 24h)
    min_volatility = 0.02
    
    # Calcular volatilidad real de 24h
    high = max(prices_24h)
    low = min(prices_24h)
    volatility = (high - low) / low
    
    # ATR filter (ATR debe ser > 1% del precio)
    atr_ok = atr > (current_price * 0.01)
    
    return volatility > min_volatility and atr_ok

# Usar antes de entry
if check_volatility(prices_24h, atr, price):
    open_position()
else:
    log("Volatilidad muy baja - Esperar...")
```

---

#### **Filtro 3: Filtro de RSI**

**Código a Agregar:**
```python
# Agregar función de RSI filter
def check_rsi_for_entry(rsi, direction):
    """
    Filtro de RSI para evitar entradas en extremos
    
    Returns:
        True: RSI permite entrada
        False: RSI en extremo (rechazar)
    """
    if direction == 'LONG':
        # Evitar LONG en sobrecompra (RSI > 70)
        return rsi < 70
    
    elif direction == 'SHORT':
        # Evitar SHORT en sobreventa (RSI < 30)
        return rsi > 30
    
    return True

# Usar antes de entry
if check_rsi_for_entry(rsi, direction):
    open_position()
else:
    log(f"RSI en extremo ({rsi}) - Rechazar entrada")
```

---

### **Cambio 4: Reducir Risk/Trade**

#### **Opción A: Reducir a 5%**

**Parámetro a Modificar:** `master_orchestrator.py`

```python
# Cambiar de 10% a 5%
MAX_RISK_PER_TRADE = 0.05  # 5% del capital
```

**Impacto:**
- ✅ 10 losses consecutivos = -50% (en lugar de -90%)
- ✅ Drawdown más lento y controlable
- ✅ Más margen para recuperación
- ✅ Menos presión psicológica

**Trade-off:**
- ⚠️ P&L por trade más pequeño
- ⚠️ Más tiempo para alcanzar objetivos

---

#### **Opción B: Dynamic Risk Based on Confidence**

**Código a Agregar:**
```python
# Agregar sistema de riesgo dinámico
def calculate_position_size(capital, confidence_score, max_risk=0.05):
    """
    Calcular tamaño basado en confianza de la señal
    
    confidence_score: 0.0 - 1.0 (baja - alta)
    max_risk: 5% máximo
    """
    # Riesgo dinámico: 1% (baja confianza) a 5% (alta confianza)
    dynamic_risk = 0.01 + (confidence_score * 0.04)
    dynamic_risk = min(dynamic_risk, max_risk)
    
    return capital * dynamic_risk

# Ejemplo de uso
confidence_score = calculate_confidence(ema, rsi, volatility)
position_size = calculate_position_size(capital, confidence_score)
```

**Beneficios:**
- ✅ Solo arriesgar más en señales de alta calidad
- ✅ Proteger capital en señales inciertas
- ✅ Optimizar risk-adjusted returns

---

### **Cambio 5: Límite de Trades por Token**

**Código a Agregar:**
```python
# Agregar tracking de trades por token
class TokenTradeTracker:
    def __init__(self, max_trades_per_token=2, cooldown_hours=4):
        self.max_trades_per_token = max_trades_per_token
        self.cooldown_hours = cooldown_hours
        self.token_trades = {}  # {token: [(timestamp, pnl), ...]}
    
    def can_trade(self, token):
        """
        Verificar si se puede hacer trade en este token
        
        Returns:
            True: Permitido
            False: Rechazar (muchos trades o en cooldown)
        """
        now = time.time()
        
        if token not in self.token_trades:
            return True
        
        # Check cooldown
        recent_trades = [
            t for t in self.token_trades[token]
            if t[0] > now - (self.cooldown_hours * 3600)
        ]
        
        # Actualizar lista
        self.token_trades[token] = recent_trades
        
        # Check max trades
        if len(recent_trades) >= self.max_trades_per_token:
            log(f"Token {token} ha alcanzado límite de trades")
            return False
        
        return True
    
    def record_trade(self, token, pnl):
        """Registrar trade completado"""
        if token not in self.token_trades:
            self.token_trades[token] = []
        
        self.token_trades[token].append((time.time(), pnl))

# Usar antes de entry
trade_tracker = TokenTradeTracker(max_trades_per_token=2, cooldown_hours=4)

if trade_tracker.can_trade(token):
    open_position()
else:
    log(f"Token {token} en cooldown o límite alcanzado")
```

**Beneficios:**
- ✅ Evitar overtrading en el mismo token
- ✅ Permitir que el mercado "respire" entre trades
- ✅ Reducir correlación de posiciones
- ✅ Mejorar diversificación

---

## 📊 **MÉTRICAS DE VALIDACIÓN**

### **KPIs a Monitorear**

| KPI | Actual | Meta | Medición |
| ---- | ------- | ---- | -------- |
| **Win Rate** | 16.7% | 40%+ | Cada 10 trades |
| **Profit Factor** | TBD | 1.5+ | Cada 20 trades |
| **Sharpe Ratio** | TBD | 1.0+ | Cada 50 trades |
| **Max Drawdown** | TBD | <10% | Continuo |
| **RR Ratio** | 1.5:1 | 2.0:1 | Fijo |
| **Avg P&L per Trade** | -$0.26 | +$2.0+ | Cada 10 trades |

---

### **Fase de Testing (Próximos 7 días)**

#### **Día 1-2: Implementar Cambios**
- [ ] Cambiar SL/TP (RR 2.0:1)
- [ ] Implementar trailing stop
- [ ] Agregar filtros de entrada
- [ ] Reducir risk/trade a 5%

#### **Día 3-7: Validar**
- [ ] Monitorear Win Rate (target: 30%+ en 20 trades)
- [ ] Verificar reducción de SLs (target: <60% de trades)
- [ ] Validar trailing stop effectiveness
- [ ] Documentar comportamiento de filtros

#### **Día 8-14: Optimizar**
- [ ] Ajustar parámetros si Win Rate < 35%
- [ ] Calibrar trailing stop distance
- [ ] Optimizar filtros de volatilidad
- [ ] Validar en diferentes condiciones de mercado

---

## 📋 **LISTA DE CHECKLIST**

### **Cambios Críticos (Implementar YA)**

- [ ] **Cambiar SL a -2.5%** (de -2%)
- [ ] **Cambiar TP a +5.0%** (de +3%)
- [ ] **Actualizar RR ratio a 2.0:1** (de 1.5:1)
- [ ] **Reducir risk/trade a 5%** (de 10%)
- [ ] **Agregar trailing stop 0.5%**

### **Cambios Importantes (Implementar en 1-2 días)**

- [ ] **Agregar filtro de confirmación de tendencia** (EMA + RSI)
- [ ] **Agregar filtro de volatilidad** (ATR > 1%)
- [ ] **Agregar filtro de RSI** (no entrar en extremos)
- [ ] **Limitar trades por token** (máx 2 cada 4h)

### **Cambios Opcionales (Implementar en 1 semana)**

- [ ] **Implementar riesgo dinámico** (basado en confianza)
- [ ] **Agregar filtro de volumen** (evitar illiquid tokens)
- [ ] **Implementar adaptive TP/SL** (basado en volatilidad)
- [ ] **Agregar notificaciones de eventos de mercado**

---

## 🎯 **PROYECCIONES DE RENTABILIDAD**

### **Escenario 1: Con Cambios Implementados (Optimista)**

**Suposiciones:**
- Win Rate: 40% (meta)
- RR Ratio: 2.0:1 (nuevo)
- Risk/Trade: 5% (nuevo)
- Trades/Día: 2

**Resultados Esperados (50 trades ~25 días):**
| Métrica | Valor |
| ------- | ----- |
| Wins | 20 trades (40%) |
| Losses | 30 trades (60%) |
| Ganancia por Win | +5% de posición = +$25 |
| Pérdida por Loss | -2.5% de posición = -$12.50 |
| P&L Neto | +$187.50 |
| ROI | +37.5% en 25 días |
| Win Rate requerido | 33% (para break-even con RR 2.0:1) |

**Conclusión:** ✅ **MUY RENTABLE**

---

### **Escenario 2: Mejora Moderada (Realista)**

**Suposiciones:**
- Win Rate: 35% (mejora de 16.7% → 35%)
- RR Ratio: 2.0:1
- Risk/Trade: 5%
- Trades/Día: 2

**Resultados Esperados (50 trades):**
| Métrica | Valor |
| ------- | ----- |
| Wins | 17.5 trades (35%) |
| Losses | 32.5 trades (65%) |
| Ganancia por Win | +$25 |
| Pérdida por Loss | -$12.50 |
| P&L Neto | +$31.25 |
| ROI | +6.25% en 25 días |
| Win Rate requerido | 33% |

**Conclusión:** ✅ **RENTABLE** (aunque modestamente)

---

### **Escenario 3: Win Rate Bajo (Pesimista)**

**Suposiciones:**
- Win Rate: 25% (mejora de 16.7% → 25%)
- RR Ratio: 2.0:1
- Risk/Trade: 5%

**Resultados Esperados (50 trades):**
| Métrica | Valor |
| ------- | ----- |
| Wins | 12.5 trades (25%) |
| Losses | 37.5 trades (75%) |
| Ganancia por Win | +$25 |
| Pérdida por Loss | -$12.50 |
| P&L Neto | -$187.50 |
| ROI | -37.5% en 25 días |
| Win Rate requerido | 33% |

**Conclusión:** ❌ **NO RENTABLE**

---

### **Resumen de Requerimientos**

| Parámetro | Valor Mínimo para Rentabilidad |
| --------- | --------------------------- |
| **Win Rate con RR 2.0:1** | 33%+ |
| **Win Rate con RR 1.5:1** | 40%+ |
| **Win Rate actual** | 16.7% |
| **Gap al mínimo** | 16.3% (para RR 2.0:1) |
| **Gap al mínimo** | 23.3% (para RR 1.5:1) |

**Conclusión:** Aumentar RR ratio a 2.0:1 reduce requerimiento de Win Rate de 40% a 33%.

---

## 📝 **NOTAS ADICIONALES**

### **Comparación con FTMO Strategy**

| Aspecto | Bot Solana | FTMO Strategy | Lección |
| ------- | ---------- | ------------- | -------- |
| Win Rate | 16.7% | 59.8% | FTMO es 3.6x mejor |
| RR Ratio | 1.5:1 | 2.5:1 | FTMO es más conservador en SL |
| Risk/Trade | 10% | 0.5% | FTMO es 20x más conservador |
| SL | -2% | -1% | FTMO permite más ruido |
| TP | +3% | +2.5% | FTMO es más realista |
| Validación | Paper | Live (FTMO funded) | FTMO está probado |

**Lecciones:**
- ✅ FTMO usa RR más alto (2.5:1 vs 1.5:1)
- ✅ FTMO usa riesgo MUY bajo (0.5% vs 10%)
- ✅ FTMO tiene Win Rate alto (59.8% vs 16.7%)
- ✅ La estrategia FTMO está validada en live trading

---

### **Errores Comunes Observados**

1. **Entrar en consolidación** (SOL, BTC, WBTC múltiples trades)
   - Solución: Filtro de volatilidad
   
2. **SL muy ajustado** (83% de trades cerraron por SL)
   - Solución: Aumentar SL a -2.5% o -3%

3. **Overtrading** (3-4 trades en el mismo token en corto tiempo)
   - Solución: Limitar trades por token + cooldown

4. **Sin trailing stop** (ganancias se revierten)
   - Solución: Implementar trailing stop

5. **Risk/trade demasiado alto** (10% vs 0.5% FTMO)
   - Solución: Reducir a 5% o usar riesgo dinámico

---

## 🔗 **REFERENCIAS**

### **Archivos a Modificar**

1. **`workers/jupiter_worker.py`**
   - Cambiar `STOP_LOSS_PERCENT`
   - Cambiar `TAKE_PROFIT_PERCENT`
   - Implementar `TrailingStop` class
   - Agregar filtros de entrada
   - Implementar `TokenTradeTracker`

2. **`master_orchestrator.py`**
   - Cambiar `MAX_RISK_PER_TRADE`
   - Implementar riesgo dinámico
   - Actualizar métricas de validación

3. **`config.json`**
   - Actualizar parámetros globales
   - Agregar flags de características

---

### **Documentos Relacionados**

- `REPORTE_COMPLETO_TRADING.md` - Reporte actual del bot
- `memory/2026-02-24.md` - Memoria del día
- `FTMO_v5.mq5` - Estrategia FTMO (referencia)

---

## ✅ **RESUMEN EJECUTIVO**

### **Qué hacer AHORA (prioridad máxima)**

1. ✅ **Cambiar SL/TP:** RR 1.5:1 → 2.0:1 (SL -2.5%, TP +5%)
2. ✅ **Reducir risk/trade:** 10% → 5%
3. ✅ **Implementar trailing stop:** 0.5% trail

### **Qué hacer EN 1-2 DÍAS (prioridad alta)**

1. ✅ Agregar filtros de entrada (tendencia, volatilidad, RSI)
2. ✅ Limitar trades por token (máx 2 cada 4h)
3. ✅ Monitorear Win Rate (target: 35%+ en 20 trades)

### **Qué esperar (resultados)**

| Escenario | Win Rate | Resultado (50 trades) | Conclusión |
| --------- | --------- | --------------------- | ---------- |
| Optimista | 40%+ | +$187.50 (+37.5%) | ✅ MUY RENTABLE |
| Realista | 35% | +$31.25 (+6.25%) | ✅ RENTABLE |
| Pesimista | 25% | -$187.50 (-37.5%) | ❌ NO RENTABLE |

**Mínimo requerido:** 33% Win Rate con RR 2.0:1

---

## 📞 **SOPORTE Y CONTACTO**

- **Telegram:** @Enderjh
- **GitHub:** @enderjnets
- **Email:** [Tu email]

---

**Última actualización:** 24 Febrero 2026 | 6:30 PM MST
**Bot:** Solana Trading Bot v3.1 | Drift Protocol Simulation
**Estado:** Paper Trading

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
