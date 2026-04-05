# ============================================================================
# RESEARCHER AGENT v3.2 (With Entry Filters)
# ============================================================================

"""
🔧 MEJORAS v3.2 IMPLEMENTADAS:
1. Confirmación de tendencia (EMA alignment + RSI filter)
2. Filtro de volatilidad (mínimo 2% cambio en 24h)
3. Filtro de RSI (no entrar en extremos 30-70)
4. Límite de trades por token (máx 2 cada 4 horas)
"""

class ResearcherAgent:
    """Investiga oportunidades de mercado con lógica bidireccional

    🔧 V3.2: Mejoras implementadas
    - Filtros de entrada (tendencia, volatilidad, RSI)
    - Confirmación antes de generar señales
    """

    def __init__(self, state, trade_tracker=None):
        self.state = state
        self.trade_tracker = trade_tracker or TokenTradeTracker()

    async def run(self):
        self.state.data["agents"]["researcher"]["status"] = "running"
        self.state.save()
        self.state.log("🔍 RESEARCHER: Analizando mercado...")

        # Fetch real prices with indicators
        prices = get_real_prices()

        # Detect trend and get indicators
        market_trend, prices = detect_market_trend(prices)
        self.state.data["market_trend"] = market_trend
        self.state.data["market_indicators"] = {
            "ema_fast": next(iter(prices.values()), {}).get("ema_fast", 0) if prices else 0,
            "ema_slow": next(iter(prices.values()), {}).get("ema_slow", 0) if prices else 0,
            "rsi": next(iter(prices.values()), {}).get("rsi", 50) if prices else 50
        }

        self.state.log(f"   📊 Tendencia del mercado: {market_trend.upper()}")
        for token, data in prices.items():
            ema_fast = data.get("ema_fast", 0)
            ema_slow = data.get("ema_slow", 0)
            rsi = data.get("rsi", 50)
            self.state.log(f"   📊 {token}: ${data['price']:,.2f} ({data['change']:+.2f}%) | EMA: {ema_fast:.2f}/{ema_slow:.2f} | RSI: {rsi:.1f}")

        # Generate opportunities based on trend and signals
        opportunities = []

        for token, data in prices.items():
            current = data["price"]
            change = data["change"]
            ema_fast = data.get("ema_fast", 0)
            ema_slow = data.get("ema_slow", 0)
            rsi = data.get("rsi", 50)

            # 🔧 V3.2: Verificar si se puede hacer trade en este token
            if not self.trade_tracker.can_trade(token):
                self.state.log(f"   ⚠️ {token}: Cooldown o límite de trades alcanzado - saltando")
                continue

            # Bidirectional signal logic
            signal = None
            target = current
            confidence = 0.5
            direction = LONG

            if market_trend == "bullish":
                # PUMP signal - Buy on upward momentum
                if change > 3:
                    direction = LONG
                    signal = "pump"
                    target = current * 1.05  # V3.2: +5% TP
                    confidence = 0.7
                    self.state.log(f"   🔥 {token}: PUMP detectado ({change:+.1f}%)")

                # Trend following - LONG on uptrend
                elif ema_fast > ema_slow and change > 1:
                    direction = LONG
                    signal = "long"
                    target = current * 1.05  # V3.2: +5% TP
                    confidence = 0.6
                    self.state.log(f"   📈 {token}: LONG en alza ({change:+.1f}%)")

            elif market_trend == "bearish":
                # DIP signal - SHORT on downward momentum
                if change < -2:
                    direction = SHORT
                    signal = "short"
                    target = current * 0.95  # V3.2: -5% TP
                    confidence = 0.7
                    self.state.log(f"   💥 {token}: SHORT en bajada ({change:+.1f}%)")

                # Don't buy dips in bearish market
                elif change < -1:
                    continue  # Skip buying opportunities in bearish market

            elif market_trend == "neutral":
                # Range trading - Buy at support, Sell at resistance
                if ema_fast < ema_slow and change > -2:
                    direction = LONG
                    signal = "long"
                    target = current * 1.05  # V3.2: +5% TP
                    confidence = 0.5
                    self.state.log(f"   📊 {token}: RANGE LONG en soporte")
                elif ema_fast > ema_slow and change > 2:
                    direction = SHORT
                    signal = "short"
                    target = current * 0.95  # V3.2: -5% TP
                    confidence = 0.5
                    self.state.log(f"   📊 {token}: RANGE SHORT en resistencia")

            # 🔧 V3.2: Aplicar filtros antes de agregar oportunidad
            if signal:
                # Filtro 1: Volatilidad mínima
                if ENTRY_FILTER_ENABLED and not check_volatility(change):
                    self.state.log(f"   ⚠️ {token}: Volatilidad muy baja ({change:+.2f}%) - rechazando")
                    continue

                # Filtro 2: Confirmación de tendencia
                if ENTRY_FILTER_ENABLED and not confirm_trend(ema_fast, ema_slow, rsi, current):
                    self.state.log(f"   ⚠️ {token}: Tendencia no confirmada - rechazando")
                    continue

                # Filtro 3: RSI en extremos
                if ENTRY_FILTER_ENABLED and not check_rsi_for_entry(rsi, direction):
                    self.state.log(f"   ⚠️ {token}: RSI en extremo ({rsi:.1f}) - rechazando")
                    continue

                opportunities.append({
                    "token": token,
                    "signal": signal,
                    "entry": current,
                    "target": round(target, 2),
                    "confidence": min(0.95, confidence),
                    "change_24h": change,
                    "direction": direction
                })

        self.state.data["agents"]["researcher"]["findings"] = opportunities
        self.state.data["agents"]["researcher"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["researcher"]["status"] = "idle"
        self.state.data["opportunities"] = opportunities
        self.state.save()

        self.state.log(f"✅ RESEARCHER: {len(opportunities)} oportunidades encontradas (después de filtros)")
        return opportunities
