# ============================================================================
# PAPER TRADING AGENT v3.2 (Drift Protocol Simulation con Mejoras)
# ============================================================================

"""
🔧 MEJORAS v3.2 IMPLEMENTADAS:
1. RR Ratio 2.0:1 (SL -2.5%, TP +5.0%)
2. Risk/Trade reducido a 5% (de 10%)
3. Trailing Stop implementado (0.5% trail)
4. Registro de trades en TokenTradeTracker
5. Mejorados cálculos de Drift Protocol
"""

class PaperTradingAgent:
    """Ejecuta trades en modo paper trading con simulación Drift Protocol

    🔧 MEJORAS v3.2:
    - Simulación Leverage (5x por defecto)
    - Trading fees (0.05% por trade)
    - Liquidation thresholds (80% collateral)
    - Borrowing fees para SHORTs
    - Funding rates (simulado)
    - Stop Loss al -2.5% (aumentado de -2.0%)
    - Take Profit al +5.0% (aumentado de +3.0%)
    - RR Ratio 2.0:1
    - Risk/Trade 5% (reducido de 10%)
    - Trailing Stop 0.5%
    """

    def __init__(self, state, trade_tracker=None):
        self.state = state
        self.trade_tracker = trade_tracker or TokenTradeTracker()

    async def run(self):
        """Procesa trades aprobados y simula ejecución Drift"""
        self.state.data["agents"]["paper_trading"] = {"status": "running", "last_run": None}

        approved = self.state.data["agents"]["auditor"].get("approved", [])
        prices = self.state.data.get("current_prices", {})
        positions = self.state.data.get("paper_positions", [])

        # Get current prices for each token
        current_prices = {}
        for token, data in prices.items():
            if isinstance(data, dict):
                current_prices[token] = data.get("price", 0)

        # Get current capital and Drift params
        capital = self.state.data.get("paper_capital", 500.0)

        drift_params = self.state.data.get("drift_simulation", {
            "leverage": LEVERAGE,
            "trading_fee_pct": TRADING_FEE_PCT,
            "liquidation_threshold": LIQUIDATION_THRESHOLD,
            "borrowing_fee_hourly": BORROWING_FEE_HOURLY,
            "funding_rate_pct": FUNDING_RATE_PCT
        })

        leverage = drift_params.get("leverage", LEVERAGE)
        trading_fee_pct = drift_params.get("trading_fee_pct", TRADING_FEE_PCT)
        liquidation_threshold = drift_params.get("liquidation_threshold", LIQUIDATION_THRESHOLD)
        borrowing_fee_hourly = drift_params.get("borrowing_fee_hourly", BORROWING_FEE_HOURLY)
        funding_rate_pct = drift_params.get("funding_rate_pct", FUNDING_RATE_PCT)

        # Initialize trailing stops for open positions
        if not hasattr(self, 'trailing_stops'):
            self.trailing_stops = {}

        # Open new positions from approved trades
        for strat in approved:
            token = strat.get("token", "")
            if not token:
                continue

            # 🔧 V3.2: Verificar si se puede hacer trade en este token
            if not self.trade_tracker.can_trade(token):
                self.state.log(f"   ⚠️ {token}: Cooldown o límite de trades - saltando")
                continue

            # Check if we already have a position for this token
            existing = [p for p in positions if p["token"] == token and p.get("status") == "open"]
            if existing:
                continue

            # Max 5 posiciones abiertas a la vez
            if len([p for p in positions if p.get("status") == "open"]) >= 5:
                break

            # Open new paper position
            entry_price = current_prices.get(token, strat.get("entry", 0))
            if entry_price <= 0:
                continue

            # 🔧 V3.2: Calculate position size (5% of capital per trade)
            position_size = capital * MAX_RISK_PER_TRADE  # 5% instead of 10%

            # Calculate exposure with leverage
            exposure = position_size * leverage

            # Calculate trading fee
            trading_fee = calculate_trading_fee(exposure, trading_fee_pct)

            direction = strat.get("direction", LONG)
            signal = strat.get("signal", "unknown")

            # Calculate SL and TP based on direction (V3.2: new values)
            if direction == LONG:
                stop_loss = entry_price * (1 + STOP_LOSS_PCT / 100)
                take_profit = entry_price * (1 + TAKE_PROFIT_PCT / 100)
            else:
                stop_loss = entry_price * (1 - STOP_LOSS_PCT / 100)
                take_profit = entry_price * (1 - TAKE_PROFIT_PCT / 100)

            # Calculate liquidation price
            liquidation_price = calculate_liquidation_price(entry_price, direction, leverage, liquidation_threshold)

            # 🔧 V3.2: Initialize trailing stop for this position
            if TRAILING_STOP_ENABLED:
                self.trailing_stops[token] = TrailingStop(
                    initial_sl_percent=abs(STOP_LOSS_PCT),
                    trail_percent=TRAILING_STOP_PCT
                )

            position = {
                "token": token,
                "signal": signal,
                "entry_price": entry_price,
                "entry_time": datetime.now().isoformat(),
                "status": "open",
                "position_size": position_size,
                "exposure": exposure,
                "direction": direction,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "liquidation_price": liquidation_price,
                "trading_fee": trading_fee,
                "sharpe": strat.get("sharpe", 0),
                "win_rate": strat.get("win_rate", 0)
            }

            positions.append(position)
            
            # 🔧 V3.2: Registrar trade en el tracker
            self.trade_tracker.record_trade(token, pnl=0.0)

            direction_symbol = "📈 LONG" if direction == LONG else "📉 SHORT"
            self.state.log(f"📝 DRIFT: {direction_symbol} {token} @ ${entry_price:.2f} ({signal}) | Leverage: {leverage}x | Exposure: ${exposure:.2f} | Fee: ${trading_fee:.4f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | Liq: ${liquidation_price:.2f}")

        # Update open positions with current prices & calculate P&L
        open_pnl = 0.0
        closed_positions = []

        # Get stats
        stats = self.state.data.get("stats", {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_fees": 0.0,
            "total_borrowing_fees": 0.0,
            "total_funding_received": 0.0,
            "total_funding_paid": 0.0
        })

        for pos in positions:
            if pos.get("status") != "open":
                continue

            token = pos["token"]
            entry = pos["entry_price"]
            current = current_prices.get(token, entry)
            position_size = pos.get("position_size", 50)
            exposure = pos.get("exposure", 0)
            direction = pos.get("direction", LONG)
            entry_time_str = pos.get("entry_time", datetime.now().isoformat())

            if current <= 0:
                continue

            # Calculate hours held
            entry_time = datetime.fromisoformat(entry_time_str)
            hours_held = (datetime.now() - entry_time).total_seconds() / 3600

            # Calculate raw P&L
            if direction == LONG:
                pnl_pct = ((current - entry) / entry) * 100
            else:
                pnl_pct = ((entry - current) / entry) * 100

            pnl_value = (pnl_pct / 100) * position_size

            # Calculate additional fees
            borrowing_fee = 0.0
            funding_paid, funding_received = 0.0, 0.0

            if direction == SHORT:
                borrowing_fee = calculate_borrowing_fee(hours_held, position_size, borrowing_fee_hourly, leverage)

            funding_paid, funding_received = calculate_funding_rate(hours_held, position_size, funding_rate_pct, leverage)

            # Calculate total fees
            total_fees = pos.get("trading_fee", 0) + borrowing_fee + funding_paid - funding_received

            # 🔧 V3.2: Update trailing stop and get dynamic SL
            if TRAILING_STOP_ENABLED and token in self.trailing_stops:
                new_sl, sl_triggered = self.trailing_stops[token].update_trailing_stop(current, direction)
                if sl_triggered:
                    stop_loss = new_sl
                else:
                    pos["trailing_stop"] = new_sl

            # Net P&L after fees
            net_pnl_value = pnl_value - total_fees

            pos["current_price"] = current
            pos["pnl_pct"] = pnl_pct
            pos["pnl_value"] = net_pnl_value
            pos["hours_held"] = hours_held
            pos["borrowing_fee"] = borrowing_fee
            pos["funding_paid"] = funding_paid
            pos["funding_received"] = funding_received
            pos["total_fees"] = total_fees

            # Close position if conditions met
            should_close = False
            close_reason = ""

            # Check liquidation
            if direction == LONG and current <= pos.get("liquidation_price", 0):
                should_close = True
                close_reason = "LIQUIDATED"
                net_pnl_value = -position_size  # Total loss
            elif direction == SHORT and current >= pos.get("liquidation_price", 999999):
                should_close = True
                close_reason = "LIQUIDATED"
                net_pnl_value = -position_size  # Total loss

            # Check SL/TP (V3.2: using new STOP_LOSS_PCT and TAKE_PROFIT_PCT)
            if direction == LONG:
                if pnl_pct <= STOP_LOSS_PCT:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
            else:
                if pnl_pct >= -STOP_LOSS_PCT:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
                elif pnl_pct <= -TAKE_PROFIT_PCT:
                    should_close = True
                    close_reason = "STOP_LOSS"

            # Close after 24 hours
            if hours_held > 24 and not should_close:
                should_close = True
                close_reason = "TIMEOUT"

            if should_close:
                pos["status"] = "closed"
                pos["close_price"] = current
                pos["close_time"] = datetime.now().isoformat()
                pos["pnl_final"] = net_pnl_value
                pos["close_reason"] = close_reason
                closed_positions.append(pos)

                # Update stats
                stats["total_trades"] += 1
                stats["total_fees"] += total_fees
                stats["total_borrowing_fees"] += borrowing_fee
                stats["total_funding_paid"] += funding_paid
                stats["total_funding_received"] += funding_received

                if net_pnl_value > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

                self.state.log(f"📝 DRIFT: Closed {token} @ ${current:.2f} | P&L: ${net_pnl_value:+.2f} ({pnl_pct:+.1f}%) | Reason: {close_reason} | Fees: ${total_fees:.4f}")

        # Remove closed positions from open list
        positions = [p for p in positions if p.get("status") != "closed"]

        # Calculate total P&L
        total_pnl = sum(p.get("pnl_final", 0) for p in closed_positions)
        open_pnl = sum(p.get("pnl_value", 0) for p in positions)

        # Calculate win rate
        if stats["total_trades"] > 0:
            stats["win_rate"] = (stats["wins"] / stats["total_trades"]) * 100

        # Update state
        self.state.data["paper_positions"] = positions
        if "paper_history" not in self.state.data:
            self.state.data["paper_history"] = []
        self.state.data["paper_history"].extend(closed_positions)
        self.state.data["total_pnl"] = total_pnl
        self.state.data["paper_capital"] = capital
        self.state.data["stats"] = stats
        self.state.data["daily_pnl"] = ((capital - 500) / 500) * 100

        self.state.data["agents"]["paper_trading"] = {
            "status": "idle",
            "last_run": datetime.now().isoformat(),
            "open_positions": len(positions),
            "closed_today": len(closed_positions),
            "total_pnl": total_pnl,
            "daily_pnl_pct": self.state.data["daily_pnl"],
            "capital": capital,
            "win_rate": stats["win_rate"],
            "total_fees": stats["total_fees"]
        }
        self.state.save()

        if positions:
            long_count = len([p for p in positions if p.get("direction") == LONG])
            short_count = len([p for p in positions if p.get("direction") == SHORT])
            self.state.log(f"📝 DRIFT: {len(positions)} posiciones abiertas | LONG: {long_count} | SHORT: {short_count} | P&L abierto: ${open_pnl:+.2f} | Fees totales: ${stats['total_fees']:.4f}")
        if closed_positions:
            self.state.log(f"📝 DRIFT: {len(closed_positions)} cerradas | Capital: ${capital:.2f} | Win Rate: {stats['win_rate']:.1f}% | Fees: ${stats['total_fees']:.4f}")
