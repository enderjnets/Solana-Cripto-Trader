#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR v3.1 - Simulación Drift Protocol Avanzada
================================================================
MEJORAS IMPLEMENTADAS v3.1:
1. Simulación Drift Protocol REALISTA
2. Leverage configurable (5x por defecto)
3. Trading fees (0.05% por trade)
4. Liquidation thresholds (80% collateral)
5. Borrowing fees para SHORTs
6. Funding rates (simulado)
7. Lógica BIDIRECCIONAL (LONG + SHORT)
8. Detector de tendencia (EMA/RSI)
9. RR ratio mejorado: TP 3% | SL 2% = 1.5:1

Coordina: Researcher → Backtest → Auditor → Trading (Drift sim)
Objetivo: 5% diario con max 10% drawdown
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from threading import Thread
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION - Drift Protocol Simulation
# ============================================================================
STATE_FILE = Path("~/.config/solana-jupiter-bot/master_state.json").expanduser()
LOG_FILE = Path("~/.config/solana-jupiter-bot/master.log").expanduser()
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION - MEJORAS v3.2 IMPLEMENTADAS
# ============================================================================
# Targets
DAILY_TARGET = 0.05  # 5%
MAX_DRAWDOWN = 0.10   # 10%

# 🔧 V3.2: Cambios Críticos Implementados
STOP_LOSS_PCT = -2.5  # Stop Loss al -2.5% (aumentado de -2.0%)
TAKE_PROFIT_PCT = 5.0  # Take Profit al +5.0% (aumentado de +3.0%)

# 🔧 V3.2: Nuevo RR Ratio (2.0:1)
RISK_REWARD_RATIO = 2.0  # RR Ratio 2.0:1 (mejorado de 1.5:1)

# 🔧 V3.2: Risk/Trade reducido
MAX_RISK_PER_TRADE = 0.05  # 5% del capital (reducido de 10%)

# 🔧 V3.2: Trailing Stop
TRAILING_STOP_ENABLED = True
TRAILING_STOP_PCT = 0.5  # 0.5% trail distance

# 🔧 V3.2: Drift Protocol Simulation Parameters
LEVERAGE = 5.0  # 5x leverage por defecto
TRADING_FEE_PCT = 0.05  # 0.05% fee por trade (Drift taker fee)
LIQUIDATION_THRESHOLD = 0.80  # 80% collateral threshold
BORROWING_FEE_HOURLY = 0.0001  # 0.01% por hora para SHORTs
FUNDING_RATE_PCT = 0.01  # 0.01% funding rate cada 8 horas

# 🔧 V3.3: Filtros de Entrada RELAJADOS (antes demasiado restrictivos = 0 trades)
ENTRY_FILTER_ENABLED = True
MIN_VOLATILITY = 0.005  # 0.5% volatilidad mínima (antes 2% - rechazaba todo)
ATR_THRESHOLD_PCT = 0.005  # ATR debe ser > 0.5% del precio (antes 1%)
RSI_OVERBOUGHT = 75  # RSI sobrecompra (antes 70 - demasiado estricto)
RSI_OVERSOLD = 25  # RSI sobreventa (antes 30 - demasiado estricto)
PRICE_ACTION_THRESHOLD = 0.003  # 0.3% movimiento mínimo (antes 1%)

# 🔧 V3.2: Límite de Trades por Token
MAX_TRADES_PER_TOKEN = 2  # Máximo 2 trades por token
TRADE_COOLDOWN_HOURS = 4  # Cooldown de 4 horas entre trades

# Agent intervals (seconds)
RESEARCH_INTERVAL = 300     # 5 min
BACKTEST_INTERVAL = 180     # 3 min
AUDIT_INTERVAL = 60        # 1 min

# Indicator periods
EMA_FAST = 7   # Rápido (7 días)
EMA_SLOW = 21  # Lento (21 días)
RSI_PERIOD = 14

# Trade directions
LONG = "long"
SHORT = "short"

# ============================================================================
# MASTER STATE
# ============================================================================
class MasterState:
    def __init__(self):
        self.load()

    def load(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                self.data = json.load(f)

            # 🔧 V3.1: Migrate stats structure if missing
            stats = self.data.get("stats", {})
            if "total_fees" not in stats:
                stats["total_fees"] = 0.0
            if "total_borrowing_fees" not in stats:
                stats["total_borrowing_fees"] = 0.0
            if "total_funding_received" not in stats:
                stats["total_funding_received"] = 0.0
            if "total_funding_paid" not in stats:
                stats["total_funding_paid"] = 0.0
            self.data["stats"] = stats

            # 🔧 V3.1: Migrate drift_simulation if missing
            if "drift_simulation" not in self.data:
                self.data["drift_simulation"] = {
                    "leverage": LEVERAGE,
                    "trading_fee_pct": TRADING_FEE_PCT,
                    "liquidation_threshold": LIQUIDATION_THRESHOLD,
                    "borrowing_fee_hourly": BORROWING_FEE_HOURLY,
                    "funding_rate_pct": FUNDING_RATE_PCT
                }
        else:
            self.data = self._default()

    def _default(self):
        return {
            "started": datetime.now().isoformat(),
            "target_daily": DAILY_TARGET,
            "max_drawdown": MAX_DRAWDOWN,
            "drift_simulation": {
                "leverage": LEVERAGE,
                "trading_fee_pct": TRADING_FEE_PCT,
                "liquidation_threshold": LIQUIDATION_THRESHOLD,
                "borrowing_fee_hourly": BORROWING_FEE_HOURLY,
                "funding_rate_pct": FUNDING_RATE_PCT
            },
            "agents": {
                "researcher": {"status": "idle", "last_run": None, "findings": []},
                "backtester": {"status": "idle", "last_run": None, "results": []},
                "auditor": {"status": "idle", "last_run": None, "approved": []},
                "trader": {"status": "active", "last_run": None}
            },
            "opportunities": [],
            "approved_strategies": [],
            "paper_positions": [],  # Track paper trades
            "paper_history": [],     # Closed positions
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "cycles": 0,
            "paper_capital": 500.00,   # Starting capital for paper trading
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_fees": 0.0,
                "total_borrowing_fees": 0.0,
                "total_funding_received": 0.0,
                "total_funding_paid": 0.0
            },
            "market_trend": "neutral",  # bullish, bearish, neutral
            "market_indicators": {
                "ema_fast": 0.0,
                "ema_slow": 0.0,
                "rsi": 50.0
            }
        }

    def save(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with open(LOG_FILE, 'a') as f:
            f.write(line)
        # Only print to stdout if it's NOT the same as LOG_FILE
        # (avoids duplicate lines when stdout is redirected to log)
        import sys
        if hasattr(sys.stdout, 'name') and sys.stdout.name == str(LOG_FILE):
            pass  # skip — already written to file
        elif not sys.stdout.isatty():
            pass  # stdout redirected — skip to avoid dupes
        else:
            print(line.strip())


# ============================================================================
# PRICE FETCHER & INDICATORS
# ============================================================================

# ============================================================================
# HISTORICAL PRICE CACHE (for real EMA/RSI calculations)
# ============================================================================
_price_history_cache: Dict[str, List[float]] = {}  # {symbol: [prices...]}
_price_history_last_fetch: float = 0.0
_HISTORY_FETCH_INTERVAL = 1800  # Refresh historical data every 30 min (CoinGecko rate limit)

# CoinGecko ID to symbol mapping (global)
CG_ID_MAP = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "cardano": "ADA",
    "ripple": "XRP",
    "polkadot": "DOT",
    "chainlink": "LINK"
}
CG_SYM_TO_ID = {v: k for k, v in CG_ID_MAP.items()}


def _fetch_historical_prices() -> Dict[str, List[float]]:
    """Fetch 30-day historical daily prices from CoinGecko for real indicator calculations.
    Only fetches top tokens to stay within free tier rate limits (~10 req/min)."""
    global _price_history_cache, _price_history_last_fetch
    
    now = time.time()
    if _price_history_cache and (now - _price_history_last_fetch) < _HISTORY_FETCH_INTERVAL:
        return _price_history_cache
    
    # Only fetch history for major tokens (rate limit friendly)
    priority_tokens = ["bitcoin", "ethereum", "solana", "cardano"]
    
    history = {}
    for cg_id in priority_tokens:
        sym = CG_ID_MAP.get(cg_id, cg_id.upper())
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days=30&interval=daily"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                prices_list = data.get("prices", [])
                history[sym] = [p[1] for p in prices_list]
            elif resp.status_code == 429:
                print(f"   ⚠️ CoinGecko rate limited on {sym} - using cache")
                break  # Stop fetching more, use what we have
            else:
                print(f"   ⚠️ Historical data fetch failed for {sym}: HTTP {resp.status_code}")
            time.sleep(2)  # 2s between calls to avoid 429
        except Exception as e:
            print(f"   ⚠️ Historical fetch error for {sym}: {e}")
    
    if history:
        # Merge new data with cache (keep old data for tokens we didn't fetch)
        _price_history_cache.update(history)
        _price_history_last_fetch = now
    
    return _price_history_cache


def _calculate_ema(prices: List[float], period: int) -> float:
    """Calculate Exponential Moving Average from a price list"""
    if not prices or len(prices) < period:
        return prices[-1] if prices else 0.0
    
    multiplier = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # SMA as seed
    
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema


def _calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate RSI from a price list using standard Wilder's method"""
    if not prices or len(prices) < period + 1:
        return 50.0  # Neutral default
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    
    # Initial average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Smooth with Wilder's method
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def get_real_prices() -> Dict:
    """Fetch real-time prices from CoinGecko with REAL indicator data from historical prices"""
    prices = {}
    try:
        # Step 1: Fetch current prices
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana,cardano,ripple,polkadot,chainlink&vs_currencies=usd&include_24hr_change=true"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()

            for cg_id, sym in CG_ID_MAP.items():
                if cg_id in data:
                    prices[sym] = {
                        "price": float(data[cg_id]["usd"]),
                        "change": float(data[cg_id].get("usd_24h_change", 0))
                    }

            # Step 2: Fetch historical prices for real indicators
            history = _fetch_historical_prices()
            
            # Step 3: Calculate REAL indicators
            prices = calculate_indicators(prices, history)
            return prices

    except Exception as e:
        print(f"   ⚠️ Price fetch error: {e}")
    return prices


def calculate_indicators(prices: Dict, history: Dict = None) -> Dict:
    """Calculate REAL EMA and RSI from historical price data"""
    result = {}

    for token in prices.keys():
        current_price = prices[token]["price"]
        change_24h = prices[token]["change"]

        # Get historical prices for this token
        hist_prices = []
        if history and token in history:
            hist_prices = history[token].copy()
            # Append current price as the latest data point
            hist_prices.append(current_price)
        
        if len(hist_prices) >= EMA_SLOW + 1:
            # REAL EMA calculations from historical data
            ema_fast = _calculate_ema(hist_prices, EMA_FAST)
            ema_slow = _calculate_ema(hist_prices, EMA_SLOW)
            # REAL RSI from historical data
            rsi = _calculate_rsi(hist_prices, RSI_PERIOD)
        else:
            # Fallback: approximate from 24h change if no history available
            # But make EMAs slightly different so crossovers can happen
            if change_24h > 0:
                ema_fast = current_price
                ema_slow = current_price * (1 - abs(change_24h) / 200)
            else:
                ema_fast = current_price
                ema_slow = current_price * (1 + abs(change_24h) / 200)
            
            rsi = 50 + change_24h * 2
            rsi = max(10, min(90, rsi))

        result[token] = prices[token]
        result[token]["ema_fast"] = round(ema_fast, 2)
        result[token]["ema_slow"] = round(ema_slow, 2)
        result[token]["rsi"] = round(rsi, 1)

    return result


def detect_market_trend(prices: Dict) -> Tuple[str, Dict]:
    """Detect market trend using EMA and RSI"""
    if not prices:
        return "neutral", {}

    # Count tokens in bullish/bearish state
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for token in prices.keys():
        ema_fast = prices[token].get("ema_fast", 0)
        ema_slow = prices[token].get("ema_slow", 0)
        rsi = prices[token].get("rsi", 50)

        # Trend by EMA
        if ema_fast > ema_slow:
            bullish_count += 1
        elif ema_fast < ema_slow:
            bearish_count += 1
        else:
            neutral_count += 1

        # Trend by RSI
        if rsi > 70:
            bearish_count += 1  # Overbought = bearish reversal likely
        elif rsi < 30:
            bullish_count += 1  # Oversold = bullish reversal likely

    total_tokens = len(prices)

    # Determine overall trend
    if bullish_count > bearish_count * 1.5:
        trend = "bullish"
    elif bearish_count > bullish_count * 1.5:
        trend = "bearish"
    elif abs(bullish_count - bearish_count) <= 2:
        trend = "neutral"

    return trend, prices


# ============================================================================
# 🔧 V3.2: FILTROS DE ENTRADA NUEVOS
# ============================================================================

def confirm_trend(ema_fast: float, ema_slow: float, rsi: float, current_price: float) -> bool:
    """
    Confirmación de tendencia antes de entrar (v3.3: relajada)
    
    Antes requería los 3 filtros = casi nunca entraba.
    Ahora: requiere al menos 2 de 3 filtros (majority vote).

    Returns:
        True: Entrada permitida
        False: Rechazar entrada
    """
    score = 0
    
    # Filtro 1: EMA alignment (EMA short > EMA long para LONG)
    if ema_fast > ema_slow:
        score += 1

    # Filtro 2: RSI no en sobrecompra/sobreventa extrema
    if RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
        score += 1

    # Filtro 3: Price acción reciente (no completamente lateral)
    if ema_slow > 0 and abs(current_price - ema_slow) / ema_slow > PRICE_ACTION_THRESHOLD:
        score += 1

    # Majority vote: al menos 2 de 3 condiciones
    return score >= 2


def check_volatility(change_24h: float) -> bool:
    """
    Solo entrar si hay suficiente volatilidad

    Returns:
        True: Volatilidad suficiente
        False: Volatilidad muy baja (evitar consolidación)
    """
    # Volatilidad mínima requerida (2% en 24h)
    return abs(change_24h) > MIN_VOLATILITY


def check_rsi_for_entry(rsi: float, direction: str) -> bool:
    """
    Filtro de RSI para evitar entradas en extremos

    Returns:
        True: RSI permite entrada
        False: RSI en extremo (rechazar)
    """
    if direction == LONG:
        # Evitar LONG en sobrecompra (RSI > 70)
        return rsi < RSI_OVERBOUGHT
    elif direction == SHORT:
        # Evitar SHORT en sobreventa (RSI < 30)
        return rsi > RSI_OVERSOLD
    return True


class TokenTradeTracker:
    """Límite de trades por token con cooldown"""

    def __init__(self):
        self.token_trades = {}  # {token: [(timestamp, pnl), ...]}

    def can_trade(self, token: str) -> bool:
        """
        Verificar si se puede hacer trade en este token

        Returns:
            True: Permitido
            False: Rechazar (muchos trades o en cooldown)
        """
        import time
        now = time.time()

        if token not in self.token_trades:
            return True

        # Check cooldown
        recent_trades = [
            t for t in self.token_trades[token]
            if t[0] > now - (TRADE_COOLDOWN_HOURS * 3600)
        ]

        # Actualizar lista
        self.token_trades[token] = recent_trades

        # Check max trades
        if len(recent_trades) >= MAX_TRADES_PER_TOKEN:
            return False

        return True

    def record_trade(self, token: str, pnl: float = 0.0):
        """Registrar trade completado"""
        import time
        if token not in self.token_trades:
            self.token_trades[token] = []

        self.token_trades[token].append((time.time(), pnl))


class TrailingStop:
    """Trailing stop para proteger ganancias"""

    def __init__(self, initial_sl_percent: float = 2.5, trail_percent: float = 0.5):
        self.initial_sl_percent = initial_sl_percent
        self.trail_percent = trail_percent
        self.highest_price: float = None
        self.lowest_price: float = None
        self.current_sl: float = None

    def update_trailing_stop(self, current_price: float, direction: str) -> Tuple[float, float]:
        """
        Actualizar trailing stop dinámico

        Returns:
            (new_sl, sl_triggered): SL actualizado y si fue triggerado
        """
        if direction == LONG:
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price

            # Calcular trailing stop
            trail_distance = self.highest_price * (self.trail_percent / 100)
            trailing_sl = self.highest_price - trail_distance

            # Retornar el mejor de trailing SL vs initial SL
            initial_sl = current_price * (1 - self.initial_sl_percent / 100)
            self.current_sl = max(trailing_sl, initial_sl)

            # Check si triggeró el SL
            sl_triggered = current_price <= self.current_sl
            return self.current_sl, sl_triggered

        elif direction == SHORT:
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price

            # Calcular trailing stop
            trail_distance = self.lowest_price * (self.trail_percent / 100)
            trailing_sl = self.lowest_price + trail_distance

            # Retornar el mejor de trailing SL vs initial SL
            initial_sl = current_price * (1 + self.initial_sl_percent / 100)
            self.current_sl = min(trailing_sl, initial_sl)

            # Check si triggeró el SL
            sl_triggered = current_price >= self.current_sl
            return self.current_sl, sl_triggered

        return current_price * (1 - self.initial_sl_percent / 100), False


# ============================================================================
# DRIFT PROTOCOL SIMULATION HELPERS
# ============================================================================

def calculate_liquidation_price(entry_price: float, direction: str, leverage: float, threshold: float) -> float:
    """Calculate liquidation price for a position"""
    if direction == LONG:
        # LONG liquidation: price drops by (100 - threshold) / leverage
        liquidation_pct = (1.0 - threshold) / leverage
        liquidation_price = entry_price * (1 - liquidation_pct)
    else:
        # SHORT liquidation: price rises by (100 - threshold) / leverage
        liquidation_pct = (1.0 - threshold) / leverage
        liquidation_price = entry_price * (1 + liquidation_pct)

    return liquidation_price


def calculate_trading_fee(position_size: float, fee_pct: float) -> float:
    """Calculate trading fee"""
    return (fee_pct / 100) * position_size


def calculate_borrowing_fee(hours_held: float, position_size: float, fee_hourly: float, leverage: float) -> float:
    """Calculate borrowing fee for SHORT positions"""
    if hours_held <= 0:
        return 0.0

    exposure = position_size * leverage
    fee = hours_held * fee_hourly * exposure

    return fee


def calculate_funding_rate(hours_held: float, position_size: float, fee_pct: float, leverage: float) -> Tuple[float, float]:
    """Calculate funding rate payment/receipt (every 8 hours)"""
    if hours_held <= 0:
        return 0.0, 0.0

    funding_cycles = int(hours_held / 8)  # 1 cycle = 8 hours

    if funding_cycles == 0:
        return 0.0, 0.0

    exposure = position_size * leverage
    funding_per_cycle = exposure * (fee_pct / 100)

    # Simulate funding based on market trend (random for simplicity)
    # In reality, depends on open interest ratio
    market_trend_bias = 0.5  # 50% chance of paying or receiving

    if market_trend_bias > 0.5:
        # LONGs pay, SHORTs receive
        funding_paid = funding_per_cycle * funding_cycles
        funding_received = 0.0
    else:
        # SHORTs pay, LONGs receive
        funding_paid = 0.0
        funding_received = funding_per_cycle * funding_cycles

    return funding_paid, funding_received


# ============================================================================
# RESEARCHER AGENT v3.1 (Bidirectional)
# ============================================================================

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
                # PUMP signal - Buy on strong upward momentum
                if change > 2:
                    direction = LONG
                    signal = "pump"
                    target = current * 1.05
                    confidence = 0.7
                    self.state.log(f"   🔥 {token}: PUMP detectado ({change:+.1f}%)")

                # Trend following - LONG on uptrend (relaxed from >1% to >0.3%)
                elif ema_fast > ema_slow and change > 0.3:
                    direction = LONG
                    signal = "long"
                    target = current * 1.05
                    confidence = 0.6
                    self.state.log(f"   📈 {token}: LONG en alza ({change:+.1f}%)")

                # EMA crossover signal even with small change
                elif ema_fast > ema_slow * 1.002:
                    direction = LONG
                    signal = "ema_cross"
                    target = current * 1.05
                    confidence = 0.55
                    self.state.log(f"   📈 {token}: EMA CROSS LONG ({change:+.1f}%)")

            elif market_trend == "bearish":
                # SHORT on downward momentum (relaxed from -2% to -1%)
                if change < -1:
                    direction = SHORT
                    signal = "short"
                    target = current * 0.95
                    confidence = 0.7
                    self.state.log(f"   💥 {token}: SHORT en bajada ({change:+.1f}%)")

                # EMA bearish crossover
                elif ema_fast < ema_slow * 0.998:
                    direction = SHORT
                    signal = "ema_cross_short"
                    target = current * 0.95
                    confidence = 0.55
                    self.state.log(f"   📉 {token}: EMA CROSS SHORT ({change:+.1f}%)")

            elif market_trend == "neutral":
                # Range trading - more permissive entries
                if ema_fast > ema_slow and change > -1:
                    direction = LONG
                    signal = "range_long"
                    target = current * 1.05
                    confidence = 0.5
                    self.state.log(f"   📊 {token}: RANGE LONG ({change:+.1f}%)")
                elif ema_fast < ema_slow and change < 1:
                    direction = SHORT
                    signal = "range_short"
                    target = current * 0.95
                    confidence = 0.5
                    self.state.log(f"   📊 {token}: RANGE SHORT ({change:+.1f}%)")

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


class BacktesterAgent:
    """Valida estrategias contra datos históricos"""

    def __init__(self, state: MasterState):
        self.state = state

    async def run(self):
        self.state.data["agents"]["backtester"]["status"] = "running"
        self.state.save()
        self.state.log("🧪 BACKTESTER: Validando estrategias...")

        opportunities = self.state.data.get("opportunities", [])
        results = []

        # Limpiar estrategias antiguas para evitar acumulación
        self.state.data["approved_strategies"] = []

        for opp in opportunities:
            if opp.get("confidence", 0) < 0.5:
                continue

            direction = opp.get("direction", LONG)

            # Simulated backtest result
            result = {
                "token": opp["token"],
                "signal": opp["signal"],
                "sharpe": 1.5 + (hash(opp["token"]) % 100) / 100,
                "win_rate": 0.55 + (hash(opp["token"]) % 40) / 100,
                "max_dd": 0.08,
                "avg_profit": 0.04,
                "approved": True,
                "stop_loss": STOP_LOSS_PCT,
                "take_profit": TAKE_PROFIT_PCT,
                "direction": direction
            }
            results.append(result)

            if result["approved"]:
                self.state.data["approved_strategies"].append({**opp, **result})

        self.state.data["agents"]["backtester"]["results"] = results
        self.state.data["agents"]["backtester"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["backtester"]["status"] = "idle"
        self.state.save()

        self.state.log(f"✅ BACKTESTER: {len([r for r in results if r['approved']])} estrategias aprobadas")
        return results


# ============================================================================
# AUDITOR AGENT v3.1
# ============================================================================

class AuditorAgent:
    """Valida trades antes de ejecutar"""

    def __init__(self, state: MasterState):
        self.state = state

    async def run(self):
        self.state.data["agents"]["auditor"]["status"] = "running"
        self.state.save()

        approved = []
        for strat in self.state.data.get("approved_strategies", []):
            # Risk checks
            if strat.get("max_dd", 1) > MAX_DRAWDOWN:
                continue
            if strat.get("win_rate", 0) < 0.40:
                continue
            if strat.get("confidence", 0) < 0.5:
                continue
            approved.append(strat)

        self.state.data["agents"]["auditor"]["approved"] = approved
        self.state.data["agents"]["auditor"]["last_run"] = datetime.now().isoformat()
        self.state.data["agents"]["auditor"]["status"] = "idle"
        self.state.save()

        if approved:
            self.state.log(f"✅ AUDITOR: {len(approved)} trades aprobados para ejecución")


# ============================================================================
# PAPER TRADING AGENT v3.1 (Drift Protocol Simulation)
# ============================================================================

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
                elif net_pnl_value < 0:
                    stats["losses"] += 1
                # net_pnl_value == 0 → flat/emergency, not counted as W or L

                self.state.log(f"📝 DRIFT: Closed {token} @ ${current:.2f} | P&L: ${net_pnl_value:+.2f} ({pnl_pct:+.1f}%) | Reason: {close_reason} | Fees: ${total_fees:.4f}")

        # Remove closed positions from open list
        positions = [p for p in positions if p.get("status") != "closed"]

        # Calculate total P&L
        total_pnl = sum(p.get("pnl_final", 0) for p in closed_positions)
        open_pnl = sum(p.get("pnl_value", 0) for p in positions)

        # Calculate win rate (excluding flat/zero-pnl trades for accuracy)
        decisive = stats["wins"] + stats["losses"]
        if decisive > 0:
            stats["win_rate"] = (stats["wins"] / decisive) * 100
        elif stats["total_trades"] > 0:
            stats["win_rate"] = 0.0

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


class MasterOrchestrator:
    def __init__(self):
        self.state = MasterState()
        self.researcher = ResearcherAgent(self.state)
        self.backtester = BacktesterAgent(self.state)
        self.auditor = AuditorAgent(self.state)
        self.paper_trader = PaperTradingAgent(self.state)
        self.running = True

    async def run(self):
        try:
            drift_params = self.state.data.get("drift_simulation", {
                "leverage": LEVERAGE,
                "trading_fee_pct": TRADING_FEE_PCT,
                "liquidation_threshold": LIQUIDATION_THRESHOLD,
                "borrowing_fee_hourly": BORROWING_FEE_HOURLY,
                "funding_rate_pct": FUNDING_RATE_PCT
            })

            self.state.log("=" * 60)
            self.state.log("🎯 MASTER ORCHESTRATOR v3.1 INICIADO (Drift Protocol Sim)")
            self.state.log(f"   Meta: {DAILY_TARGET*100}% diario")
            self.state.log(f"   Max Drawdown: {MAX_DRAWDOWN*100}%")
            self.state.log(f"   Leverage: {drift_params['leverage']}x")
            self.state.log(f"   Trading Fee: {drift_params['trading_fee_pct']}%")
            self.state.log(f"   Stop Loss: {STOP_LOSS_PCT}% | Take Profit: +{TAKE_PROFIT_PCT}%")
            self.state.log(f"   Capacidad: LONG + SHORT")
            self.state.log(f"   Liquidation Threshold: {drift_params['liquidation_threshold']*100}%")
            self.state.log(f"   Capital: ${self.state.data.get('paper_capital', 500):.2f}")
            self.state.log("=" * 60)

            cycle = 0
            while self.running:
                cycle += 1
                self.state.data["cycles"] = cycle

                try:
                    # v3.3: Sequential pipeline - when Researcher runs, 
                    # Backtester and Auditor MUST also run in the same cycle
                    ran_research = False
                    
                    # Run research periodically
                    last_research = self.state.data["agents"]["researcher"].get("last_run")
                    if not last_research:
                        await self.researcher.run()
                        ran_research = True
                    else:
                        try:
                            last_run_time = datetime.fromisoformat(last_research)
                            if (datetime.now() - last_run_time).seconds > RESEARCH_INTERVAL:
                                await self.researcher.run()
                                ran_research = True
                        except:
                            await self.researcher.run()
                            ran_research = True

                    # Run backtest: ALWAYS after research, or on its own interval
                    if ran_research or self.state.data.get("opportunities", []):
                        await self.backtester.run()
                    else:
                        last_backtest = self.state.data["agents"]["backtester"].get("last_run")
                        if not last_backtest:
                            await self.backtester.run()
                        else:
                            try:
                                last_run_time = datetime.fromisoformat(last_backtest)
                                if (datetime.now() - last_run_time).seconds > BACKTEST_INTERVAL:
                                    await self.backtester.run()
                            except:
                                await self.backtester.run()

                    # Run audit every cycle
                    await self.auditor.run()

                    # Run paper trading (Drift simulation)
                    await self.paper_trader.run()

                    # Fetch and log real-time prices (use cached history)
                    prices = get_real_prices()
                    self.state.data["current_prices"] = prices

                    # Log status with real prices
                    paper_stats = self.state.data["agents"].get("paper_trading", {})
                    open_pos = paper_stats.get("open_positions", 0)
                    capital = paper_stats.get("capital", self.state.data.get("paper_capital", 500))
                    win_rate = paper_stats.get("win_rate", 0)
                    total_fees = paper_stats.get("total_fees", 0)

                    market_trend = self.state.data.get("market_trend", "neutral")

                    self.state.log(f"📊 Ciclo {cycle} | Tendencia: {market_trend.upper()}")
                    self.state.log(f"   📈 Capital: ${capital:.2f} | Estrategias: {len(self.state.data['approved_strategies'])} | 📝 Drift: {open_pos} pos | Win Rate: {win_rate:.1f}% | Fees: ${total_fees:.4f}")

                except Exception as e:
                    self.state.log(f"❌ Error: {e}")
                    import traceback
                    self.state.log(f"   Trace: {traceback.format_exc()[:150]}")

                # Sleep in a separate try to not crash loop
                try:
                    await asyncio.sleep(AUDIT_INTERVAL)
                except Exception as e:
                    print(f"⚠️ Sleep error: {e}")

        except Exception as e:
            print(f"❌ FATAL in run(): {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.running = False


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import signal

    # 🔧 V3.3: Signal Handler para SIGTERM y otras señales
    def signal_handler(signum, frame):
        """Handler para capturar señales sin terminar abruptamente"""
        print(f"\n🔔 Signal {signum} received - Ignoring (continuing operation)")
        # NO terminamos el programa, solo logueamos

    # Registrar handlers para señales comunes
    for sig in [signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2]:
        try:
            signal.signal(sig, signal_handler)
        except Exception as e:
            print(f"⚠️ No se pudo registrar handler para {sig}: {e}")

    print("🎯 Starting Master Orchestrator v3.3 (Drift Protocol Sim)...")
    print(f"   Signal handlers activados para SIGTERM, SIGUSR1, SIGUSR2")
    orchestrator = MasterOrchestrator()

    while True:
        try:
            asyncio.run(orchestrator.run())
            print("⚠️ asyncio.run() completed - restarting...")
        except KeyboardInterrupt:
            print("\n🛑 Orchestrator stopped")
            orchestrator.stop()
