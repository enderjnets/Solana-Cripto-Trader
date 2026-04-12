#!/usr/bin/env python3
"""
🧠 Agente 3: Strategy — Stack Completo de Indicadores
=======================================================
INDICADORES IMPLEMENTADOS:
  Tendencia:    EMA 7, EMA 21, EMA 50, SMA 20, VWAP proxy
  Momentum:     RSI 14, MACD (12/26/9), ROC (Rate of Change)
  Volatilidad:  ATR 14, Bollinger Bands (20/2), Keltner Channel
  Volumen:      OBV (On Balance Volume), Volume Rate
  Patrones:     Divergencia RSI, Golden/Death Cross EMA
  Confirmación: Mínimo 3/6 indicadores alineados para señal

ESTRATEGIAS:
  1. Trend Momentum  — EMA + RSI + MACD + volumen
  2. Breakout        — BB superior + ATR + OBV confirma
  3. Oversold Bounce — RSI + BB inferior + divergencia
  4. Golden Cross    — EMA 7 cruza EMA 21 con volumen
  5. MACD Cross      — MACD cruza señal con tendencia

SL/TP DINÁMICOS basados en ATR (no fijos):
  SL = entrada - 1.5 × ATR
  TP = entrada + 3.0 × ATR  (RR 2:1 mínimo)
"""

import json
import logging
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MARKET_FILE        = DATA_DIR / "market_latest.json"
RISK_FILE          = DATA_DIR / "risk_report.json"
OUTPUT_FILE        = DATA_DIR / "signals_latest.json"
PRICE_HISTORY      = DATA_DIR / "price_history.json"
VOLUME_HISTORY     = DATA_DIR / "volume_history.json"
AUTO_LEARNER_FILE  = DATA_DIR / "auto_learner_state.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("strategy")

# ─── Parámetros ──────────────────────────────────────────────────────────────

# Historial mínimo por indicador
MIN_RSI         = 15    # RSI(14) necesita 15 puntos
MIN_MACD        = 35    # MACD(26) + señal(9) necesita 35 puntos
MIN_BB          = 22    # BB(20) necesita 22 puntos
MIN_ATR         = 15    # ATR(14) necesita 15 puntos
MIN_EMA_SLOW    = 52    # EMA(50) necesita 52 puntos
MIN_GOLDEN_X    = 22    # Golden Cross EMA(7/21) necesita 22 puntos

# Períodos
RSI_PERIOD      = 14
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
BB_STD          = 2.0
ATR_PERIOD      = 14
EMA_FAST        = 7
EMA_MID         = 21
EMA_SLOW        = 50
ROC_PERIOD      = 10

# Fibonacci Golden Pocket (optimizado via 3.31M simulaciones — 2026-04-10)
FIB_LOOKBACK        = 80      # ventana de velas para swing high/low
FIB_GP_LOW          = 61.8    # Golden Pocket inicio (%)
FIB_GP_HIGH         = 65.0    # Golden Pocket fin (%)
FIB_GP_TOLERANCE    = 0.8     # ±0.8% tolerancia en la zona
FIB_OTE_HIGH        = 79.0    # OTE zone límite superior (%)
FIB_RSI_THRESHOLD   = 50      # RSI máximo para confirmar GP long entry
FIB_SL_BUFFER       = 0.010   # 1.0% debajo de nivel 78.6% para SL
FIB_TP_EXTENSION    = 1.272   # extensión 127.2% para TP

# Filtros de entrada — OPTIMIZADO 2026-03-27 (post-drawdown crítico)
MIN_CONFIDENCE          = 0.65   # Sim: 0.60 óptimo; 0.65 conservador para robustez
MIN_INDICATORS_ALIGNED  = 3      # Subido de 2 — requiere más confirmación
ATR_SL_MULTIPLIER       = 0.8    # Sim: 0.6x óptimo (41.7M sims 2026-04-10); 0.8x para robustez real
ATR_TP_MULTIPLIER       = 4.0    # Sim: 4.0x óptimo en todas las estrategias (RR 5:1)
MIN_ATR_PCT             = 0.010  # Subido de 0.008 — filtra más activos planos
RSI_OVERBOUGHT          = 75     # Subido de 70 — más conservador para shorts
RSI_OVERSOLD            = 25     # Bajado de 30 — más conservador para longs
MOMENTUM_24H_MIN        = 3.0    # Subido de 2.0 — solo momentum fuerte
BB_SQUEEZE_THRESHOLD    = 0.015  # Bajado de 0.02 — squeeze más estricto
MIN_VOLUME_24H          = 1000000  # NUEVO: mínimo $1M volumen 24h
MAX_ATR_PCT             = 0.05   # NUEVO: máximo 5% ATR (evita alta volatilidad)
CONSECUTIVE_LOSS_COOLDOWN = 2    # NUEVO: cooldown después de 2 pérdidas

# ─── Persistencia ────────────────────────────────────────────────────────────

def load_auto_learner_params() -> dict:
    """Carga parámetros del auto_learner. Retorna defaults si el archivo no existe."""
    defaults = {"sl_pct": 0.025, "tp_pct": 0.05}
    try:
        if AUTO_LEARNER_FILE.exists():
            data = json.loads(AUTO_LEARNER_FILE.read_text())
            params = data.get("params", {})
            sl_pct = float(params.get("sl_pct", defaults["sl_pct"]))
            tp_pct = float(params.get("tp_pct", defaults["tp_pct"]))
            return {"sl_pct": sl_pct, "tp_pct": tp_pct}
    except Exception:
        pass
    return defaults


def load_json(path: Path, default=None):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default if default is not None else {}


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def update_price_history(tokens: dict) -> dict:
    history = load_json(PRICE_HISTORY, {})
    vol_history = load_json(VOLUME_HISTORY, {})
    now = datetime.now(timezone.utc).isoformat()

    for symbol, data in tokens.items():
        price = data.get("price", 0)
        volume = data.get("volume_24h", 0)
        if price <= 0:
            continue

        if symbol not in history:
            history[symbol] = []
        history[symbol].append({"price": price, "ts": now})
        history[symbol] = history[symbol][-500:]  # 5 horas de historia a 60s/ciclo

        if symbol not in vol_history:
            vol_history[symbol] = []
        vol_history[symbol].append({"volume": volume, "ts": now})
        vol_history[symbol] = vol_history[symbol][-500:]

    save_json(PRICE_HISTORY, history)
    save_json(VOLUME_HISTORY, vol_history)
    return history, vol_history


# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS
# ══════════════════════════════════════════════════════════════════════════════

def ema(prices: list, period: int) -> Optional[float]:
    """Exponential Moving Average."""
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    val = sum(prices[:period]) / period  # SMA inicial
    for p in prices[period:]:
        val = p * k + val * (1 - k)
    return round(val, 8)


def ema_series(prices: list, period: int) -> list:
    """Serie completa de EMA."""
    if len(prices) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(prices[:period]) / period]
    for p in prices[period:]:
        result.append(p * k + result[-1] * (1 - k))
    return result


def sma(prices: list, period: int) -> Optional[float]:
    """Simple Moving Average."""
    if len(prices) < period:
        return None
    return round(sum(prices[-period:]) / period, 8)


def rsi(prices: list, period: int = RSI_PERIOD) -> Optional[float]:
    """RSI con Wilder's smoothing (más preciso que SMA)."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains  = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]

    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def rsi_series(prices: list, period: int = RSI_PERIOD) -> list:
    """Serie completa de RSI para detectar divergencias."""
    if len(prices) < period + 1:
        return []
    results = []
    for i in range(period, len(prices)):
        window = prices[i - period:i + 1]
        results.append(rsi(window, period) or 50)
    return results


def macd(prices: list, fast=MACD_FAST, slow=MACD_SLOW, signal_period=MACD_SIGNAL) -> Optional[dict]:
    """MACD completo: línea MACD, señal y histograma."""
    if len(prices) < slow + signal_period:
        return None

    ema_fast_s = ema_series(prices, fast)
    ema_slow_s = ema_series(prices, slow)

    # Alinear por longitud
    min_len = min(len(ema_fast_s), len(ema_slow_s))
    macd_line = [ema_fast_s[-(min_len - i)] - ema_slow_s[-(min_len - i)]
                 for i in range(min_len)]

    if len(macd_line) < signal_period:
        return None

    signal_line_val = ema(macd_line, signal_period)
    if signal_line_val is None:
        return None

    macd_val   = macd_line[-1]
    histogram  = macd_val - signal_line_val
    prev_hist  = (macd_line[-2] - ema(macd_line[:-1], signal_period)) if len(macd_line) > 1 else 0

    return {
        "macd":      round(macd_val, 8),
        "signal":    round(signal_line_val, 8),
        "histogram": round(histogram, 8),
        "prev_hist": round(prev_hist or 0, 8),
        "bullish_cross": macd_val > signal_line_val and (prev_hist or 0) <= 0,
        "bearish_cross": macd_val < signal_line_val and (prev_hist or 0) >= 0,
        "histogram_growing": histogram > (prev_hist or 0),
    }


def bollinger_bands(prices: list, period=BB_PERIOD, std_mult=BB_STD) -> Optional[dict]:
    """Bollinger Bands: upper, middle (SMA), lower, width, %B."""
    if len(prices) < period:
        return None
    window = prices[-period:]
    mid = sum(window) / period
    variance = sum((p - mid) ** 2 for p in window) / period
    std = math.sqrt(variance)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid if mid > 0 else 0
    current = prices[-1]
    pct_b = (current - lower) / (upper - lower) if (upper - lower) > 0 else 0.5

    return {
        "upper":    round(upper, 8),
        "middle":   round(mid, 8),
        "lower":    round(lower, 8),
        "width":    round(width, 4),
        "pct_b":    round(pct_b, 4),  # 0=lower, 0.5=middle, 1=upper
        "squeeze":  width < BB_SQUEEZE_THRESHOLD,
        "above_upper": current > upper,
        "below_lower": current < lower,
    }


def atr(highs: list, lows: list, closes: list, period=ATR_PERIOD) -> Optional[float]:
    """Average True Range. Si no hay high/low, aproxima con precio."""
    if len(closes) < period + 1:
        return None
    # Si no tenemos high/low reales, estimar con ±0.5% del precio
    if not highs or not lows:
        highs  = [c * 1.005 for c in closes]
        lows   = [c * 0.995 for c in closes]

    trs = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    if len(trs) < period:
        return None

    # Wilder smoothing
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return round(atr_val, 8)


def obv(prices: list, volumes: list) -> Optional[float]:
    """On Balance Volume — acumulación/distribución."""
    if len(prices) < 2 or len(volumes) < 2:
        return None
    min_len = min(len(prices), len(volumes))
    prices  = prices[-min_len:]
    volumes = volumes[-min_len:]
    obv_val = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv_val += volumes[i]
        elif prices[i] < prices[i-1]:
            obv_val -= volumes[i]
    return obv_val


def roc(prices: list, period=ROC_PERIOD) -> Optional[float]:
    """Rate of Change — velocidad del precio."""
    if len(prices) < period + 1:
        return None
    old = prices[-(period + 1)]
    if old == 0:
        return None
    return round((prices[-1] - old) / old * 100, 4)


def vwap_proxy(prices: list, volumes: list) -> Optional[float]:
    """VWAP proxy (sin datos tick-by-tick, aproximado con precio × volumen)."""
    min_len = min(len(prices), len(volumes))
    if min_len < 2:
        return None
    prices  = prices[-min_len:]
    volumes = volumes[-min_len:]
    total_vol = sum(volumes)
    if total_vol == 0:
        return None
    vwap_val = sum(p * v for p, v in zip(prices, volumes)) / total_vol
    return round(vwap_val, 8)


def detect_rsi_divergence(prices: list, rsi_vals: list, lookback=10) -> Optional[str]:
    """
    Detecta divergencias RSI/precio:
    - Bullish: precio hace mínimo más bajo pero RSI hace mínimo más alto
    - Bearish: precio hace máximo más alto pero RSI hace máximo más bajo
    """
    if len(prices) < lookback or len(rsi_vals) < lookback:
        return None

    p = prices[-lookback:]
    r = rsi_vals[-lookback:]

    price_low1  = min(p[:lookback//2])
    price_low2  = min(p[lookback//2:])
    rsi_low1    = min(r[:lookback//2])
    rsi_low2    = min(r[lookback//2:])

    price_high1 = max(p[:lookback//2])
    price_high2 = max(p[lookback//2:])
    rsi_high1   = max(r[:lookback//2])
    rsi_high2   = max(r[lookback//2:])

    if price_low2 < price_low1 * 0.99 and rsi_low2 > rsi_low1 * 1.02:
        return "bullish"  # precio baja pero RSI sube = rebote probable
    if price_high2 > price_high1 * 1.01 and rsi_high2 < rsi_high1 * 0.98:
        return "bearish"  # precio sube pero RSI baja = caída probable
    return None


def golden_cross(prices: list) -> Optional[str]:
    """Detecta Golden Cross (EMA7 > EMA21) o Death Cross (EMA7 < EMA21)."""
    if len(prices) < MIN_GOLDEN_X + 2:
        return None

    # EMA actuales
    e7_now  = ema(prices, EMA_FAST)
    e21_now = ema(prices, EMA_MID)

    # EMA anteriores (sin último punto)
    e7_prev  = ema(prices[:-1], EMA_FAST)
    e21_prev = ema(prices[:-1], EMA_MID)

    if None in (e7_now, e21_now, e7_prev, e21_prev):
        return None

    if e7_prev <= e21_prev and e7_now > e21_now:
        return "golden"   # cruce alcista
    if e7_prev >= e21_prev and e7_now < e21_now:
        return "death"    # cruce bajista
    return None



# ══════════════════════════════════════════════════════════════════════════════
# FIBONACCI GOLDEN POCKET — Simulado 3.31M veces, PF 2.88 vs baseline 0.063
# ══════════════════════════════════════════════════════════════════════════════

def calc_fibonacci_levels(prices: list, lookback: int = FIB_LOOKBACK) -> Optional[dict]:
    """
    Calcula niveles Fibonacci Golden Pocket usando los últimos `lookback` precios.
    
    Golden Pocket (61.8-65% retracement): zona de mayor probabilidad de reversión
    según el PDF de estrategias Fibonacci. Optimizado con 3.31M simulaciones.
    
    Returns dict con:
      - retracement_down: % pullback desde swing high (para LONG setup)
      - retracement_up:   % bounce desde swing low (para SHORT setup)
      - in_gp_long:       True si precio en zona GP para entrada LONG
      - in_gp_short:      True si precio en zona GP para entrada SHORT
      - in_ote_long/short: True si en zona OTE más amplia (61.8-79%)
      - fib_786, fib_618, fib_1272: niveles absolutos para SL/TP
    """
    if len(prices) < lookback:
        return None
    window   = prices[-lookback:]
    swing_hi = max(window)
    swing_lo = min(window)
    rng      = swing_hi - swing_lo
    if rng == 0 or rng / swing_hi < 0.002:  # ignora rangos < 0.2%
        return None
    current = prices[-1]

    # % de pullback DESDE el swing high (setup LONG: precio cayó y rebotó)
    retrace_down = (swing_hi - current) / rng * 100
    # % de bounce DESDE el swing low (setup SHORT: precio subió desde el mínimo)
    retrace_up   = (current - swing_lo) / rng * 100

    tol = FIB_GP_TOLERANCE
    gp_lo = FIB_GP_LOW - tol
    gp_hi = FIB_GP_HIGH + tol
    ote_hi = FIB_OTE_HIGH + tol

    in_gp_long    = gp_lo <= retrace_down <= gp_hi
    in_gp_short   = gp_lo <= retrace_up   <= gp_hi
    in_ote_long   = gp_lo <= retrace_down <= ote_hi
    in_ote_short  = gp_lo <= retrace_up   <= ote_hi
    near_gp_long  = (gp_hi < retrace_down <= gp_hi + 4.0) or (gp_lo - 4.0 <= retrace_down < gp_lo)
    near_gp_short = (gp_hi < retrace_up   <= gp_hi + 4.0) or (gp_lo - 4.0 <= retrace_up   < gp_lo)

    return {
        "swing_hi":      round(swing_hi, 8),
        "swing_lo":      round(swing_lo, 8),
        "retrace_down":  round(retrace_down, 1),
        "retrace_up":    round(retrace_up, 1),
        "in_gp_long":    in_gp_long,
        "in_gp_short":   in_gp_short,
        "in_ote_long":   in_ote_long,
        "in_ote_short":  in_ote_short,
        "near_gp_long":  near_gp_long,
        "near_gp_short": near_gp_short,
        # Niveles absolutos (para SL/TP con Fibonacci)
        "fib_786":  round(swing_hi - 0.786 * rng, 8),  # 78.6% — SL zona para longs
        "fib_618":  round(swing_hi - 0.618 * rng, 8),  # 61.8% — inicio Golden Pocket
        "fib_500":  round(swing_hi - 0.500 * rng, 8),  # 50%   — mitad del rango
        "fib_382":  round(swing_hi - 0.382 * rng, 8),  # 38.2% — nivel de continuación
        "fib_1272": round(swing_lo  + 1.272 * rng, 8), # 127.2% ext — TP para LONG
        "fib_1618": round(swing_lo  + 1.618 * rng, 8), # 161.8% ext — TP agresivo
        # Equivalentes para SHORT (extensiones hacia abajo)
        "fib_786_short":  round(swing_lo + 0.786 * rng, 8),
        "fib_618_short":  round(swing_lo + 0.618 * rng, 8),
        "fib_1272_short": round(swing_hi - 1.272 * rng, 8),
    }

# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO COMPLETO DE INDICADORES POR TOKEN
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(symbol: str, token: dict, price_hist: list, vol_hist: list) -> dict:
    """Calcula todos los indicadores disponibles para un token."""
    prices  = [p["price"] for p in price_hist]
    volumes = [v.get("volume", 0) for v in vol_hist] if vol_hist else []
    n       = len(prices)

    ind = {
        "symbol":   symbol,
        "price":    token.get("price", 0),
        "change_5m": token.get("price_5min_change_pct", 0),
        "change_24h": token.get("price_24h_change_pct", 0),
        "volume_24h": token.get("volume_24h", 0),
        "market_cap": token.get("market_cap", 0),
        "n_prices":   n,
    }

    # ── EMA ──
    ind["ema7"]  = ema(prices, EMA_FAST)  if n >= EMA_FAST  else None
    ind["ema21"] = ema(prices, EMA_MID)   if n >= EMA_MID   else None
    ind["ema50"] = ema(prices, EMA_SLOW)  if n >= EMA_SLOW  else None
    ind["sma20"] = sma(prices, 20)        if n >= 20        else None

    # Tendencia general
    if ind["ema7"] and ind["ema21"]:
        ind["trend"] = "up" if ind["ema7"] > ind["ema21"] else "down"
        ind["ema_gap_pct"] = round((ind["ema7"] - ind["ema21"]) / ind["ema21"] * 100, 3)
    else:
        ind["trend"] = "unknown"
        ind["ema_gap_pct"] = 0

    # ── RSI ──
    ind["rsi"] = rsi(prices) if n >= MIN_RSI else None
    rsi_s      = rsi_series(prices) if n >= MIN_RSI else []

    # ── MACD ──
    macd_data   = macd(prices) if n >= MIN_MACD else None
    ind["macd"] = macd_data

    # ── Bollinger Bands ──
    bb          = bollinger_bands(prices) if n >= MIN_BB else None
    ind["bb"]   = bb

    # ── ATR ──
    closes      = prices
    atr_val     = atr([], [], closes) if n >= MIN_ATR else None
    ind["atr"]  = atr_val
    ind["atr_pct"] = round(atr_val / prices[-1] * 100, 4) if (atr_val and prices[-1]) else None

    # ── OBV ──
    if volumes and n >= 5:
        ind["obv"]       = obv(prices, volumes)
        prev_volumes     = volumes[:-5] if len(volumes) > 5 else volumes
        ind["obv_prev"]  = obv(prices[:-5], prev_volumes) if len(prices) > 5 else None
        if ind["obv"] and ind["obv_prev"]:
            ind["obv_trend"] = "up" if ind["obv"] > ind["obv_prev"] else "down"
        else:
            ind["obv_trend"] = "unknown"
    else:
        ind["obv"] = ind["obv_prev"] = None
        ind["obv_trend"] = "unknown"

    # ── ROC ──
    ind["roc"] = roc(prices) if n >= ROC_PERIOD + 1 else None

    # ── VWAP ──
    if volumes:
        ind["vwap"] = vwap_proxy(prices, volumes)
        if ind["vwap"] and prices:
            ind["price_vs_vwap"] = round((prices[-1] - ind["vwap"]) / ind["vwap"] * 100, 3)
        else:
            ind["price_vs_vwap"] = 0
    else:
        ind["vwap"] = None
        ind["price_vs_vwap"] = 0

    # ── Divergencia RSI ──
    ind["rsi_divergence"] = detect_rsi_divergence(prices, rsi_s) if rsi_s else None

    # ── Golden/Death Cross ──
    ind["cross"] = golden_cross(prices) if n >= MIN_GOLDEN_X + 2 else None

    # ── Keltner Channel (EMA + ATR) ──
    if ind["ema21"] and atr_val:
        ind["kc_upper"] = round(ind["ema21"] + 2 * atr_val, 8)
        ind["kc_lower"] = round(ind["ema21"] - 2 * atr_val, 8)
        price = prices[-1]
        ind["kc_position"] = "above" if price > ind["kc_upper"] else ("below" if price < ind["kc_lower"] else "inside")
    else:
        ind["kc_upper"] = ind["kc_lower"] = None
        ind["kc_position"] = "unknown"

    # ── Fibonacci Golden Pocket ── (lb=80, GP 61.8-65%, tol=0.8%)
    ind["fib"] = calc_fibonacci_levels(prices, FIB_LOOKBACK) if n >= FIB_LOOKBACK else None

    # ── Stochastic K (14 períodos — usando closes como proxy de high/low) ──
    # Validado: mejor indicador de extremos en sim 357,984 combos (2026-04-12)
    # Con RSI ≤35: WR=64.5%, PF=38.5x en 30m scalping
    if n >= 14:
        low_14  = min(prices[-14:])
        high_14 = max(prices[-14:])
        ind["stoch_k"] = round((prices[-1] - low_14) / (high_14 - low_14) * 100, 2) if (high_14 - low_14) > 0 else 50.0
    else:
        ind["stoch_k"] = None

    # Pasar price_hist para que las funciones de régimen lo usen en las estrategias
    ind["_price_hist"] = list(price_hist[-100:]) if price_hist else []
    return ind


# ══════════════════════════════════════════════════════════════════════════════
# SCORING SYSTEM — Múltiples indicadores dan confianza
# ══════════════════════════════════════════════════════════════════════════════

def score_long(ind: dict) -> tuple[float, list]:
    """
    Puntúa la probabilidad de un trade LONG.
    Retorna (score 0-1, lista de razones).
    """
    # ── Fear & Greed penaliza LONGs pero NO los bloquea ──
    # En spot/paper mode no podemos shortear, así que necesitamos
    # poder hacer longs incluso en mercados con miedo (oversold bounces)
    fear_greed = ind.get("fear_greed", 50)

    score   = 0.0
    reasons = []

    if fear_greed <= 20:
        score -= 0.15
        reasons.append(f"😨 Fear & Greed {fear_greed} — Extreme Fear, penalización LONG ⚠️")
    elif fear_greed <= 35:
        score -= 0.05
        reasons.append(f"😰 Fear & Greed {fear_greed} — Fear, cautela LONG ⚠️")
    price   = ind["price"]
    rsi_val = ind.get("rsi")
    bb      = ind.get("bb")
    macd_d  = ind.get("macd")
    atr_val = ind.get("atr")

    # ── 1. Tendencia (EMA) ── peso 0.20
    if ind["trend"] == "up":
        score += 0.20
        reasons.append(f"EMA7>{ind['ema7']:.4f} > EMA21>{ind['ema21']:.4f} ✅")
    elif ind["trend"] == "down":
        score -= 0.10
        reasons.append("Tendencia bajista ⚠️")
    elif ind["trend"] == "unknown":
        score += 0.05  # Sin datos suficientes, neutral positivo leve

    # ── 2. RSI ── peso 0.18
    if rsi_val is not None:
        if 45 <= rsi_val <= 65:
            score += 0.18
            reasons.append(f"RSI {rsi_val:.0f} zona neutral-alcista ✅")
        elif rsi_val < 35:
            score += 0.15  # Sobrevendido = rebote probable
            reasons.append(f"RSI {rsi_val:.0f} sobrevendido ✅")
        elif rsi_val > 75:
            score -= 0.12
            reasons.append(f"RSI {rsi_val:.0f} sobrecomprado ⚠️")
        else:
            score += 0.08
            reasons.append(f"RSI {rsi_val:.0f} aceptable")

    # ── 3. MACD ── peso 0.18
    if macd_d:
        if macd_d["bullish_cross"]:
            score += 0.18
            reasons.append("MACD cruce alcista ✅")
        elif macd_d["histogram"] > 0 and macd_d["histogram_growing"]:
            score += 0.12
            reasons.append(f"MACD histograma creciendo ✅")
        elif macd_d["macd"] > 0:
            score += 0.08
            reasons.append("MACD positivo")
        elif macd_d["bearish_cross"]:
            score -= 0.12
            reasons.append("MACD cruce bajista ⚠️")

    # ── 4. Bollinger Bands ── peso 0.15
    if bb:
        pct_b = bb["pct_b"]
        if pct_b < 0.20:
            score += 0.15
            reasons.append(f"Precio en BB inferior ({pct_b:.2f}) ✅")
        elif 0.35 <= pct_b <= 0.65:
            score += 0.10
            reasons.append(f"Precio en BB medio ✅")
        elif pct_b > 0.85:
            score -= 0.08
            reasons.append(f"Precio en BB superior ({pct_b:.2f}) ⚠️")
        if bb["squeeze"]:
            score += 0.08
            reasons.append("BB Squeeze — breakout inminente ✅")

    # ── 5. ATR / Volatilidad ── peso 0.10
    if ind.get("atr_pct"):
        if 0.5 <= ind["atr_pct"] <= 5.0:
            score += 0.10
            reasons.append(f"ATR {ind['atr_pct']:.2f}% — volatilidad saludable ✅")
        elif ind["atr_pct"] > 8.0:
            score -= 0.05
            reasons.append(f"ATR {ind['atr_pct']:.2f}% — demasiado volátil ⚠️")
        elif ind["atr_pct"] < 0.3:
            score -= 0.08
            reasons.append(f"ATR {ind['atr_pct']:.2f}% — sin movimiento ⚠️")

    # ── 6. OBV ── peso 0.10
    if ind.get("obv_trend") == "up":
        score += 0.10
        reasons.append("OBV alcista — acumulación ✅")
    elif ind.get("obv_trend") == "down":
        score -= 0.05
        reasons.append("OBV bajista — distribución ⚠️")

    # ── 7. ROC ── peso 0.08
    roc_val = ind.get("roc")
    if roc_val is not None:
        if roc_val > 2.0:
            score += 0.08
            reasons.append(f"ROC +{roc_val:.1f}% aceleración ✅")
        elif roc_val < -3.0:
            score -= 0.06
            reasons.append(f"ROC {roc_val:.1f}% desaceleración ⚠️")

    # ── 8. Divergencia RSI ── peso 0.10
    div = ind.get("rsi_divergence")
    if div == "bullish":
        score += 0.12
        reasons.append("Divergencia RSI alcista ✅")
    elif div == "bearish":
        score -= 0.10
        reasons.append("Divergencia RSI bajista ⚠️")

    # ── 9. Golden Cross ── peso 0.12
    cross = ind.get("cross")
    if cross == "golden":
        score += 0.15
        reasons.append("Golden Cross EMA7/EMA21 ✅")
    elif cross == "death":
        score -= 0.12
        reasons.append("Death Cross EMA7/EMA21 ⚠️")

    # ── 10. VWAP ── peso 0.08
    pvwap = ind.get("price_vs_vwap", 0)
    if pvwap and -1.0 <= pvwap <= 1.0:
        score += 0.08
        reasons.append(f"Precio cerca de VWAP ({pvwap:+.2f}%) ✅")
    elif pvwap and pvwap > 3.0:
        score -= 0.05
        reasons.append(f"Precio muy sobre VWAP ({pvwap:+.2f}%) ⚠️")

    # ── 11. Keltner Channel ── peso 0.06
    if ind.get("kc_position") == "below":
        score += 0.08
        reasons.append("Precio bajo Keltner Channel — rebote potencial ✅")
    elif ind.get("kc_position") == "above":
        score += 0.04
        reasons.append("Precio sobre Keltner Channel — momentum fuerte")

    # ── 12. Cambio 24h como contexto macro ──
    c24 = ind.get("change_24h", 0)
    if 3 <= c24 <= 15:
        score += 0.06
        reasons.append(f"24h +{c24:.1f}% — momentum macro ✅")
    elif c24 > 20:
        score -= 0.05
        reasons.append(f"24h +{c24:.1f}% — posible sobreextensión ⚠️")
    elif c24 < -10:
        score += 0.04  # Oversold opportunity
        reasons.append(f"24h {c24:.1f}% — posible rebote")

    # ── 13. Fibonacci Golden Pocket ── (PF 2.88 en 3.31M sims, lb=80 óptimo)
    fib = ind.get("fib")
    if fib:
        ret_down = fib["retrace_down"]
        rsi_ok   = (rsi_val is None) or (rsi_val <= FIB_RSI_THRESHOLD)
        if fib["in_gp_long"] and rsi_ok:
            score += 0.35
            reasons.append(f"FIB 🎯 Golden Pocket LONG {ret_down:.1f}% retrace RSI≤{FIB_RSI_THRESHOLD} ✅")
        elif fib["in_gp_long"]:
            score += 0.20
            reasons.append(f"FIB Golden Pocket LONG {ret_down:.1f}% retrace ✅")
        elif fib["in_ote_long"] and rsi_ok:
            score += 0.15
            reasons.append(f"FIB OTE Zone LONG {ret_down:.1f}% retrace ✅")
        elif fib["near_gp_long"]:
            score += 0.08
            reasons.append(f"FIB acercándose a GP {ret_down:.1f}% retrace")

    return round(min(max(score, 0), 1), 3), reasons


def score_short(ind: dict) -> tuple[float, list]:
    """Puntúa la probabilidad de un trade SHORT — análisis independiente.
    
    Dos estrategias:
    1. Reversión de sobrecompra (RSI alto, BB superior)
    2. Trend Following bajista (Fear & Greed bajo, tendencia confirmada)
    """
    score = 0.30  # Base neutral-baja
    reasons = []

    rsi_val = ind.get("rsi")
    macd_d  = ind.get("macd")
    bb      = ind.get("bb")
    fear_greed = ind.get("fear_greed", 50)  # Default neutral

    # ══════════════════════════════════════════════════════════════════
    # NUEVO: Trend Following SHORT en mercados bajistas
    # ══════════════════════════════════════════════════════════════════

    # Guard: penalizar SHORT si el precio 24h es alcista (mercado recuperándose)
    c24_guard = ind.get('change_24h', 0)
    if c24_guard > 8:
        score -= 0.18
        reasons.append(f"📈 24h +{c24_guard:.1f}% momentum alcista fuerte — SHORT muy arriesgado ⚠️⚠️⚠️")
    elif c24_guard > 4:
        score -= 0.10
        reasons.append(f"📈 24h +{c24_guard:.1f}% momentum alcista — SHORT arriesgado ⚠️⚠️")

    # Fear & Greed + acción de precio real — NO shortear solo por pánico si el precio sube
    c24_fg = ind.get('change_24h', 0)
    if fear_greed <= 20:
        if c24_fg < -2 and ind.get('trend') == 'down':
            score += 0.20
            reasons.append(f"😨 F&G {fear_greed} + precio {c24_fg:.1f}% + EMA down — trend following SHORT ✅✅")
        elif c24_fg < 0:
            score += 0.08
            reasons.append(f"😨 F&G {fear_greed} Extreme Fear, precio {c24_fg:.1f}% — SHORT parcial ⚠️")
        else:
            score -= 0.05
            reasons.append(f"😨 F&G {fear_greed} pero precio {c24_fg:+.1f}% — mercado rebotando, SHORT arriesgado ⚠️⚠️")
    elif fear_greed <= 35:
        if c24_fg < 0:
            score += 0.12
            reasons.append(f"😰 F&G {fear_greed} Fear + precio {c24_fg:.1f}% — favorable SHORT ✅")
        else:
            score += 0.03
            reasons.append(f"😰 F&G {fear_greed} Fear pero precio {c24_fg:+.1f}% — SHORT con cautela ⚠️")
    elif fear_greed >= 75:
        score -= 0.10
        reasons.append(f"🤑 Fear & Greed {fear_greed} — Greed extremo, SHORT arriesgado ⚠️")

    # ── 1. Tendencia bajista ── peso 0.15 (aumentado en mercado Fear)
    if ind.get("trend") == "down":
        base_trend_score = 0.15
        if fear_greed <= 25:
            base_trend_score = 0.22  # Bonus en mercado Fear
        score += base_trend_score
        reasons.append("Tendencia bajista EMA ✅")
    elif ind.get("trend") == "up":
        score -= 0.10
        reasons.append("Tendencia alcista — SHORT riesgoso ⚠️")

    # ── 2. RSI — adaptado para trend following
    if rsi_val:
        if rsi_val > 80:
            score += 0.18
            reasons.append(f"RSI {rsi_val:.0f} extremo sobrecomprado ✅✅")
        elif rsi_val > RSI_OVERBOUGHT:
            score += 0.12
            reasons.append(f"RSI {rsi_val:.0f} sobrecomprado ✅")
        elif rsi_val < 30 and fear_greed <= 25:
            # En mercado Fear extremo, RSI bajo = momentum bajista fuerte
            score += 0.08
            reasons.append(f"RSI {rsi_val:.0f} + Fear extremo = momentum bajista ✅")
        elif rsi_val < 40 and fear_greed > 40:
            # Solo penalizar en mercados neutrales/alcistas
            score -= 0.08
            reasons.append(f"RSI {rsi_val:.0f} bajo — no ideal para SHORT ⚠️")

    # ── 3. MACD bearish ── peso 0.12
    if macd_d:
        if macd_d.get("bearish_cross"):
            score += 0.12
            reasons.append("MACD cruce bajista ✅")
        elif macd_d.get("bullish_cross"):
            score -= 0.08
            reasons.append("MACD cruce alcista — contradice SHORT ⚠️")
        hist = macd_d.get("histogram", 0)
        if hist and hist < -0.5:
            score += 0.05
            reasons.append("MACD histograma negativo fuerte ✅")

    # ── 4. Bollinger Band superior ── peso 0.10
    if bb:
        pct_b = bb.get("pct_b", 0.5)
        if pct_b > 0.95:
            score += 0.12
            reasons.append(f"Precio sobre BB superior ({pct_b:.2f}) ✅✅")
        elif pct_b > 0.85:
            score += 0.08
            reasons.append(f"Precio en BB superior ({pct_b:.2f}) ✅")
        elif pct_b < 0.3:
            score -= 0.08
            reasons.append(f"Precio en BB inferior — no SHORT ⚠️")

    # ── 5. Death Cross ── peso 0.10
    if ind.get("cross") == "death":
        score += 0.12
        reasons.append("Death Cross EMA7/EMA21 ✅")
    elif ind.get("cross") == "golden":
        score -= 0.10
        reasons.append("Golden Cross — contradice SHORT ⚠️")

    # ── 6. Divergencia RSI bearish ── peso 0.10
    if ind.get("rsi_divergence") == "bearish":
        score += 0.12
        reasons.append("Divergencia RSI bajista ✅")
    elif ind.get("rsi_divergence") == "bullish":
        score -= 0.08
        reasons.append("Divergencia RSI alcista — contradice SHORT ⚠️")

    # ── 7. OBV distribución ── peso 0.08
    if ind.get("obv_trend") == "down":
        score += 0.08
        reasons.append("OBV distribución — venta institucional ✅")
    elif ind.get("obv_trend") == "up":
        score -= 0.05
        reasons.append("OBV acumulación — contradice SHORT ⚠️")

    # ── 8. ROC negativo ── peso 0.06
    roc_val = ind.get("roc")
    if roc_val is not None:
        if roc_val < -3.0:
            score += 0.08
            reasons.append(f"ROC {roc_val:.1f}% momentum bajista ✅")
        elif roc_val > 3.0:
            score -= 0.05
            reasons.append(f"ROC +{roc_val:.1f}% momentum alcista ⚠️")

    # ── 9. Sobreextensión 24h ── peso 0.08 (adaptado para trend following)
    c24 = ind.get("change_24h", 0)
    if c24 > 20:
        score += 0.10
        reasons.append(f"24h +{c24:.1f}% sobreextendido — reversión probable ✅")
    elif c24 > 10:
        score += 0.06
        reasons.append(f"24h +{c24:.1f}% rally extendido ✅")
    elif c24 < -10 and fear_greed > 40:
        # Solo penalizar caídas en mercados neutrales/alcistas
        score -= 0.08
        reasons.append(f"24h {c24:.1f}% ya cayó mucho — no SHORT ⚠️")
    elif c24 < -5 and fear_greed <= 25:
        # En mercado Fear extremo, caídas moderadas = trend following
        score += 0.10
        reasons.append(f"24h {c24:.1f}% + Fear extremo = continuación bajista ✅")

    # ── 10. VWAP resistencia ── peso 0.06
    pvwap = ind.get("price_vs_vwap", 0)
    if pvwap and pvwap > 3.0:
        score += 0.06
        reasons.append(f"Precio muy sobre VWAP (+{pvwap:.1f}%) — regresión probable ✅")

    # ── 11. Keltner Channel superior ── peso 0.05
    if ind.get("kc_position") == "above":
        score += 0.06
        reasons.append("Sobre Keltner Channel — sobreextensión ✅")

    # ── 12. Fibonacci Golden Pocket SHORT ── (bounce a 61.8-65% = short opportunity)
    fib = ind.get("fib")
    if fib:
        ret_up = fib["retrace_up"]
        rsi_short_ok = (rsi_val is None) or (rsi_val >= 50)
        if fib["in_gp_short"] and rsi_short_ok:
            score += 0.25
            reasons.append(f"FIB 🎯 Golden Pocket SHORT {ret_up:.1f}% bounce RSI≥50 ✅")
        elif fib["in_gp_short"]:
            score += 0.15
            reasons.append(f"FIB Golden Pocket SHORT {ret_up:.1f}% bounce ✅")
        elif fib["in_ote_short"] and rsi_short_ok:
            score += 0.10
            reasons.append(f"FIB OTE Zone SHORT {ret_up:.1f}% bounce ✅")

    if not reasons:
        reasons = [f"Score SHORT base {score:.2f}"]

    return round(min(max(score, 0), 1), 3), reasons


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIAS
# ══════════════════════════════════════════════════════════════════════════════

def build_signal(symbol, direction, strategy, ind, score, reasons, risk_eval):
    """Construye el diccionario de señal estandarizado."""
    price   = ind["price"]
    atr_val = ind.get("atr") or (price * 0.02)  # 2% fallback si no hay ATR

    # Usar parámetros del auto_learner si están disponibles
    al_params = load_auto_learner_params()
    al_sl_pct = al_params["sl_pct"]
    al_tp_pct = al_params["tp_pct"]

    # SL/TP primario basado en ATR (dinámica); luego ajustado por pct mínimo del auto_learner
    atr_sl = ATR_SL_MULTIPLIER * atr_val
    atr_tp = ATR_TP_MULTIPLIER * atr_val
    # Si el SL ATR es menor que el mínimo del auto_learner, usar el pct del auto_learner
    min_sl_abs = price * al_sl_pct
    min_tp_abs = price * al_tp_pct
    effective_sl = max(atr_sl, min_sl_abs)
    effective_tp = max(atr_tp, min_tp_abs)

    sl = round(price - effective_sl, 8) if direction == "long" \
         else round(price + effective_sl, 8)
    tp = round(price + effective_tp, 8) if direction == "long" \
         else round(price - effective_tp, 8)

    # Si estamos en Fibonacci Golden Pocket, usar niveles Fibonacci para SL/TP
    # (PF 2.88 en simulación vs ATR baseline — SL=fib_786×0.99, TP=fib_1272)
    fib_used = False
    fib = ind.get("fib")
    if fib and direction == "long" and fib.get("in_gp_long"):
        fib_sl = round(fib["fib_786"] * (1 - FIB_SL_BUFFER), 8)
        fib_tp = fib["fib_1272"]
        # Solo usar Fibonacci SL/TP si mejoran el RR vs ATR
        atr_rr = effective_tp / effective_sl if effective_sl > 0 else 0
        fib_rr_raw = abs(fib_tp - price) / abs(price - fib_sl) if abs(price - fib_sl) > 0 else 0
        if fib_rr_raw >= 1.5:  # mínimo RR 1.5:1
            sl = fib_sl
            tp = fib_tp
            fib_used = True
    elif fib and direction == "short" and fib.get("in_gp_short"):
        fib_sl = round(fib["fib_786_short"] * (1 + FIB_SL_BUFFER), 8)
        fib_tp = fib["fib_1272_short"]
        fib_rr_raw = abs(fib_tp - price) / abs(price - fib_sl) if abs(price - fib_sl) > 0 else 0
        if fib_rr_raw >= 1.5:
            sl = fib_sl
            tp = fib_tp
            fib_used = True

    rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0

    risk_sizing = risk_eval.get("position_size", {})

    return {
        "symbol":              symbol,
        "direction":           direction,
        "strategy":            strategy,
        "confidence":          score,
        "price":               price,
        "sl_price":            sl,
        "tp_price":            tp,
        "rr_ratio":            round(rr, 2),
        "atr":                 atr_val,
        "rsi":                 ind.get("rsi"),
        "ema7":                ind.get("ema7"),
        "ema21":               ind.get("ema21"),
        "macd_hist":           ind.get("macd", {}).get("histogram") if ind.get("macd") else None,
        "bb_pct_b":            ind.get("bb", {}).get("pct_b") if ind.get("bb") else None,
        "obv_trend":           ind.get("obv_trend"),
        "roc":                 ind.get("roc"),
        "divergence":          ind.get("rsi_divergence"),
        "cross":               ind.get("cross"),
        "trend":               ind.get("trend"),
        "n_indicators":        len([r for r in reasons if "✅" in r]),
        "reasons":             reasons[:8],
        "suggested_size_usd":  risk_sizing.get("position_size_usd", 100),
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "fib_golden_pocket":   fib_used,
        "fib_retrace":         fib["retrace_down"] if (fib and direction == "long") else (fib["retrace_up"] if fib else None),
    }


def strategy_trend_momentum(symbol, ind, risk_eval) -> Optional[dict]:
    """
    Trend Momentum: tendencia EMA + RSI confirmado + MACD + OBV.
    Mínimo 2 indicadores alcistas para entrar.
    """
    if not risk_eval.get("approved"):
        return None
    if ind["price"] <= 0:
        return None

    long_score, long_reasons  = score_long(ind)
    short_score, short_reasons = score_short(ind)

    best_score = max(long_score, short_score)
    if best_score < MIN_CONFIDENCE:
        return None

    direction = "long" if long_score >= short_score else "short"
    score     = long_score if direction == "long" else short_score
    reasons   = long_reasons if direction == "long" else short_reasons

    bullish_confirmations = len([r for r in reasons if "✅" in r])
    if bullish_confirmations < MIN_INDICATORS_ALIGNED:
        return None

    return build_signal(symbol, direction, "trend_momentum", ind, score, reasons, risk_eval)


def strategy_breakout(symbol, ind, risk_eval) -> Optional[dict]:
    """
    Breakout: BB squeeze + precio rompe BB superior + OBV confirma.
    """
    if not risk_eval.get("approved"):
        return None
    bb  = ind.get("bb")
    obv = ind.get("obv_trend")
    rsi_val = ind.get("rsi")
    if not bb:
        return None

    # Condición: precio sobre BB superior O saliendo de squeeze
    if not (bb["above_upper"] or (bb["squeeze"] and ind.get("change_5m", 0) > 1.0)):
        return None

    # OBV debe confirmar (acumulación)
    if obv == "down":
        return None

    # Sim: RSI <= 80 es el filtro óptimo (PF 2.55 en 7.2M sims)
    if rsi_val and rsi_val > 80:
        return None

    score_base = 0.50
    reasons = [f"BB breakout (pct_b={bb['pct_b']:.2f}) ✅"]
    if bb["squeeze"]:
        score_base += 0.10
        reasons.append("BB Squeeze previo ✅")
    if obv == "up":
        score_base += 0.12
        reasons.append("OBV confirma breakout ✅")
    if ind.get("macd", {}) and ind["macd"].get("histogram", 0) > 0:
        score_base += 0.08
        reasons.append("MACD positivo ✅")
    if ind["trend"] == "up":
        score_base += 0.08
        reasons.append("Tendencia EMA alcista ✅")
    if rsi_val and 55 <= rsi_val <= 75:
        score_base += 0.07
        reasons.append(f"RSI {rsi_val:.0f} zona breakout ✅")

    return build_signal(symbol, "long", "breakout", ind, min(score_base, 0.95), reasons, risk_eval)


def strategy_oversold_bounce(symbol, ind, risk_eval) -> Optional[dict]:
    """
    Oversold Bounce: RSI < 30, precio en/bajo BB inferior, divergencia alcista.
    """
    if not risk_eval.get("approved"):
        return None
    rsi_val = ind.get("rsi")
    bb      = ind.get("bb")
    c24     = ind.get("change_24h", 0)

    # Sim: RSI<30 + c24<-8% (condición AND más estricta que OR evita señales falsas)
    # OR condition daba 44.7% señal rate — demasiado permisivo
    oversold_rsi    = rsi_val is not None and rsi_val < RSI_OVERSOLD  # default 25
    oversold_price  = c24 < -8.0  # Caída fuerte como proxy

    # Requiere AL MENOS una condición fuerte; si solo c24 aplica, también necesita RSI bajo 40
    if not oversold_rsi and not oversold_price:
        return None
    if oversold_price and not oversold_rsi:
        # Solo c24 trigger: exigir RSI < 40 como filtro adicional
        if rsi_val is not None and rsi_val >= 40:
            return None

    score_base = 0.45
    reasons    = []

    if oversold_rsi:
        score_base += 0.15
        reasons.append(f"RSI {rsi_val:.0f} sobrevendido ✅")
    if oversold_price:
        score_base += 0.10
        reasons.append(f"Caída 24h {c24:.1f}% — rebote potencial ✅")
    if bb and bb["below_lower"]:
        score_base += 0.12
        reasons.append("Precio bajo BB inferior ✅")
    if ind.get("rsi_divergence") == "bullish":
        score_base += 0.15
        reasons.append("Divergencia RSI alcista ✅")
    if ind.get("obv_trend") == "up":
        score_base += 0.08
        reasons.append("OBV acumulación ✅")
    if ind.get("kc_position") == "below":
        score_base += 0.08
        reasons.append("Bajo Keltner Channel ✅")
    if ind.get("macd", {}) and ind["macd"].get("bullish_cross"):
        score_base += 0.10
        reasons.append("MACD cruce alcista ✅")

    # Mercado bajista general reduce confianza
    if ind["trend"] == "down":
        score_base -= 0.08
        reasons.append("Contratendencia bajista ⚠️")

    if score_base < MIN_CONFIDENCE or not reasons:
        return None

    return build_signal(symbol, "long", "oversold_bounce", ind, min(score_base, 0.92), reasons, risk_eval)



# ── Régimen de Mercado ────────────────────────────────────────────────────────
def efficiency_ratio(price_hist: list, period: int = 14) -> float:
    """
    Kaufman Efficiency Ratio.
    0 = caótico/mean-reverting → scalping tiene ventaja estadística
    1 = tendencia pura          → scalping es peligroso
    Umbral bloqueo: ER > 0.55
    """
    if len(price_hist) < period + 1:
        return 0.5
    prices = [p["price"] for p in price_hist[-(period + 1):]]
    net  = abs(prices[-1] - prices[0])
    path = sum(abs(prices[i] - prices[i - 1]) for i in range(1, len(prices)))
    return net / path if path > 0 else 0.5


def hurst_exponent(price_hist: list, min_len: int = 50) -> float:
    """
    Hurst exponent R/S simplificado.
    H < 0.45 → mean-reverting (rebotes predecibles)
    H = 0.5  → caminata aleatoria
    H > 0.55 → trending (memoria positiva)
    """
    import math as _math
    prices = [p["price"] for p in price_hist[-100:]]
    n = len(prices)
    if n < min_len:
        return 0.5
    try:
        returns = [_math.log(prices[i] / prices[i - 1]) for i in range(1, n) if prices[i - 1] > 0]
        rs_pairs = []
        for lag in [8, 16, 32]:
            if len(returns) < lag * 2:
                continue
            segs = [returns[i:i + lag] for i in range(0, len(returns) - lag + 1, lag)]
            rs_vals = []
            for seg in segs:
                if not seg:
                    continue
                m = sum(seg) / len(seg)
                cum = [sum(seg[:k + 1]) - m * (k + 1) for k in range(len(seg))]
                R = max(cum) - min(cum)
                S = (_math.sqrt(sum((x - m) ** 2 for x in seg) / len(seg)))
                if S > 0:
                    rs_vals.append(R / S)
            if rs_vals:
                rs_pairs.append((_math.log(lag), _math.log(sum(rs_vals) / len(rs_vals))))
        if len(rs_pairs) < 2:
            return 0.5
        xs = [p[0] for p in rs_pairs]
        ys = [p[1] for p in rs_pairs]
        xm = sum(xs) / len(xs)
        ym = sum(ys) / len(ys)
        num = sum((xs[i] - xm) * (ys[i] - ym) for i in range(len(xs)))
        den = sum((xs[i] - xm) ** 2 for i in range(len(xs)))
        H = num / den if den != 0 else 0.5
        return max(0.0, min(1.0, H))
    except Exception:
        return 0.5


def market_regime(price_hist: list, symbol: str = "") -> str:
    """
    Clasifica el régimen de mercado combinando ER, Hurst y ADX (30m OHLCV).
    'mean_reverting' → scalping con ventaja estadística
    'trending'       → scalping peligroso, usar momentum/cross
    'neutral'        → condición ambigua
    """
    er = efficiency_ratio(price_hist, period=14)
    H  = hurst_exponent(price_hist, min_len=50)

    # ADX real desde velas 30m (más preciso que ER/Hurst solos)
    adx = None
    if symbol:
        try:
            import candle_aggregator as _ca
            adx = _ca.get_adx(symbol, period=14)
        except Exception:
            pass

    # ADX > 25 confirma tendencia fuerte (más peso que ER/Hurst)
    if adx is not None:
        if adx > 28:
            return "trending"
        if adx < 18 and (er < 0.50 or H < 0.52):
            return "mean_reverting"

    # Fallback: ER + Hurst
    if er > 0.60 or H > 0.58:
        return "trending"
    if er < 0.42 and H < 0.48:
        return "mean_reverting"
    return "neutral"

def strategy_golden_cross(symbol, ind, risk_eval) -> Optional[dict]:
    """Golden Cross EMA7/EMA21 con volumen y RSI confirmando."""
    if not risk_eval.get("approved"):
        return None
    if ind.get("cross") != "golden":
        return None

    rsi_val = ind.get("rsi")
    macd_d  = ind.get("macd")

    # Sim: RSI 50-70 + MACD confirmado = PF 3.088 (mejor config en 11.5M sims)
    # MACD histograma positivo es filtro obligatorio para calidad
    if not macd_d or macd_d.get("histogram", 0) <= 0:
        return None  # Golden Cross sin MACD positivo = señal débil

    if rsi_val and (rsi_val < 50 or rsi_val > 75):
        return None  # RSI fuera de zona óptima 50-75

    score_base = 0.60
    reasons = ["Golden Cross EMA7/EMA21 ✅"]

    if rsi_val and 50 <= rsi_val <= 70:
        score_base += 0.12
        reasons.append(f"RSI {rsi_val:.0f} zona óptima (50-70) ✅")
    if macd_d.get("histogram", 0) > 0:
        score_base += 0.10
        reasons.append("MACD histograma positivo ✅")
    if macd_d.get("bullish_cross"):
        score_base += 0.08
        reasons.append("MACD cruce alcista confirmado ✅")
    if ind.get("obv_trend") == "up":
        score_base += 0.10
        reasons.append("OBV acumulación ✅")
    if ind.get("bb", {}) and ind["bb"].get("pct_b", 0.5) > 0.45:
        score_base += 0.06
        reasons.append("BB posición favorable ✅")

    return build_signal(symbol, "long", "golden_cross", ind, min(score_base, 0.92), reasons, risk_eval)


def strategy_death_cross(symbol, ind, risk_eval) -> Optional[dict]:
    """Death Cross EMA7/EMA21 — señal SHORT para Pure Mode y Wild Mode.
    Sim validado: WR~65%, PF~3.0 en 1h×30d (2026-04-12)."""
    if not risk_eval.get("approved"):
        return None
    if ind.get("cross") != "death":
        return None

    rsi_val = ind.get("rsi")
    macd_d  = ind.get("macd")

    if not macd_d or macd_d.get("histogram", 0) >= 0:
        return None  # Death Cross sin MACD negativo = señal débil

    if rsi_val and (rsi_val < 30 or rsi_val > 60):
        return None  # Muy sobrevendido (riesgo rebote) o muy alto = skip

    score_base = 0.60
    reasons = ["Death Cross EMA7/EMA21 ✅"]

    if rsi_val and 40 <= rsi_val <= 58:
        score_base += 0.12
        reasons.append(f"RSI {rsi_val:.0f} zona óptima short (40-58) ✅")
    if macd_d.get("histogram", 0) < 0:
        score_base += 0.10
        reasons.append("MACD histograma negativo ✅")
    if macd_d.get("bearish_cross"):
        score_base += 0.08
        reasons.append("MACD cruce bajista confirmado ✅")
    if ind.get("obv_trend") == "down":
        score_base += 0.10
        reasons.append("OBV distribución ✅")
    if ind.get("bb", {}) and ind["bb"].get("pct_b", 0.5) < 0.55:
        score_base += 0.06
        reasons.append("BB posición desfavorable ✅")

    return build_signal(symbol, "short", "death_cross", ind, min(score_base, 0.92), reasons, risk_eval)


def strategy_macd_cross(symbol, ind, risk_eval) -> Optional[dict]:
    """MACD Cross con tendencia EMA y RSI confirman."""
    if not risk_eval.get("approved"):
        return None
    macd_d = ind.get("macd")
    if not macd_d:
        return None

    direction  = None
    score_base = 0.45
    reasons    = []

    if macd_d["bullish_cross"]:
        direction  = "long"
        score_base = 0.50
        reasons.append("MACD cruce alcista ✅")
    elif macd_d["bearish_cross"]:
        direction  = "short"
        score_base = 0.50
        reasons.append("MACD cruce bajista ✅")
    else:
        return None

    rsi_val = ind.get("rsi")
    if direction == "long":
        if ind["trend"] == "up":
            score_base += 0.12
            reasons.append("Tendencia EMA alcista ✅")
        if rsi_val and 40 <= rsi_val <= 65:
            score_base += 0.10
            reasons.append(f"RSI {rsi_val:.0f} ✅")
        if ind.get("obv_trend") == "up":
            score_base += 0.08
            reasons.append("OBV acumulación ✅")
        if macd_d["histogram_growing"]:
            score_base += 0.08
            reasons.append("Histograma MACD creciendo ✅")
    else:  # short
        if ind["trend"] == "down":
            score_base += 0.12
            reasons.append("Tendencia EMA bajista ✅")
        if rsi_val and rsi_val > 60:
            score_base += 0.10
            reasons.append(f"RSI {rsi_val:.0f} sobrecomprado ✅")
        if ind.get("obv_trend") == "down":
            score_base += 0.08
            reasons.append("OBV distribución ✅")

    if score_base < MIN_CONFIDENCE:
        return None

    return build_signal(symbol, direction, "macd_cross", ind, min(score_base, 0.90), reasons, risk_eval)




def strategy_stoch_rsi_scalp(symbol: str, ind: dict, risk_eval: dict) -> Optional[dict]:
    """
    PRIMARY Wild Mode: Scalping bidireccional Stochastic K + RSI.
    Doble confirmación de extremo — validado como mejor estrategia Wild Mode.

    LONG:  Stoch K ≤ 20  AND  RSI ≤ 35  → precio doblemente sobrevendido
    SHORT: Stoch K ≥ 80  AND  RSI ≥ 65  → precio doblemente sobrecomprado

    Simulación 357,984 combos (30m × 15d, Kraken, 2026-04-12):
      + Martingala: WR=64.5%, PF=38.5x, Daily=+0.64%, DD=-0.44%
      + Sin mart.:  WR=~35%, PF=~2.5x (estrategia selectiva = pocos trades)

    Solo tokens core para martingala (SOL,ETH,XRP,BTC,JUP).
    exit_mode=fixed para cierre rápido.
    """
    if not risk_eval.get("approved"):
        return None
    # ── Filtro de régimen: stoch_rsi solo funciona en mercados caóticos ──
    _ph = ind.get("_price_hist", [])
    if len(_ph) >= 50:
        _regime = market_regime(_ph, symbol)
        if _regime == "trending":
            log.debug(f"⛔ stoch_rsi_scalp bloqueado {symbol}: tendencia detectada (ER/Hurst)")
            return None

    price   = ind.get("price", 0)
    rsi_val = ind.get("rsi")
    stoch_k = ind.get("stoch_k")
    atr_pct = ind.get("atr_pct", 0)
    fg      = ind.get("fear_greed", 50)
    bb      = ind.get("bb", {})
    obv_tr  = ind.get("obv_trend", "unknown")
    diverg  = ind.get("rsi_divergence", None)

    if price <= 0 or rsi_val is None or stoch_k is None:
        return None
    if atr_pct is not None and atr_pct > 8.0:   # Evitar tokens extremadamente volátiles
        return None

    # ── SCALP LONG: Stoch K ≤ 20 AND RSI ≤ 35 ──────────────────────────────
    if stoch_k <= 20 and rsi_val <= 35 and fg <= 70:
        score = 0.62
        reasons = [f"Stoch K {stoch_k:.0f} sobrevendido + RSI {rsi_val:.0f} — doble extremo ✅"]

        if rsi_val <= 28:
            score += 0.10
            reasons.append(f"RSI {rsi_val:.0f} extremo (<28) ✅")
        if stoch_k <= 10:
            score += 0.08
            reasons.append(f"Stoch K {stoch_k:.0f} extremo (<10) ✅")
        if bb and bb.get("below_lower"):
            score += 0.07
            reasons.append("Precio bajo BB inferior ✅")
        if obv_tr == "up":
            score += 0.06
            reasons.append("OBV rebote alcista ✅")
        if diverg == "bullish":
            score += 0.08
            reasons.append("Divergencia RSI bullish ✅")
        if fg <= 20:
            score += 0.06
            reasons.append(f"Extreme Fear {fg} — rebote probable ✅")

        if score >= 0.76 and len(reasons) >= 2:
            sig = build_signal(symbol, "long", "stoch_rsi_scalp", ind, min(score, 0.93), reasons, risk_eval)
            if sig:
                sig["exit_mode"] = "fixed"
                sig["trailing_pct"] = 0.0
                sig["mart_eligible"] = symbol in {"SOL", "ETH", "XRP", "BTC", "JUP"}
            return sig

    # ── SCALP SHORT: Stoch K ≥ 80 AND RSI ≥ 65 ──────────────────────────────
    if stoch_k >= 80 and rsi_val >= 65 and fg >= 25:
        score = 0.62
        reasons = [f"Stoch K {stoch_k:.0f} sobrecomprado + RSI {rsi_val:.0f} — doble extremo ✅"]

        if rsi_val >= 72:
            score += 0.10
            reasons.append(f"RSI {rsi_val:.0f} extremo (>72) ✅")
        if stoch_k >= 90:
            score += 0.08
            reasons.append(f"Stoch K {stoch_k:.0f} extremo (>90) ✅")
        if bb and bb.get("above_upper"):
            score += 0.07
            reasons.append("Precio sobre BB superior ✅")
        if obv_tr == "down":
            score += 0.06
            reasons.append("OBV distribución bajista ✅")
        if diverg == "bearish":
            score += 0.08
            reasons.append("Divergencia RSI bearish ✅")
        if fg >= 80:
            score += 0.06
            reasons.append(f"Extreme Greed {fg} — caída probable ✅")

        if score >= 0.76 and len(reasons) >= 2:
            sig = build_signal(symbol, "short", "stoch_rsi_scalp", ind, min(score, 0.93), reasons, risk_eval)
            if sig:
                sig["exit_mode"] = "fixed"
                sig["trailing_pct"] = 0.0
                sig["mart_eligible"] = symbol in {"SOL", "ETH", "XRP", "BTC", "JUP"}
            return sig

    return None


def strategy_rsi_bb_scalp(symbol: str, ind: dict, risk_eval: dict) -> Optional[dict]:
    """
    SECONDARY Wild Mode: RSI extremo + precio toca banda Bollinger.
    WR=73-76% con martingala, PF=17-20x (sim 357,984 combos 2026-04-12).

    LONG:  RSI ≤ 30  AND  BB pct_b < 0.05  (precio en la banda inferior)
    SHORT: RSI ≥ 70  AND  BB pct_b > 0.95  (precio en la banda superior)
    """
    if not risk_eval.get("approved"):
        return None
    # ── Filtro de régimen: rsi_bb también requiere mercado caótico ──
    _ph2 = ind.get("_price_hist", [])
    if len(_ph2) >= 50:
        _regime2 = market_regime(_ph2, symbol)
        if _regime2 == "trending":
            log.debug(f"⛔ rsi_bb_scalp bloqueado {symbol}: tendencia detectada")
            return None

    price   = ind.get("price", 0)
    rsi_val = ind.get("rsi")
    bb      = ind.get("bb", {})
    atr_pct = ind.get("atr_pct", 0)
    fg      = ind.get("fear_greed", 50)
    obv_tr  = ind.get("obv_trend", "unknown")
    diverg  = ind.get("rsi_divergence", None)

    if price <= 0 or rsi_val is None or not bb:
        return None
    if atr_pct is not None and atr_pct > 8.0:
        return None

    bb_pct_b = bb.get("pct_b", 0.5)

    # ── LONG: RSI extremo + BB inferior ──────────────────────────────────────
    if rsi_val <= 30 and bb_pct_b is not None and bb_pct_b < 0.05 and fg <= 70:
        score = 0.64
        reasons = [f"RSI {rsi_val:.0f} oversold + precio en BB inferior ({bb_pct_b:.2f}) ✅"]

        if rsi_val <= 25:
            score += 0.10
            reasons.append(f"RSI {rsi_val:.0f} extremo ✅")
        if bb_pct_b < 0.02:
            score += 0.08
            reasons.append("BB inferior extremo (<0.02) ✅")
        if obv_tr == "up":
            score += 0.06
            reasons.append("OBV acumulación ✅")
        if diverg == "bullish":
            score += 0.08
            reasons.append("Divergencia RSI bullish ✅")
        if fg <= 20:
            score += 0.06
            reasons.append(f"Extreme Fear {fg} ✅")

        if score >= 0.76 and len(reasons) >= 2:
            sig = build_signal(symbol, "long", "rsi_bb_scalp", ind, min(score, 0.93), reasons, risk_eval)
            if sig:
                sig["exit_mode"] = "fixed"
                sig["trailing_pct"] = 0.0
                sig["mart_eligible"] = symbol in {"SOL", "ETH", "XRP", "BTC", "JUP"}
            return sig

    # ── SHORT: RSI extremo + BB superior ─────────────────────────────────────
    if rsi_val >= 70 and bb_pct_b is not None and bb_pct_b > 0.95 and fg >= 25:
        score = 0.64
        reasons = [f"RSI {rsi_val:.0f} overbought + precio en BB superior ({bb_pct_b:.2f}) ✅"]

        if rsi_val >= 75:
            score += 0.10
            reasons.append(f"RSI {rsi_val:.0f} extremo ✅")
        if bb_pct_b > 0.98:
            score += 0.08
            reasons.append("BB superior extremo (>0.98) ✅")
        if obv_tr == "down":
            score += 0.06
            reasons.append("OBV distribución ✅")
        if diverg == "bearish":
            score += 0.08
            reasons.append("Divergencia RSI bearish ✅")
        if fg >= 80:
            score += 0.06
            reasons.append(f"Extreme Greed {fg} ✅")

        if score >= 0.76 and len(reasons) >= 2:
            sig = build_signal(symbol, "short", "rsi_bb_scalp", ind, min(score, 0.93), reasons, risk_eval)
            if sig:
                sig["exit_mode"] = "fixed"
                sig["trailing_pct"] = 0.0
                sig["mart_eligible"] = symbol in {"SOL", "ETH", "XRP", "BTC", "JUP"}
            return sig

    return None


def strategy_scalping(symbol: str, ind: dict, risk_eval: dict) -> Optional[dict]:
    """
    Scalping bidireccional RSI+ATR — suplementario en Wild Mode (COMBO).
    
    LONG:  RSI ≤ 30 + caída > 0.4×ATR → SL/TP calculados por auto-learner (exit_mode=fixed)
    SHORT: RSI ≥ 70 + subida > 0.4×ATR → SL/TP calculados por auto-learner (exit_mode=fixed)
    
    Validado: mejor variante en simulación 1h × 30 días (SOL,ETH,XRP).
    BTC excluido (históricamente negativo en scalping).
    exit_mode="fixed" para cierre rápido sin trailing.
    Confidence mínima 0.68 — más exigente que swing (0.55).
    """
    if not risk_eval.get("approved"):
        return None

    # BTC no funciona bien para scalping (0/12 configs positivas en simulación)
    if symbol == "BTC":
        return None

    price   = ind.get("price", 0)
    rsi_val = ind.get("rsi", 50)
    atr_val = ind.get("atr", 0)
    atr_pct = ind.get("atr_pct", 0)
    fg      = ind.get("fear_greed", 50)
    change  = ind.get("change_5m", 0)
    bb      = ind.get("bb", {})
    obv_tr  = ind.get("obv_trend", "unknown")
    diverg  = ind.get("rsi_divergence", None)

    if price <= 0 or atr_pct < 0.008:  # ATR mínimo 0.8% del precio
        return None

    atr_thr = (atr_val / price * 100) * 0.4  # 0.4×ATR como mínimo de movimiento

    # ── SCALP LONG: RSI oversold + caída confirmada ─────────────────────────
    if rsi_val <= 30 and fg <= 65:
        drop = abs(min(change, 0))
        score = 0.55
        reasons = [f"RSI {rsi_val:.0f} oversold — scalp setup ✅"]

        if drop >= atr_thr:
            score += 0.15
            reasons.append(f"Drop {drop:.1f}% ≥ 0.4×ATR ({atr_thr:.1f}%) ✅")
        if obv_tr == "up":
            score += 0.08
            reasons.append("OBV rebote alcista ✅")
        if diverg == "bullish":
            score += 0.08
            reasons.append("Divergencia RSI bullish ✅")
        if bb.get("below_lower"):
            score += 0.07
            reasons.append("Precio bajo BB inferior ✅")
        if fg <= 20:
            score += 0.08
            reasons.append(f"Extreme Fear {fg} — rebote probable ✅")
        elif fg <= 35:
            score += 0.04

        if score >= 0.68 and len(reasons) >= 2:
            sig = build_signal(symbol, "long", "scalping", ind, min(score, 0.90), reasons, risk_eval)
            if sig:
                sig["exit_mode"] = "fixed"   # Scalp: sin trailing, cierre rápido
                sig["trailing_pct"] = 0.0
            return sig

    # ── SCALP SHORT: RSI overbought + subida confirmada ─────────────────────
    if rsi_val >= 70 and fg >= 20:  # No shortar en panic extremo (riesgo de rebote violento)
        rise = max(change, 0)
        score = 0.55
        reasons = [f"RSI {rsi_val:.0f} overbought — scalp short setup ✅"]

        if rise >= atr_thr:
            score += 0.15
            reasons.append(f"Subida {rise:.1f}% ≥ 0.4×ATR ({atr_thr:.1f}%) ✅")
        if obv_tr == "down":
            score += 0.08
            reasons.append("OBV distribución bajista ✅")
        if diverg == "bearish":
            score += 0.08
            reasons.append("Divergencia RSI bearish ✅")
        if bb.get("above_upper"):
            score += 0.07
            reasons.append("Precio sobre BB superior ✅")
        if fg >= 80:
            score += 0.08
            reasons.append(f"Extreme Greed {fg} — caída probable ✅")
        elif fg >= 65:
            score += 0.04

        if score >= 0.68 and len(reasons) >= 2:
            sig = build_signal(symbol, "short", "scalping", ind, min(score, 0.90), reasons, risk_eval)
            if sig:
                sig["exit_mode"] = "fixed"
                sig["trailing_pct"] = 0.0
            return sig

    return None

# ── Strategy Lists by Mode (sim validado 357,984 combos Kraken 5m-1d, 2026-04-12) ──
# PURE: golden+death cross primero (WR=73%, PF=5.42) + swing conservadoras
STRATEGIES_PURE  = [strategy_golden_cross, strategy_death_cross,
                    strategy_oversold_bounce, strategy_breakout,
                    strategy_trend_momentum]

# WILD: scalping agresivo bidireccional primero (stoch+rsi=doble extremo)
# stoch_rsi_scalp → rsi_bb_scalp → golden/death cross → macd → rsi+atr scalp
STRATEGIES_WILD  = [strategy_stoch_rsi_scalp, strategy_rsi_bb_scalp,
                    strategy_golden_cross, strategy_death_cross,
                    strategy_macd_cross, strategy_scalping]

# COMBO mantenido por compatibilidad (alias de WILD)
STRATEGIES_COMBO = STRATEGIES_WILD

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(debug: bool = False, wild_mode: bool = False) -> dict:
    log.info("=" * 50)
    log.info("🧠 STRATEGY AGENT — stack completo de indicadores")
    log.info("=" * 50)

    market = load_json(MARKET_FILE)
    risk   = load_json(RISK_FILE)

    if not market or not risk:
        log.error("❌ Faltan datos — abortando")
        return {}

    if risk.get("portfolio_status") in ("PAUSED", "STOPPED"):
        log.warning(f"⛔ Sistema {risk['portfolio_status']} — sin señales")
        result = {"timestamp": datetime.now(timezone.utc).isoformat(), "signals": [], "blocked": True}
        save_json(OUTPUT_FILE, result)
        return result

    tokens = market.get("tokens", {})
    evaluations = risk.get("evaluations", {})

    # Actualizar historial
    price_hist, vol_hist = update_price_history(tokens)
    # Actualizar velas 30m para ADX real
    try:
        import candle_aggregator as _ca
        _ca.update(price_hist)
    except Exception as _ca_e:
        log.debug(f"candle_aggregator: {_ca_e}")

    strategies = STRATEGIES_WILD if wild_mode else STRATEGIES_PURE
    mode_label = "WILD (stoch_rsi+rsi_bb scalping, 6 strats)" if wild_mode else "PURE (golden+death cross, swing, 5 strats)"
    log.info(f"   📊 Strategy mode: {mode_label}")

    signals = []
    strategy_counts = {s.__name__: 0 for s in strategies}
    indicator_summary = {}

    for symbol, token_data in tokens.items():
        risk_eval    = evaluations.get(symbol, {"approved": False})
        p_hist       = price_hist.get(symbol, [])
        v_hist       = vol_hist.get(symbol, [])

        # FILTRO 1: Volumen O Liquidez mínima — evita shitcoins ilíquidos
        # Usar liquidez como fallback cuando CoinGecko bloquea (vol=0)
        volume_24h = token_data.get("volume_24h", 0)
        liquidity = token_data.get("liquidity", 0)
        has_sufficient_liquidity = volume_24h >= MIN_VOLUME_24H or liquidity >= MIN_VOLUME_24H
        
        if not has_sufficient_liquidity:
            if debug:
                log.info(f"  ⏭️ {symbol}: Vol ${volume_24h/1e6:.2f}M + Liq ${liquidity/1e6:.2f}M < ${MIN_VOLUME_24H/1e6:.1f}M mínimo — SKIP")
            continue

        ind = compute_indicators(symbol, token_data, p_hist, v_hist)
        
        # Agregar Fear & Greed al diccionario de indicadores para trend following
        fg = market.get("fear_greed", {})
        ind["fear_greed"] = fg.get("value", 50) if isinstance(fg, dict) else 50
        
        # FILTRO 2: ATR máximo — evita alta volatilidad extrema
        atr_pct = ind.get("atr_pct", 0)
        if atr_pct and atr_pct > MAX_ATR_PCT * 100:
            if debug:
                log.info(f"  ⏭️ {symbol}: ATR {atr_pct:.2f}% > {MAX_ATR_PCT*100:.1f}% máximo — SKIP")
            continue
        indicator_summary[symbol] = {
            "n": ind["n_prices"],
            "rsi": ind.get("rsi"),
            "trend": ind.get("trend"),
            "atr_pct": ind.get("atr_pct"),
            "macd": ind.get("macd", {}).get("histogram") if ind.get("macd") else None,
        }

        if debug:
            log.info(f"  {symbol}: n={ind['n_prices']} RSI={ind.get('rsi','?')} "
                     f"trend={ind.get('trend','?')} ATR%={ind.get('atr_pct','?')}")

        for strat_fn in strategies:
            sig = strat_fn(symbol, ind, risk_eval)
            if sig and sig["confidence"] >= MIN_CONFIDENCE:
                signals.append(sig)
                strategy_counts[strat_fn.__name__] += 1
                break  # Una señal por token

    signals.sort(key=lambda x: x["confidence"], reverse=True)

    result = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "signals":           signals,
        "total_signals":     len(signals),
        "by_strategy":       strategy_counts,
        "indicator_summary": indicator_summary,
    }
    save_json(OUTPUT_FILE, result)

    if signals:
        log.info(f"🎯 {len(signals)} señales generadas:")
        for sig in signals:
            arrow = "🟢" if sig["direction"] == "long" else "🔴"
            n_ok  = sig.get("n_indicators", 0)
            log.info(f"  {arrow} {sig['symbol']:8s} [{sig['strategy']:18s}] "
                     f"conf={sig['confidence']:.2f} | {n_ok} indicadores ✅ | {sig['reasons'][0]}")
    else:
        log.info("📊 Sin señales en este ciclo")
        if debug:
            for sym, s in indicator_summary.items():
                log.info(f"  {sym}: n={s['n']} rsi={s['rsi']} trend={s['trend']}")

    by = {k.replace("strategy_",""):v for k,v in strategy_counts.items() if v > 0}
    log.info(f"📊 Estrategias activas: {by if by else 'ninguna'}")
    log.info(f"💾 Guardado en {OUTPUT_FILE}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(debug=args.debug)


# ── E5: Mean Reversion Detection ──────────────────────────────
def detect_mean_reversion(symbol: str, rsi_val: float, tokens_to_prefer: list) -> dict:
    """Detect extreme oversold for mean reversion long opportunity."""
    if rsi_val is None or rsi_val > 20:
        return None
    if symbol not in tokens_to_prefer:
        return None
    return {
        "symbol": symbol,
        "direction": "long",
        "strategy": "mean_reversion",
        "confidence": 0.75,
        "bypass_fg_block": True,
        "exit_mode": "fixed",
    }
