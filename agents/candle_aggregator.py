"""
30m OHLCV Aggregator — agrega el historial de precios en velas de 30 minutos.
Habilita:
  - Stochastic K con true high/low (más preciso)
  - ADX real (Average Directional Index)
  - Análisis multi-timeframe (30m vs ciclo de 60s)

Uso:
    import candle_aggregator as ca
    ca.update(price_history_dict)   # llamar al inicio de cada ciclo
    candles = ca.get("SOL", n=50)   # obtener últimas N velas de un símbolo
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("candle_aggregator")

DATA_DIR = Path(__file__).parent / "data"
OHLCV_DIR = DATA_DIR / "ohlcv_30m"
OHLCV_DIR.mkdir(parents=True, exist_ok=True)

CANDLE_SECONDS = 1800   # 30 minutos
MAX_CANDLES    = 100    # ~50 horas de historia


def _candle_bucket(ts: float) -> int:
    """Número de vela (floor division de timestamp / 1800)."""
    return int(ts) // CANDLE_SECONDS


def _candle_path(symbol: str) -> Path:
    return OHLCV_DIR / f"{symbol.upper()}.json"


def _load_candles(symbol: str) -> list:
    p = _candle_path(symbol)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_candles(symbol: str, candles: list):
    try:
        _candle_path(symbol).write_text(json.dumps(candles[-MAX_CANDLES:]))
    except Exception as e:
        log.warning(f"ohlcv_30m write error {symbol}: {e}")


def update(price_history: dict):
    """
    Agrega entradas del historial de precios en velas 30m.

    Args:
        price_history: dict {symbol: [{"price": float, "ts": float}, ...]}
                       (contenido de price_history.json)
    """
    for symbol, entries in price_history.items():
        if not entries:
            continue
        candles = _load_candles(symbol)
        existing_buckets = {c["bucket"] for c in candles}

        # Agrupar entradas por bucket de 30m
        buckets: dict = {}
        for e in entries:
            ts = e.get("ts") or e.get("timestamp", time.time())
            price = e.get("price", 0)
            vol   = e.get("volume", 0)
            if price <= 0:
                continue
            b = _candle_bucket(float(ts))
            if b not in buckets:
                buckets[b] = {"prices": [], "volumes": [], "ts": float(ts)}
            buckets[b]["prices"].append(price)
            buckets[b]["volumes"].append(vol)

        # Construir velas completas (todas excepto la vela en curso)
        now_bucket = _candle_bucket(time.time())
        new_candles = []
        for b, data in sorted(buckets.items()):
            if b == now_bucket:
                continue  # vela en curso — no finalizarla
            if b in existing_buckets:
                continue  # ya existe
            prices = data["prices"]
            new_candles.append({
                "bucket": b,
                "ts":     b * CANDLE_SECONDS,
                "open":   prices[0],
                "high":   max(prices),
                "low":    min(prices),
                "close":  prices[-1],
                "volume": sum(data["volumes"]),
                "n":      len(prices),
            })

        if new_candles:
            candles.extend(new_candles)
            candles.sort(key=lambda c: c["bucket"])
            _save_candles(symbol, candles)
            log.debug(f"ohlcv_30m {symbol}: +{len(new_candles)} velas (total {len(candles)})")


def get(symbol: str, n: int = 50) -> list:
    """Retorna las últimas N velas 30m de un símbolo (más antiguas primero)."""
    candles = _load_candles(symbol)
    return candles[-n:] if candles else []


def compute_adx(candles: list, period: int = 14) -> Optional[float]:
    """
    Average Directional Index (ADX) usando true high/low de las velas 30m.
    < 20  → mercado sin tendencia (scalping tiene ventaja)
    20-25 → tendencia emergente
    > 25  → tendencia establecida (scalping peligroso)
    > 40  → tendencia muy fuerte
    Retorna None si no hay suficientes datos.
    """
    if len(candles) < period + 2:
        return None

    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    closes = [c["close"] for c in candles]

    # True Range y Directional Movement
    tr_list, pdm_list, ndm_list = [], [], []
    for i in range(1, len(candles)):
        h, l, c   = highs[i],  lows[i],  closes[i]
        ph, pl, pc = highs[i-1], lows[i-1], closes[i-1]

        tr  = max(h - l, abs(h - pc), abs(l - pc))
        pdm = max(h - ph, 0) if (h - ph) > (pl - l) else 0
        ndm = max(pl - l, 0) if (pl - l) > (h - ph) else 0

        tr_list.append(tr)
        pdm_list.append(pdm)
        ndm_list.append(ndm)

    if len(tr_list) < period:
        return None

    # Wilder smoothing (EMA con alpha=1/period)
    def wilder_smooth(data, p):
        result = [sum(data[:p]) / p]
        for v in data[p:]:
            result.append(result[-1] * (p - 1) / p + v)
        return result

    atr_s  = wilder_smooth(tr_list,  period)
    pdm_s  = wilder_smooth(pdm_list, period)
    ndm_s  = wilder_smooth(ndm_list, period)

    dx_list = []
    for i in range(len(atr_s)):
        if atr_s[i] == 0:
            continue
        pdi = 100 * pdm_s[i] / atr_s[i]
        ndi = 100 * ndm_s[i] / atr_s[i]
        denom = pdi + ndi
        if denom == 0:
            continue
        dx_list.append(100 * abs(pdi - ndi) / denom)

    if len(dx_list) < period:
        return None

    adx_s = wilder_smooth(dx_list, period)
    return round(adx_s[-1], 2) if adx_s else None


def get_adx(symbol: str, period: int = 14) -> Optional[float]:
    """Calcula ADX para un símbolo usando sus velas 30m almacenadas."""
    candles = get(symbol, n=period * 3)
    return compute_adx(candles, period)
