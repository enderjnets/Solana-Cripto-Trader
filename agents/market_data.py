#!/usr/bin/env python3
"""
🌐 Agente 1: Market Data
Obtiene precios reales de tokens Solana via Jupiter Price API v2.
Detecta momentum, calcula cambios 24h y guarda market_latest.json.

Uso:
    python3 market_data.py          # Ejecuta y guarda datos
    python3 market_data.py --debug  # Con output detallado
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "market_latest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("market_data")

# FIX 3.2: Rate limiting para APIs
import threading as _threading

class _RateLimiter:
    def __init__(self, calls_per_minute=30):
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self._lock = _threading.Lock()
    def wait(self):
        import time as _t
        with self._lock:
            now = _t.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                _t.sleep(self.min_interval - elapsed)
            self.last_call = _t.time()

_jupiter_limiter = _RateLimiter(calls_per_minute=30)
_coingecko_limiter = _RateLimiter(calls_per_minute=10)


# Tokens a monitorear — mint addresses en Solana
TOKEN_MINTS = {
    "SOL":      "So11111111111111111111111111111111111111112",
    "BTC":      "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh",   # Wrapped BTC (Wormhole)
    "ETH":      "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",   # Wrapped ETH (Wormhole)
    "JUP":      "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "BONK":     "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "RAY":      "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "PENGU":    "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv",
    "FARTCOIN": "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump",
    "MOODENG":  "ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY",
    "GOAT":     "CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump",
    "WIF":      "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
    "POPCAT":   "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
}

JUPITER_PRICE_URL = "https://lite-api.jup.ag/price/v3"
COINGECKO_FEAR_GREED = "https://api.alternative.me/fng/?limit=1"

# Timeout para requests
REQUEST_TIMEOUT = 15

# ─── Funciones de API ─────────────────────────────────────────────────────────

def fetch_jupiter_prices(mints: dict) -> dict:
    """Obtiene precios de Jupiter Price API v3 (gratis, sin API key).
    
    Respuesta v3: { "<mint>": { "usdPrice": float, "priceChange24h": float, "liquidity": float } }
    """
    mint_ids = ",".join(mints.values())
    try:
        resp = requests.get(
            JUPITER_PRICE_URL,
            params={"ids": mint_ids},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        # v3 returns the data directly (not wrapped in "data" key)
        return resp.json()
    except requests.RequestException as e:
        log.error(f"❌ Jupiter API error: {e}")
        return {}


def fetch_coingecko_fear_greed() -> dict:
    """Obtiene Fear & Greed Index de CoinGecko/alternative.me (gratis)."""
    try:
        _coingecko_limiter.wait()  # FIX 3.2
        resp = requests.get(COINGECKO_FEAR_GREED, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        fng = data.get("data", [{}])[0]
        return {
            "value": int(fng.get("value", 50)),
            "label": fng.get("value_classification", "Neutral"),
            "timestamp": fng.get("timestamp", "")
        }
    except Exception as e:
        log.warning(f"⚠️  Fear/Greed API error: {e}")
        return {"value": 50, "label": "Neutral", "timestamp": ""}


def load_previous_prices() -> dict:
    """Carga precios anteriores para calcular cambio 5min."""
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE) as f:
                old = json.load(f)
            return {
                t: d["price"]
                for t, d in old.get("tokens", {}).items()
                if d.get("price")
            }
        except Exception:
            pass
    return {}


# ─── Lógica Principal ─────────────────────────────────────────────────────────

def get_active_token_mints() -> dict:
    """
    Retorna los tokens activos: core + dinámicos del scanner.
    """
    active_mints = dict(TOKEN_MINTS)  # Empezar con tokens core
    
    # Agregar tokens dinámicos del scanner
    tracked_file = DATA_DIR / "tracked_tokens.json"
    if tracked_file.exists():
        try:
            tracked = json.loads(tracked_file.read_text())
            for symbol, data in tracked.get("tokens", {}).items():
                if symbol not in active_mints and data.get("address"):
                    active_mints[symbol] = data["address"]
        except Exception as e:
            log.warning(f"Error cargando tokens dinámicos: {e}")
    
    return active_mints


def build_market_data(debug: bool = False) -> dict:
    """Construye el snapshot completo del mercado."""
    # Usar tokens dinámicos (core + descubiertos)
    active_mints = get_active_token_mints()
    log.info(f"🌐 Obteniendo precios de Jupiter... ({len(active_mints)} tokens)")

    raw_prices = fetch_jupiter_prices(active_mints)
    prev_prices = load_previous_prices()
    fear_greed = fetch_coingecko_fear_greed()

    tokens = {}
    momentum_tokens = []
    now = datetime.now(timezone.utc).isoformat()

    for symbol, mint in active_mints.items():
        # Jupiter v3: keyed by mint address directly
        price_data = raw_prices.get(mint, {})
        price = float(price_data.get("usdPrice", 0)) if price_data else 0
        change_24h_jup = float(price_data.get("priceChange24h", 0)) if price_data else 0
        liquidity = float(price_data.get("liquidity", 0)) if price_data else 0

        if price == 0:
            log.warning(f"⚠️  Sin precio para {symbol}")
            tokens[symbol] = {
                "symbol": symbol,
                "mint": mint,
                "price": 0,
                "price_24h_change_pct": 0,
                "price_5min_change_pct": 0,
                "volume_24h": 0,
                "liquidity": 0,
                "momentum": False,
                "error": "no_price",
            }
            continue

        # Cambio en 5min (vs snapshot anterior)
        prev = prev_prices.get(symbol, price)
        change_5min = ((price - prev) / prev * 100) if prev > 0 else 0

        # Momentum: >3% cambio en 5min
        has_momentum = abs(change_5min) >= 3.0

        token_info = {
            "symbol": symbol,
            "mint": mint,
            "price": round(price, 8),
            "price_5min_change_pct": round(change_5min, 4),
            # Jupiter v3 gives priceChange24h directly!
            "price_24h_change_pct": round(change_24h_jup, 4),
            "liquidity": round(liquidity, 2),
            "volume_24h": 0,   # Will be enriched by CoinGecko if available
            "momentum": has_momentum,
            "last_updated": now,
        }
        tokens[symbol] = token_info

        if has_momentum:
            momentum_tokens.append({
                "symbol": symbol,
                "change_5min": round(change_5min, 2),
                "direction": "up" if change_5min > 0 else "down",
                "price": round(price, 8),
            })

        if debug:
            arrow = "🟢" if change_5min >= 0 else "🔴"
            log.info(f"  {arrow} {symbol:10s} ${price:>14.6f}  5min: {change_5min:+.2f}%")

    # Enriquecer con CoinGecko 24h data para tokens principales
    cg_prices = fetch_coingecko_24h()
    for symbol, cg in cg_prices.items():
        if symbol in tokens:
            tokens[symbol]["price_24h_change_pct"] = cg.get("change_24h", 0)
            tokens[symbol]["volume_24h"] = cg.get("volume_24h", 0)
            tokens[symbol]["market_cap"] = cg.get("market_cap", 0)

    result = {
        "timestamp": now,
        "cycle": int(time.time()),
        "tokens": tokens,
        "momentum_alerts": momentum_tokens,
        "fear_greed": fear_greed,
        "total_tokens": len(tokens),
        "prices_ok": sum(1 for t in tokens.values() if t["price"] > 0),
    }

    return result


def fetch_coingecko_24h() -> dict:
    """
    Obtiene cambio 24h, volumen y market cap de CoinGecko (gratis).
    Solo para tokens principales (SOL, BTC, ETH, etc.)
    """
    CG_IDS = {
        "SOL":      "solana",
        "BTC":      "bitcoin",
        "ETH":      "ethereum",
        "JUP":      "jupiter-exchange-solana",
        "BONK":     "bonk",
        "RAY":      "raydium",
        "WIF":      "dogwifcoin",
        "POPCAT":   "popcat",
        "GOAT":     "goatseus-maximus",
        "MOODENG":  "moo-deng",
        "FARTCOIN": "fartcoin",
        "PENGU":    "pudgy-penguins",
    }
    ids_str = ",".join(CG_IDS.values())
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        _coingecko_limiter.wait()  # FIX 3.2
        resp = requests.get(url, params={
            "vs_currency": "usd",
            "ids": ids_str,
            "order": "market_cap_desc",
            "per_page": 20,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # Invertir mapa CG_IDS para lookup
        cg_to_symbol = {v: k for k, v in CG_IDS.items()}
        result = {}
        for coin in data:
            symbol = cg_to_symbol.get(coin["id"])
            if symbol:
                result[symbol] = {
                    "change_24h": round(coin.get("price_change_percentage_24h") or 0, 2),
                    "volume_24h": coin.get("total_volume", 0),
                    "market_cap": coin.get("market_cap", 0),
                }
        return result
    except Exception as e:
        log.warning(f"⚠️  CoinGecko 24h error: {e}")
        return {}


# ─── Multi-Timeframe (Hourly Trend) ──────────────────────────────────────────

HOURLY_TOKENS = {"SOL": "solana", "BTC": "bitcoin", "ETH": "ethereum", "JUP": "jupiter-exchange-solana"}

def fetch_hourly_trends() -> dict:
    """Fetch 24h hourly candle data from CoinGecko for trend analysis."""
    trends = {}
    for symbol, cg_id in HOURLY_TOKENS.items():
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
            _coingecko_limiter.wait()  # FIX 3.2
            resp = requests.get(url, params={"vs_currency": "usd", "days": "1"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            prices = data.get("prices", [])
            if len(prices) >= 2:
                first_price = prices[0][1]
                last_price = prices[-1][1]
                change_pct = ((last_price - first_price) / first_price) * 100
                if change_pct > 1.5:
                    trend = "up"
                elif change_pct < -1.5:
                    trend = "down"
                else:
                    trend = "sideways"
                trends[symbol] = {
                    "price_1h_trend": trend,
                    "trend_change_pct": round(change_pct, 2),
                    "hourly_high": round(max(p[1] for p in prices), 4),
                    "hourly_low": round(min(p[1] for p in prices), 4),
                }
            # m3: removed - rate limiter handles timing
        # time.sleep(1.5)  # Rate limit CoinGecko free tier
        except Exception as e:
            log.warning(f"⚠️ Hourly trend failed for {symbol}: {e}")
            trends[symbol] = {"price_1h_trend": "unknown", "trend_change_pct": 0}
    return trends


def run(debug: bool = False) -> dict:
    """Entry point principal."""
    log.info("=" * 50)
    log.info("🌐 MARKET DATA AGENT — iniciando")
    log.info("=" * 50)

    t0 = time.time()
    market = build_market_data(debug=debug)

    # Add hourly trends for top tokens
    try:
        hourly = fetch_hourly_trends()
        for symbol, trend_data in hourly.items():
            if symbol in market.get("tokens", {}):
                market["tokens"][symbol].update(trend_data)
        trend_strs = [f"{s}={d.get('price_1h_trend', '?')}" for s, d in hourly.items()]
        log.info(f"📊 Hourly trends: {', '.join(trend_strs)}")
    except Exception as e:
        log.warning(f"⚠️ Hourly trends failed: {e}")

    elapsed = time.time() - t0

    # Guardar
    with open(OUTPUT_FILE, "w") as f:
        json.dump(market, f, indent=2)

    prices_ok = market["prices_ok"]
    total = market["total_tokens"]
    fg = market["fear_greed"]
    momentum = market["momentum_alerts"]

    log.info(f"✅ Precios obtenidos: {prices_ok}/{total} tokens")
    log.info(f"😨 Fear & Greed: {fg['value']} — {fg['label']}")

    if momentum:
        log.info(f"🚀 Momentum detectado ({len(momentum)} tokens):")
        for m in momentum:
            arrow = "⬆️" if m["direction"] == "up" else "⬇️"
            log.info(f"   {arrow} {m['symbol']} {m['change_5min']:+.2f}% @ ${m['price']}")
    else:
        log.info("📊 Sin momentum > 3% en este ciclo")

    log.info(f"💾 Guardado en {OUTPUT_FILE} ({elapsed:.1f}s)")
    return market


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Data Agent")
    parser.add_argument("--debug", action="store_true", help="Output detallado por token")
    args = parser.parse_args()

    result = run(debug=args.debug)
    sys.exit(0)
