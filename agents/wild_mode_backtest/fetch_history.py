"""
fetch_history.py — Descarga OHLCV histórico para los tokens principales.

Fuentes (en orden):
1. CoinGecko /coins/{id}/market_chart  (público, rate-limited)
2. Binance klines (para SOL/BTC/ETH)
3. Birdeye (para tokens Solana memes — necesita API key)

Output:
  agents/data/history/<TOKEN>_1h.json   list[(ts, open, high, low, close, vol)]
  agents/data/history/fear_greed_120d.json
"""
from __future__ import annotations
import json
import time
import logging
from pathlib import Path
import urllib.request
import urllib.error

log = logging.getLogger('fetch_history')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'data' / 'history'
DATA_DIR.mkdir(parents=True, exist_ok=True)

DAYS = 90  # 90 days of hourly candles

# CoinGecko IDs for the relevant tokens
COINGECKO_IDS = {
    'SOL': 'solana',
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'JUP': 'jupiter-exchange-solana',
    'BONK': 'bonk',
    'RAY': 'raydium',
    'WIF': 'dogwifcoin',
    'PENGU': 'pudgy-penguins',
    'POPCAT': 'popcat',
    'FARTCOIN': 'fartcoin',
    'MOODENG': 'moo-deng',
}

# Binance symbols (USDT pairs) for the majors
BINANCE_SYMBOLS = {
    'SOL': 'SOLUSDT',
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
}


def http_get(url: str, timeout: int = 30) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'wild-mode-backtest/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        log.warning(f"HTTPError {e.code} for {url}")
        return None
    except Exception as e:
        log.warning(f"Error {type(e).__name__} for {url}: {e}")
        return None


def fetch_binance_klines(symbol: str, interval: str = '1h', days: int = 90) -> list | None:
    """Returns list of [open_time, open, high, low, close, vol]."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000
    all_candles = []
    cur = start_ms
    while cur < end_ms:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={cur}&limit=1000"
        data = http_get(url)
        if not data:
            break
        if not isinstance(data, list) or len(data) == 0:
            break
        for k in data:
            all_candles.append([
                int(k[0]),       # open time ms
                float(k[1]),     # open
                float(k[2]),     # high
                float(k[3]),     # low
                float(k[4]),     # close
                float(k[5]),     # volume
            ])
        last_ts = int(data[-1][0])
        if last_ts <= cur:
            break
        cur = last_ts + 3600 * 1000
        time.sleep(0.1)
    return all_candles if all_candles else None


def fetch_coingecko(token: str, cg_id: str, days: int = 90) -> list | None:
    """Coingecko market_chart returns hourly data for 1-90d range."""
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={days}&interval=hourly"
    data = http_get(url)
    if not data:
        return None
    prices = data.get('prices', [])
    volumes = data.get('total_volumes', [])
    if not prices:
        return None
    candles = []
    vol_map = {int(v[0]): float(v[1]) for v in volumes}
    for p in prices:
        ts = int(p[0])
        price = float(p[1])
        # Coingecko hourly already gives close-only; synthesize OHLC
        candles.append([ts, price, price, price, price, vol_map.get(ts, 0.0)])
    return candles


def fetch_token(token: str) -> bool:
    out_file = DATA_DIR / f'{token}_1h.json'
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text())
            if len(existing) >= 24 * 60:  # at least 60d
                log.info(f"  {token}: already cached ({len(existing)} candles), skip")
                return True
        except Exception:
            pass

    # Try Binance first for majors
    candles = None
    if token in BINANCE_SYMBOLS:
        log.info(f"  {token}: fetching Binance {BINANCE_SYMBOLS[token]}...")
        candles = fetch_binance_klines(BINANCE_SYMBOLS[token], '1h', DAYS)

    # Fallback to Coingecko
    if not candles and token in COINGECKO_IDS:
        log.info(f"  {token}: fetching CoinGecko {COINGECKO_IDS[token]}...")
        candles = fetch_coingecko(token, COINGECKO_IDS[token], DAYS)
        time.sleep(2)  # rate limit

    if not candles:
        log.warning(f"  {token}: NO data available")
        return False

    out_file.write_text(json.dumps(candles))
    log.info(f"  {token}: saved {len(candles)} candles -> {out_file.name}")
    return True


def fetch_fear_greed(days: int = 120) -> bool:
    out = DATA_DIR / 'fear_greed_120d.json'
    if out.exists():
        try:
            existing = json.loads(out.read_text())
            if len(existing) >= 60:
                log.info(f"FG: cached ({len(existing)} entries), skip")
                return True
        except Exception:
            pass
    url = f"https://api.alternative.me/fng/?limit={days}"
    data = http_get(url)
    if not data or 'data' not in data:
        log.warning("FG fetch failed")
        return False
    entries = []
    for d in data['data']:
        try:
            entries.append({
                'timestamp': int(d['timestamp']),
                'value': int(d['value']),
                'classification': d.get('value_classification', 'unknown'),
            })
        except Exception:
            continue
    out.write_text(json.dumps(entries))
    log.info(f"FG: saved {len(entries)} entries")
    return True


def main():
    log.info(f"Fetching {DAYS}d hourly history for {len(COINGECKO_IDS)} tokens...")
    ok = 0
    for tok in COINGECKO_IDS.keys():
        if fetch_token(tok):
            ok += 1
    log.info(f"Tokens fetched: {ok}/{len(COINGECKO_IDS)}")
    fetch_fear_greed()
    log.info("Done.")


if __name__ == '__main__':
    main()
