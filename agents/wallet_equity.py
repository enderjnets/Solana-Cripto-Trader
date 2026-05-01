"""On-chain wallet equity helper — shared between dashboard and risk_manager.

v2.12.30: extracted from dashboard/app.py v2.12.28 for reuse in
agents/risk_manager.py::calculate_drawdown.

v2.13.1: RPC error handling per-token with retry + cache fallback. Prevents
false 0 balances when public RPC rate-limits (429) on getTokenAccountsByOwner.

v2.13.2: Persistent disk cache (wallet_equity_cache.json) so fallback survives
process restarts. Prevents dashboard from showing $12 equity after reboot when
RPC is rate-limited.

Returns wallet_total = USDC + SOL * SOL_price + JUP * JUP_price + ETH * ETH_price
from on-chain RPC query, cached 60s. Jupiter Prices API is primary source;
market_latest.json is fallback when Jupiter DNS flaps (observed in production).

Safe no-op for paper mode: returns None when HOT_WALLET_ADDRESS is unset or
MOCK_NO_PUBKEY, letting callers fall back to their legacy bot-equity calc.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

# agents/ lives next to data/ — cross-boundary path
DATA = Path(__file__).resolve().parent / "data"

CACHE_TTL_SEC = 60
_CACHE = {"ts": 0.0, "data": None}

MINT_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
MINT_JUP = "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"
MINT_ETH = "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"
MINT_BTC = "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh"
MINT_SOL_WRAPPED = "So11111111111111111111111111111111111111112"

_DISK_CACHE_PATH = DATA / "wallet_equity_cache.json"


def _load_disk_cache():
    """Load last known good equity from disk (survives process restarts)."""
    try:
        if _DISK_CACHE_PATH.exists():
            with open(_DISK_CACHE_PATH) as f:
                return json.load(f)
    except Exception as e:
        log.debug(f"disk cache load failed: {e}")
    return None


def _save_disk_cache(data: dict):
    """Persist successful equity snapshot to disk."""
    try:
        _DISK_CACHE_PATH.write_text(json.dumps(data))
    except Exception as e:
        log.debug(f"disk cache save failed: {e}")


def _load_market_latest_prices():
    """Fallback prices from market_latest.json (bot updates it each cycle)."""
    try:
        raw = (DATA / "market_latest.json").read_text()
        data = json.loads(raw)
        tokens = data.get("tokens", {}) if isinstance(data, dict) else {}
        return {
            "SOL": float(tokens.get("SOL", {}).get("price", 0) or 0),
            "JUP": float(tokens.get("JUP", {}).get("price", 0) or 0),
            "ETH": float(tokens.get("ETH", {}).get("price", 0) or 0),
            "BTC": float(tokens.get("BTC", {}).get("price", 0) or 0),
        }
    except Exception:
        return {"SOL": 0.0, "JUP": 0.0, "ETH": 0.0, "BTC": 0.0}


def _fetch_token_balance(wallet: str, mint: str, rpc: str, timeout: int = 5, retries: int = 2) -> float:
    """Query SPL token balance with retry and explicit error checking.

    Raises RuntimeError on RPC failure so callers can fallback to cache.
    """
    import requests

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "getTokenAccountsByOwner",
        "params": [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(rpc, json=payload, timeout=timeout).json()
            if "error" in r:
                raise RuntimeError(f"RPC error: {r['error']}")
            total = sum(
                float(a["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"] or 0)
                for a in r.get("result", {}).get("value", [])
            )
            return total
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1)
                continue
    raise RuntimeError(f"getTokenAccountsByOwner failed after {retries + 1} attempts: {last_err}")


def fetch_wallet_equity(force_refresh: bool = False):
    """Query on-chain wallet balances × market prices.

    Returns dict:
        {
          "wallet_total": float,     # total USD equity
          "balances": {USDC, SOL, JUP, ETH},
          "prices":   {SOL, JUP, ETH},
          "ts":       float,
        }
    or None if unavailable (paper mode, HOT_WALLET_ADDRESS missing, RPC failure).
    Cached 60s to avoid RPC spam.
    """
    try:
        import requests
    except ImportError:
        return None

    now = time.time()
    if (
        not force_refresh
        and _CACHE["data"] is not None
        and (now - _CACHE["ts"] < CACHE_TTL_SEC)
    ):
        return _CACHE["data"]

    wallet = os.environ.get("HOT_WALLET_ADDRESS", "").strip()
    if not wallet or wallet == "MOCK_NO_PUBKEY":
        return None

    rpc = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

    # Load disk cache for fallback across restarts
    disk_cache = _load_disk_cache()

    try:
        balances = {}
        rpc_errors = 0
        rpc_total = 0

        # Native SOL
        r = requests.post(
            rpc,
            json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet]},
            timeout=5,
        ).json()
        if "error" in r:
            log.warning(f"SOL balance RPC error: {r['error']}")
            rpc_errors += 1
            balances["SOL"] = _CACHE["data"]["balances"].get("SOL", 0) if _CACHE["data"] else (disk_cache["balances"].get("SOL", 0) if disk_cache else 0)
        else:
            balances["SOL"] = r["result"]["value"] / 1e9

        # SPL tokens — each with independent error handling + cache fallback
        for sym, mint in (("USDC", MINT_USDC), ("JUP", MINT_JUP), ("ETH", MINT_ETH), ("BTC", MINT_BTC)):
            rpc_total += 1
            try:
                balances[sym] = _fetch_token_balance(wallet, mint, rpc)
            except Exception as e:
                log.warning(f"Token balance fetch failed for {sym}: {e}")
                rpc_errors += 1
                # fallback chain: memory cache -> disk cache -> 0
                if _CACHE["data"] and (now - _CACHE["ts"] < 300):
                    cached_val = _CACHE["data"]["balances"].get(sym, 0)
                    log.info(f"Using memory cache for {sym}: {cached_val}")
                    balances[sym] = cached_val
                elif disk_cache and disk_cache.get("balances"):
                    cached_val = disk_cache["balances"].get(sym, 0)
                    log.info(f"Using disk cache for {sym}: {cached_val}")
                    balances[sym] = cached_val
                else:
                    balances[sym] = 0

        # Prices — Jupiter primary
        sol_px = jup_px = eth_px = btc_px = 0.0
        try:
            ids = f"{MINT_SOL_WRAPPED},{MINT_JUP},{MINT_ETH},{MINT_BTC}"
            px_r = requests.get(f"https://price.jup.ag/v6/price?ids={ids}", timeout=5).json()
            px_data = px_r.get("data", {})
            sol_px = float(px_data.get(MINT_SOL_WRAPPED, {}).get("price", 0) or 0)
            jup_px = float(px_data.get(MINT_JUP, {}).get("price", 0) or 0)
            eth_px = float(px_data.get(MINT_ETH, {}).get("price", 0) or 0)
            btc_px = float(px_data.get(MINT_BTC, {}).get("price", 0) or 0)
        except Exception:
            pass

        # Fallback: market_latest.json (bot cache) when Jupiter price == 0
        if sol_px == 0 or jup_px == 0 or eth_px == 0:
            fb = _load_market_latest_prices()
            if sol_px == 0:
                sol_px = fb["SOL"]
            if jup_px == 0:
                jup_px = fb["JUP"]
            if eth_px == 0:
                eth_px = fb["ETH"]
            if btc_px == 0:
                btc_px = fb["BTC"]

        prices = {"SOL": sol_px, "JUP": jup_px, "ETH": eth_px, "BTC": btc_px}

        wallet_total = (
            balances["USDC"]
            + balances["SOL"] * sol_px
            + balances["JUP"] * jup_px
            + balances["ETH"] * eth_px
            + balances["BTC"] * btc_px
        )

        # Heuristic: if >50% of token queries failed AND wallet_total looks implausibly low
        # compared to disk cache, return disk cache instead to avoid false panic.
        if rpc_errors > rpc_total // 2 and disk_cache:
            disk_total = disk_cache.get("wallet_total", 0)
            if disk_total > 0 and wallet_total < disk_total * 0.5:
                log.warning(f" wallet_total ${wallet_total:.2f} suspiciously low vs disk cache ${disk_total:.2f} after {rpc_errors}/{rpc_total} RPC errors — using disk cache")
                return disk_cache

        # Jupiter Perps value (collateral locked in contract not in wallet balances)
        perps_value = 0.0
        try:
            if os.environ.get("JUP_PERP_ENABLED", "").lower() == "true":
                try:
                    import jupiter_perp_adapter as _jpa
                except ImportError:
                    from agents import jupiter_perp_adapter as _jpa
                _snapshot = _jpa.get_account_snapshot()
                if _snapshot and _snapshot.get("positions"):
                    perps_value = float(_snapshot.get("total_size_usd", 0.0))
        except Exception:
            pass

        total_equity = wallet_total + perps_value

        result = {
            "wallet_total": round(wallet_total, 4),
            "total_equity": round(total_equity, 4),
            "jupiter_perps_value": round(perps_value, 4),
            "balances": {k: round(v, 6) for k, v in balances.items()},
            "prices": {k: round(v, 6) for k, v in prices.items()},
            "ts": now,
        }
        _CACHE["ts"] = now
        _CACHE["data"] = result
        _save_disk_cache(result)
        return result
    except Exception as e:
        log.warning(f"fetch_wallet_equity failed: {e}")
        if _CACHE["data"] is not None:
            log.info("Returning memory stale cache due to RPC failure")
            return _CACHE["data"]
        if disk_cache is not None:
            log.info("Returning disk cache due to RPC failure")
            return disk_cache
        return None


if __name__ == "__main__":
    # CLI self-test
    import sys
    # Load .env if running from CLI
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    data = fetch_wallet_equity(force_refresh=True)
    if data is None:
        print("wallet equity unavailable (paper mode or RPC down)")
        sys.exit(1)
    print(json.dumps(data, indent=2))
