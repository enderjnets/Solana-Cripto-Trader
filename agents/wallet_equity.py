"""On-chain wallet equity helper — shared between dashboard and risk_manager.

v2.12.30: extracted from dashboard/app.py v2.12.28 for reuse in
agents/risk_manager.py::calculate_drawdown, which was silently using only the
bot's tracked capital_usd (excluded SOL fuel + dust residues, reporting false
high drawdown that triggered spurious emergency_close events).

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
MINT_SOL_WRAPPED = "So11111111111111111111111111111111111111112"


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
        }
    except Exception:
        return {"SOL": 0.0, "JUP": 0.0, "ETH": 0.0}


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

    try:
        balances = {}

        # Native SOL
        r = requests.post(
            rpc,
            json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet]},
            timeout=5,
        ).json()
        balances["SOL"] = r["result"]["value"] / 1e9

        # SPL tokens
        for sym, mint in (("USDC", MINT_USDC), ("JUP", MINT_JUP), ("ETH", MINT_ETH)):
            r = requests.post(
                rpc,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "getTokenAccountsByOwner",
                    "params": [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
                },
                timeout=5,
            ).json()
            total = sum(
                float(a["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"] or 0)
                for a in r.get("result", {}).get("value", [])
            )
            balances[sym] = total

        # Prices — Jupiter primary
        sol_px = jup_px = eth_px = 0.0
        try:
            ids = f"{MINT_SOL_WRAPPED},{MINT_JUP},{MINT_ETH}"
            px_r = requests.get(f"https://price.jup.ag/v6/price?ids={ids}", timeout=5).json()
            px_data = px_r.get("data", {})
            sol_px = float(px_data.get(MINT_SOL_WRAPPED, {}).get("price", 0) or 0)
            jup_px = float(px_data.get(MINT_JUP, {}).get("price", 0) or 0)
            eth_px = float(px_data.get(MINT_ETH, {}).get("price", 0) or 0)
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

        prices = {"SOL": sol_px, "JUP": jup_px, "ETH": eth_px}

        wallet_total = (
            balances["USDC"]
            + balances["SOL"] * sol_px
            + balances["JUP"] * jup_px
            + balances["ETH"] * eth_px
        )

        result = {
            "wallet_total": round(wallet_total, 4),
            "balances": {k: round(v, 6) for k, v in balances.items()},
            "prices": {k: round(v, 6) for k, v in prices.items()},
            "ts": now,
        }
        _CACHE["ts"] = now
        _CACHE["data"] = result
        return result
    except Exception as e:
        log.warning(f"fetch_wallet_equity failed: {e}")
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
