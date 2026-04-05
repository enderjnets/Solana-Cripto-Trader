#!/usr/bin/env python3
"""
🔗 Agente: Leverage Manager - Drift Protocol Integration
=========================================================
Gestiona posiciones con apalancamiento en Drift Protocol.

Características:
- Apalancamiento ajustable (1-10x)
- LONG y SHORT trading
- Colateral management
- Risk-aware leverage sizing
- Integración con Risk Manager

Uso:
    python3 leverage_manager.py          # Gestión paper
    python3 leverage_manager.py --live  # Modo real (requiere wallet)
"""

import os
import json
import logging
import requests
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
import random

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MARKET_FILE = DATA_DIR / "market_latest.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
LEVERAGE_STATE = DATA_DIR / "leverage_state.json"
OUTPUT_FILE = DATA_DIR / "leverage_signals.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("leverage_manager")

# ─── Parámetros de Leverage ────────────────────────────────────────────────

MAX_LEVERAGE = 10.0     # Drift permite hasta 10x
DEFAULT_LEVERAGE = 3.0   # Default leverage
MIN_LEVERAGE = 1.0     # Mínimo leverage
MAX_COLLATERAL_PCT = 0.5 # Máximo 50% del capital en colateral

# Leverage sizing basado en confianza
LEVERAGE_TIERS = {
    "CONSERVATIVE": (1.0, 2.0),   # 1-2x para baja confianza
    "MODERATE": (2.0, 5.0),        # 2-5x para confianza media
    "AGGRESSIVE": (5.0, 10.0),     # 5-10x para alta confianza
}

# Risk parameters para perps
SL_PERP_PCT = 0.03   # 3% SL (sin leverage)
TP_PERP_PCT = 0.05   # 5% TP (sin leverage)
# Con leverage 5x: 3% = 15% PnL, 5% = 25% PnL

# ─── Drift Token Addresses ───────────────────────────────────────────────────

DRIFT_TOKENS = {
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tFm7mkBZKaUUTwJWS3d",
}

# ─── Carga / Guardado de Estado ───────────────────────────────────────────────

def load_portfolio() -> dict:
    """Carga portfolio principal."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"capital_usd": 500.0, "initial_capital": 500.0, "positions": []}


def load_leverage_state() -> dict:
    """Carga estado de positions con leverage."""
    if LEVERAGE_STATE.exists():
        with open(LEVERAGE_STATE) as f:
            return json.load(f)
    return {
        "collateral_usd": 0.0,
        "positions": [],
        "total_exposure_usd": 0.0,
        "max_leverage": DEFAULT_LEVERAGE,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


def save_leverage_state(state: dict):
    """Guarda estado de positions con leverage."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(LEVERAGE_STATE, "w") as f:
        json.dump(state, f, indent=2)


def load_market() -> dict:
    """Carga datos de mercado."""
    if MARKET_FILE.exists():
        with open(MARKET_FILE) as f:
            return json.load(f)
    return {"tokens": {}}


# ─── Cálculos de Leverage ──────────────────────────────────────────────────

def calculate_leverage_size(
    capital_usd: float,
    leverage: float,
    price: float,
    confidence: float = 0.5
) -> dict:
    """
    Calcula tamaño de posición con leverage.

    Ejemplo con 5x leverage:
    - Capital: $500
    - Colateral: $50 (10%)
    - Leverage: 5x
    - Exposición: $250 ($50 * 5)
    - Si +5% → PnL: +$12.50 (25% del colateral)
    """
    # Tier de confianza
    if confidence < 0.6:
        min_lev, max_lev = LEVERAGE_TIERS["CONSERVATIVE"]
    elif confidence < 0.8:
        min_lev, max_lev = LEVERAGE_TIERS["MODERATE"]
    else:
        min_lev, max_lev = LEVERAGE_TIERS["AGGRESSIVE"]

    # Cap leverage según tier
    leverage = max(min_lev, min(max_lev, leverage))

    # Colateral (10% del capital por trade)
    collateral = capital_usd * 0.1

    # Exposición con leverage
    exposure_usd = collateral * leverage
    tokens = exposure_usd / price if price > 0 else 0

    return {
        "collateral_usd": round(collateral, 2),
        "leverage": round(leverage, 1),
        "exposure_usd": round(exposure_usd, 2),
        "tokens": round(tokens, 6),
        "tier": get_tier_name(confidence, leverage)
    }


def get_tier_name(confidence: float, leverage: float) -> str:
    """Determina el tier de risk."""
    if confidence < 0.6:
        return "CONSERVATIVE"
    elif confidence < 0.8:
        return "MODERATE"
    else:
        return "AGGRESSIVE"


def calculate_perp_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    leverage: float,
    collateral: float
) -> dict:
    """
    Calcula PnL para position de perp con leverage.

    Ejemplo:
    - LONG, entry $100, exit $105, leverage 5x, collateral $10
    - Precio subió 5%
    - Con leverage 5x: PnL = 5% * 5 = 25%
    - PnL USD = $10 * 0.25 = $2.50
    """
    if direction == "long":
        price_change_pct = (exit_price - entry_price) / entry_price
    else:  # short
        price_change_pct = (entry_price - exit_price) / entry_price

    # PnL con leverage
    pnl_pct = price_change_pct * leverage
    pnl_usd = collateral * pnl_pct

    return {
        "pnl_pct": round(pnl_pct * 100, 2),
        "pnl_usd": round(pnl_usd, 2),
        "price_change_pct": round(price_change_pct * 100, 2),
        "effective_leverage": round(leverage, 1)
    }


# ─── Gestión de Posiciones ─────────────────────────────────────────────────

def open_leverage_position(
    symbol: str,
    direction: str,
    leverage: float,
    confidence: float,
    portfolio: dict,
    market: dict
) -> Optional[dict]:
    """
    Abre posición con leverage (paper trading).
    """
    if symbol not in DRIFT_TOKENS:
        log.warning(f"⚠️ {symbol} no soportado en Drift")
        return None

    price = market.get("tokens", {}).get(symbol, {}).get("price", 0)
    if price <= 0:
        return None

    # Calcular tamaño
    sizing = calculate_leverage_size(portfolio["capital_usd"], leverage, price, confidence)

    # SL/TP ajustados para perps (menos estrictos que spot)
    if direction == "long":
        sl_price = price * (1 - SL_PERP_PCT)
        tp_price = price * (1 + TP_PERP_PCT)
    else:
        sl_price = price * (1 + SL_PERP_PCT)
        tp_price = price * (1 - TP_PERP_PCT)

    position = {
        "id": f"LEVERAGE_{symbol}_{int(datetime.now().timestamp())}",
        "symbol": symbol,
        "direction": direction,
        "strategy": "leverage_perp",
        "entry_price": price,
        "collateral_usd": sizing["collateral_usd"],
        "leverage": sizing["leverage"],
        "exposure_usd": sizing["exposure_usd"],
        "tokens": sizing["tokens"],
        "sl_price": round(sl_price, 8),
        "tp_price": round(tp_price, 8),
        "pnl_usd": 0.0,
        "pnl_pct": 0.0,
        "status": "open",
        "open_time": datetime.now(timezone.utc).isoformat(),
        "close_time": None,
        "close_reason": None,
        "mode": "paper",
        "tier": sizing["tier"],
        "confidence": confidence
    }

    return position


def update_leverage_positions(
    state: dict,
    market: dict,
    portfolio: dict
) -> List[dict]:
    """
    Actualiza PnL de positions con leverage y cierra SL/TP.
    """
    closed = []
    remaining = []

    for pos in state["positions"]:
        if pos.get("status") != "open":
            continue

        symbol = pos["symbol"]
        current_price = market.get("tokens", {}).get(symbol, {}).get("price", 0)

        if current_price <= 0:
            remaining.append(pos)
            continue

        # Calcular PnL
        pnl = calculate_perp_pnl(
            pos["direction"],
            pos["entry_price"],
            current_price,
            pos["leverage"],
            pos["collateral_usd"]
        )

        pos["pnl_usd"] = pnl["pnl_usd"]
        pos["pnl_pct"] = pnl["pnl_pct"]

        # Check SL/TP
        hit_sl = False
        hit_tp = False

        if pos["direction"] == "long":
            hit_sl = current_price <= pos["sl_price"]
            hit_tp = current_price >= pos["tp_price"]
        else:
            hit_sl = current_price >= pos["sl_price"]
            hit_tp = current_price <= pos["tp_price"]

        if hit_sl or hit_tp:
            close_reason = "TP" if hit_tp else "SL"
            pos["status"] = "closed"
            pos["close_time"] = datetime.now(timezone.utc).isoformat()
            pos["close_reason"] = close_reason
            pos["close_price"] = current_price

            # Devolver colateral + PnL
            portfolio["capital_usd"] += pos["collateral_usd"] + pos["pnl_usd"]

            closed.append(pos)
            log.info(f"  {'✅' if pos['pnl_usd'] > 0 else '❌'} [{close_reason}] {symbol} (leverage {pos['leverage']}x) | P&L: ${pos['pnl_usd']:+.2f} ({pos['pnl_pct']:+.1f}%)")
        else:
            remaining.append(pos)

    state["positions"] = remaining

    # Actualizar colateral y exposición
    state["collateral_usd"] = sum(p.get("collateral_usd", 0) for p in remaining)
    state["total_exposure_usd"] = sum(p.get("exposure_usd", 0) for p in remaining)

    return closed


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(debug: bool = False) -> dict:
    log.info("=" * 50)
    log.info("🔗 LEVERAGE MANAGER — Drift Protocol")
    log.info("=" * 50)

    market = load_market()
    portfolio = load_portfolio()
    state = load_leverage_state()

    # Actualizar posiciones
    closed = update_leverage_positions(state, market, portfolio)

    # Métricas
    total_collateral = state["collateral_usd"]
    total_exposure = state["total_exposure_usd"]
    effective_leverage = total_exposure / total_collateral if total_collateral > 0 else 0

    log.info(f"💰 Colateral: ${total_collateral:.2f}")
    log.info(f"🔗 Exposición: ${total_exposure:.2f}")
    log.info(f"⚡ Leverage efectivo: {effective_leverage:.1f}x")
    log.info(f"📊 Posiciones: {len(state['positions'])}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collateral_usd": round(total_collateral, 2),
        "exposure_usd": round(total_exposure, 2),
        "effective_leverage": round(effective_leverage, 2),
        "open_positions": len(state["positions"]),
        "closed_this_cycle": len(closed),
        "positions": state["positions"]
    }

    # Guardar estado
    save_leverage_state(state)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"💾 Guardado en {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leverage Manager - Drift Protocol")
    parser.add_argument("--debug", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run(debug=args.debug)
