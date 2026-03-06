#!/usr/bin/env python3
"""
🛡️ Agente 2: Risk Manager
Calcula tamaños de posición, stop loss, take profit y límites de exposición.
Usa Kelly Criterion simplificado (2% riesgo por trade).

Uso:
    python3 risk_manager.py          # Genera risk_report.json
    python3 risk_manager.py --debug  # Con output detallado
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MARKET_FILE  = DATA_DIR / "market_latest.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
OUTPUT_FILE  = DATA_DIR / "risk_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("risk_manager")

# ─── Parámetros de Riesgo ────────────────────────────────────────────────────

RISK_PER_TRADE_PCT    = 0.02   # 2% del capital por trade (Kelly simplificado)
SL_PCT                = 0.025  # Stop loss: 2.5% fijo
TP_MULTIPLIER         = 2.0    # TP = 2x SL → 5%
MAX_OPEN_POSITIONS    = 5      # Máximo de posiciones simultáneas
MAX_DRAWDOWN_PCT      = 0.10   # 10% drawdown máximo antes de parar
PAUSE_DRAWDOWN_PCT    = 0.08   # 8% → PAUSED
MIN_POSITION_USD      = 5.0    # Mínimo en dólares por posición
MAX_SINGLE_EXPOSURE   = 0.20   # Máximo 20% del capital en un solo token

# ─── Carga de Datos ───────────────────────────────────────────────────────────

def load_market() -> dict:
    if not MARKET_FILE.exists():
        log.error(f"❌ No existe {MARKET_FILE} — ejecuta market_data.py primero")
        return {}
    with open(MARKET_FILE) as f:
        return json.load(f)


def load_portfolio() -> dict:
    """Carga portfolio actual. Si no existe, devuelve portfolio vacío inicial."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    # Portfolio inicial por defecto
    return {
        "capital_usd": 500.0,
        "initial_capital": 500.0,
        "positions": [],
        "status": "ACTIVE",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Cálculos de Riesgo ──────────────────────────────────────────────────────

def calculate_drawdown(portfolio: dict) -> float:
    """Calcula el drawdown actual como porcentaje del capital inicial.
    IMPORTANTE: capital_usd es solo el capital LIBRE (no invertido).
    El equity real = capital_libre + invertido_en_posiciones + P&L_abierto.
    """
    initial  = portfolio.get("initial_capital", 500.0)
    free_cap = portfolio.get("capital_usd", 500.0)

    # Sumar capital invertido en posiciones abiertas + P&L no realizado
    invested    = 0.0
    unrealized  = 0.0
    for pos in portfolio.get("positions", []):
        if pos.get("status") == "open":
            invested   += pos.get("size_usd", 0)
            unrealized += pos.get("pnl_usd", 0)

    # Equity real = lo que tenemos libre + lo que tenemos invertido ± P&L
    total_value = free_cap + invested + unrealized
    if initial <= 0:
        return 0.0
    drawdown = (initial - total_value) / initial
    return max(0.0, drawdown)


def calculate_position_size(capital: float, price: float, sl_pct: float = SL_PCT) -> dict:
    """
    Kelly Criterion simplificado:
    - Riesgo máximo por trade: 2% del capital
    - Tamaño = (capital * risk_pct) / sl_pct
    - Convertido a tokens = tamaño_usd / precio
    """
    risk_amount = capital * RISK_PER_TRADE_PCT
    position_size_usd = risk_amount / sl_pct

    # Cap: no más de MAX_SINGLE_EXPOSURE del capital
    max_position = capital * MAX_SINGLE_EXPOSURE
    position_size_usd = min(position_size_usd, max_position)

    tokens = (position_size_usd / price) if price > 0 else 0

    return {
        "risk_amount_usd": round(risk_amount, 2),
        "position_size_usd": round(position_size_usd, 2),
        "tokens": round(tokens, 6),
        "sl_pct": sl_pct,
        "tp_pct": round(sl_pct * TP_MULTIPLIER, 4),
    }


def evaluate_token(symbol: str, token_data: dict, capital: float,
                   open_positions: list, portfolio_status: str) -> dict:
    """Evalúa si un token puede recibir una nueva posición."""
    price = token_data.get("price", 0)
    result = {
        "symbol": symbol,
        "price": price,
        "approved": False,
        "reason": "",
        "position_size": None,
    }

    # Check 1: Sistema pausado
    if portfolio_status == "PAUSED":
        result["reason"] = "SISTEMA_PAUSED_DRAWDOWN"
        return result

    # Check 2: Precio disponible
    if price <= 0:
        result["reason"] = "SIN_PRECIO"
        return result

    # Check 3: Capital mínimo
    if capital < MIN_POSITION_USD * 2:
        result["reason"] = f"CAPITAL_INSUFICIENTE (${capital:.2f})"
        return result

    # Check 4: Ya tiene posición abierta
    open_tokens = [p.get("symbol") for p in open_positions]
    if symbol in open_tokens:
        result["reason"] = "POSICION_YA_ABIERTA"
        return result

    # Check 5: Máximo de posiciones
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        result["reason"] = f"MAX_POSICIONES ({MAX_OPEN_POSITIONS})"
        return result

    # Calcular tamaño de posición
    sizing = calculate_position_size(capital, price)

    if sizing["position_size_usd"] < MIN_POSITION_USD:
        result["reason"] = f"POSICION_MUY_PEQUEÑA (${sizing['position_size_usd']:.2f})"
        return result

    result["approved"] = True
    result["reason"] = "OK"
    result["position_size"] = sizing
    result["sl_price"] = round(price * (1 - sizing["sl_pct"]), 8)
    result["tp_price"] = round(price * (1 + sizing["tp_pct"]), 8)
    return result


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(debug: bool = False) -> dict:
    log.info("=" * 50)
    log.info("🛡️  RISK MANAGER — iniciando")
    log.info("=" * 50)

    market = load_market()
    portfolio = load_portfolio()

    free_capital   = portfolio.get("capital_usd", 500.0)
    initial        = portfolio.get("initial_capital", 500.0)
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    portfolio_status = portfolio.get("status", "ACTIVE")

    # Equity real = capital libre + invertido en posiciones + P&L abierto
    invested   = sum(p.get("size_usd", 0) for p in open_positions)
    unrealized = sum(p.get("pnl_usd", 0) for p in open_positions)
    capital    = free_capital + invested + unrealized  # equity total

    # Calcular drawdown
    drawdown = calculate_drawdown(portfolio)
    drawdown_pct = round(drawdown * 100, 2)

    # Actualizar estado por drawdown
    if drawdown >= MAX_DRAWDOWN_PCT:
        portfolio_status = "STOPPED"
        log.warning(f"🚨 DRAWDOWN CRÍTICO: {drawdown_pct:.1f}% — SISTEMA DETENIDO")
    elif drawdown >= PAUSE_DRAWDOWN_PCT:
        portfolio_status = "PAUSED"
        log.warning(f"⛔ Drawdown {drawdown_pct:.1f}% > 8% — SISTEMA PAUSADO")
    else:
        log.info(f"📊 Capital: ${capital:.2f} | Drawdown: {drawdown_pct:.2f}% | Posiciones: {len(open_positions)}/{MAX_OPEN_POSITIONS}")

    # Evaluar cada token
    tokens = market.get("tokens", {})
    evaluations = {}
    approved_count = 0

    for symbol, token_data in tokens.items():
        eval_result = evaluate_token(
            symbol, token_data, capital, open_positions, portfolio_status
        )
        evaluations[symbol] = eval_result

        if eval_result["approved"]:
            approved_count += 1
            if debug:
                sz = eval_result["position_size"]
                log.info(f"  ✅ {symbol}: ${sz['position_size_usd']:.2f} | SL: {sz['sl_pct']*100:.1f}% | TP: {sz['tp_pct']*100:.1f}%")
        else:
            if debug:
                log.info(f"  ❌ {symbol}: {eval_result['reason']}")

    # Calcular exposición total actual
    total_exposure = sum(p.get("size_usd", 0) for p in open_positions)
    exposure_pct = (total_exposure / capital * 100) if capital > 0 else 0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capital_usd": round(capital, 2),
        "initial_capital": round(initial, 2),
        "drawdown_pct": drawdown_pct,
        "portfolio_status": portfolio_status,
        "open_positions": len(open_positions),
        "max_positions": MAX_OPEN_POSITIONS,
        "total_exposure_usd": round(total_exposure, 2),
        "exposure_pct": round(exposure_pct, 2),
        "risk_per_trade_pct": RISK_PER_TRADE_PCT * 100,
        "sl_pct": SL_PCT * 100,
        "tp_pct": SL_PCT * TP_MULTIPLIER * 100,
        "tokens_approved": approved_count,
        "evaluations": evaluations,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"✅ Tokens aprobados: {approved_count}/{len(tokens)}")
    log.info(f"💾 Guardado en {OUTPUT_FILE}")
    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk Manager Agent")
    parser.add_argument("--debug", action="store_true", help="Output detallado por token")
    args = parser.parse_args()

    run(debug=args.debug)
    sys.exit(0)
