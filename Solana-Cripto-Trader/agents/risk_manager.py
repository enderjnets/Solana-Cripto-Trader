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
from typing import Optional, List, Dict

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MARKET_FILE  = DATA_DIR / "market_latest.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
RESEARCH_FILE = DATA_DIR / "research_latest.json"
OUTPUT_FILE  = DATA_DIR / "risk_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("risk_manager")

# ─── Parámetros de Riesgo ────────────────────────────────────────────────────

RISK_PER_TRADE_PCT    = 0.015  # 1.5% del capital por trade (era 2% — más conservador)
SL_PCT                = 0.03   # Stop loss: 3% (era 2.5% — más room)
TP_MULTIPLIER         = 2.5    # TP = 2.5x SL → 7.5% (era 2x → más runway)
MAX_OPEN_POSITIONS    = 3      # Máximo 3 posiciones (era 5 — más concentrado)
MAX_DRAWDOWN_PCT      = 0.10   # 10% drawdown máximo antes de parar
PAUSE_DRAWDOWN_PCT    = 0.08   # 8% → PAUSED
MIN_POSITION_USD      = 8.0    # Mínimo $8 margen por posición
MAX_SINGLE_EXPOSURE   = 0.25   # Máximo 25% del capital en margen

# ─── Drift Protocol Parameters ───────────────────────────────────────────────
DEFAULT_LEVERAGE      = 3      # 3x leverage por defecto
MAX_LEVERAGE          = 10     # Máximo 10x
MAINTENANCE_MARGIN    = 0.05   # 5% del notional

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


def load_research() -> dict:
    """Carga el análisis de mercado del AI Researcher."""
    if RESEARCH_FILE.exists():
        with open(RESEARCH_FILE) as f:
            return json.load(f)
    return {"trend": "NEUTRAL", "confidence": 0.5}


# ─── Cálculos de Riesgo ──────────────────────────────────────────────────────

def calculate_drawdown(portfolio: dict) -> float:
    """Calcula el drawdown actual como porcentaje del capital inicial.
    IMPORTANTE: capital_usd es solo el capital LIBRE (no invertido).
    El equity real = capital_libre + invertido_en_posiciones + P&L_abierto.
    """
    initial  = portfolio.get("initial_capital", 500.0)
    free_cap = portfolio.get("capital_usd", 500.0)

    # Sumar margen invertido en posiciones abiertas + P&L no realizado
    invested    = 0.0
    unrealized  = 0.0
    for pos in portfolio.get("positions", []):
        if pos.get("status") == "open":
            invested   += pos.get("margin_usd", pos.get("size_usd", 0))
            unrealized += pos.get("pnl_usd", 0)

    # Equity real = lo que tenemos libre + margen invertido ± P&L (con leverage)
    total_value = free_cap + invested + unrealized
    if initial <= 0:
        return 0.0
    drawdown = (initial - total_value) / initial
    return max(0.0, drawdown)


def evaluate_emergency_close(portfolio: dict, research: dict, market: dict) -> dict:
    """
    Evalúa si se deben cerrar todas las posiciones por emergencia.

    Causas de cierre:
    1. 5 posiciones todas LONG y tendencia BEARISH (confianza > 70%)
    2. 5 posiciones todas SHORT y tendencia BULLISH (confianza > 70%)
    3. Fear & Greed en extremos (<10 o >90) con todas las posiciones en contra
    4. Drawdown > 8% y todas las posiciones negativas

    Retorna:
        {"emergency_close": bool, "reason": str, "symbols": List[str]}
    """
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]

    if not open_positions or len(open_positions) < MAX_OPEN_POSITIONS:
        return {"emergency_close": False, "reason": "", "symbols": []}

    # Obtener tendencia del research
    trend = research.get("trend", "NEUTRAL").upper()
    confidence = research.get("confidence", 0)

    # Obtener Fear & Greed del market
    fear_greed_value = market.get("fear_greed", {}).get("value", 50)

    # Analizar direcciones de posiciones
    long_positions = [p for p in open_positions if p.get("direction") == "long"]
    short_positions = [p for p in open_positions if p.get("direction") == "short"]
    all_long = len(long_positions) == len(open_positions)
    all_short = len(short_positions) == len(open_positions)

    # Calcular P&L de posiciones
    negative_positions = [p for p in open_positions if p.get("pnl_usd", 0) < 0]
    all_negative = len(negative_positions) == len(open_positions)

    # Causa 1: 5 LONGs + tendencia BEARISH (alta confianza)
    if all_long and trend == "BEARISH" and confidence >= 0.7:
        return {
            "emergency_close": True,
            "reason": f"Tendencia BEARISH ({int(confidence*100)}%) con {len(open_positions)} LONGs",
            "symbols": [p["symbol"] for p in open_positions]
        }

    # Causa 2: 5 SHORTs + tendencia BULLISH (alta confianza)
    if all_short and trend == "BULLISH" and confidence >= 0.7:
        return {
            "emergency_close": True,
            "reason": f"Tendencia BULLISH ({int(confidence*100)}%) con {len(open_positions)} SHORTs",
            "symbols": [p["symbol"] for p in open_positions]
        }

    # Causa 3: Fear & Greed extremo + todas posiciones en contra
    if fear_greed_value < 10 and all_long and all_negative:
        return {
            "emergency_close": True,
            "reason": f"Fear & Greed EXTREMO ({fear_greed_value}) - mercado pánico, cerrar LONGs",
            "symbols": [p["symbol"] for p in open_positions]
        }

    if fear_greed_value > 90 and all_short and all_negative:
        return {
            "emergency_close": True,
            "reason": f"Fear & Greed EXTREMO ({fear_greed_value}) - euforia, cerrar SHORTs",
            "symbols": [p["symbol"] for p in open_positions]
        }

    # Causa 4: Drawdown alto + todas posiciones negativas
    drawdown = calculate_drawdown(portfolio)
    if drawdown >= 0.08 and all_negative:
        return {
            "emergency_close": True,
            "reason": f"Drawdown {int(drawdown*100)}% con todas las posiciones perdiendo",
            "symbols": [p["symbol"] for p in open_positions]
        }

    return {"emergency_close": False, "reason": "", "symbols": []}


def calculate_position_size(capital: float, price: float, sl_pct: float = SL_PCT,
                            leverage: int = DEFAULT_LEVERAGE) -> dict:
    """
    Drift Protocol position sizing:
    - Riesgo máximo por trade: 1.5% del capital
    - margin_usd = riesgo / sl_pct (lo que pone el trader)
    - notional = margin * leverage (tamaño real de la posición)
    - tokens = notional / precio
    - liquidation_price calculado basado en margen
    """
    leverage = max(1, min(leverage, MAX_LEVERAGE))

    risk_amount = capital * RISK_PER_TRADE_PCT
    margin_usd = risk_amount / sl_pct

    # Cap: margen no más de MAX_SINGLE_EXPOSURE del capital
    max_margin = capital * MAX_SINGLE_EXPOSURE
    margin_usd = min(margin_usd, max_margin)

    # Notional = margen * leverage
    notional_usd = margin_usd * leverage
    tokens = (notional_usd / price) if price > 0 else 0

    # Margen de mantenimiento
    margin_maintenance = notional_usd * MAINTENANCE_MARGIN

    return {
        "risk_amount_usd": round(risk_amount, 2),
        "margin_usd": round(margin_usd, 2),
        "position_size_usd": round(margin_usd, 2),     # Compat: = margin
        "notional_usd": round(notional_usd, 2),
        "leverage": leverage,
        "tokens": round(tokens, 6),
        "sl_pct": sl_pct,
        "tp_pct": round(sl_pct * TP_MULTIPLIER, 4),
        "margin_maintenance": round(margin_maintenance, 4),
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

    # Calcular tamaño de posición con leverage
    sizing = calculate_position_size(capital, price, leverage=DEFAULT_LEVERAGE)

    if sizing["margin_usd"] < MIN_POSITION_USD:
        result["reason"] = f"MARGEN_MUY_PEQUEÑO (${sizing['margin_usd']:.2f})"
        return result

    result["approved"] = True
    result["reason"] = "OK"
    result["position_size"] = sizing
    result["sl_price_long"] = round(price * (1 - sizing["sl_pct"]), 8)
    result["tp_price_long"] = round(price * (1 + sizing["tp_pct"]), 8)
    result["sl_price_short"] = round(price * (1 + sizing["sl_pct"]), 8)
    result["tp_price_short"] = round(price * (1 - sizing["tp_pct"]), 8)
    # Compat
    result["sl_price"] = result["sl_price_long"]
    result["tp_price"] = result["tp_price_long"]
    # Liquidation prices
    margin_ratio = sizing["margin_usd"] / sizing["notional_usd"] if sizing["notional_usd"] > 0 else 1
    result["liq_price_long"] = round(price * (1 - margin_ratio), 8)
    result["liq_price_short"] = round(price * (1 + margin_ratio), 8)
    return result


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(debug: bool = False) -> dict:
    log.info("=" * 50)
    log.info("🛡️  RISK MANAGER — iniciando")
    log.info("=" * 50)

    market = load_market()
    portfolio = load_portfolio()
    research = load_research()

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

    # Evaluación de cierre de emergencia
    emergency_eval = evaluate_emergency_close(portfolio, research, market)
    if emergency_eval["emergency_close"]:
        log.error(f"🚨 EMERGENCY CLOSE: {emergency_eval['reason']}")
        log.error(f"    Cerrar: {', '.join(emergency_eval['symbols'])}")

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
                log.info(f"  ✅ {symbol}: Margin ${sz['margin_usd']:.2f} | {sz['leverage']}x → Notional ${sz['notional_usd']:.2f} | SL: {sz['sl_pct']*100:.1f}% | TP: {sz['tp_pct']*100:.1f}%")
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
        "emergency_close": {
            "triggered": emergency_eval["emergency_close"],
            "reason": emergency_eval["reason"],
            "symbols": emergency_eval["symbols"]
        }
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
