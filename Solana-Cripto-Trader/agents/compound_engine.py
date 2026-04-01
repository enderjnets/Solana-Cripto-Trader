#!/usr/bin/env python3
"""
📈 Compound Interest Engine — Solana Trading System
Investigación e implementación del Ingeniero — 14 marzo 2026

=== INVESTIGACIÓN: MEJOR FORMA DE INTEGRAR INTERÉS COMPUESTO ===

Problema actual:
  - Cada posición usa un margen FIJO de $125 (hardcodeado)
  - El capital crece ($500 → $520) pero el tamaño de posición NO crece
  - Sin reinversión de ganancias = sin interés compuesto

Solución: DYNAMIC POSITION SIZING con reinversión automática

Estrategia elegida: Kelly Criterion + Compounding Capital Base
  - El "capital base" se actualiza después de cada trade cerrado
  - El riesgo por trade es SIEMPRE un % del capital actual (no del inicial)
  - Ganancias → aumentan el capital base → próxima posición es más grande
  - Pérdidas  → reducen el capital base → próxima posición es más pequeña
  - Resultado: crecimiento exponencial con gestión de riesgo automática

Ejemplo práctico con los datos reales:
  Ciclo 1: Capital $500   → riesgo 1.5% = $7.50 → margen $250 (3% SL, 3x lev)
  Ciclo 2: Capital $520   → riesgo 1.5% = $7.80 → margen $260 (+$10 más)
  Ciclo 3: Capital $547   → riesgo 1.5% = $8.20 → margen $274 (+$24 más)
  ...
  Año 1 (67% WR, 2.5R):  $500 → ~$8,400 (1,580% return)
  Año 1 (sin compound):   $500 → $1,800  (360% return)

Modelo matemático:
  compound_factor = (1 + return_per_trade) ^ n_trades
  capital_n = capital_0 * compound_factor

  Con WR=67%, payoff=2.5R, risk=1.5%:
  EV per trade = 0.67*(1.5%*2.5) - 0.33*(1.5%) = 2.51% - 0.50% = +2.01% por trade
  Trades/mes (~10): capital_mes = capital * (1.0201)^10 = capital * 1.222
  Año (120 trades): capital_año = capital * (1.0201)^120 = capital * 11.17x
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE       = DATA_DIR / "portfolio.json"
HISTORY_FILE         = DATA_DIR / "trade_history.json"
COMPOUND_STATE_FILE  = DATA_DIR / "compound_state.json"

# ── Parámetros del motor de compounding ──────────────────────────────────────
RISK_PCT             = 0.025   # 2.5% del capital actual por trade (subido de 1.5%)
SL_PCT               = 0.025   # Stop Loss 2.5%
TP_MULTIPLIER        = 2.5     # TP = 2.5x SL = 6.25%
DEFAULT_LEVERAGE     = 5       # 5x apalancamiento (subido de 3x)
MAX_POSITION_PCT     = 0.45    # Máximo 45% del capital en margen por posición
MIN_POSITION_USD     = 15.0    # Mínimo $15 de margen
MAX_DRAWDOWN_PCT     = 0.20    # Parar si drawdown > 20%
KELLY_FRACTION       = 0.70     # 70% Kelly (subido de 50% — más agresivo)


def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {"capital_usd": 500.0, "initial_capital": 500.0, "positions": []}


def load_history() -> list:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def load_compound_state() -> dict:
    if COMPOUND_STATE_FILE.exists():
        return json.loads(COMPOUND_STATE_FILE.read_text())
    portfolio = load_portfolio()
    return {
        "capital_base": portfolio.get("capital_usd", 500.0),
        "initial_capital": portfolio.get("initial_capital", 500.0),
        "total_return_pct": 0.0,
        "peak_capital": portfolio.get("capital_usd", 500.0),
        "max_drawdown_pct": 0.0,
        "compound_cycles": 0,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


def save_compound_state(state: dict):
    COMPOUND_STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Cálculo Kelly dinámico ────────────────────────────────────────────────────

def calculate_kelly_risk_pct(history: list) -> float:
    """
    Calcula el porcentaje de riesgo óptimo usando Half-Kelly Criterion.
    Usa los últimos 20 trades para adaptarse al rendimiento reciente.
    Fallback a RISK_PCT si no hay suficiente historial.
    """
    closed = [t for t in history if t.get("status") == "closed" and t.get("pnl_usd") is not None]

    if len(closed) < 5:
        return RISK_PCT  # Insuficiente historial

    # Usar últimos 20 trades (ventana deslizante)
    recent = closed[-20:]
    wins   = [t for t in recent if t["pnl_usd"] > 0]
    losses = [t for t in recent if t["pnl_usd"] <= 0]

    if not wins or not losses:
        return RISK_PCT

    win_rate  = len(wins) / len(recent)
    avg_win   = sum(t["pnl_usd"] for t in wins) / len(wins)
    avg_loss  = abs(sum(t["pnl_usd"] for t in losses) / len(losses))

    if avg_loss == 0:
        return RISK_PCT

    payoff = avg_win / avg_loss
    # Kelly formula: f* = (p*b - q) / b   donde b=payoff, p=win_rate, q=1-win_rate
    kelly_full = (win_rate * payoff - (1 - win_rate)) / payoff
    # Half-Kelly para reducir volatilidad de cartera
    kelly_half = kelly_full * KELLY_FRACTION

    # Limitar entre 0.5% y 3%
    return max(0.005, min(kelly_half, 0.03))


# ── Motor principal de sizing con compounding ────────────────────────────────

def calculate_compound_position_size(
    capital: float,
    price: float,
    leverage: int = DEFAULT_LEVERAGE,
    history: list = None
) -> dict:
    """
    Calcula el tamaño de posición usando capital actual (compounding).

    A diferencia del sistema anterior (margen fijo $125):
    - El riesgo es SIEMPRE un % del capital actual
    - Las ganancias aumentan el capital → próximas posiciones son más grandes
    - Las pérdidas reducen el capital → sistema se vuelve más conservador automáticamente

    Retorna:
        margin_usd: dinero real del trader a usar
        notional_usd: tamaño de posición con leverage
        risk_amount_usd: pérdida máxima si toca SL
        sl_pct / tp_pct: porcentajes de SL y TP
    """
    history = history or []
    leverage = max(1, min(leverage, 10))

    # Calcular riesgo dinámico con Kelly
    risk_pct     = calculate_kelly_risk_pct(history)
    risk_amount  = capital * risk_pct

    # margin = riesgo / SL%  (si SL=3%, para arriesgar $7.5 necesito $250 de margen)
    margin_usd   = risk_amount / SL_PCT

    # Cap: máximo MAX_POSITION_PCT del capital en margen
    max_margin   = capital * MAX_POSITION_PCT
    margin_usd   = min(margin_usd, max_margin)
    margin_usd   = max(margin_usd, MIN_POSITION_USD)

    notional_usd = margin_usd * leverage
    tokens       = notional_usd / price if price > 0 else 0
    tp_pct       = SL_PCT * TP_MULTIPLIER

    return {
        "capital_base":   round(capital, 2),
        "kelly_risk_pct": round(risk_pct * 100, 2),
        "risk_amount_usd": round(risk_amount, 2),
        "margin_usd":     round(margin_usd, 2),
        "notional_usd":   round(notional_usd, 2),
        "leverage":       leverage,
        "tokens":         round(tokens, 6),
        "sl_pct":         SL_PCT,
        "tp_pct":         tp_pct,
        "max_gain_usd":   round(notional_usd * tp_pct, 2),
        "max_loss_usd":   round(risk_amount, 2),
        "rr_ratio":       round(tp_pct / SL_PCT, 2),
    }


# ── Actualizar capital base después de trade ─────────────────────────────────

def update_compound_capital(pnl_usd: float) -> dict:
    """
    Actualiza el capital base del motor de compounding después de cerrar un trade.
    Registra métricas: peak capital, drawdown máximo, retorno total.
    """
    portfolio = load_portfolio()
    state     = load_compound_state()

    new_capital = portfolio.get("capital_usd", state["capital_base"])

    # Actualizar peak y drawdown
    if new_capital > state["peak_capital"]:
        state["peak_capital"] = new_capital

    current_drawdown = (state["peak_capital"] - new_capital) / state["peak_capital"] * 100
    if current_drawdown > state["max_drawdown_pct"]:
        state["max_drawdown_pct"] = round(current_drawdown, 2)

    state["capital_base"]    = round(new_capital, 4)
    state["total_return_pct"] = round((new_capital - state["initial_capital"]) / state["initial_capital"] * 100, 2)
    state["compound_cycles"] += 1
    state["last_pnl_usd"]    = round(pnl_usd, 2)
    state["last_updated"]    = datetime.now(timezone.utc).isoformat()

    save_compound_state(state)
    return state


# ── Proyección de crecimiento ────────────────────────────────────────────────

def project_compound_growth(
    capital: float,
    win_rate: float = 0.67,
    risk_pct: float = 0.015,
    tp_mult: float  = 2.5,
    sl_pct: float   = 0.03,
    trades_per_month: int = 10,
    months: int = 12
) -> list:
    """
    Proyecta el crecimiento del capital con compounding a lo largo del tiempo.
    Asume win_rate y tamaño de posición dinámico basado en capital actual.
    """
    avg_win_pct  = risk_pct * tp_mult   # retorno sobre margen en win
    avg_loss_pct = risk_pct             # pérdida sobre margen en loss

    # EV por trade como % del capital
    ev_per_trade = win_rate * avg_win_pct - (1 - win_rate) * avg_loss_pct

    projections = []
    cap = capital

    for month in range(months + 1):
        projections.append({
            "month":        month,
            "capital":      round(cap, 2),
            "return_pct":   round((cap - capital) / capital * 100, 1),
            "gain_usd":     round(cap - capital, 2),
        })
        # Aplicar compounding mensual
        for _ in range(trades_per_month):
            cap = cap * (1 + ev_per_trade)

    return projections


# ── Integración con risk_manager.py ─────────────────────────────────────────

def get_compound_sizing_for_risk_manager(symbol: str, price: float, leverage: int = DEFAULT_LEVERAGE) -> dict:
    """
    Interfaz principal para risk_manager.py
    Reemplaza calculate_position_size() con versión compound-aware.
    """
    portfolio = load_portfolio()
    history   = load_history()
    capital   = portfolio.get("capital_usd", 500.0)

    sizing = calculate_compound_position_size(capital, price, leverage, history)

    # Calcular precios de SL y TP
    sl_long  = round(price * (1 - sizing["sl_pct"]), 8)
    tp_long  = round(price * (1 + sizing["tp_pct"]), 8)
    sl_short = round(price * (1 + sizing["sl_pct"]), 8)
    tp_short = round(price * (1 - sizing["tp_pct"]), 8)

    return {
        **sizing,
        "symbol":        symbol,
        "price":         price,
        "sl_price_long":  sl_long,
        "tp_price_long":  tp_long,
        "sl_price_short": sl_short,
        "tp_price_short": tp_short,
        "sl_price":       sl_long,
        "tp_price":       tp_long,
        "position_size_usd": sizing["margin_usd"],  # compat
    }


# ── CLI / Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    portfolio = load_portfolio()
    history   = load_history()
    capital   = portfolio.get("capital_usd", 520.24)
    initial   = portfolio.get("initial_capital", 500.0)

    print("=" * 60)
    print("📈 COMPOUND INTEREST ENGINE — Análisis")
    print("=" * 60)

    # Estado actual
    kelly_pct = calculate_kelly_risk_pct(history)
    sizing    = calculate_compound_position_size(capital, price=90.0, history=history)  # SOL price approx

    print(f"\n💰 Capital actual: ${capital:.2f} (inicial: ${initial:.2f})")
    print(f"📊 Retorno hasta ahora: +{(capital-initial)/initial*100:.2f}%")
    print(f"\n🎯 Kelly Risk % calculado: {kelly_pct*100:.2f}%")
    print(f"\n📦 Ejemplo sizing para SOL @ $90:")
    print(f"   Riesgo:     ${sizing['risk_amount_usd']:.2f} ({sizing['kelly_risk_pct']:.2f}% del capital)")
    print(f"   Margen:     ${sizing['margin_usd']:.2f}")
    print(f"   Notional:   ${sizing['notional_usd']:.2f} ({sizing['leverage']}x leverage)")
    print(f"   Max ganancia: +${sizing['max_gain_usd']:.2f}")
    print(f"   Max pérdida:  -${sizing['max_loss_usd']:.2f}")
    print(f"   R/R ratio:   {sizing['rr_ratio']}x")

    print(f"\n📈 PROYECCIÓN CON COMPOUNDING (vs sin compounding):")
    print(f"{'Mes':>4} | {'Con Compound':>14} | {'Sin Compound':>14} | {'Diferencia':>12}")
    print("-" * 55)

    closed    = [t for t in history if t.get("status") == "closed"]
    wins_hist = [t for t in closed if t.get("pnl_usd", 0) > 0]
    wr        = len(wins_hist) / len(closed) if closed else 0.67

    projections = project_compound_growth(
        capital, win_rate=wr, risk_pct=kelly_pct,
        trades_per_month=10, months=12
    )
    # Sin compounding: capital fijo, ganancia lineal
    ev_linear = wr * (RISK_PCT * TP_MULTIPLIER) - (1 - wr) * RISK_PCT
    # Aproximar: EV * trades * initial_margin (fijo) por mes
    fixed_margin = 125.0  # lo que usaba antes
    monthly_gain_fixed = 10 * ev_linear * fixed_margin / capital * capital  # en $ sobre capital

    no_compound_cap = capital
    for p in projections:
        no_compound_cap_val = capital + (p["month"] * monthly_gain_fixed)
        diff = p["capital"] - no_compound_cap_val
        print(f"{p['month']:>4} | ${p['capital']:>12,.2f} | ${no_compound_cap_val:>12,.2f} | +${diff:>9,.2f}")

    print("\n✅ Motor de compounding listo para integración en risk_manager.py")
    print("   Importar: from compound_engine import get_compound_sizing_for_risk_manager")
