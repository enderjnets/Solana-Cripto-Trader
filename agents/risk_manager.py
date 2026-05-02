#!/usr/bin/env python3
"""
🛡️ Agente 2: Risk Manager
Calcula tamaños de posición, stop loss, take profit y límites de exposición.
Usa Kelly Criterion simplificado (2% riesgo por trade).

NUEVA FUNCIÓN: evaluate_position_decision()
  → Usa LLM (Claude Sonnet 4.6) + métricas cuantitativas para decidir
    si una posición abierta debe CERRARSE, MANTENERSE o REDUCIRSE.
  → Factores: distancia TP/SL, P&L, horas abiertas, tendencia, Fear & Greed,
    RSI, momentum reciente y ratio riesgo/beneficio restante.

Uso:
    python3 risk_manager.py          # Genera risk_report.json
    python3 risk_manager.py --debug  # Con output detallado
    python3 risk_manager.py --decide # Evalúa todas las posiciones abiertas
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

try:
    from llm_config import call_llm
except ImportError:
    def call_llm(prompt, system="", max_tokens=2000):
        return None

try:
    from compound_engine import (
        calculate_compound_position_size,
        get_compound_sizing_for_risk_manager,
        update_compound_capital,
        calculate_kelly_risk_pct,
    )
    COMPOUND_ENABLED = True
except ImportError:
    COMPOUND_ENABLED = False

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MARKET_FILE        = DATA_DIR / "market_latest.json"
PORTFOLIO_FILE     = DATA_DIR / "portfolio.json"
RESEARCH_FILE      = DATA_DIR / "research_latest.json"
OUTPUT_FILE        = DATA_DIR / "risk_report.json"
AUTO_LEARNER_FILE  = DATA_DIR / "auto_learner_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("risk_manager")

# ─── Parámetros de Riesgo — OPTIMIZADO 2026-03-27 (post-drawdown crítico) ────

RISK_PER_TRADE_PCT    = 0.025  # E2: Raised from 1% to 2.5%  # 1.0% del capital por trade
SL_PCT                = 0.025  # Stop loss: 2.5% — crypto necesita espacio (1.5% causó 70% SL hits)
TP_MULTIPLIER         = 2.0    # TP = 2x SL → 5.0% (R:R 1:2)
MAX_OPEN_POSITIONS    = 4      # AUDIT FIX: Aligned with auto_learner moderado profile (was 6)
MAX_DRAWDOWN_PCT      = 0.15   # FIX 2.5: 15% drawdown (ampliado de 10% para 5x leverage)
PAUSE_DRAWDOWN_PCT    = 0.10   # FIX 2.5: 10% pause (ampliado de 6% para 5x leverage)
MIN_POSITION_USD      = 2.0    # Lowered from $5 for small capital ($100+)    # Mínimo $5 margen (era $8 — ajustado a capital bajo)
MAX_SINGLE_EXPOSURE   = 0.40   # Máximo 40% del capital en margen (ajustado 2026-03-31 para mercado lento — posiciones más grandes)


# ─── Auto-Learner Override ───────────────────────────────────────────────────
# Si auto_learner_state.json existe, usar sus parámetros (no hardcoded)
_AUTO_LEARNER_CACHE = None

def _load_auto_learner_params():
    """Carga parámetros del auto_learner. Cacheado por sesión."""
    global _AUTO_LEARNER_CACHE
    if _AUTO_LEARNER_CACHE is not None:
        return _AUTO_LEARNER_CACHE
    try:
        if AUTO_LEARNER_FILE.exists():
            data = json.loads(AUTO_LEARNER_FILE.read_text())
            params = data.get("params", {})
            if params:
                _AUTO_LEARNER_CACHE = params
                log.info(f"   🧠 Auto-Learner params loaded: SL={params.get('sl_pct',0):.3f} TP={params.get('tp_pct',0):.3f} RPT={params.get('risk_per_trade',0):.3f}")
                return params
    except Exception as e:
        log.warning(f"   ⚠️ Auto-Learner load error: {e}")
    _AUTO_LEARNER_CACHE = {}
    return _AUTO_LEARNER_CACHE

# Apply auto_learner overrides (can be called dynamically after learner runs)
def reload_auto_learner_params():
    """Fuerza relectura de auto_learner (llamar después de que learner genera nuevos params).
       AUDIT FIX: actually updates MAX_OPEN_POSITIONS global from learner state."""
    global _AUTO_LEARNER_CACHE, SL_PCT, TP_MULTIPLIER, RISK_PER_TRADE_PCT, MAX_OPEN_POSITIONS
    _AUTO_LEARNER_CACHE = None
    params = _load_auto_learner_params()
    if params and 'max_positions' in params:
        try:
            _new_max = int(params.get('max_positions', MAX_OPEN_POSITIONS))
            if _new_max != MAX_OPEN_POSITIONS and 1 <= _new_max <= 10:
                log.info(f"   🧠 Auto-learner override: MAX_OPEN_POSITIONS {MAX_OPEN_POSITIONS} → {_new_max}")
                MAX_OPEN_POSITIONS = _new_max
        except Exception as _e:
            log.warning(f"   ⚠️ MAX_OPEN_POSITIONS override error: {_e}")

# Load at module import time
_learner_params = _load_auto_learner_params()
if _learner_params:
    # Override with learned values if available
    pass  # Constants stay as defaults; functions read from file at runtime

# ─── Volatility-Adaptive Sizing Parameters (2026-03-31, orden de Ender) ──────
# Cuando la volatilidad es BAJA, el precio se mueve poco → necesitamos MÁS
# notional para alcanzar el TP en un tiempo razonable.
# Cuando la volatilidad es ALTA, el precio se mueve rápido → MENOS notional.
#
# Fórmula: vol_factor = REFERENCE_VOL / actual_atr_pct (clamped a [MIN, MAX])
REFERENCE_VOL         = 0.025  # Volatilidad de referencia = SL_PCT (2.5%)
VOL_MULT_MIN          = 0.5    # Mínimo multiplicador (alta volatilidad → posición más chica)
VOL_MULT_MAX          = 3.0    # Máximo multiplicador (baja volatilidad → posición más grande)
VOL_ATR_FLOOR         = 0.003  # ATR mínimo para evitar divisiones extremas (0.3%)

# ─── Drift Protocol Parameters ───────────────────────────────────────────────
DEFAULT_LEVERAGE      = 3      # 3x leverage por defecto (overridden by auto_learner)
MAX_LEVERAGE          = 10     # Máximo 10x
MAINTENANCE_MARGIN    = 0.05   # 5% del notional

# Leverage tiers from auto_learner
LEVERAGE_TIERS = {1: 2, 2: 3, 3: 5, 4: 7, 5: 10}


def load_auto_learner_leverage() -> int:
    """Lee leverage_tier del auto_learner y retorna el leverage efectivo."""
    try:
        if AUTO_LEARNER_FILE.exists():
            data = json.loads(AUTO_LEARNER_FILE.read_text())
            tier = int(data.get("params", {}).get("leverage_tier", 2))
            return LEVERAGE_TIERS.get(tier, DEFAULT_LEVERAGE)
    except Exception:
        pass
    return DEFAULT_LEVERAGE

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
    """Calcula el drawdown basado en equity TOTAL del wallet on-chain.

    v2.12.30 FIX: anteriormente usaba portfolio.capital_usd + invested + unrealized,
    que EXCLUÍA SOL fuel (~$14) y dust JUP/ETH (~$3). Consecuencia: reportaba
    drawdown 20.87% cuando el wallet real estaba en +1.5% (ganancia). Ese DD falso
    disparaba evaluate_emergency_close Cause #4 en cada trade → 95% emergency closes.

    Fuente primaria: wallet_equity.fetch_wallet_equity() — query on-chain
    (USDC + SOL + JUP + ETH × Jupiter prices, 60s cache).
    Fallback: cálculo bot-equity legacy si wallet no disponible (paper mode, RPC fail).
    """
    initial = portfolio.get("initial_capital", 500.0)
    if initial <= 0:
        return 0.0

    # v2.12.30: primary source — on-chain wallet total equity
    # v2.13.3 FIX: Jupiter Perps collateral is locked in the perps contract and
    # excluded from wallet balances. We must add back perps position value to get
    # true equity, otherwise every perp trade triggers a false drawdown pause.
    try:
        from wallet_equity import fetch_wallet_equity
        we = fetch_wallet_equity()
        if we and we.get("wallet_total") is not None:
            wallet_total = float(we["wallet_total"])
            # Add Jupiter Perps position value (collateral locked in contract)
            perps_value = 0.0
            try:
                import os
                if os.environ.get("JUP_PERP_ENABLED", "").lower() == "true":
                    # Robust import: works whether agents/ is in sys.path or not
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
            return max(0.0, (initial - total_equity) / initial)
    except Exception as _e:
        log.warning(f"calculate_drawdown: wallet_equity unavailable, falling back to bot equity ({_e})")

    # Fallback (paper mode or RPC failure): legacy bot-equity calculation
    free_cap = portfolio.get("capital_usd", 500.0)
    invested = 0.0
    unrealized = 0.0
    for pos in portfolio.get("positions", []):
        if pos.get("status") == "open":
            invested += pos.get("margin_usd", 0)
            unrealized += pos.get("pnl_usd", 0)
    total_value = free_cap + invested + unrealized
    return max(0.0, (initial - total_value) / initial)


def evaluate_emergency_close(portfolio: dict, research: dict, market: dict) -> dict:
    """
    Evalúa si se deben cerrar todas las posiciones por emergencia.

    Causas de cierre:
    1. 5 posiciones todas LONG y tendencia BEARISH (confianza > 70%)
    2. 5 posiciones todas SHORT y tendencia BULLISH (confianza > 70%)
    3. Fear & Greed en extremos (<10 o >90) con todas las posiciones en contra
    4. Drawdown > 10% y todas las posiciones negativas

    Retorna:
        {"emergency_close": bool, "reason": str, "symbols": List[str]}
    """
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]

    if not open_positions:  # M6: removed MAX_OPEN_POSITIONS guard
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

    # Causa 1: 5 LONGs + tendencia BEARISH (alta confianza) — threshold 0.80 para evitar falsos positivos en borde
    if all_long and trend == "BEARISH" and confidence >= 0.80:
        return {
            "emergency_close": True,
            "reason": f"Tendencia BEARISH ({int(confidence*100)}%) con {len(open_positions)} LONGs",
            "symbols": [p["symbol"] for p in open_positions]
        }

    # Causa 2: 5 SHORTs + tendencia BULLISH (alta confianza) — threshold 0.80 para evitar falsos positivos en borde
    # Adicionalmente: no cerrar si las posiciones tienen < 5 min (evitar cierre inmediato tras apertura)
    if all_short and trend == "BULLISH" and confidence >= 0.80:
        now = datetime.now(timezone.utc)
        for pos in open_positions:
            try:
                opened_at = datetime.fromisoformat(pos.get("opened_at", pos.get("open_time", "")).replace("Z", "+00:00"))
                age_minutes = (now - opened_at).total_seconds() / 60
                if age_minutes < 5:
                    log.info(f"   🛡️ Emergency close SKIPPED: {pos['symbol']} abierta hace solo {age_minutes:.1f}min (< 5min protection)")
                    continue  # Skip this position
            except:
                pass
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
    # v2.12.30: defensive guards to prevent spurious closes on micro-dips.
    # Previously fired on ANY negative position when stale DD >= 8% (Bug #1).
    # Today (2026-04-23): 73 of 77 trades (95%) closed here with 1.6min median hold.
    drawdown = calculate_drawdown(portfolio)
    if drawdown >= 0.08 and all_negative:
        # Guard A: min position age >= 5min (same pattern as Cause #2)
        now = datetime.now(timezone.utc)
        min_age_ok = True
        for pos in open_positions:
            try:
                opened_at = datetime.fromisoformat(
                    str(pos.get("open_time", pos.get("opened_at", ""))).replace("Z", "+00:00")
                )
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                age_min = (now - opened_at).total_seconds() / 60
                if age_min < 5:
                    min_age_ok = False
                    log.info(
                        f"   🛡️ Emergency close SKIPPED (age): {pos.get('symbol','?')} "
                        f"abierto hace {age_min:.1f}min < 5min protection"
                    )
                    break
            except Exception:
                # If open_time unparseable, don't block — conservative default
                pass

        # Guard B: min loss threshold — need >= 1% loss to count (ignore entry slippage)
        MIN_LOSS_PCT = 0.01
        try:
            max_loss = max(
                abs(float(p.get("pnl_pct", 0) or 0)) for p in open_positions
            )
        except Exception:
            max_loss = 0.0
        min_loss_ok = max_loss >= MIN_LOSS_PCT
        if not min_loss_ok:
            log.info(
                f"   🛡️ Emergency close SKIPPED (loss): max_loss={max_loss*100:.2f}% "
                f"< {MIN_LOSS_PCT*100:.0f}% — within normal entry slippage"
            )

        if min_age_ok and min_loss_ok:
            return {
                "emergency_close": True,
                "reason": f"Drawdown {int(drawdown*100)}% con todas las posiciones perdiendo >=1% por >=5min",
                "symbols": [p["symbol"] for p in open_positions]
            }

    return {"emergency_close": False, "reason": "", "symbols": []}


def _kelly_risk_pct() -> float:
    """Calculate half-Kelly risk percentage from auto_learner stats."""
    try:
        if AUTO_LEARNER_FILE.exists():
            state = json.loads(AUTO_LEARNER_FILE.read_text())
            params = state.get("params", {})
            # Need win rate and payoff ratio from trade history
            th_file = DATA_DIR / "trade_history.json"
            if th_file.exists():
                trades = json.loads(th_file.read_text())
                if isinstance(trades, list) and len(trades) >= 10:
                    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
                    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
                    if wins and losses:
                        wr = len(wins) / len(trades)
                        avg_win = sum(t["pnl_usd"] for t in wins) / len(wins)
                        avg_loss = abs(sum(t["pnl_usd"] for t in losses) / len(losses))
                        if avg_loss > 0:
                            payoff = avg_win / avg_loss
                            kelly = wr - (1 - wr) / payoff
                            # Half-Kelly for safety, capped between 0.5% and 3%
                            half_kelly = max(0.005, min(kelly * 0.5, 0.03))
                            log.info(f"📐 Kelly: WR={wr:.1%} Payoff={payoff:.2f} → half-Kelly={half_kelly:.1%}")
                            return half_kelly
    except Exception as e:
        log.warning(f"⚠️ Kelly calculation failed: {e}")
    return RISK_PER_TRADE_PCT  # Fallback to default


def volatility_multiplier(atr_pct: float = None) -> float:
    """
    Calcula el multiplicador de posición basado en volatilidad (ATR%).
    
    - Baja volatilidad (ATR < REFERENCE_VOL) → multiplicador > 1 → posición más grande
    - Alta volatilidad (ATR > REFERENCE_VOL) → multiplicador < 1 → posición más chica
    - Sin datos de ATR → multiplicador = 1.0 (sin ajuste)
    
    Ejemplo con REFERENCE_VOL=2.5%:
      ATR 0.5% → factor 5.0 → clamped 3.0 → notional × 3
      ATR 1.0% → factor 2.5 → notional × 2.5
      ATR 2.5% → factor 1.0 → sin cambio
      ATR 5.0% → factor 0.5 → notional × 0.5
    """
    if atr_pct is None or atr_pct <= 0:
        return 1.0
    
    # Piso para evitar divisiones extremas
    effective_atr = max(atr_pct / 100.0, VOL_ATR_FLOOR)  # atr_pct viene en %, convertir a decimal
    
    factor = REFERENCE_VOL / effective_atr
    clamped = max(VOL_MULT_MIN, min(factor, VOL_MULT_MAX))
    
    return round(clamped, 2)


def calculate_position_size(capital: float, price: float, sl_pct: float = None,
                            leverage: int = DEFAULT_LEVERAGE,
                            atr_pct: float = None) -> dict:
    """
    Drift Protocol position sizing con ajuste de volatilidad:
    - Riesgo máximo por trade: Kelly Criterion (half-Kelly) or 1.5% fallback
    - margin_usd = riesgo / sl_pct (lo que pone el trader)
    - notional = margin * leverage (tamaño real de la posición)
    - vol_factor ajusta el margen según volatilidad actual (ATR%)
    - tokens = notional / precio
    - liquidation_price calculado basado en margen
    """
    leverage = max(1, min(leverage, MAX_LEVERAGE))

    kelly_pct = _kelly_risk_pct()
    risk_amount = capital * kelly_pct
    margin_usd = risk_amount / sl_pct

    # ─── Volatility-Adaptive Sizing ───────────────────────────────────────
    vol_mult = volatility_multiplier(atr_pct)
    margin_usd *= vol_mult

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
        "vol_multiplier": vol_mult,
        "atr_pct_used": atr_pct,
    }


def evaluate_token(symbol: str, token_data: dict, capital: float,
                   open_positions: list, portfolio_status: str,
                   atr_pct: float = None) -> dict:
    """Evalúa si un token puede recibir una nueva posición.
    atr_pct: ATR% del token (de indicator_summary) para vol-adaptive sizing.
    """
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
    _p2 = _load_auto_learner_params()
    _max_pos = _p2.get("max_positions", MAX_OPEN_POSITIONS) if _p2 else MAX_OPEN_POSITIONS
    if len(open_positions) >= _max_pos:
        result["reason"] = f"MAX_POSICIONES ({_max_pos})"
        return result

    # Calcular tamaño de posición — Compound Engine (si disponible) o fallback
    effective_leverage = load_auto_learner_leverage()
    if COMPOUND_ENABLED:
        history = []
        th_file = DATA_DIR / "trade_history.json"
        if th_file.exists():
            th_data = json.loads(th_file.read_text())
            history = th_data.get("trades", []) if isinstance(th_data, dict) else []
        sizing = calculate_compound_position_size(capital, price, effective_leverage, history)
        log.info(f"   📈 Compound sizing: capital=${capital:.2f} → margen=${sizing['margin_usd']:.2f} (kelly={sizing['kelly_risk_pct']:.2f}%)")
    else:
        sizing = calculate_position_size(capital, price, leverage=effective_leverage, atr_pct=atr_pct)
    
    vol_mult = sizing.get("vol_multiplier", 1.0)
    if vol_mult != 1.0:
        log.info(f"   📊 Vol-adaptive sizing: ATR={atr_pct:.2f}% → mult={vol_mult}x → margin=${sizing['margin_usd']:.2f}")

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


# ─── Smart Rotation (Opción 1) ───────────────────────────────────────────────

def check_stale_losing_positions(portfolio: dict, max_hours: int = 48, improvement_hours: int = 12) -> list:
    """
    Detecta posiciones perdedoras que han estado abiertas > max_hours sin mejora.
    Retorna lista de posiciones a cerrar.
    """
    positions_to_close = []
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    
    now = datetime.now(timezone.utc)
    
    for pos in open_positions:
        # Verificar si es perdedora
        pnl_pct = pos.get("pnl_pct", 0)
        if pnl_pct >= 0:
            continue  # Solo perdedoras
        
        # Verificar tiempo abierto (campo puede ser "open_time" o "opened_at")
        opened_at_str = pos.get("open_time") or pos.get("opened_at")
        if not opened_at_str:
            continue
        
        try:
            opened_at = datetime.fromisoformat(opened_at_str.replace('Z', '+00:00'))
            hours_open = (now - opened_at).total_seconds() / 3600
            
            if hours_open < max_hours:
                continue  # No ha pasado suficiente tiempo
            
            # Verificar si ha mejorado en las últimas improvement_hours
            # Si no tiene historial de P&L, asumimos que no ha mejorado
            recent_pnl_history = pos.get("pnl_history", [])
            if len(recent_pnl_history) > 0:
                # Tomar el P&L de hace improvement_hours
                cutoff_time = now - __import__('datetime').timedelta(hours=improvement_hours)
                old_pnl = None
                for entry in recent_pnl_history:
                    entry_time = datetime.fromisoformat(entry.get("timestamp", "").replace('Z', '+00:00'))
                    if entry_time <= cutoff_time:
                        old_pnl = entry.get("pnl_pct", 0)
                
                if old_pnl is not None and pnl_pct > old_pnl:
                    # Ha mejorado, no cerrar
                    continue
            
            # Cumple todos los criterios: cerrar
            positions_to_close.append({
                "symbol": pos.get("symbol"),
                "reason": f"STALE_LOSING: {hours_open:.1f}h abierto, P&L {pnl_pct:.2f}%",
                "pnl_usd": pos.get("pnl_usd", 0),
                "pnl_pct": pnl_pct,
                "hours_open": hours_open
            })
            
        except Exception as e:
            log.warning(f"Error checking position {pos.get('symbol')}: {e}")
            continue
    
    return positions_to_close


# ─── Intelligent Position Decision ──────────────────────────────────────────

DECISIONS_FILE = DATA_DIR / "position_decisions.json"


def _quant_score(pos: dict, market: dict, research: dict) -> dict:
    """
    Calcula un score cuantitativo de -100 a +100 para una posición abierta.
    Positivo = señal de CERRAR (tomar ganancias / cortar pérdidas)
    Negativo = señal de MANTENER (dejar correr)
    """
    symbol      = pos["symbol"]
    direction   = pos["direction"]
    entry       = pos["entry_price"]
    current     = pos["current_price"]
    sl          = pos["sl_price"]
    tp          = pos["tp_price"]
    pnl_pct     = pos.get("pnl_pct", 0)
    margin      = pos.get("margin_usd", 0)
    leverage    = pos.get("leverage", 3)

    score       = 0
    reasons     = []

    # ── Factor 1: Distancia al TP (0-40 pts) ──────────────────────────────
    # FIX 2026-04-05: Umbrales MUY relajados para evitar cierres prematuros
    # Solo cerrar si está AL BORDE del TP y tiene ganancia sólida
    if direction == "long":
        dist_tp_pct = (tp - current) / current * 100 if tp > current else 0
        dist_sl_pct = (current - sl) / current * 100 if current > sl else 0
    else:
        dist_tp_pct = (current - tp) / current * 100 if current > tp else 0
        dist_sl_pct = (sl - current) / current * 100 if sl > current else 0

    # TP muy cerca + ganancia alta → cerrar para asegurar (era < 0.5% = 40pts → ahora < 2.0% AND pnl_pct > 3%)
    if dist_tp_pct < 2.0 and pnl_pct > 3:
        score += 40
        reasons.append(f"TP_MUY_CERCA_GANANCIA ({dist_tp_pct:.2f}%, pnl={pnl_pct:.1f}%)")
    elif dist_tp_pct < 4.0 and pnl_pct > 5:
        score += 25
        reasons.append(f"TP_CERCA_GANANCIA ({dist_tp_pct:.2f}%, pnl={pnl_pct:.1f}%)")
    elif dist_tp_pct < 6.0:
        score += 10
        reasons.append(f"TP_MODERADO ({dist_tp_pct:.2f}%)")

    # SL muy cerca + posición vieja → cerrar (era < 0.3% → ahora + hours_open > 0.5)
    if dist_sl_pct < 0.3 and hours_open > 0.5:
        score += 45
        reasons.append(f"SL_PELIGRO ({dist_sl_pct:.2f}%, {hours_open:.1f}h)")
    elif dist_sl_pct < 0.7 and hours_open > 1.0:
        score += 25
        reasons.append(f"SL_CERCA ({dist_sl_pct:.2f}%, {hours_open:.1f}h)")

    # ── Factor 2: Risk/Reward restante ────────────────────────────────────
    if dist_sl_pct > 0:
        rr_remaining = dist_tp_pct / dist_sl_pct
        if rr_remaining < 0.5:
            score += 20
            reasons.append(f"RR_MALO ({rr_remaining:.2f}x)")
        elif rr_remaining > 2.0:
            score -= 15
            reasons.append(f"RR_BUENO ({rr_remaining:.2f}x)")

    # ── Factor 3: Horas abierto ───────────────────────────────────────────
    # v2.12.9 fix: positions usan campo "open_time", no "opened_at"
    opened_at_str = pos.get("open_time") or pos.get("opened_at", "")
    hours_open = 0
    if opened_at_str:
        try:
            opened_at = datetime.fromisoformat(opened_at_str.replace('Z', '+00:00'))
            hours_open = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
        except Exception:
            pass

    if hours_open > 96:
        score += 20
        reasons.append(f"MUY_ANTIGUA ({hours_open:.0f}h)")
    elif hours_open > 72:
        score += 10
        reasons.append(f"ANTIGUA ({hours_open:.0f}h)")

    # ── Factor 4: P&L actual ──────────────────────────────────────────────
    # FIX 2026-04-05: Solo cerrar posiciones perdedoras extremas, no ganadores
    # Posición MUY ganadora (>10%) → considerar cerrar si otras condiciones coinciden
    if pnl_pct > 12:
        score += 15
        reasons.append(f"GANANCIA_MUY_ALTA ({pnl_pct:.1f}%)")

    # Posición perdedora sin señal de recuperación (era < -5 → ahora < -8)
    if pnl_pct < -8:
        score += 20
        reasons.append(f"PERDIDA_ALTA ({pnl_pct:.1f}%)")

    # ── Factor 5: Tendencia de mercado ────────────────────────────────────
    trend      = research.get("trend", "NEUTRAL").upper()
    confidence = research.get("confidence", 0.5)

    if direction == "long" and trend == "BEARISH" and confidence > 0.75:
        score += 20
        reasons.append(f"TENDENCIA_EN_CONTRA (BEARISH {confidence:.0%})")
    elif direction == "short" and trend == "BULLISH" and confidence > 0.75:
        score += 20
        reasons.append(f"TENDENCIA_EN_CONTRA (BULLISH {confidence:.0%})")
    elif direction == "long" and trend == "BULLISH" and confidence > 0.75:
        score -= 15
        reasons.append(f"TENDENCIA_FAVOR (BULLISH {confidence:.0%})")
    elif direction == "short" and trend == "BEARISH" and confidence > 0.75:
        score -= 15
        reasons.append(f"TENDENCIA_FAVOR (BEARISH {confidence:.0%})")

    # ── Factor 6: Fear & Greed ────────────────────────────────────────────
    fg = market.get("fear_greed", {}).get("value", 50)
    if fg > 80 and direction == "long":
        score += 10
        reasons.append(f"EUFORIA_EXTREMA (F&G={fg})")
    elif fg < 20 and direction == "short":
        score += 10
        reasons.append(f"PANICO_EXTREMO (F&G={fg})")

    return {
        "symbol": symbol,
        "score": score,
        "reasons": reasons,
        "dist_tp_pct": round(dist_tp_pct, 3),
        "dist_sl_pct": round(dist_sl_pct, 3),
        "hours_open": round(hours_open, 1),
        "pnl_pct": round(pnl_pct, 2),
        "rr_remaining": round(dist_tp_pct / dist_sl_pct, 3) if dist_sl_pct > 0 else 0,
    }


def _get_trade_history_context(symbol: str, max_trades: int = 5) -> str:
    """Build enriched context from recent trade history for LLM prompt."""
    try:
        _th_file = DATA_DIR / "trade_history.json"
        if not _th_file.exists():
            return ""
        _trades = json.loads(_th_file.read_text())
        if not isinstance(_trades, list):
            return ""
        _recent = _trades[-max_trades:]
        _lines = []
        for t in _recent:
            _lines.append(f"  {t.get('symbol')} {t.get('direction')} | {t.get('close_reason')} | PnL: ${t.get('pnl_usd',0):.2f}")
        from collections import defaultdict
        _strat = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for t in _trades:
            s = t.get("strategy", "unknown")
            if t.get("pnl_usd", 0) > 0:
                _strat[s]["wins"] += 1
            else:
                _strat[s]["losses"] += 1
            _strat[s]["pnl"] += t.get("pnl_usd", 0)
        _wr_lines = []
        for s, d in _strat.items():
            _total = d["wins"] + d["losses"]
            if _total > 0:
                _wr = d["wins"] / _total * 100
                _wr_lines.append(f"  {s}: {_wr:.0f}% WR ({d['wins']}W/{d['losses']}L) | Total PnL: ${d['pnl']:.2f}")
        _ctx = "HISTORIAL RECIENTE DE TRADES:" + chr(10) + chr(10).join(_lines)
        if _wr_lines:
            _ctx += chr(10) + "WIN RATE POR ESTRATEGIA:" + chr(10) + chr(10).join(_wr_lines)
        return _ctx
    except Exception:
        return ""


def _llm_decision(pos: dict, quant: dict, market: dict, research: dict,
                   portfolio_context: dict = None) -> dict:
    """
    Pide al LLM (Claude Sonnet 4.6) su recomendación basada en el análisis cuantitativo.
    Retorna: {"action": "CLOSE"|"HOLD"|"REDUCE", "confidence": 0-1, "reasoning": str}
    
    portfolio_context: dict con info del portafolio coordinado:
      - total_positions, portfolio_pnl, portfolio_target,
      - other_positions (lista de {symbol, pnl_usd, pnl_pct, direction})
    """
    symbol    = pos["symbol"]
    direction = pos["direction"].upper()
    entry     = pos["entry_price"]
    current   = pos["current_price"]
    sl        = pos["sl_price"]
    tp        = pos["tp_price"]
    pnl_usd   = pos.get("pnl_usd", 0)
    pnl_pct   = pos.get("pnl_pct", 0)
    leverage  = pos.get("leverage", 3)

    # Construir sección de contexto de portafolio
    portfolio_section = ""
    if portfolio_context:
        total_pos = portfolio_context.get("total_positions", 1)
        port_pnl = portfolio_context.get("portfolio_pnl", 0)
        port_target = portfolio_context.get("portfolio_target", 2.0)
        others = portfolio_context.get("other_positions", [])
        
        portfolio_section = f"""
CONTEXTO DE PORTAFOLIO COORDINADO:
- Posiciones abiertas: {total_pos}
- P&L total del portafolio: ${port_pnl:.2f}
- Target de ganancia del portafolio: ${port_target:.2f}
- Progreso hacia target: {port_pnl/port_target*100:.0f}% {'✅' if port_pnl >= port_target else ''}
- Otras posiciones: {', '.join(f"{o['symbol']} {o['direction']} ${o['pnl_usd']:.2f}" for o in others) if others else 'ninguna'}

REGLA DE PORTAFOLIO (referencial):
- El target de ${port_target:.2f} es una guía a largo plazo, NO una razón para cerrar apresuradamente
- Con menos de 5 trades cerradas, el win rate histórico NO es estadísticamente significativo — ignóralo
- Una posición con P&L entre -1% y +1% tras menos de 1 hora es RUIDO NORMAL, no justifica cierre
"""

    _history_ctx = _get_trade_history_context(symbol)
    _funding = pos.get("funding_accumulated", 0)

    prompt = f"""Eres un gestor de riesgo experto en crypto. Analiza esta posición y decide: CLOSE, HOLD, REDUCE o TIGHTEN.

POSICIÓN ACTUAL:
- Symbol: {symbol}
- Dirección: {direction}
- Entrada: ${entry:.6f}
- Precio actual: ${current:.6f}
- Stop Loss: ${sl:.6f} ({quant['dist_sl_pct']:.2f}% de distancia)
- Take Profit: ${tp:.6f} ({quant['dist_tp_pct']:.2f}% de distancia)
- P&L actual: ${pnl_usd:.2f} ({pnl_pct:.2f}% sobre margen)
- Leverage: {leverage}x
- Horas abierta: {quant['hours_open']:.1f}h
- R/R restante: {quant['rr_remaining']:.2f}x
- Funding acumulado: ${_funding:.4f}

ANÁLISIS CUANTITATIVO:
- Score: {quant['score']}/100 (>50 = señal de cerrar)
- Factores: {', '.join(quant['reasons']) if quant['reasons'] else 'ninguno'}

CONTEXTO DE MERCADO:
- Tendencia: {research.get('trend', 'NEUTRAL')} (confianza: {research.get('confidence', 0.5):.0%})
- Fear & Greed: {market.get('fear_greed', {}).get('value', 50)}/100
{portfolio_section}
{_history_ctx}

REGLAS DE DECISIÓN:
1. CLOSE si: TP a <1% Y score>30, O SL a <0.5%, O R/R < 0.5x Y perdedor
2. CLOSE si: score >= 50 (señal cuantitativa fuerte de salida) Y confianza del mercado < 50%
3. HOLD si: score < 30, tendencia a favor, R/R > 1.0x, y no hay señal de reversión clara
4. REDUCE si: ganancia >8% pero aún tiene potencial — cerrar 50% y mover SL a entrada
5. TIGHTEN si: ganancia >5% y el trailing stop es muy ancho — ajustar trailing para proteger más profit
REGLA DE ORO: Score < 20 + P&L > -2% + <1h abierta = NO cerrar. Es ruido de mercado, no señal.

MODO DEVIL'S ADVOCATE:
Aunque tu instinto sea HOLD, argumenta brevemente (1 frase) por qué CLOSE podría ser correcto. PERO: si score < 20 y P&L no es catastrófico (<-3%), el argumento devil's advocate DEBE ser débil — no inventes razones para cerrar una posición sana.

Responde SOLO en JSON válido:
{{"action": "CLOSE|HOLD|REDUCE|TIGHTEN", "confidence": 0.0-1.0, "reasoning": "máx 2 oraciones", "trailing_pct": 0.005}}"""

    try:
        from lang_utils import lang_directive, get_user_language
        _user_lang = get_user_language()
        prompt += lang_directive(_user_lang)
    except Exception:
        _user_lang = 'es'  # Fail-safe

    system = "Eres un risk manager de crypto. Responde siempre con JSON válido únicamente, sin markdown."

    response = call_llm(prompt, system=system, max_tokens=4000)

    if not response:
        # Fallback cuantitativo si LLM falla — threshold más alto para evitar cierres excesivos
        if quant["score"] >= 65:
            action = "CLOSE"
        elif quant["score"] >= 40:
            action = "REDUCE"
        else:
            action = "HOLD"
        return {
            "action": action,
            "confidence": 0.5,
            "reasoning": f"Decisión cuantitativa (LLM no disponible). Score: {quant['score']}",
            "lang": "es",
            "source": "quantitative_fallback"
        }

    # Parse JSON response
    try:
        # Strip markdown if present
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        result = json.loads(text)
        result["source"] = "llm_claude_sonnet"
        result["lang"] = _user_lang
        return result
    except Exception:
        # Try extracting action from text
        action = "HOLD"
        if "CLOSE" in response.upper():
            action = "CLOSE"
        elif "TIGHTEN" in response.upper():
            action = "TIGHTEN"
        elif "REDUCE" in response.upper():
            action = "REDUCE"
        # v2.9.1: sanitizar reasoning si la respuesta parece error/log
        _lower = (response or "").lower()
        _error_markers = (
            "error:", "traceback", "unauthorized", "failed to refresh",
            "codex_login", "401 ", "403 ", "invalid_api_key",
        )
        _is_error = any(m in _lower for m in _error_markers)
        if _is_error:
            _clean_reason = f"LLM returned error response -- using quant fallback (action: {action})"
        else:
            _clean_reason = response[:200]
        return {
            "action": action,
            "confidence": 0.5,
            "reasoning": _clean_reason,
            "lang": _user_lang,
            "source": "llm_text_fallback"
        }


# v2.12.5-live position throttle: skip LLM eval si posición calm + última eval <3min
_POSITION_EVAL_THROTTLE_SEC = 180    # 3 min
_POSITION_CALM_PNL_BAND     = 0.005  # ±0.5% de pnl_pct fraccional
_POSITION_CALM_SLTP_DIST    = 0.03   # 3% de distancia a SL o TP


def _should_throttle_position_eval(pos: dict) -> tuple:
    """Returns (throttle: bool, cached_decision: dict|None).
    Skip LLM si: pnl_pct <0.5% AND distSL>3% AND distTP>3% AND lastDecision<180s."""
    try:
        import time as _t
        pnl_pct_frac = abs(float(pos.get('pnl_pct', 0) or 0) / 100.0)
        if pnl_pct_frac >= _POSITION_CALM_PNL_BAND:
            return (False, None)
        cp = float(pos.get('current_price', 0) or 0)
        sl = float(pos.get('sl_price', 0) or 0)
        tp = float(pos.get('tp_price', 0) or 0)
        if cp > 0 and sl > 0 and tp > 0:
            dist_sl = abs(cp - sl) / cp
            dist_tp = abs(cp - tp) / cp
            if dist_sl < _POSITION_CALM_SLTP_DIST or dist_tp < _POSITION_CALM_SLTP_DIST:
                return (False, None)
        # Check last decision age
        if not DECISIONS_FILE.exists():
            return (False, None)
        data = json.loads(DECISIONS_FILE.read_text())
        decisions = data.get('decisions', []) if isinstance(data, dict) else []
        pos_id = pos.get('id')
        from datetime import datetime
        for d in reversed(decisions):
            if d.get('position_id') != pos_id:
                continue
            ts = d.get('ts') or d.get('timestamp')
            if not ts:
                continue
            age = _t.time() - datetime.fromisoformat(str(ts).replace('Z','+00:00')).timestamp()
            if age < _POSITION_EVAL_THROTTLE_SEC:
                return (True, d)
            break
    except Exception:
        pass
    return (False, None)


def evaluate_position_decision(portfolio: dict, market: dict, research: dict) -> list:
    """
    Evalúa CADA posición abierta y genera una recomendación de acción.

    Para cada posición:
    1. Calcula score cuantitativo (distancias TP/SL, R/R, horas, tendencia)
    2. Consulta al LLM con todo el contexto
    3. Combina ambos para generar recomendación final

    Retorna lista de decisiones con:
        symbol, action (CLOSE/HOLD/REDUCE), confidence, score, reasoning
    """
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]

    if not open_positions:
        return []

    decisions = []

    # Construir contexto de portafolio para decisiones coordinadas
    portfolio_pnl = sum(p.get("pnl_usd", 0) for p in open_positions)
    portfolio_target = 2.0  # PORTFOLIO_TP_USD — hardcoded aquí, sync con executor.py

    for pos in open_positions:
        symbol = pos["symbol"]

        # v2.12.5-live: throttle — skip LLM si posición calm y reciente
        _throttle, _cached = _should_throttle_position_eval(pos)
        if _throttle and _cached:
            log.info(f"   💾 {symbol}: throttled (calm, last eval <{_POSITION_EVAL_THROTTLE_SEC}s) — reuse: {_cached.get('action','HOLD')}")
            decisions.append({
                'symbol': symbol,
                'action': _cached.get('action', 'HOLD'),
                'confidence': float(_cached.get('confidence', 0.6)),
                'score': _cached.get('score', 0),
                'reasoning': 'Throttled (position calm + recent eval): ' + str(_cached.get('reasoning', ''))[:100],
                'llm_reasoning': _cached.get('llm_reasoning', ''),
                'llm_reasoning_lang': _cached.get('llm_reasoning_lang', 'es'),
                'position_id': pos.get('id'),
                'source': 'throttle_cache',
            })
            continue
        log.info(f"   🧠 Evaluando {symbol}...")

        # Step 1: Quantitative analysis
        quant = _quant_score(pos, market, research)

        # Step 2: Build portfolio context for coordinated decisions
        other_positions = [
            {"symbol": p["symbol"], "direction": p["direction"],
             "pnl_usd": round(p.get("pnl_usd", 0), 2),
             "pnl_pct": round(p.get("pnl_pct", 0), 2)}
            for p in open_positions if p["symbol"] != symbol
        ]
        portfolio_context = {
            "total_positions": len(open_positions),
            "portfolio_pnl": round(portfolio_pnl, 2),
            "portfolio_target": portfolio_target,
            "other_positions": other_positions,
        }

        # Step 3: LLM recommendation with portfolio context
        llm_rec = _llm_decision(pos, quant, market, research, portfolio_context=portfolio_context)

        # Step 3: Combine (LLM has 60% weight, quant has 40%)
        # If both agree → high confidence final decision
        # If they disagree → use LLM but lower confidence
        quant_action = "CLOSE" if quant["score"] >= 50 else ("REDUCE" if quant["score"] >= 25 else "HOLD")
        llm_action   = llm_rec.get("action", "HOLD")

        if quant_action == llm_action:
            final_action     = llm_action
            final_confidence = min(1.0, llm_rec.get("confidence", 0.7) * 1.2)  # boost when aligned
            alignment        = "ALIGNED"
        else:
            # LLM wins but with reduced confidence
            final_action     = llm_action
            final_confidence = llm_rec.get("confidence", 0.6) * 0.8
            alignment        = f"SPLIT (quant={quant_action}, llm={llm_action})"

        decision = {
            "symbol":           symbol,
            "direction":        pos["direction"],
            "action":           final_action,
            "confidence":       round(final_confidence, 2),
            "alignment":        alignment,
            "quant_score":      quant["score"],
            "quant_action":     quant_action,
            "quant_reasons":    quant["reasons"],
            "llm_action":       llm_action,
            "llm_confidence":   llm_rec.get("confidence", 0),
            "llm_reasoning":    llm_rec.get("reasoning", ""),
            "llm_reasoning_lang": llm_rec.get("lang", "es"),
            "llm_source":       llm_rec.get("source", "unknown"),
            "pnl_usd":          pos.get("pnl_usd", 0),
            "pnl_pct":          pos.get("pnl_pct", 0),
            "dist_tp_pct":      quant["dist_tp_pct"],
            "dist_sl_pct":      quant["dist_sl_pct"],
            "rr_remaining":     quant["rr_remaining"],
            "hours_open":       quant["hours_open"],
            "evaluated_at":     datetime.now(timezone.utc).isoformat(),
        }
        decisions.append(decision)

        icon = "🔴" if final_action == "CLOSE" else ("🟡" if final_action == "REDUCE" else "🟢")
        log.info(f"   {icon} {symbol}: {final_action} (conf {final_confidence:.0%}) — {llm_rec.get('reasoning', '')[:80]}")

    # Save decisions
    DECISIONS_FILE.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decisions": decisions
    }, indent=2))

    # v2.13.6: LLM audit trail — track every LLM decision for post-hoc analysis
    try:
        _audit_path = DATA_DIR / "llm_audit.json"
        _audit = {"version": "1.0", "entries": [], "summary": {}}
        if _audit_path.exists():
            _audit = json.loads(_audit_path.read_text())
        _audit.setdefault("entries", [])
        _audit.setdefault("summary", {
            "total_evaluations": 0,
            "llm_recommended_close": 0,
            "llm_recommended_reduce": 0,
            "llm_recommended_tighten": 0,
            "llm_recommended_hold": 0,
            "executed_close": 0,
            "executed_reduce": 0,
            "executed_tighten": 0,
            "avg_pnl_when_llm_said_close": 0.0,
            "avg_pnl_when_llm_said_hold": 0.0,
        })
        _now = datetime.now(timezone.utc).isoformat()
        for d in decisions:
            _entry = {
                "evaluated_at": _now,
                "symbol": d["symbol"],
                "position_id": d.get("position_id"),
                "llm_action": d.get("llm_action", "HOLD"),
                "llm_confidence": d.get("llm_confidence", 0),
                "quant_action": d.get("quant_action", "HOLD"),
                "quant_score": d.get("quant_score", 0),
                "final_action": d["action"],
                "pnl_usd_at_eval": d.get("pnl_usd", 0),
                "pnl_pct_at_eval": d.get("pnl_pct", 0),
                "hours_open": d.get("hours_open", 0),
                "llm_reasoning": d.get("llm_reasoning", "")[:200],
                "executed": False,
                "close_reason": None,
                "final_pnl_usd": None,
                "final_pnl_pct": None,
            }
            _audit["entries"].append(_entry)
            _audit["summary"]["total_evaluations"] += 1
            _act = d.get("llm_action", "HOLD")
            if _act == "CLOSE":
                _audit["summary"]["llm_recommended_close"] += 1
            elif _act == "REDUCE":
                _audit["summary"]["llm_recommended_reduce"] += 1
            elif _act == "TIGHTEN":
                _audit["summary"]["llm_recommended_tighten"] += 1
            else:
                _audit["summary"]["llm_recommended_hold"] += 1
        # Cap entries to 5000 to prevent unbounded growth
        if len(_audit["entries"]) > 5000:
            _audit["entries"] = _audit["entries"][-5000:]
        _audit_path.write_text(json.dumps(_audit, indent=2))
    except Exception as _e:
        log.debug(f"llm_audit write failed: {_e}")

    return decisions


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

    # Equity real = capital libre + margen invertido en posiciones + P&L abierto
    # IMPORTANTE: usar margin_usd (capital real del trader), NO size_usd (notional = margen × leverage)
    invested   = sum(p.get("margin_usd", 0) for p in open_positions)
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

    # Cargar indicadores técnicos para vol-adaptive sizing
    indicators = {}
    signals_file = DATA_DIR / "signals_latest.json"
    try:
        if signals_file.exists():
            sig_data = json.loads(signals_file.read_text())
            indicators = sig_data.get("indicator_summary", {})
    except Exception:
        pass

    # Evaluar cada token
    tokens = market.get("tokens", {})
    evaluations = {}
    approved_count = 0

    for symbol, token_data in tokens.items():
        # Obtener ATR% del token para vol-adaptive sizing
        token_atr_pct = None
        if symbol in indicators:
            token_atr_pct = indicators[symbol].get("atr_pct")

        eval_result = evaluate_token(
            symbol, token_data, capital, open_positions, portfolio_status,
            atr_pct=token_atr_pct
        )
        evaluations[symbol] = eval_result

        if eval_result["approved"]:
            approved_count += 1
            if debug:
                sz = eval_result["position_size"]
                vol_info = f" | Vol×{sz.get('vol_multiplier', 1.0)}" if sz.get('vol_multiplier', 1.0) != 1.0 else ""
                log.info(f"  ✅ {symbol}: Margin ${sz['margin_usd']:.2f} | {sz['leverage']}x → Notional ${sz['notional_usd']:.2f} | SL: {sz['sl_pct']*100:.1f}% | TP: {sz['tp_pct']*100:.1f}%{vol_info}")
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
    parser.add_argument("--debug",  action="store_true", help="Output detallado por token")
    parser.add_argument("--decide", action="store_true", help="Evalúa todas las posiciones abiertas con LLM")
    args = parser.parse_args()

    if args.decide:
        log.info("=" * 50)
        log.info("🧠 RISK MANAGER — Evaluando decisiones de posiciones")
        log.info("=" * 50)
        portfolio = load_portfolio()
        market    = load_market()
        research  = load_research()
        decisions = evaluate_position_decision(portfolio, market, research)

        if not decisions:
            log.info("ℹ️  No hay posiciones abiertas para evaluar")
        else:
            print("\n" + "=" * 60)
            print("📋 RECOMENDACIONES DE POSICIONES")
            print("=" * 60)
            for d in decisions:
                icon = "🔴" if d["action"] == "CLOSE" else ("🟡" if d["action"] == "REDUCE" else "🟢")
                print(f"\n{icon} {d['symbol']} {d['direction'].upper()}")
                print(f"   Acción:     {d['action']} (confianza: {d['confidence']:.0%})")
                print(f"   Score:      {d['quant_score']}/100 | R/R restante: {d['rr_remaining']:.2f}x")
                print(f"   TP dist:    {d['dist_tp_pct']:.2f}% | SL dist: {d['dist_sl_pct']:.2f}%")
                print(f"   P&L:        ${d['pnl_usd']:.2f} ({d['pnl_pct']:.2f}%)")
                print(f"   Horas:      {d['hours_open']:.1f}h")
                print(f"   Razones:    {', '.join(d['quant_reasons']) if d['quant_reasons'] else 'ninguna'}")
                print(f"   LLM:        {d['llm_reasoning']}")
                print(f"   Alineación: {d['alignment']}")
            print(f"\n💾 Guardado en: {DECISIONS_FILE}")
    else:
        run(debug=args.debug)

    sys.exit(0)
