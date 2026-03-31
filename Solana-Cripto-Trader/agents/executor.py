#!/usr/bin/env python3
"""
⚡ Agente 4: Executor
Abre y cierra posiciones según señales del Strategy Agent.
Persiste estado real en portfolio.json y trade_history.json.

MODOS:
  - safe=True  → Paper trading (precio real Jupiter, sin wallet)
  - safe=False → Trades reales via Jupiter Swap API (requiere keypair)

Uso:
    python3 executor.py          # Paper trading (default)
    python3 executor.py --live   # Modo real (requiere configuración de wallet)
    python3 executor.py --debug  # Con output detallado
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

try:
    from compound_engine import update_compound_capital
    _COMPOUND_ENABLED = True
except ImportError:
    _COMPOUND_ENABLED = False

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SIGNALS_FILE       = DATA_DIR / "signals_latest.json"
SIGNALS_LLM_FILE   = DATA_DIR / "strategy_llm.json"
MARKET_FILE        = DATA_DIR / "market_latest.json"
RISK_FILE          = DATA_DIR / "risk_report.json"
PORTFOLIO_FILE     = DATA_DIR / "portfolio.json"
HISTORY_FILE       = DATA_DIR / "trade_history.json"

# Máxima antigüedad de señales LLM para considerarlas válidas (en segundos)
LLM_SIGNALS_MAX_AGE_SEC = 600  # 10 minutos (5 ciclos de 2min)

# .env del proyecto para Telegram/wallet
ENV_FILE = Path(__file__).parent.parent / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("executor")

# ─── Parámetros ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 500.0   # Capital paper inicial
PAPER_MODE      = True    # Cambia a False para trades reales

# ─── Risk Management (ajustado 31-Mar-2026 — orden de Ender) ─────────────────
MIN_CONFIDENCE      = 0.70     # Bajado de 0.85 para aprovechar más señales en extremos (2026-03-31)
BLOCK_LONGS_FG      = 35       # Bloquear LONGs si Fear & Greed < 35 (mercado bajista)
MAX_TRADES_PER_DAY  = 0        # 0 = sin límite

# ─── Regla de Ender (31-Mar-2026): $4 min profit, $2 max risk ────────────
MAX_RISK_USD        = 2.00     # Máximo $2 de riesgo por POSICIÓN individual (pérdida en SL + fees)

# ─── Portfolio Take Profit (31-Mar-2026) — orden de Ender ─────────────────
PORTFOLIO_TP_USD     = 2.00     # Cerrar todo si el P&L combinado llega a $2
PORTFOLIO_MIN_USD    = 1.00     # Si está en $1+ pero AI duda, cerrar y asegurar $1

# ─── Coordinated Portfolio Sizing (31-Mar-2026 v2) — orden de Ender ───────
# El sizing se calcula a nivel de PORTAFOLIO, no por posición aislada.
# La IA decide cuántas posiciones abrir (N), y el sistema distribuye el
# target de ganancias entre todas ellas, ajustando por volatilidad.
# Cuando el P&L combinado llega al PORTFOLIO_TP → cerrar TODO.
PORTFOLIO_MAX_RISK_USD = 10.00  # Riesgo total máximo del portafolio (todas las posiciones)
MIN_PROFIT_PER_POS_USD = 0.50   # Profit mínimo por posición ($0.50, flexible según N)

# ─── Drift Protocol Simulation ───────────────────────────────────────────────
TAKER_FEE           = 0.001    # 0.1% taker fee (Drift Protocol)
MAKER_FEE           = 0.001    # 0.1% maker fee (Drift Protocol)
DEFAULT_LEVERAGE    = 3        # 3x leverage por defecto
MAX_LEVERAGE        = 10       # Máximo 10x
MAINTENANCE_MARGIN  = 0.05     # 5% margen de mantenimiento
FUNDING_RATE        = 0.0001   # 0.01% por hora (funding rate simulado)

# ─── Carga / Guardado de Estado ──────────────────────────────────────────────

def load_portfolio() -> dict:
    """Carga portfolio desde disco. Crea uno nuevo si no existe."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "capital_usd": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "positions": [],
        "status": "ACTIVE",
        "mode": "paper",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
    }


def save_portfolio(portfolio: dict):
    portfolio["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def load_history() -> list:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_signals() -> dict:
    """
    Carga señales priorizando AI Strategy (strategy_llm.json) sobre técnico (signals_latest.json).
    
    Lógica:
    1. Si strategy_llm.json existe y tiene señales recientes (< 10 min) → usar señales IA
    2. Fallback → signals_latest.json (señales técnicas de strategy.py)
    """
    # Intentar cargar señales del AI Strategy primero
    if SIGNALS_LLM_FILE.exists():
        try:
            with open(SIGNALS_LLM_FILE) as f:
                llm_data = json.load(f)
            
            # Verificar antigüedad del archivo
            generated_at = llm_data.get("generated_at", "")
            if generated_at:
                try:
                    ts = datetime.fromisoformat(generated_at)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
                    
                    llm_signals = llm_data.get("signals", [])
                    # Solo usar señales LLM si son recientes y hay señales válidas
                    if age_sec <= LLM_SIGNALS_MAX_AGE_SEC and llm_signals:
                        # Normalizar formato: ai_strategy usa sl_price/tp_price directamente
                        # pero necesita direction != "none"
                        valid_llm_signals = [
                            s for s in llm_signals
                            if s.get("direction", "none") not in ("none", "")
                        ]
                        if valid_llm_signals:
                            log.info(f"🤖 Usando señales AI Strategy ({len(valid_llm_signals)} señales, {age_sec:.0f}s de antigüedad)")
                            return {"signals": valid_llm_signals, "source": "ai_strategy"}
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            log.warning(f"⚠️ Error leyendo strategy_llm.json: {e}")
    
    # Fallback: señales técnicas de strategy.py
    if not SIGNALS_FILE.exists():
        return {"signals": [], "source": "none"}
    
    log.info("📊 Usando señales técnicas (signals_latest.json) — fallback")
    with open(SIGNALS_FILE) as f:
        data = json.load(f)
    data["source"] = "technical"
    return data


def load_market() -> dict:
    if not MARKET_FILE.exists():
        return {"tokens": {}}
    with open(MARKET_FILE) as f:
        return json.load(f)


def get_current_price(symbol: str, market: dict) -> float:
    """Obtiene precio actual del snapshot de mercado."""
    token_data = market.get("tokens", {}).get(symbol, {})
    return float(token_data.get("price", 0))


def load_risk_report() -> dict:
    """Carga el reporte del Risk Manager."""
    if RISK_FILE.exists():
        with open(RISK_FILE) as f:
            return json.load(f)
    return {}


def close_positions_emergency(portfolio: dict, symbols: list, market: dict, history: list) -> list:
    """
    Cierra posiciones por emergencia (sin importar SL/TP).
    Usado cuando Risk Manager detecta condiciones extremas.
    """
    closed = []
    now = datetime.now(timezone.utc).isoformat()

    for pos in portfolio["positions"]:
        if pos.get("status") != "open":
            continue

        if pos["symbol"] in symbols:
            # Actualizar precio actual
            pos["current_price"] = get_current_price(pos["symbol"], market)

            pos["status"] = "closed"
            pos["close_time"] = now
            pos["close_reason"] = "EMERGENCY_CLOSE"
            pos["close_price"] = pos["current_price"]

            # Actualizar P&L con leverage
            notional = pos.get("notional_value", pos.get("size_usd", 0))
            margin = pos.get("margin_usd", pos.get("size_usd", 0))

            pnl_pct = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"]
            if pos["direction"] == "short":
                pnl_pct = -pnl_pct
            pnl_usd = notional * pnl_pct + pos.get("funding_accumulated", 0)

            pos["pnl_pct"] = round(pnl_pct * 100, 4)
            pos["pnl_usd"] = round(pnl_usd, 4)

            # Actualizar stats del portfolio
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            if pnl_usd > 0:
                portfolio["wins"] = portfolio.get("wins", 0) + 1
            else:
                portfolio["losses"] = portfolio.get("losses", 0) + 1

            # Devolver margen + P&L al portfolio
            returned = max(0, margin + pnl_usd)
            portfolio["capital_usd"] += returned

            # Agregar al historial
            history.append({
                "id": pos["id"],
                "symbol": pos["symbol"],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": pos["close_price"],
                "size_usd": pos["size_usd"],
                "pnl_usd": pos["pnl_usd"],
                "pnl_pct": pos["pnl_pct"],
                "open_time": pos["open_time"],
                "close_time": pos["close_time"],
                "close_reason": "EMERGENCY_CLOSE",
                "strategy": pos.get("strategy", "unknown"),
            })

            closed.append(pos)
            log.error(f"🚨 EMERGENCY CLOSE: {pos['symbol']} | P&L: ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%)")

    return closed


# ─── Paper Trading ────────────────────────────────────────────────────────────

def get_fear_greed_index() -> int:
    """Obtiene el Fear & Greed index actual (0-100)."""
    try:
        fg_file = DATA_DIR / "market_latest.json"
        if fg_file.exists():
            with open(fg_file) as f:
                data = json.load(f)
                return int(data.get("fear_greed", {}).get("value", 50))
    except Exception:
        pass
    return 50  # Neutral por defecto


def paper_open_position(signal: dict, portfolio: dict, market: dict) -> Optional[dict]:
    """
    Abre una posición en paper trading simulando Drift Protocol.
    - Leverage configurable (default 3x, max 10x)
    - Solo se descuenta el margen del capital
    - Calcula precio de liquidación
    """
    symbol = signal["symbol"]
    direction = signal.get("direction", "")
    confidence = signal.get("confidence", 0)
    
    # ─── Risk Filters (27-Mar-2026) ───────────────────────────────────────────
    # 1. Filtrar por confidence mínima
    if confidence < MIN_CONFIDENCE:
        log.info(f"⏭️  Señal {symbol} ignorada: confidence {confidence:.2f} < {MIN_CONFIDENCE}")
        return None
    
    # 2. Bloquear LONGs en mercado bajista (Fear & Greed < threshold)
    if direction == "long":
        fear_greed = get_fear_greed_index()
        if fear_greed < BLOCK_LONGS_FG:
            log.info(f"⏭️  LONG {symbol} bloqueado: Fear & Greed {fear_greed} < {BLOCK_LONGS_FG} (mercado bajista)")
            return None
    
    price = get_current_price(symbol, market)

    if price <= 0:
        log.warning(f"⚠️  Sin precio para {symbol}, no se puede abrir posición")
        return None

    # Determinar leverage (signal puede sugerirlo, si no, default)
    leverage = signal.get("leverage", DEFAULT_LEVERAGE)
    leverage = max(1, min(leverage, MAX_LEVERAGE))

    # ─── Position Sizing basado en Regla de Ender ─────────────────────────
    # Regla: ganar mín $4 neto después de comisiones, arriesgar máx $2
    #
    # Cálculo:
    #   max_risk = notional * sl_pct + notional * fee * 2  ≤  MAX_RISK_USD
    #   min_profit = notional * tp_pct - notional * fee * 2  ≥  MIN_NET_PROFIT_USD
    #
    # De max_risk: notional ≤ MAX_RISK_USD / (sl_pct + fee*2)
    # De min_profit: notional ≥ MIN_NET_PROFIT_USD / (tp_pct - fee*2)

    # SL/TP: usar los del signal si existen, sino defaults
    sl_pct = 0.02   # 2% SL (ajustado para regla de Ender)
    tp_pct = 0.05   # 5% TP (2.5:1 R:R → $1.09 profit, $0.50 risk)

    if signal.get("sl_price", 0) > 0 and signal.get("tp_price", 0) > 0:
        if signal["direction"] == "long":
            sl_pct = abs(price - signal["sl_price"]) / price
            tp_pct = abs(signal["tp_price"] - price) / price
        else:
            sl_pct = abs(signal["sl_price"] - price) / price
            tp_pct = abs(price - signal["tp_price"]) / price

    # Asegurar R:R mínimo de 2:1 para cumplir la regla
    if tp_pct < sl_pct * 2:
        tp_pct = sl_pct * 2.5  # Forzar 2.5:1 si el signal no lo da
        log.info(f"📐 TP ajustado a {tp_pct*100:.1f}% para mantener R:R ≥ 2.5:1")

    fee_round_trip = TAKER_FEE * 2  # Entry + exit fees

    # ─── Volatility-Adaptive + Coordinated Sizing (2026-03-31 v2) ────────
    # El sistema coordina el sizing de TODAS las posiciones del portafolio:
    # 1. La IA decide cuántas posiciones abrir (N)
    # 2. El profit target se distribuye: target_por_pos = PORTFOLIO_TP / N
    # 3. El notional se calcula para alcanzar ese target, ajustado por ATR
    # 4. Portfolio TP cierra TODO cuando se alcanza el target combinado
    REFERENCE_VOL = 0.025    # 2.5% = volatilidad "normal" de referencia
    VOL_MULT_MIN  = 0.5      # Alta vol → posición más chica
    VOL_MULT_MAX  = 3.0      # Baja vol → posición hasta 3x más grande
    VOL_ATR_FLOOR = 0.003    # Piso para evitar extremos (0.3%)

    # Obtener ATR% del token desde signals_latest.json
    token_atr_pct = None
    try:
        signals_file = DATA_DIR / "signals_latest.json"
        if signals_file.exists():
            sig_data = json.loads(signals_file.read_text())
            ind_summary = sig_data.get("indicator_summary", {})
            token_atr_pct = ind_summary.get(symbol, {}).get("atr_pct")
    except Exception:
        pass

    vol_factor = 1.0
    if token_atr_pct and token_atr_pct > 0:
        effective_atr = max(token_atr_pct / 100.0, VOL_ATR_FLOOR)
        vol_factor = REFERENCE_VOL / effective_atr
        vol_factor = max(VOL_MULT_MIN, min(vol_factor, VOL_MULT_MAX))

    # Usar sizing coordinado si viene del orchestrator
    coordinated_risk = signal.get("_coordinated_risk", MAX_RISK_USD)
    coordinated_profit = signal.get("_coordinated_profit_target", PORTFOLIO_TP_USD)
    n_positions = signal.get("_coordinated_n", 1)

    # Aplicar factor de volatilidad al riesgo coordinado
    adjusted_risk = coordinated_risk * vol_factor
    # Hard cap: adjusted_risk × N no puede exceder PORTFOLIO_MAX_RISK_USD
    max_adj_per_pos = PORTFOLIO_MAX_RISK_USD / max(n_positions, 1)
    adjusted_risk = min(adjusted_risk, max_adj_per_pos)

    log.info(f"📊 Coordinated sizing: {n_positions} pos | risk=${coordinated_risk:.2f}×vol{vol_factor:.1f}=${adjusted_risk:.2f} | target=${coordinated_profit:.2f}/pos"
             + (f" | ATR={token_atr_pct:.2f}%" if token_atr_pct else ""))

    # Calcular notional óptimo
    max_notional_by_risk = adjusted_risk / (sl_pct + fee_round_trip)
    min_notional_by_profit = coordinated_profit / (tp_pct - fee_round_trip)

    if min_notional_by_profit > max_notional_by_risk:
        # Imposible cumplir ambas reglas → ajustar TP para que funcione
        needed_tp = (coordinated_profit / max_notional_by_risk) + fee_round_trip
        log.info(f"📐 Ajustando TP de {tp_pct*100:.1f}% a {needed_tp*100:.1f}% para cumplir regla coordinada")
        tp_pct = needed_tp

    notional_value = max_notional_by_risk  # Usar max permitido por riesgo
    margin_usd = notional_value / leverage

    # Verificar capital suficiente
    if portfolio["capital_usd"] < margin_usd:
        log.warning(f"⚠️  Capital insuficiente: ${portfolio['capital_usd']:.2f} < ${margin_usd:.2f}")
        return None

    # Verificar que el trade cumple la regla antes de ejecutar
    expected_profit = notional_value * tp_pct - notional_value * fee_round_trip
    expected_risk = notional_value * sl_pct + notional_value * fee_round_trip
    log.info(f"💰 Sizing: notional=${notional_value:.1f} margin=${margin_usd:.1f} SL={sl_pct*100:.1f}% TP={tp_pct*100:.1f}%")
    log.info(f"   Expected profit: ${expected_profit:.2f} | Max risk: ${expected_risk:.2f} | R:R 1:{expected_profit/expected_risk:.1f}")

    if expected_profit < coordinated_profit - 0.01:  # epsilon para rounding
        log.warning(f"⚠️  Trade rechazado: profit esperado ${expected_profit:.2f} < ${coordinated_profit:.2f}")
        return None

    # Fees sobre el notional
    fee_entry = notional_value * TAKER_FEE
    tokens = (notional_value - fee_entry) / price

    # Margen de mantenimiento
    margin_maintenance = notional_value * MAINTENANCE_MARGIN

    # Precio de liquidación
    if signal["direction"] == "long":
        liq_price = price * (1 - (margin_usd - fee_entry) / notional_value)
    else:
        liq_price = price * (1 + (margin_usd - fee_entry) / notional_value)

    # Calcular SL/TP prices
    sl_price = price * (1 - sl_pct) if signal["direction"] == "long" else price * (1 + sl_pct)
    tp_price = price * (1 + tp_pct) if signal["direction"] == "long" else price * (1 - tp_pct)

    position = {
        "id": f"{symbol}_{int(time.time())}",
        "symbol": symbol,
        "direction": signal["direction"],
        "strategy": signal["strategy"],
        "entry_price": round(price, 8),
        "current_price": round(price, 8),
        "margin_usd": round(margin_usd, 2),          # Lo que puso el trader
        "notional_value": round(notional_value, 2),   # Tamaño real de la posición
        "leverage": leverage,
        "size_usd": round(notional_value, 2),         # Compat: = notional
        "tokens": round(tokens, 6),
        "sl_price": round(sl_price, 8),
        "tp_price": round(tp_price, 8),
        "liquidation_price": round(liq_price, 8),
        "margin_maintenance": round(margin_maintenance, 4),
        "fee_entry": round(fee_entry, 4),
        "funding_accumulated": 0.0,
        "pnl_usd": 0.0,
        "pnl_pct": 0.0,
        "status": "open",
        "open_time": datetime.now(timezone.utc).isoformat(),
        "close_time": None,
        "mode": "paper",
        "confidence": signal.get("confidence", 0),
        "last_funding_time": datetime.now(timezone.utc).isoformat(),
        # Trailing stop support
        "exit_mode": signal.get("exit_mode", "fixed"),  # "fixed" or "trailing"
        "trailing_pct": signal.get("trailing_pct", 0.0),  # e.g. 0.02 = 2% pullback
        "peak_price": round(price, 8),  # highest price seen (for long) / lowest (for short)
        "trailing_sl": 0.0,  # dynamic SL that follows the peak
    }

    # Descontar solo el MARGEN del capital (no el notional completo)
    portfolio["capital_usd"] = round(portfolio["capital_usd"] - margin_usd, 2)
    portfolio["positions"].append(position)

    log.info(f"    📐 Leverage: {leverage}x | Margen: ${margin_usd:.2f} | Notional: ${notional_value:.2f} | Liq: ${liq_price:.6f}")

    return position


def apply_funding_rate(pos: dict) -> float:
    """
    Simula el funding rate de Drift Protocol.
    Se aplica cada hora sobre el notional value.
    Longs pagan cuando funding > 0, shorts pagan cuando funding < 0.
    Retorna el monto de funding aplicado (negativo = pagado, positivo = recibido).
    """
    now = datetime.now(timezone.utc)
    last_funding_str = pos.get("last_funding_time", pos.get("open_time"))
    try:
        last_funding = datetime.fromisoformat(last_funding_str)
        if last_funding.tzinfo is None:
            last_funding = last_funding.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        last_funding = now

    # Calcular horas transcurridas desde último funding
    hours_elapsed = (now - last_funding).total_seconds() / 3600.0
    if hours_elapsed < 1.0:
        return 0.0

    # Número de períodos de funding a aplicar
    periods = int(hours_elapsed)
    notional = pos.get("notional_value", pos.get("size_usd", 0))

    # Funding rate simulado: longs pagan, shorts reciben (mercado normal)
    # En mercado bajista sería al revés, pero simplificamos con rate fijo
    funding_per_period = notional * FUNDING_RATE
    total_funding = funding_per_period * periods

    # Longs pagan funding (negativo), shorts reciben (positivo)
    if pos["direction"] == "long":
        funding_impact = -total_funding
    else:
        funding_impact = total_funding

    pos["funding_accumulated"] = round(pos.get("funding_accumulated", 0) + funding_impact, 6)
    pos["last_funding_time"] = now.isoformat()

    if abs(funding_impact) > 0.001:
        direction_text = "pagó" if funding_impact < 0 else "recibió"
        log.info(f"    💸 {pos['symbol']} {direction_text} ${abs(funding_impact):.4f} funding ({periods}h)")

    return funding_impact


def paper_update_positions(portfolio: dict, market: dict, history: list) -> list:
    """
    Actualiza P&L de todas las posiciones abiertas.
    Simula Drift Protocol: leverage, liquidación, funding rate.
    Cierra automáticamente si toca SL, TP o precio de liquidación.
    """
    closed = []
    remaining = []

    for pos in portfolio["positions"]:
        if pos["status"] != "open":
            remaining.append(pos)
            continue

        symbol = pos["symbol"]
        current_price = get_current_price(symbol, market)

        if current_price <= 0:
            remaining.append(pos)
            continue

        # Actualizar precio actual
        pos["current_price"] = round(current_price, 8)

        # Aplicar funding rate
        funding_impact = apply_funding_rate(pos)

        # Calcular P&L basado en NOTIONAL (leverage amplifica ganancias Y pérdidas)
        notional = pos.get("notional_value", pos.get("size_usd", 0))
        margin = pos.get("margin_usd", pos.get("size_usd", 0))
        leverage = pos.get("leverage", 1)

        if pos["direction"] == "long":
            price_pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        else:
            price_pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

        # P&L = movimiento de precio * notional + funding acumulado
        pnl_from_price = notional * price_pnl_pct
        funding_total = pos.get("funding_accumulated", 0)
        pnl_usd = pnl_from_price + funding_total

        # P&L % relativo al MARGEN (no al notional) — refleja retorno real del trader
        pnl_pct_on_margin = (pnl_usd / margin * 100) if margin > 0 else 0

        pos["pnl_usd"] = round(pnl_usd, 4)
        pos["pnl_pct"] = round(pnl_pct_on_margin, 4)

        # ─── Verificar LIQUIDACIÓN (Drift Protocol) ─────────────────────
        liq_price = pos.get("liquidation_price", 0)
        hit_liquidation = False

        if liq_price > 0:
            if pos["direction"] == "long":
                hit_liquidation = current_price <= liq_price
            else:
                hit_liquidation = current_price >= liq_price

        if hit_liquidation:
            pos["status"] = "closed"
            pos["close_time"] = datetime.now(timezone.utc).isoformat()
            pos["close_reason"] = "LIQUIDATED"
            pos["close_price"] = current_price

            # En liquidación se pierde todo el margen
            pos["pnl_usd"] = round(-margin, 4)
            pos["pnl_pct"] = -100.0
            pos["fee_exit"] = 0.0

            # No se devuelve nada — margen perdido completamente
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            portfolio["losses"] = portfolio.get("losses", 0) + 1

            log.error(f"  💀 [LIQUIDATED] {symbol} {leverage}x {pos['direction']} | "
                      f"Perdido: ${margin:.2f} margen completo | Liq: ${liq_price:.6f}")

            history.append({**pos})
            closed.append(pos)
            continue

        # ─── Trailing Stop Logic ─────────────────────────────────────────
        exit_mode = pos.get("exit_mode", "fixed")
        trailing_pct = pos.get("trailing_pct", 0.0)

        if exit_mode == "trailing" and trailing_pct > 0:
            if pos["direction"] == "long":
                # Update peak if price made new high
                if current_price > pos.get("peak_price", pos["entry_price"]):
                    pos["peak_price"] = round(current_price, 8)
                # Trailing SL follows peak downward
                pos["trailing_sl"] = round(pos["peak_price"] * (1 - trailing_pct), 8)
            else:
                # Short: update peak (lowest) if price made new low
                if current_price < pos.get("peak_price", pos["entry_price"]):
                    pos["peak_price"] = round(current_price, 8)
                pos["trailing_sl"] = round(pos["peak_price"] * (1 + trailing_pct), 8)

        # ─── Verificar SL/TP ─────────────────────────────────────────────
        hit_sl = False
        hit_tp = False
        hit_trailing = False

        if pos["direction"] == "long":
            hit_sl = current_price <= pos["sl_price"]
            hit_tp = current_price >= pos["tp_price"]
            # Trailing SL only activates after price has moved past entry (in profit)
            if exit_mode == "trailing" and pos.get("trailing_sl", 0) > pos["entry_price"]:
                hit_trailing = current_price <= pos["trailing_sl"]
        else:
            hit_sl = current_price >= pos["sl_price"]
            hit_tp = current_price <= pos["tp_price"]
            if exit_mode == "trailing" and pos.get("trailing_sl", 0) < pos["entry_price"]:
                hit_trailing = current_price >= pos["trailing_sl"]

        if hit_sl or hit_tp or hit_trailing:
            if hit_trailing:
                close_reason = "TRAILING_SL"
            elif hit_tp:
                close_reason = "TP"
            else:
                close_reason = "SL"
            pos["status"] = "closed"
            pos["close_time"] = datetime.now(timezone.utc).isoformat()
            pos["close_reason"] = close_reason
            pos["close_price"] = current_price

            # Fee de salida sobre el notional
            fee_exit = notional * TAKER_FEE
            pos["fee_exit"] = round(fee_exit, 4)

            # Devolver margen + P&L - fees
            returned = margin + pnl_usd - fee_exit
            returned = max(0, returned)  # No puede ser negativo (ya se descontó margen)
            portfolio["capital_usd"] = round(portfolio["capital_usd"] + returned, 2)

            # Estadísticas
            net_pnl = pnl_usd - fee_exit
            is_win = net_pnl > 0
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            if is_win:
                portfolio["wins"] = portfolio.get("wins", 0) + 1
            else:
                portfolio["losses"] = portfolio.get("losses", 0) + 1

            result_emoji = "✅" if is_win else "❌"
            log.info(f"  {result_emoji} [{close_reason}] {symbol} {leverage}x {pos['direction']} | "
                     f"P&L: ${net_pnl:+.2f} ({pnl_pct_on_margin:+.1f}% on margin) | "
                     f"Funding: ${funding_total:+.4f}")

            history.append({**pos})
            closed.append(pos)

            # 📈 Compound Engine: actualizar capital base tras cada cierre
            if _COMPOUND_ENABLED:
                try:
                    state = update_compound_capital(net_pnl)
                    log.info(f"   📈 Compound: capital_base=${state['capital_base']:.2f} | "
                             f"return={state['total_return_pct']:.2f}% | "
                             f"cycles={state['compound_cycles']}")
                except Exception as e:
                    log.warning(f"   ⚠️ Compound update error: {e}")
        else:
            remaining.append(pos)

    portfolio["positions"] = remaining
    return closed


# ─── Ejecución Real (Jupiter Swap API) ───────────────────────────────────────

def real_open_position(signal: dict, portfolio: dict) -> Optional[dict]:
    """
    Ejecuta trade real via Jupiter Swap API.
    REQUIERE: keypair configurado en .env (HOT_WALLET_PRIVATE_KEY)
    """
    log.warning("⚠️  Modo LIVE no implementado aún — activar manualmente")
    log.warning("   Para activar trades reales, configura HOT_WALLET_PRIVATE_KEY en .env")
    log.warning("   y revisa MAINNET_GUIDE.md para el proceso seguro")
    return None


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(safe: bool = True, debug: bool = False) -> dict:
    log.info("=" * 50)
    mode_label = "📄 PAPER" if safe else "🔴 LIVE"
    log.info(f"⚡ EXECUTOR — {mode_label} MODE")
    log.info("=" * 50)

    portfolio = load_portfolio()
    history = load_history()
    signals_data = load_signals()
    market = load_market()
    risk_report = load_risk_report()

    signals = signals_data.get("signals", [])
    portfolio_status = portfolio.get("status", "ACTIVE")

    # Chequear emergency close del Risk Manager
    emergency_close = risk_report.get("emergency_close", {})
    if emergency_close.get("triggered", False):
        symbols_to_close = emergency_close.get("symbols", [])
        reason = emergency_close.get("reason", "")
        log.warning(f"⚠️ RISK MANAGER solicitó emergency close: {reason}")
        emergency_closed = close_positions_emergency(portfolio, symbols_to_close, market, history)
        if emergency_closed:
            log.error(f"🚨 {len(emergency_closed)} posiciones cerradas por emergencia")

    # Actualizar precios y cerrar posiciones que tocaron SL/TP
    open_before = len([p for p in portfolio["positions"] if p.get("status") == "open"])
    closed_this_cycle = paper_update_positions(portfolio, market, history)
    open_after = len([p for p in portfolio["positions"] if p.get("status") == "open"])

    if closed_this_cycle:
        log.info(f"🔒 {len(closed_this_cycle)} posición(es) cerrada(s) este ciclo")

    # Estado del sistema
    if portfolio_status in ("PAUSED", "STOPPED"):
        log.warning(f"⛔ Sistema {portfolio_status} — no se abren nuevas posiciones")
        save_portfolio(portfolio)
        save_history(history)
        return {
            "status": portfolio_status,
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # ── Safety Nets ──────────────────────────────────────────────────────
    # Kill switch: if STOP_TRADING file exists, don't open new positions
    STOP_FILE = DATA_DIR / "STOP_TRADING"
    if STOP_FILE.exists():
        log.warning("🛑 KILL SWITCH ACTIVE — STOP_TRADING file detected. No new positions.")
        _save_portfolio(portfolio)
        return {
            "status": "kill_switch_active",
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # Max daily loss: if realized P&L today < -$25, stop trading
    MAX_DAILY_LOSS = 25.0
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_pnl = sum(
        t.get("pnl_usd", 0) for t in closed_this_cycle
    )
    # Also count previously closed trades today from history
    if HISTORY_FILE.exists():
        try:
            all_history = json.loads(HISTORY_FILE.read_text())
            today_pnl += sum(
                t.get("pnl_usd", 0) for t in all_history
                if isinstance(t, dict) and t.get("closed_at", "").startswith(today_str)
            )
        except Exception:
            pass

    if today_pnl < -MAX_DAILY_LOSS:
        log.warning(f"🛑 MAX DAILY LOSS hit: ${today_pnl:.2f} < -${MAX_DAILY_LOSS}. Stopping new trades.")
        _save_portfolio(portfolio)
        return {
            "status": "max_daily_loss",
            "daily_pnl": round(today_pnl, 2),
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # ── Max trades per day: 0 = sin límite (orden Ender 2026-03-29) ──
    MAX_TRADES_PER_DAY = 0
    today_trade_count = portfolio.get("total_trades", 0)
    if MAX_TRADES_PER_DAY > 0 and today_trade_count >= MAX_TRADES_PER_DAY:
        log.warning(f"🛑 MAX TRADES/DAY hit: {today_trade_count} >= {MAX_TRADES_PER_DAY}. No new trades today.")
        save_portfolio(portfolio)
        save_history(history)
        return {
            "status": "max_daily_trades",
            "trades_today": today_trade_count,
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # ─── Coordinated Portfolio Opening (v2) ─────────────────────────────
    opened = []
    open_count = len([p for p in portfolio["positions"] if p.get("status") == "open"])

    MAX_POSITIONS = 5  # Ajustado 2026-03-31 (orden de Ender) — modo moderado
    slots_available = MAX_POSITIONS - open_count

    if slots_available <= 0:
        log.info(f"📊 Máximo de posiciones alcanzado ({MAX_POSITIONS})")
    else:
        # Paso 1: Filtrar señales válidas (no duplicadas, con confidence)
        valid_signals = []
        open_symbols = {p["symbol"] for p in portfolio["positions"] if p.get("status") == "open"}
        for signal in signals:
            sym = signal["symbol"]
            if sym in open_symbols:
                if debug:
                    log.info(f"  ⏭️  {sym}: posición ya abierta, skip")
                continue
            if signal.get("confidence", 0) < MIN_CONFIDENCE:
                continue
            valid_signals.append(signal)
            if len(valid_signals) >= slots_available:
                break

        # Paso 2: Calcular sizing coordinado para N posiciones
        n_planned = len(valid_signals)
        if n_planned > 0:
            # Calcular riesgo ACTUAL de posiciones ya abiertas
            existing_risk = 0.0
            for epos in portfolio.get("positions", []):
                if epos.get("status") != "open":
                    continue
                e_entry = epos.get("entry_price", 0)
                e_notional = epos.get("notional_value", 0)
                e_sl = epos.get("sl_price", 0)
                if e_entry > 0 and e_notional > 0:
                    sl_dist = abs(e_sl - e_entry) / e_entry
                    existing_risk += e_notional * sl_dist + e_notional * TAKER_FEE * 2
            
            # Presupuesto de riesgo restante para nuevas posiciones
            remaining_risk_budget = max(0, PORTFOLIO_MAX_RISK_USD - existing_risk)
            
            if remaining_risk_budget <= 0:
                log.warning(f"⚠️ Sin presupuesto de riesgo: existente ${existing_risk:.2f} ≥ cap ${PORTFOLIO_MAX_RISK_USD:.2f}")
                n_planned = 0
                valid_signals = []
            else:
                # El profit target por posición se distribuye entre todas (existentes + nuevas)
                total_positions = open_count + n_planned
                profit_per_pos = max(PORTFOLIO_TP_USD / total_positions, MIN_PROFIT_PER_POS_USD)
                # El riesgo por posición = presupuesto RESTANTE / N nuevas posiciones
                risk_per_pos = remaining_risk_budget / n_planned
                # También respetar el límite individual
                risk_per_pos = min(risk_per_pos, MAX_RISK_USD)

                log.info(f"📊 Coordinated sizing: {n_planned} nuevas + {open_count} existentes")
                log.info(f"   Risk existente: ${existing_risk:.2f} | Restante: ${remaining_risk_budget:.2f}")
                log.info(f"   Target: ${profit_per_pos:.2f}/pos | Risk: ${risk_per_pos:.2f}/pos")

            # Inyectar sizing coordinado en cada signal
            for sig in valid_signals:
                sig["_coordinated_n"] = n_planned
                sig["_coordinated_risk"] = risk_per_pos
                sig["_coordinated_profit_target"] = profit_per_pos

        # Paso 3: Abrir posiciones con sizing coordinado
        for signal in valid_signals:
            if safe:
                pos = paper_open_position(signal, portfolio, market)
            else:
                pos = real_open_position(signal, portfolio)

            if pos:
                opened.append(pos)
                open_count += 1
                open_symbols.add(signal["symbol"])
                arrow = "🟢" if pos["direction"] == "long" else "🔴"
                lev = pos.get("leverage", 1)
                margin = pos.get("margin_usd", pos.get("size_usd", 0))
                notional = pos.get("notional_value", pos.get("size_usd", 0))
                log.info(f"  {arrow} ABIERTA {signal['symbol']} [{signal['strategy']}] "
                         f"{lev}x | Margen: ${margin:.2f} | Notional: ${notional:.2f} @ ${pos['entry_price']:.6f}"
                         f" | Coord: {n_planned} pos, target ${profit_per_pos:.2f}/pos")

    # Calcular métricas actuales
    total_trades = portfolio.get("total_trades", 0)
    wins = portfolio.get("wins", 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    open_positions = [p for p in portfolio["positions"] if p.get("status") == "open"]
    unrealized_pnl = sum(p.get("pnl_usd", 0) for p in open_positions)

    # Guardar
    save_portfolio(portfolio)
    save_history(history)

    log.info(f"💰 Capital: ${portfolio['capital_usd']:.2f}")
    log.info(f"📊 Posiciones abiertas: {len(open_positions)}")
    log.info(f"📈 P&L no realizado: ${unrealized_pnl:+.2f}")
    log.info(f"🏆 Win rate: {win_rate:.1f}% ({wins}W/{total_trades-wins}L de {total_trades} trades)")
    log.info(f"💾 Guardado en {PORTFOLIO_FILE}")

    return {
        "status": "OK",
        "mode": "paper" if safe else "live",
        "capital": portfolio["capital_usd"],
        "open_positions": len(open_positions),
        "opened_this_cycle": len(opened),
        "closed_this_cycle": len(closed_this_cycle),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor Agent")
    parser.add_argument("--live", action="store_true",
                        help="Activar modo real (cuidado: trades reales en Solana)")
    parser.add_argument("--debug", action="store_true", help="Output detallado")
    args = parser.parse_args()

    safe_mode = not args.live
    if not safe_mode:
        log.warning("🚨 MODO LIVE ACTIVADO — se ejecutarán trades reales en Solana")
        confirm = input("Confirmar modo LIVE (escribe 'SI' para continuar): ")
        if confirm.strip() != "SI":
            log.info("Cancelado.")
            sys.exit(0)

    run(safe=safe_mode, debug=args.debug)
    sys.exit(0)
