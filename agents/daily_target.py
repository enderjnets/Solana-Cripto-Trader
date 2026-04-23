#!/usr/bin/env python3
"""
📊 Daily Profit Target Manager
IA evalúa condiciones de mercado y cierra todas las posiciones cuando:
1. P&L diario alcanza target (3-5%)
2. Y condiciones de overbought detectadas (RSI > 70)
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
SIGNALS_FILE = DATA_DIR / "signals_latest.json"
DAILY_TARGET_FILE = DATA_DIR / "daily_target_state.json"

# Config
TARGET_MIN_PCT = 0.05  # 5% mínimo (was 3%)
TARGET_MAX_PCT = 0.05  # E7: 5% realistic target (was 10%)  # 10% máximo (was 5% — raised to avoid premature locking with leverage)
RSI_OVERBOUGHT = 70    # RSI > 70 = overbought


def load_portfolio() -> dict:
    """Carga portfolio actual"""
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {"capital_usd": 500.0, "initial_capital": 500.0, "positions": []}


def load_signals() -> dict:
    """Carga señales de mercado"""
    if SIGNALS_FILE.exists():
        return json.loads(SIGNALS_FILE.read_text())
    return {}


def load_daily_state() -> dict:
    """Carga estado del target diario"""
    if DAILY_TARGET_FILE.exists():
        return json.loads(DAILY_TARGET_FILE.read_text())
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "starting_capital": None,
        "target_reached": False,
        "closed_at": None
    }


def save_daily_state(state: dict):
    """Guarda estado del target diario"""
    DAILY_TARGET_FILE.write_text(json.dumps(state, indent=2))


def calculate_daily_pnl(portfolio: dict, state: dict) -> dict:
    """Calcula P&L del día actual"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Si es nuevo día, resetear con EQUITY (no solo capital_usd)
    if state.get("date") != today:
        # Calcular equity actual para usarla como starting point
        _positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
        _invested = sum(p.get("margin_usd", 0) for p in _positions)
        _unrealized = sum(p.get("pnl_usd", 0) for p in _positions)
        _equity_now = portfolio.get("capital_usd", 1000.0) + _invested + _unrealized
        state = {
            "date": today,
            "starting_capital": _equity_now,  # FIX: usar equity, no capital_usd
            "target_reached": False,
            "closed_at": None
        }
        save_daily_state(state)
    
    initial = portfolio.get("initial_capital", 500.0)
    current_capital = portfolio.get("capital_usd", 1000.0)
    
    # P&L total (incluye posiciones abiertas)
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    invested = sum(p.get("margin_usd", 0) for p in open_positions)
    unrealized_pnl = sum(p.get("pnl_usd", 0) for p in open_positions)
    equity = current_capital + invested + unrealized_pnl
    
    # P&L del día
    daily_start = state.get("starting_capital", initial)
    daily_pnl_usd = equity - daily_start
    # Usar initial_capital como denominador para % consistente
    base_capital = portfolio.get("initial_capital", daily_start)
    daily_pnl_pct = (daily_pnl_usd / base_capital) if base_capital > 0 else 0
    
    return {
        "equity": equity,
        "daily_start": daily_start,
        "daily_pnl_usd": daily_pnl_usd,
        "daily_pnl_pct": daily_pnl_pct,
        "state": state
    }


def evaluate_market_conditions(signals: dict) -> dict:
    """Evalúa condiciones de mercado para decidir target"""
    if not signals or "signals" not in signals:
        return {"overbought": False, "avg_rsi": 50, "trend": "neutral"}
    
    all_signals = signals.get("signals", [])
    if not all_signals:
        return {"overbought": False, "avg_rsi": 50, "trend": "neutral"}
    
    # Calcular RSI promedio
    rsi_values = [s.get("rsi", 50) for s in all_signals if "rsi" in s]
    avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 50
    
    # Detectar overbought
    overbought = avg_rsi > RSI_OVERBOUGHT
    
    # Detectar trend general
    trends = [s.get("trend", "neutral") for s in all_signals if "trend" in s]
    up_count = trends.count("up")
    down_count = trends.count("down")
    
    if up_count > down_count:
        trend = "up"
    elif down_count > up_count:
        trend = "down"
    else:
        trend = "neutral"
    
    return {
        "overbought": overbought,
        "avg_rsi": round(avg_rsi, 2),
        "trend": trend,
        "signals_count": len(all_signals)
    }


def evaluate_daily_target(portfolio: dict, signals: dict) -> dict:
    """
    Evalúa si se debe cerrar todo por profit target diario.
    Retorna decisión con posiciones a cerrar.
    """
    state = load_daily_state()
    daily = calculate_daily_pnl(portfolio, state)
    market = evaluate_market_conditions(signals)
    
    daily_pnl_pct = daily["daily_pnl_pct"]
    
    # Determinar target dinámico (3-5%)
    # Si mercado está muy overbought, usar target más bajo (3%)
    # Si está neutral, usar target más alto (5%)
    if market["overbought"]:
        target_pct = TARGET_MIN_PCT
        reason_target = "OVERBOUGHT_DETECTED"
    else:
        target_pct = TARGET_MAX_PCT
        reason_target = "STANDARD_TARGET"
    
    # Verificar si se alcanzó target
    target_reached = daily_pnl_pct >= target_pct
    
    # Decisión de cierre
    should_close_all = False
    close_reason = ""
    
    # v2.12.26 safety check: daily_pnl_pct > 20% en un solo día es casi imposible
    # con 3-5 trades de $10 cada uno. Si aparece, es casi seguro que starting_capital
    # está stale (positions abiertas al day-rollover). NO disparar close en ese caso.
    _SAFETY_MAX_PNL_PCT = 0.20
    _suspicious = daily_pnl_pct > _SAFETY_MAX_PNL_PCT

    if target_reached and market["overbought"]:
        if _suspicious:
            import logging as _log
            _log.getLogger(__name__).warning(
                f"⚠️ DAILY_TARGET_REACHED skipped — daily_pnl_pct={daily_pnl_pct*100:.1f}% "
                f">{_SAFETY_MAX_PNL_PCT*100:.0f}% parece irreal (posible starting_capital stale "
                f"${daily['daily_start']:.2f}). v2.12.26 safety."
            )
        else:
            should_close_all = True
            close_reason = f"DAILY_TARGET_REACHED: {daily_pnl_pct*100:.2f}% >= {target_pct*100:.1f}% + OVERBOUGHT (RSI {market['avg_rsi']:.1f})"
    elif target_reached and daily_pnl_pct >= TARGET_MAX_PCT:
        if _suspicious:
            import logging as _log
            _log.getLogger(__name__).warning(
                f"⚠️ DAILY_TARGET_MAX_REACHED skipped — daily_pnl_pct={daily_pnl_pct*100:.1f}% "
                f">{_SAFETY_MAX_PNL_PCT*100:.0f}% parece irreal (posible starting_capital stale "
                f"${daily['daily_start']:.2f}). v2.12.26 safety."
            )
        else:
            # Si llegó al 5% o más, cerrar sin importar overbought
            should_close_all = True
            close_reason = f"DAILY_TARGET_MAX_REACHED: {daily_pnl_pct*100:.2f}% >= {TARGET_MAX_PCT*100:.1f}%"
    
    # Actualizar estado
    state["current_pnl_pct"] = daily_pnl_pct
    state["target_pct"] = target_pct
    state["target_reached"] = target_reached
    
    if should_close_all:
        state["target_reached"] = True
        state["closed_at"] = datetime.now(timezone.utc).isoformat()
    
    save_daily_state(state)
    
    # Lista de posiciones a cerrar
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    positions_to_close = []
    
    if should_close_all:
        for pos in open_positions:
            positions_to_close.append({
                "symbol": pos.get("symbol"),
                "pnl_usd": pos.get("pnl_usd", 0),
                "pnl_pct": pos.get("pnl_pct", 0),
                "reason": close_reason
            })
    
    return {
        "should_close_all": should_close_all,
        "close_reason": close_reason if should_close_all else None,
        "positions_to_close": positions_to_close,
        "daily_pnl_pct": round(daily_pnl_pct * 100, 2),
        "daily_pnl_usd": round(daily["daily_pnl_usd"], 2),
        "target_pct": round(target_pct * 100, 1),
        "market_conditions": market,
        "state": state
    }


if __name__ == "__main__":
    print("=" * 80)
    print("📊 DAILY TARGET MANAGER")
    print("=" * 80)
    
    portfolio = load_portfolio()
    signals = load_signals()
    
    result = evaluate_daily_target(portfolio, signals)
    
    print(f"📅 P&L Diario: {result['daily_pnl_pct']:.2f}% (${result['daily_pnl_usd']:.2f})")
    print(f"🎯 Target: {result['target_pct']:.1f}%")
    print(f"📊 Mercado: RSI {result['market_conditions']['avg_rsi']:.1f} | Overbought: {result['market_conditions']['overbought']}")
    print()
    
    if result["should_close_all"]:
        print(f"🚨 CERRAR TODO: {result['close_reason']}")
        print(f"   Posiciones a cerrar: {len(result['positions_to_close'])}")
        for pos in result["positions_to_close"]:
            print(f"     • {pos['symbol']}: {pos['pnl_pct']:.2f}%")
    else:
        print("✅ Mantener posiciones abiertas")
        print(f"   Faltan: {result['target_pct'] - result['daily_pnl_pct']:.2f}% para target")
    
    print()
    print("=" * 80)
