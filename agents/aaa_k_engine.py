#!/usr/bin/env python3
"""
AAA-K Engine — Agente de Autoaprendizaje Acelerado (Kimi 2.6)
"El Estratega" — Motor principal que orquesta el ciclo de trading conservador.

Ciclo: Cada 2 minutos
  1. Leer market data del live bot
  2. Actualizar P&L de posiciones abiertas (SL/TP/liq/time)
  3. Si hay posiciones para cerrar, cerrarlas y registrar
  4. Analizar portafolio con Kimi (cada 30 min)
  5. Decidir nuevas posiciones con Kimi
  6. Abrir posiciones si hay señal válida
  7. Guardar estado

Uso:
    python3 aaa_k_engine.py          # Modo normal
    python3 aaa_k_engine.py --debug  # Output detallado
    python3 aaa_k_engine.py --once   # Un solo ciclo (para testing)
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# ─── Setup paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from aaa_shared import (
    paper_open_position, paper_update_positions,
    load_portfolio, save_portfolio,
    load_trade_history, append_trades,
    load_market_data, load_signals,
    update_equity_history, calculate_metrics,
    get_current_price, get_token_liquidity,
)
from aaa_k_brain import make_trading_decision, analyze_portfolio_health

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("aaa_k_engine")

# ─── Config ─────────────────────────────────────────────────────────────────

AGENT_NAME = "AAA-K"
CYCLE_INTERVAL_SEC = 120       # 2 minutos
PORTFOLIO_ANALYSIS_INTERVAL = 15  # Cada 15 ciclos = 30 min
MAX_POSITIONS = 10
INITIAL_CAPITAL = 50000.0

# ─── Main Cycle ─────────────────────────────────────────────────────────────

def cycle(debug: bool = False) -> dict:
    """Ejecuta un ciclo completo del agente K."""
    log.info("=" * 60)
    log.info(f"🧠 {AGENT_NAME} — Ciclo {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    log.info("=" * 60)

    # 1. Cargar estado
    portfolio = load_portfolio(AGENT_NAME)
    trade_history = load_trade_history(AGENT_NAME)
    market = load_market_data()
    signals = load_signals()

    if debug:
        log.info(f"   Capital: ${portfolio.get('capital_usd', 0):.2f}")
        log.info(f"   Posiciones abiertas: {len([p for p in portfolio.get('positions', []) if p.get('status') == 'open'])}")
        log.info(f"   Trades históricos: {len(trade_history)}")

    # 2. Actualizar posiciones abiertas (SL/TP/liq/time)
    closed_positions = paper_update_positions(portfolio, market, agent=AGENT_NAME)
    if closed_positions:
        trade_history = append_trades(closed_positions, AGENT_NAME)
        if debug:
            for pos in closed_positions:
                log.info(f"   📤 Cerrado: {pos['symbol']} {pos['close_reason']} PnL=${pos.get('pnl_usd', 0):+.2f}")

    # 3. Análisis de portafolio (cada 30 min)
    cycle_count = portfolio.get("cycle_count", 0)
    portfolio["cycle_count"] = cycle_count + 1

    if cycle_count % PORTFOLIO_ANALYSIS_INTERVAL == 0 and cycle_count > 0:
        log.info(f"🩺 Análisis de portafolio (ciclo {cycle_count})...")
        health = analyze_portfolio_health(portfolio, market, trade_history)
        log.info(f"   Health Score: {health.get('health_score', 0)}/100")
        if health.get("positions_to_close"):
            log.info(f"   Recomendado cerrar: {health['positions_to_close']}")
            # Cerrar posiciones recomendadas
            for sym in health["positions_to_close"]:
                for pos in portfolio.get("positions", []):
                    if pos.get("status") == "open" and pos.get("symbol") == sym:
                        current_price = get_current_price(sym, market)
                        if current_price > 0:
                            notional = pos.get("notional_value", 0)
                            margin = pos.get("margin_usd", 0)
                            direction = pos["direction"]
                            entry = pos["entry_price"]
                            funding = pos.get("funding_accumulated", 0)
                            fee_exit_est = notional * 0.002  # aprox
                            if direction == "long":
                                price_pnl_pct = (current_price - entry) / entry
                            else:
                                price_pnl_pct = (entry - current_price) / entry
                            pnl = notional * price_pnl_pct + funding - pos.get("fee_entry", 0) - fee_exit_est
                            pos["status"] = "closed"
                            pos["close_time"] = datetime.now(timezone.utc).isoformat()
                            pos["close_reason"] = "PORTFOLIO_REBALANCE"
                            pos["close_price"] = round(current_price, 8)
                            pos["pnl_usd"] = round(pnl, 4)
                            pos["pnl_pct"] = round((pnl / margin * 100) if margin > 0 else 0, 4)
                            portfolio["capital_usd"] = round(portfolio["capital_usd"] + margin + pnl, 2)
                            trade_history = append_trades([pos], AGENT_NAME)
                            log.info(f"   📤 Rebalance cerrado: {sym} PnL=${pnl:+.2f}")
                        break
        if health.get("lessons"):
            log.info(f"   Lecciones: {health['lessons']}")

    # 4. Decidir nuevas posiciones
    open_count = len([p for p in portfolio.get("positions", []) if p.get("status") == "open"])

    if open_count < MAX_POSITIONS:
        log.info(f"🧠 Consultando a Kimi 2.6 ({open_count}/{MAX_POSITIONS} posiciones)...")
        decision = make_trading_decision(market, portfolio, trade_history, max_positions=MAX_POSITIONS)

        log.info(f"   Decisión: {decision.get('action')} | Conf: {decision.get('confidence', 0):.0%}")
        if decision.get("reasoning"):
            log.info(f"   Reasoning: {decision['reasoning']}")

        if decision.get("action") == "OPEN" and decision.get("confidence", 0) >= 0.65:
            symbol = decision.get("symbol", "").upper()
            direction = decision.get("direction", "long").lower()

            # Validar que no esté ya abierto
            already_open = any(
                p.get("status") == "open" and p.get("symbol") == symbol
                for p in portfolio.get("positions", [])
            )
            if already_open:
                log.info(f"   ⏭️ {symbol} ya tiene posición abierta")
            else:
                price = get_current_price(symbol, market)
                if price > 0:
                    capital = portfolio.get("capital_usd", 0)
                    margin_pct = decision.get("margin_pct", 0.02)
                    margin = capital * margin_pct
                    leverage = min(decision.get("leverage", 2), 10)
                    sl_pct = decision.get("sl_pct", 0.04)
                    tp_pct = decision.get("tp_pct", 0.08)

                    # Validar liquidez mínima
                    liq = get_token_liquidity(symbol, market)
                    if liq < 500_000:
                        log.warning(f"   ⚠️ Liquidez insuficiente para {symbol}: ${liq:.0f}")
                    else:
                        pos = paper_open_position(
                            symbol=symbol,
                            direction=direction,
                            entry_price=price,
                            margin_usd=margin,
                            leverage=leverage,
                            sl_pct=sl_pct,
                            tp_pct=tp_pct,
                            portfolio=portfolio,
                            market=market,
                            strategy="kimi_llm",
                            agent=AGENT_NAME,
                        )
                        if pos:
                            log.info(f"   ✅ Posición abierta: {symbol} {direction.upper()}")
                else:
                    log.warning(f"   ⚠️ Sin precio para {symbol}")

        elif decision.get("action") == "CLOSE" and decision.get("close_target"):
            target = decision["close_target"]
            for pos in portfolio.get("positions", []):
                if pos.get("status") == "open" and pos.get("symbol") == target:
                    current_price = get_current_price(target, market)
                    if current_price > 0:
                        notional = pos.get("notional_value", 0)
                        margin = pos.get("margin_usd", 0)
                        direction = pos["direction"]
                        entry = pos["entry_price"]
                        funding = pos.get("funding_accumulated", 0)
                        fee_exit_est = notional * 0.002
                        if direction == "long":
                            price_pnl_pct = (current_price - entry) / entry
                        else:
                            price_pnl_pct = (entry - current_price) / entry
                        pnl = notional * price_pnl_pct + funding - pos.get("fee_entry", 0) - fee_exit_est
                        pos["status"] = "closed"
                        pos["close_time"] = datetime.now(timezone.utc).isoformat()
                        pos["close_reason"] = "LLM_DECISION"
                        pos["close_price"] = round(current_price, 8)
                        pos["pnl_usd"] = round(pnl, 4)
                        pos["pnl_pct"] = round((pnl / margin * 100) if margin > 0 else 0, 4)
                        portfolio["capital_usd"] = round(portfolio["capital_usd"] + margin + pnl, 2)
                        trade_history = append_trades([pos], AGENT_NAME)
                        log.info(f"   📤 LLM cerró: {target} PnL=${pnl:+.2f}")
                    break
    else:
        log.info(f"   Máximo de posiciones alcanzado ({MAX_POSITIONS})")

    # 5. Actualizar equity history
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    total_equity = portfolio["capital_usd"] + sum(p.get("margin_usd", 0) + p.get("pnl_usd", 0) for p in open_positions)
    update_equity_history(total_equity, AGENT_NAME)

    # 6. Calcular y loggear métricas
    metrics = calculate_metrics(trade_history, capital_start=INITIAL_CAPITAL)
    log.info(f"📊 Métricas: WR={metrics['win_rate']:.1f}% PF={metrics['profit_factor']:.2f} Sharpe={metrics['sharpe_ratio']:.2f} DD={metrics['max_drawdown_pct']:.1f}% PnL=${metrics['total_pnl']:+.2f}")

    # 7. Guardar estado
    save_portfolio(portfolio, AGENT_NAME)

    return {
        "agent": AGENT_NAME,
        "cycle": cycle_count,
        "capital": portfolio["capital_usd"],
        "open_positions": open_count,
        "closed_this_cycle": len(closed_positions),
        "metrics": metrics,
    }


def run_continuous(debug: bool = False):
    """Ejecuta el agente en modo continuo."""
    log.info(f"🚀 {AGENT_NAME} iniciado en modo CONTINUO")
    log.info(f"   Intervalo: {CYCLE_INTERVAL_SEC}s")
    log.info(f"   Capital inicial: ${INITIAL_CAPITAL:,.2f}")
    log.info(f"   Max posiciones: {MAX_POSITIONS}")

    while True:
        try:
            t0 = time.time()
            cycle(debug=debug)
            elapsed = time.time() - t0
            sleep_time = max(1, CYCLE_INTERVAL_SEC - elapsed)
            if debug:
                log.info(f"   Ciclo completado en {elapsed:.1f}s. Durmiendo {sleep_time:.0f}s...")
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            log.info("🛑 Detenido por usuario")
            break
        except Exception as e:
            log.error(f"❌ Error en ciclo: {e}")
            import traceback
            log.debug(traceback.format_exc())
            time.sleep(30)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAA-K Engine")
    parser.add_argument("--debug", action="store_true", help="Output detallado")
    parser.add_argument("--once", action="store_true", help="Un solo ciclo")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    if args.once:
        result = cycle(debug=args.debug)
        print(json.dumps(result, indent=2))
    else:
        run_continuous(debug=args.debug)
