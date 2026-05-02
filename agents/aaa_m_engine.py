#!/usr/bin/env python3
"""
AAA-M Engine — Agente de Autoaprendizaje Acelerado (MiniMax M2.7)
"El Cazador" — Motor principal de scalping agresivo.

Ciclo: Cada 30 segundos
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

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
from aaa_m_brain import make_trading_decision, analyze_recent_trades
from aaa_m_evolution import (
    load_config, get_effective_params, ParameterApplier,
    check_and_rollback_if_needed, record_baseline_sharpe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("aaa_m_engine")

AGENT_NAME = "AAA-M"
CYCLE_INTERVAL_SEC = 30
MAX_POSITIONS = 20
INITIAL_CAPITAL = 50000.0
SELF_ANALYSIS_INTERVAL = 720  # Cada 720 ciclos = 6h


def cycle(debug: bool = False) -> dict:
    log.info("=" * 60)
    log.info(f"⚡ {AGENT_NAME} — Ciclo {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    log.info("=" * 60)

    # -- Load dynamic config (hot-reload) --
    evo_config = load_config()
    params = get_effective_params(evo_config)
    if debug:
        log.info(f"   Params: SL={params['default_sl_pct']:.1%} TP={params['default_tp_pct']:.1%} "
                 f"LEV={params['default_leverage']}x MOM>{params['min_momentum_pct']}%")

    portfolio = load_portfolio(AGENT_NAME)
    trade_history = load_trade_history(AGENT_NAME)
    market = load_market_data()

    if debug:
        log.info(f"   Capital: ${portfolio.get('capital_usd', 0):.2f}")
        log.info(f"   Posiciones abiertas: {len([p for p in portfolio.get('positions', []) if p.get('status') == 'open'])}")
        log.info(f"   Trades historicos: {len(trade_history)}")

    # 1. Actualizar posiciones
    closed_positions = paper_update_positions(portfolio, market, agent=AGENT_NAME)
    if closed_positions:
        trade_history = append_trades(closed_positions, AGENT_NAME)
        if debug:
            for pos in closed_positions:
                log.info(f"   📤 Cerrado: {pos['symbol']} {pos['close_reason']} PnL=${pos.get('pnl_usd', 0):+.2f}")

    # 2. Self-analysis cada 6h
    cycle_count = portfolio.get("cycle_count", 0)
    portfolio["cycle_count"] = cycle_count + 1

    if cycle_count % SELF_ANALYSIS_INTERVAL == 0 and cycle_count > 0 and len(trade_history) >= 5:
        log.info(f"🔄 Self-analysis (ciclo {cycle_count})...")
        analysis = analyze_recent_trades(trade_history, max_trades=20)
        log.info(f"   Analisis: {analysis.get('analysis', 'N/A')[:100]}")
        if analysis.get("recommendations"):
            log.info(f"   Recomendaciones: {analysis['recommendations']}")
        if analysis.get("param_changes"):
            log.info(f"   Parametros sugeridos: {analysis['param_changes']}")
            # -- Phase 2: Auto-apply with safety guards --
            evo_config = ParameterApplier.apply_changes(
                evo_config,
                analysis.get("param_changes", {}),
                confidence=analysis.get("confidence", 0.0),
                analysis=analysis.get("analysis", ""),
            )
            # Record baseline Sharpe for future rollback check
            current_sharpe = metrics.get("sharpe_ratio", 0.0) if 'metrics' in dir() else 0.0
            if current_sharpe != 0.0:
                record_baseline_sharpe(current_sharpe, evo_config)

            # -- Save knowledge for auto_learner integration --
            knowledge_file = Path(__file__).parent / "aaa_data" / "knowledge_m.json"
            knowledge = {"entries": [], "evolution_history": [], "last_updated": None}
            if knowledge_file.exists():
                try:
                    knowledge = json.loads(knowledge_file.read_text())
                except Exception:
                    pass
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "self_analysis",
                "analysis": analysis.get("analysis", "")[:300],
                "recommendations": analysis.get("recommendations", []),
                "param_changes": analysis.get("param_changes", {}),
                "confidence": analysis.get("confidence", 0.0),
                "trades_analyzed": min(len(trade_history), 20),
            }
            knowledge["entries"].append(entry)
            # Keep only last 10 entries
            knowledge["entries"] = knowledge["entries"][-10:]
            knowledge["evolution_history"] = evo_config.get("evolution_history", [])[-10:]
            knowledge["last_updated"] = datetime.now(timezone.utc).isoformat()
            knowledge_file.write_text(json.dumps(knowledge, indent=2))
            log.info(f"   💾 Knowledge saved to knowledge_m.json ({len(knowledge['entries'])} entries)")

    # 3. Decidir nuevas posiciones
    open_count = len([p for p in portfolio.get("positions", []) if p.get("status") == "open"])

    if open_count < MAX_POSITIONS:
        log.info(f"⚡ Consultando a MiniMax M2.7 ({open_count}/{MAX_POSITIONS} posiciones)...")
        decision = make_trading_decision(
            market, portfolio, trade_history,
            max_positions=params["max_positions"],
            dynamic_params=params,
        )

        log.info(f"   Decision: {decision.get('action')} | Conf: {decision.get('confidence', 0):.0%}")
        if decision.get("reasoning"):
            log.info(f"   Reasoning: {decision['reasoning']}")

        if decision.get("action") == "OPEN" and decision.get("confidence", 0) >= 0.60:
            symbol = decision.get("symbol", "").upper()
            direction = decision.get("direction", "long").lower()

            already_open = any(
                p.get("status") == "open" and p.get("symbol") == symbol
                for p in portfolio.get("positions", [])
            )
            if already_open:
                log.info(f"   ⏭️ {symbol} ya tiene posicion abierta")
            else:
                price = get_current_price(symbol, market)
                if price > 0:
                    capital = portfolio.get("capital_usd", 0)
                    margin_pct = decision.get("margin_pct", 0.02)
                    margin = capital * margin_pct
                    leverage = min(decision.get("leverage", params["default_leverage"]), 10)
                    sl_pct = decision.get("sl_pct", params["default_sl_pct"])
                    tp_pct = decision.get("tp_pct", params["default_tp_pct"])

                    liq = get_token_liquidity(symbol, market)
                    if liq < 200_000:
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
                            strategy="minimax_momentum",
                            agent=AGENT_NAME,
                        )
                        if pos:
                            log.info(f"   ✅ Posicion abierta: {symbol} {direction.upper()}")
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
                        log.info(f"   📤 LLM cerro: {target} PnL=${pnl:+.2f}")
                    break
    else:
        log.info(f"   Maximo de posiciones alcanzado ({MAX_POSITIONS})")

    # 4. Equity history
    open_positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    total_equity = portfolio["capital_usd"] + sum(p.get("margin_usd", 0) + p.get("pnl_usd", 0) for p in open_positions)
    update_equity_history(total_equity, AGENT_NAME)

    # 5. Metrics
    metrics = calculate_metrics(trade_history, capital_start=INITIAL_CAPITAL)

    # -- Check for Sharpe degradation and rollback if needed --
    evo_config = check_and_rollback_if_needed(metrics.get("sharpe_ratio", 0.0), evo_config)
    rolled_back = evo_config.get("last_applied") is None and evo_config.get("evolution_history") and                   evo_config["evolution_history"][-1].get("action") == "ROLLBACK"
    if rolled_back:
        log.warning("🔄 Parametros restaurados por degradacion de Sharpe")
        params = get_effective_params(evo_config)

    log.info(f"📊 Metricas: WR={metrics['win_rate']:.1f}% PF={metrics['profit_factor']:.2f} Sharpe={metrics['sharpe_ratio']:.2f} DD={metrics['max_drawdown_pct']:.1f}% PnL=${metrics['total_pnl']:+.2f}")

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
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAA-M Engine")
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
