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
from aaa_k_brain import make_trading_decision, analyze_portfolio_health, analyze_recent_trades_k
from aaa_alerts import alert_first_trade, alert_evolution_applied, alert_rollback, alert_drawdown
from aaa_k_evolution import (
    load_config, get_effective_params, ParameterApplier,
    check_and_rollback_if_needed, record_baseline_sharpe,
    get_active_variant, maybe_start_ab_test, ABTestManager,
)

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

    # -- Load dynamic config + variant (hot-reload) --
    evo_config = load_config()
    params = get_effective_params(evo_config)
    variant = get_active_variant(evo_config)
    if debug:
        log.info(f"   Params: SL={params['default_sl_pct']:.1%} TP={params['default_tp_pct']:.1%} LEV={params['default_leverage']}x")
    if variant and variant.get("name"):
        log.info(f"   Variant: {variant['name']} (addon={len(variant.get('system_prompt_addon', ''))}ch)")
    # Check A/B test evaluation
    ab_result = ABTestManager.evaluate_and_finalize(evo_config)
    if ab_result:
        log.info(f"   A/B Test result: {ab_result}")

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

        # -- Phase 2: Self-analysis every 30 cycles (1h) --
        if cycle_count % 30 == 0 and cycle_count > 0 and len(trade_history) >= 3:
            log.info(f"🔄 Self-analysis K (ciclo {cycle_count})...")
            analysis = analyze_recent_trades_k(trade_history, max_trades=20)
            log.info(f"   Analisis: {analysis.get('analysis', 'N/A')[:100]}")
            if analysis.get("recommendations"):
                log.info(f"   Recomendaciones: {analysis['recommendations']}")
            if analysis.get("param_changes"):
                log.info(f"   Parametros sugeridos: {analysis['param_changes']}")
                evo_config = ParameterApplier.apply_changes(
                    evo_config,
                    analysis.get("param_changes", {}),
                    confidence=analysis.get("confidence", 0.0),
                    analysis=analysis.get("analysis", ""),
                )
                current_sharpe = metrics.get("sharpe_ratio", 0.0) if 'metrics' in dir() else 0.0
                if current_sharpe != 0.0:
                    record_baseline_sharpe(current_sharpe, evo_config)
                # Phase 3: try to start A/B test
                test_started = maybe_start_ab_test(evo_config, analysis)
                if test_started:
                    log.info("   🧬 A/B TEST STARTED")

        # -- Save knowledge for auto_learner integration --
        knowledge_file = Path(__file__).parent / "aaa_data" / "knowledge_k.json"
        knowledge = {"entries": [], "last_updated": None}
        if knowledge_file.exists():
            try:
                knowledge = json.loads(knowledge_file.read_text())
            except Exception:
                pass
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "portfolio_health",
            "health_score": health.get("health_score", 0),
            "recommendations": health.get("recommendations", []),
            "positions_to_close": health.get("positions_to_close", []),
            "lessons": health.get("lessons", []),
            "confidence": health.get("confidence", 0.0),
        }
        knowledge["entries"].append(entry)
        knowledge["entries"] = knowledge["entries"][-10:]
        knowledge["last_updated"] = datetime.now(timezone.utc).isoformat()
        knowledge_file.write_text(json.dumps(knowledge, indent=2))
        log.info(f"   Knowledge saved to knowledge_k.json ({len(knowledge['entries'])} entries)")

    # 4. Decidir nuevas posiciones
    open_count = len([p for p in portfolio.get("positions", []) if p.get("status") == "open"])

    if open_count < MAX_POSITIONS:
        log.info(f"🧠 Consultando a Kimi 2.6 ({open_count}/{MAX_POSITIONS} posiciones)...")
        decision = make_trading_decision(
            market, portfolio, trade_history,
            max_positions=params.get("max_positions", MAX_POSITIONS),
            variant=variant,
        )

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
                    # COOLDOWN GUARD: No cerrar posiciones abiertas <30 min ago
                    opened = datetime.fromisoformat(pos["opened_at"].replace("Z", "+00:00"))
                    minutes_open = (datetime.now(timezone.utc) - opened).total_seconds() / 60
                    if minutes_open < 30:
                        log.info(f"   SHIELD COOLDOWN: Ignorando CLOSE {target} ({minutes_open:.0f}min < 30min)")
                        break
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

    # -- Check Sharpe rollback --
    evo_config = check_and_rollback_if_needed(metrics.get("sharpe_ratio", 0.0), evo_config)
    rolled_back = evo_config.get("last_applied") is None and evo_config.get("evolution_history") and                   evo_config["evolution_history"][-1].get("action") == "ROLLBACK"
    if rolled_back:
        log.warning("🔄 Parametros restaurados por degradacion de Sharpe")
        alert_rollback(AGENT_NAME, "Sharpe degradado >30%", metrics.get("sharpe_ratio", 0.0))
        params = get_effective_params(evo_config)

    log.info(f"📊 Métricas: WR={metrics['win_rate']:.1f}% PF={metrics['profit_factor']:.2f} Sharpe={metrics['sharpe_ratio']:.2f} DD={metrics['max_drawdown_pct']:.1f}% PnL=${metrics['total_pnl']:+.2f}")
    if metrics.get("max_drawdown_pct", 0) > 5.0:
        alert_drawdown(AGENT_NAME, metrics["max_drawdown_pct"])

    # -- Record A/B test cycle --
    ABTestManager.record_cycle(evo_config, variant.get("name", "v1") if variant else "v1",
                                metrics.get("sharpe_ratio", 0.0), metrics.get("total_trades", 0))

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
