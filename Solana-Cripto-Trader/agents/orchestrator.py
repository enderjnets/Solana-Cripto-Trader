#!/usr/bin/env python3
"""
🎯 Solana Trading Bot - Orchestrator Modular
Ejecuta el ciclo completo con Smart Rotation + Daily Target
"""
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("orchestrator")


def import_agents():
    """Importa módulos de agentes."""
    import market_data
    import risk_manager as rm
    import strategy
    import executor
    import reporter
    try:
        import daily_target
    except ImportError:
        daily_target = None
        log.warning("⚠️  daily_target.py no encontrado")
    return market_data, rm, strategy, executor, reporter, daily_target


def run_cycle(safe=True, debug=False):
    """Ejecuta un ciclo completo del sistema."""
    cycle_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log.info("=" * 60)
    log.info(f"🔄 CICLO INICIADO — {now}")
    mode_label = "📄 PAPER" if safe else "🔴 LIVE"
    log.info(f"   Modo: {mode_label}")
    log.info("=" * 60)
    
    results = {}
    
    # Importar agentes
    try:
        md, rm, st, ex, rp, dt = import_agents()
    except ImportError as e:
        log.error(f"❌ Error importando agentes: {e}")
        return {"status": "IMPORT_ERROR"}
    
    # Paso 1: Market Data
    log.info("━" * 40)
    log.info("🌐 [1/5] Market Data")
    try:
        result = md.run(debug=debug)
        results["market_data"] = {"ok": True, "prices": result.get("prices_ok", 0)}
        log.info(f"   → {result.get('prices_ok', 0)} precios obtenidos")
    except Exception as e:
        log.error(f"   ❌ Error: {e}")
        results["market_data"] = {"ok": False, "error": str(e)}
        return results
    
    # Paso 2: Risk Manager
    log.info("━" * 40)
    log.info("🛡️  [2/5] Risk Manager")
    try:
        result = rm.run(debug=debug)
        results["risk_manager"] = {"ok": True}
        log.info(f"   → Status: {result.get('portfolio_status', 'ACTIVE')}")
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["risk_manager"] = {"ok": False}
    
    # Paso 3: Strategy
    log.info("━" * 40)
    log.info("🧠 [3/5] Strategy")
    try:
        result = st.run(debug=debug)
        n_signals = result.get("total_signals", 0)
        results["strategy"] = {"ok": True, "signals": n_signals}
        log.info(f"   → {n_signals} señal(es)")
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["strategy"] = {"ok": False}
    
    # Paso 4: Executor
    log.info("━" * 40)
    log.info("⚡ [4/5] Executor")
    try:
        result = ex.run(safe=safe, debug=debug)
        results["executor"] = {"ok": True}
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["executor"] = {"ok": False}
    
    # Paso 4b: Smart Rotation
    log.info("━" * 40)
    log.info("🔄 [4b/5] Smart Rotation")
    
    portfolio_data = {}
    portfolio_file = DATA_DIR / "portfolio.json"
    if portfolio_file.exists():
        portfolio_data = json.loads(portfolio_file.read_text())
    
    stale_positions = []
    if portfolio_data:
        try:
            stale_positions = rm.check_stale_losing_positions(
                portfolio_data, max_hours=72, improvement_hours=24
            )
        except Exception as e:
            log.warning(f"   ⚠️ Error: {e}")
        
        if stale_positions:
            log.info(f"   → {len(stale_positions)} posición(es) a cerrar:")
            for pos in stale_positions:
                log.info(f"     - {pos['symbol']}: {pos['reason']}")
            results["smart_rotation"] = {"ok": True, "closed": len(stale_positions)}
        else:
            log.info("   → ✅ Sin posiciones a rotar")
            results["smart_rotation"] = {"ok": True, "closed": 0}
    else:
        log.info("   → ✅ Sin datos de portfolio")
        results["smart_rotation"] = {"ok": False}
    
    # Paso 4c: Daily Target
    log.info("━" * 40)
    log.info("🎯 [4c/5] Daily Target")
    
    if not dt:
        log.warning("   ⚠️ Módulo daily_target no disponible")
        results["daily_target"] = {"ok": False}
    else:
        signals_data = {}
        signals_file = DATA_DIR / "signals_latest.json"
        if signals_file.exists():
            signals_data = json.loads(signals_file.read_text())
        
        try:
            target_result = dt.evaluate_daily_target(portfolio_data, signals_data)
            
            log.info(f"   → P&L: {target_result['daily_pnl_pct']:.2f}%")
            log.info(f"   → Target: {target_result['target_pct']:.1f}%")
            log.info(f"   → RSI: {target_result['market_conditions']['avg_rsi']:.1f}")
            
            if target_result["should_close_all"]:
                log.info(f"   🚨 CERRAR TODO: {target_result['close_reason']}")
                results["daily_target"] = {
                    "ok": True,
                    "should_close": True,
                    "reason": target_result["close_reason"]
                }
            else:
                log.info("   ✅ Mantener posiciones")
                results["daily_target"] = {"ok": True, "should_close": False}
        except Exception as e:
            log.warning(f"   ⚠️ Error: {e}")
            results["daily_target"] = {"ok": False}
    
    # Paso 4d: Position Decisions (LLM + Quant)
    log.info("━" * 40)
    log.info("🧠 [4d/5] Position Decisions — LLM + Quant")
    try:
        market_data_for_dec = {}
        market_file = DATA_DIR / "market_latest.json"
        if market_file.exists():
            market_data_for_dec = json.loads(market_file.read_text())

        research_data = {}
        research_file = DATA_DIR / "research_latest.json"
        if research_file.exists():
            research_data = json.loads(research_file.read_text())

        if portfolio_data and portfolio_data.get("positions"):
            decisions = rm.evaluate_position_decision(portfolio_data, market_data_for_dec, research_data)
            close_recs  = [d for d in decisions if d["action"] == "CLOSE"  and d["confidence"] >= 0.70]
            reduce_recs = [d for d in decisions if d["action"] == "REDUCE" and d["confidence"] >= 0.70]

            if close_recs:
                log.info(f"   🔴 CERRAR ({len(close_recs)}): {', '.join(d['symbol'] for d in close_recs)}")
            if reduce_recs:
                log.info(f"   🟡 REDUCIR ({len(reduce_recs)}): {', '.join(d['symbol'] for d in reduce_recs)}")
            if not close_recs and not reduce_recs:
                log.info(f"   🟢 MANTENER todas las posiciones")

            results["position_decisions"] = {
                "ok": True,
                "evaluated": len(decisions),
                "close_signals": len(close_recs),
                "reduce_signals": len(reduce_recs),
            }
        else:
            log.info("   ℹ️  Sin posiciones abiertas")
            results["position_decisions"] = {"ok": True, "evaluated": 0}
    except Exception as e:
        log.warning(f"   ⚠️ Error en position decisions: {e}")
        results["position_decisions"] = {"ok": False}

    # Paso 5: Reporter
    log.info("━" * 40)
    log.info("📊 [5/5] Reporter")
    try:
        result = rp.run(daily=False)
        results["reporter"] = {"ok": True}
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["reporter"] = {"ok": False}
    
    # Calcular health score
    ok_count = sum(1 for r in results.values() if r.get("ok", False))
    health_score = min(10, ok_count * 2)
    
    log.info("=" * 60)
    log.info("📊 CICLO COMPLETADO")
    elapsed = time.time() - cycle_start
    log.info(f"   ⏱  Tiempo: {elapsed:.1f}s | Health: {health_score}/10")
    
    # Guardar reporte
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "health_score": health_score
    }
    report_file = DATA_DIR / "report_latest.json"
    report_file.write_text(json.dumps(report, indent=2))
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    if args.once:
        run_cycle(debug=args.debug)
    else:
        while True:
            try:
                run_cycle(debug=args.debug)
                time.sleep(60)
            except KeyboardInterrupt:
                log.info("🛑 Detenido por usuario")
                break
