#!/usr/bin/env python3
"""
🤖 ORCHESTRATOR CON AGENTES IA - Solana Trading Bot
Ejecuta el ciclo completo con IA: MarketData → AI Researcher → AI Strategy → Executor → AI Explainability → Reporter

Uso:
    python3 orchestrator_ai.py --once    # Un ciclo y termina
    python3 orchestrator_ai.py --live    # Ciclo continuo cada 60s
"""

import os
import sys
import json
import logging
import argparse
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Añadir directorio padre para imports relativos
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("orchestrator_ai")

# ─── Importar agentes ─────────────────────────────────────────────────────────

def import_agents():
    """Importa todos los módulos de agentes."""
    import market_data
    import risk_manager
    import executor
    import reporter
    import ai_researcher
    import ai_strategy
    import ai_explainability

    # Nuevos agentes opcionales
    leverage_available = True
    auto_learner_available = True

    try:
        import leverage_manager
    except ImportError:
        leverage_manager = None
        leverage_available = False

    try:
        import auto_learner
    except ImportError:
        auto_learner = None
        auto_learner_available = False

    return (market_data, risk_manager, executor, reporter,
            ai_researcher, ai_strategy, ai_explainability,
            leverage_manager, auto_learner,
            leverage_available, auto_learner_available)


# ─── Watchdog ────────────────────────────────────────────────────────────────

def run_agent(name: str, fn, retries: int = 2, **kwargs) -> tuple[bool, any]:
    """
    Ejecuta un agente con reintentos.
    Retorna (success, result).
    """
    for attempt in range(1, retries + 1):
        try:
            result = fn(**kwargs)
            return True, result
        except Exception as e:
            if attempt < retries:
                log.warning(f"⚠️  {name} falló (intento {attempt}/{retries}): {e}")
                time.sleep(2)
            else:
                log.error(f"❌ {name} falló definitivamente: {e}")
                if log.level == logging.DEBUG:
                    traceback.print_exc()
                return False, None
    return False, None


# ─── Ciclo Principal con Agentes IA ─────────────────────────────────────────────

def run_cycle(safe: bool = True, debug: bool = False, report_only: bool = False,
               leverage_mode: bool = False, auto_learn: bool = True) -> dict:
    """
    Ejecuta un ciclo completo del sistema con Agentes IA.
    Orden: MarketData → AI Researcher → Risk Manager → AI Strategy → Executor → AI Explainability → Reporter → [Leverage Manager] → [Auto-Learner]
    """
    cycle_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info("=" * 60)
    log.info(f"🔄 CICLO INICIADO — {now}")
    mode_label = "📄 PAPER" if safe else "🔴 LIVE"
    log.info(f"   Modo: {mode_label}")
    log.info("   🤖 Agentes IA: ACTIVOS")
    log.info("=" * 60)

    # Importar agentes
    try:
        (md, rm, ex, rp, ai_res, ai_strat, ai_exp,
         lev_mgr, auto_lrn, lev_avail, auto_avail) = import_agents()

        # Mostrar disponibilidad de agentes nuevos
        if lev_avail:
            log.info("✅ Leverage Manager disponible")
        else:
            log.info("⚠️  Leverage Manager no disponible")

        if auto_avail:
            log.info("✅ Auto-Learner disponible")
        else:
            log.info("⚠️  Auto-Learner no disponible")

        log.info("✅ Todos los agentes importados correctamente")
    except ImportError as e:
        log.error(f"❌ No se pudieron importar agentes: {e}")
        return {"status": "IMPORT_ERROR", "error": str(e)}

    results = {}

    if report_only:
        # Solo Reporter
        ok, result = run_agent("Reporter", rp.run, daily=False)
        results["reporter"] = {"ok": ok, "result": result}
    else:
        # ── Paso 1: Market Data ──────────────────────────────────────
        log.info("━" * 40)
        log.info("🌐 [1/9] Market Data")
        ok, result = run_agent("MarketData", md.run, debug=debug)
        results["market_data"] = {"ok": ok}
        if not ok:
            log.error("❌ MarketData falló — abortando ciclo")
            return {**results, "status": "ABORTED", "step": "market_data"}

        prices_ok = result.get("prices_ok", 0)
        log.info(f"   → {prices_ok} precios obtenidos")

        # ── Paso 2: AI Researcher ───────────────────────────────────
        log.info("━" * 40)
        log.info("🧠 [2/9] AI Researcher (Análisis de Mercado con LLM)")
        ok, result = run_agent("AI_Researcher", ai_res.run)
        results["ai_researcher"] = {"ok": ok}
        if not ok:
            log.warning("⚠️ AI Researcher falló — continuando con datos técnicos")
        else:
            trend = result.get("trend", "UNKNOWN")
            confidence = result.get("confidence", 0)
            fear_greed = result.get("fear_greed_value", 0)
            log.info(f"   → Tendencia: {trend} | Confianza: {confidence:.0%}")
            log.info(f"   → Fear & Greed: {fear_greed} ({result.get('fear_greed_label', 'N/A')})")

        # ── Paso 3: Risk Manager ─────────────────────────────────────
        log.info("━" * 40)
        log.info("🛡️  [3/9] Risk Manager")
        ok, result = run_agent("RiskManager", rm.run, debug=debug)
        results["risk_manager"] = {"ok": ok}
        if not ok:
            log.warning("⚠️ RiskManager falló — continuando sin gestión de riesgo")
        else:
            portfolio_status = result.get("portfolio_status", "ACTIVE")
            approved = result.get("tokens_approved", 0)
            log.info(f"   → Status: {portfolio_status} | {approved} tokens aprobados")

        # ── Paso 4: AI Strategy (Señales con LLM) ───────────────
        log.info("━" * 40)
        log.info("🎯 [4/9] AI Strategy (Señales con LLM)")
        ok, result = run_agent("AI_Strategy", ai_strat.run)
        results["ai_strategy"] = {"ok": ok}
        if not ok:
            log.warning("⚠️ AI Strategy falló — sin señales este ciclo")
            n_signals = 0
        else:
            signals = result.get("signals", [])
            n_signals = len(signals)
            log.info(f"   → {n_signals} señal(es) generada(s)")
            if n_signals > 0:
                for sig in signals[:3]:  # Mostrar las primeras 3
                    direction = sig.get("direction", "N/A").upper()
                    token = sig.get("token", "N/A")
                    size = sig.get("size_usd", 0)
                    log.info(f"      • {direction} {token} ${size:.2f}")

        # ── Paso 5: Executor ─────────────────────────────────────────
        log.info("━" * 40)
        log.info("⚡ [5/9] Executor")
        ok, result = run_agent("Executor", ex.run, safe=safe, debug=debug)
        results["executor"] = {"ok": ok}
        if not ok:
            log.warning("⚠️ Executor falló")
        else:
            opened = result.get("opened_this_cycle", 0)
            closed = result.get("closed_this_cycle", 0)
            capital = result.get("capital", 0)
            wr = result.get("win_rate", 0)
            log.info(f"   → Abiertos: {opened} | Cerrados: {closed} | Capital: ${capital:.2f} | WR: {wr:.1f}%")

        # ── Paso 6: AI Explainability ─────────────────────────────
        log.info("━" * 40)
        log.info("💬 [6/9] AI Explainability (Alertas y Explicaciones)")
        ok, result = run_agent("AI_Explainability", ai_exp.run)
        results["ai_explainability"] = {"ok": ok}
        if not ok:
            log.warning("⚠️ AI Explainability falló — sin explicaciones")
        else:
            summary = result.get("summary", "")
            if summary:
                log.info(f"   → {summary[:100]}...")

        # ── Paso 7: Reporter ─────────────────────────────────────
        log.info("━" * 40)
        log.info("📊 [7/9] Reporter")
        ok, result = run_agent("Reporter", rp.run)
        results["reporter"] = {"ok": ok}
        if result:
            alerts = result.get("alerts_sent", [])
            if alerts:
                log.info(f"   → 🔔 {len(alerts)} alerta(s) enviada(s)")

        # ── Paso 8: Leverage Manager (opcional) ──────────────
        if leverage_mode and lev_avail:
            log.info("━" * 40)
            log.info("🔗 [8/9] Leverage Manager (Drift Protocol)")
            ok, result = run_agent("LeverageManager", lev_mgr.run, debug=debug)
            results["leverage_manager"] = {"ok": ok}
            if not ok:
                log.warning("⚠️ LeverageManager falló")
            else:
                collateral = result.get("collateral_usd", 0)
                exposure = result.get("exposure_usd", 0)
                effective_lev = result.get("effective_leverage", 0)
                log.info(f"   → Colateral: ${collateral:.2f} | Exposición: ${exposure:.2f} | Lev: {effective_lev:.1f}x")
        elif leverage_mode and not lev_avail:
            log.warning("⚠️  Leverage mode activado pero Leverage Manager no disponible")

        # ── Paso 9: Auto-Learner (opcional) ───────────────────
        if auto_learn and auto_avail:
            log.info("━" * 40)
            log.info("🧠 [9/9] Auto-Learner (Autoaprendizaje)")
            ok, result = run_agent("AutoLearner", auto_lrn.run, debug=debug)
            results["auto_learner"] = {"ok": ok}
            if not ok:
                log.warning("⚠️ AutoLearner falló")
            else:
                confidence = result.get("confidence", 0)
                perf = result.get("performance", {})
                params = result.get("adaptive_params", {})
                log.info(f"   → Confianza: {confidence*100:.0f}% | SL: {params.get('sl_pct', 0):.2f}% | TP: {params.get('tp_pct', 0):.2f}%")
        elif auto_learn and not auto_avail:
            log.warning("⚠️  Auto-Learner no disponible")

    # ── Resumen ───────────────────────────────────────────────────────────
    elapsed = time.time() - cycle_start
    ok_count = sum(1 for v in results.values() if v.get("ok"))
    total = len(results)

    log.info("=" * 60)
    log.info(f"✅ CICLO COMPLETADO en {elapsed:.1f}s ({ok_count}/{total} agentes OK)")
    log.info("=" * 60)

    return {
        "status": "OK",
        "elapsed": round(elapsed, 1),
        "agents_ok": ok_count,
        "agents_total": total,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Loop Continuo ───────────────────────────────────────────────────────────

def run_live(safe: bool = True, debug: bool = False, interval: int = 60,
            leverage_mode: bool = False, auto_learn: bool = True):
    """Ciclo continuo, una iteración cada `interval` segundos."""
    log.info(f"🚀 Modo LIVE — ciclo cada {interval}s")
    log.info(f"   Modo trading: {'PAPER 📄' if safe else 'REAL 🔴'}")
    log.info(f"   🤖 Agentes IA: ACTIVOS")
    if leverage_mode:
        log.info(f"   🔗 Leverage Mode: ACTIVO (Drift Protocol)")
    if auto_learn:
        log.info(f"   🧠 Auto-Learning: ACTIVO")
    log.info("   Ctrl+C para detener\n")

    cycle_num = 0
    while True:
        cycle_num += 1
        log.info(f"\n{'='*60}")
        log.info(f"📍 CICLO #{cycle_num}")

        try:
            run_cycle(safe=safe, debug=debug, leverage_mode=leverage_mode, auto_learn=auto_learn)
        except KeyboardInterrupt:
            log.info("\n👋 Detenido por usuario")
            break
        except Exception as e:
            log.error(f"❌ Error inesperado en ciclo #{cycle_num}: {e}")
            traceback.print_exc()

        log.info(f"⏳ Esperando {interval}s para próximo ciclo...")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            log.info("\n👋 Detenido por usuario")
            break


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🤖 Solana Trading Bot con Agentes IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 orchestrator_ai.py --once          # Un ciclo, paper trading
  python3 orchestrator_ai.py --live          # Ciclo continuo, paper trading
  python3 orchestrator_ai.py --once --debug  # Un ciclo con output detallado
  python3 orchestrator_ai.py --report        # Solo reporte
  python3 orchestrator_ai.py --live --real   # Ciclo continuo REAL (¡cuidado!)
        """
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--once",   action="store_true", help="Ejecutar un solo ciclo")
    mode_group.add_argument("--live",   action="store_true", help="Modo continuo (cada 60s)")
    mode_group.add_argument("--report", action="store_true", help="Solo ejecutar Reporter")

    parser.add_argument("--real",    action="store_true",
                        help="ACTIVAR trades reales en Solana (default: paper)")
    parser.add_argument("--debug",   action="store_true", help="Output detallado")
    parser.add_argument("--interval", type=int, default=60,
                        help="Segundos entre ciclos en modo --live (default: 60)")
    parser.add_argument("--leverage", action="store_true",
                        help="ACTIVAR Leverage Manager (Drift Protocol con apalancamiento 1-10x)")
    parser.add_argument("--no-auto-learn", dest="auto_learn", action="store_false", default=True,
                        help="DESACTIVAR Auto-Learner (autoaprendizaje)")

    args = parser.parse_args()

    # Safety check para modo real
    safe_mode = not args.real
    if not safe_mode:
        print("\n" + "🔴" * 30)
        print("ADVERTENCIA: Modo REAL activado — se ejecutarán TRADES REALES en Solana")
        print("Asegúrate de tener HOT_WALLET_PRIVATE_KEY configurada en .env")
        print("🔴" * 30 + "\n")

    if args.once:
        run_cycle(safe=safe_mode, debug=args.debug, leverage_mode=args.leverage, auto_learn=args.auto_learn)
    elif args.live:
        run_live(safe=safe_mode, debug=args.debug, interval=args.interval,
                 leverage_mode=args.leverage, auto_learn=args.auto_learn)
    elif args.report:
        run_cycle(safe=safe_mode, report_only=True)
