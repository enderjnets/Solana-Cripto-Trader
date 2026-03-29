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

LOG_FILE = Path.home() / ".config" / "solana-jupiter-bot" / "modular.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Rotación de logs (si > 50MB) ─────────────────────────────────────────────
def _rotate_log_if_needed():
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size >= 50 * 1024 * 1024:
            old = LOG_FILE.with_suffix(".log.old")
            LOG_FILE.rename(old)
    except Exception:
        pass

_rotate_log_if_needed()

# ── Logging unificado: formato [YYYY-MM-DD HH:MM:SS] [NIVEL] mensaje ─────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),            # stdout → capturado por watchdog en LOG_FILE
    ],
)
log = logging.getLogger("orchestrator")

# ─── Circuit Breaker ─────────────────────────────────────────────────────────
# Detecta loops de emergency closes y detiene el bot automáticamente.
CIRCUIT_BREAKER_FILE = DATA_DIR / "circuit_breaker_state.json"
CIRCUIT_BREAKER_LIMIT = 10  # Máximo de emergency closes POR CICLO antes de detener el bot


def _circuit_breaker_check_and_record(n_emergency_closes: int) -> bool:
    """
    Registra el número de emergency closes del ciclo actual.
    Si supera CIRCUIT_BREAKER_LIMIT → escribe STOP_TRADING, pausa el bot y retorna True.
    Retorna True si el circuit breaker se activó (bot debe detenerse).
    """
    if n_emergency_closes < CIRCUIT_BREAKER_LIMIT:
        return False

    log.error("=" * 60)
    log.error(f"🔴 CIRCUIT BREAKER ACTIVADO: {n_emergency_closes} emergency closes en un ciclo")
    log.error(f"   Límite: {CIRCUIT_BREAKER_LIMIT} — BOT DETENIDO AUTOMÁTICAMENTE")
    log.error("=" * 60)

    # Crear STOP_TRADING para que executor no abra más posiciones
    stop_file = DATA_DIR / "STOP_TRADING"
    stop_file.write_text(
        f"Circuit breaker activado: {n_emergency_closes} emergency closes en un ciclo.\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
        f"Elimina este archivo para reanudar el bot.\n"
    )

    # Registrar estado del circuit breaker
    cb_state = {
        "triggered": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "emergency_closes_in_cycle": n_emergency_closes,
        "limit": CIRCUIT_BREAKER_LIMIT,
        "action": "STOP_TRADING file creado, portfolio pausado",
    }
    CIRCUIT_BREAKER_FILE.write_text(json.dumps(cb_state, indent=2))

    # Pausar el portfolio
    portfolio_file = DATA_DIR / "portfolio.json"
    try:
        if portfolio_file.exists():
            portfolio_data = json.loads(portfolio_file.read_text())
            portfolio_data["status"] = "PAUSED"
            portfolio_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            portfolio_file.write_text(json.dumps(portfolio_data, indent=2))
            log.error("   ✅ Portfolio marcado como PAUSED")
    except Exception as e:
        log.error(f"   ⚠️ No se pudo pausar portfolio: {e}")

    return True


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
    _cycle_emergency_closes = 0  # Circuit breaker: contador de emergency closes este ciclo
    
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

    # Definir portfolio_file aquí para que esté disponible en todos los pasos siguientes
    portfolio_file = DATA_DIR / "portfolio.json"

    # Fix 3: Snapshot de posiciones ANTES del executor para detectar recién abiertas
    pre_exec_symbols = set()
    try:
        pre_portfolio = json.loads(portfolio_file.read_text()) if portfolio_file.exists() else {}
        pre_exec_symbols = {p["symbol"] for p in pre_portfolio.get("positions", []) if p.get("status") == "open"}
    except Exception:
        pass

    try:
        result = ex.run(safe=safe, debug=debug)
        results["executor"] = {"ok": True}
    except Exception as e:
        log.warning(f"   ⚠️ Error: {e}")
        results["executor"] = {"ok": False}

    # Fix 3: Detectar símbolos recién abiertos en este ciclo
    just_opened_symbols = set()
    try:
        post_portfolio = json.loads(portfolio_file.read_text()) if portfolio_file.exists() else {}
        post_exec_symbols = {p["symbol"] for p in post_portfolio.get("positions", []) if p.get("status") == "open"}
        just_opened_symbols = post_exec_symbols - pre_exec_symbols
        if just_opened_symbols:
            log.info(f"   🛡️ Recién abiertas (protegidas de cierre este ciclo): {just_opened_symbols}")
    except Exception:
        pass

    # Paso 4b: Smart Rotation
    log.info("━" * 40)
    log.info("🔄 [4b/5] Smart Rotation")
    
    portfolio_data = {}
    if portfolio_file.exists():
        portfolio_data = json.loads(portfolio_file.read_text())
    
    stale_positions = []
    if portfolio_data:
        try:
            # OPTIMIZADO 2026-03-24: max_hours 72→96, improvement_hours 24→36
            # Las posiciones necesitan más tiempo para desarrollar momentum
            stale_positions = rm.check_stale_losing_positions(
                portfolio_data, max_hours=96, improvement_hours=36
            )
        except Exception as e:
            log.warning(f"   ⚠️ Error: {e}")
        
        if stale_positions:
            log.info(f"   → {len(stale_positions)} posición(es) a cerrar:")
            for pos in stale_positions:
                log.info(f"     - {pos['symbol']}: {pos['reason']}")
            # FIX #2: Excluir posiciones recién abiertas del Smart Rotation (cooldown same-cycle)
            symbols_to_close = [p["symbol"] for p in stale_positions if p["symbol"] not in just_opened_symbols]
            skipped_fresh = [p["symbol"] for p in stale_positions if p["symbol"] in just_opened_symbols]
            if skipped_fresh:
                log.info(f"   🛡️ Smart Rotation cooldown — skipped recién abiertas: {skipped_fresh}")
            if symbols_to_close:
                try:
                    market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                    history = json.loads((DATA_DIR / "trade_history.json").read_text()) if (DATA_DIR / "trade_history.json").exists() else []
                    closed = ex.close_positions_emergency(portfolio_data, symbols_to_close, market_data, history, reason="SMART_ROTATION")
                    ex.save_portfolio(portfolio_data)
                    (DATA_DIR / "trade_history.json").write_text(json.dumps(history, indent=2))
                    log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Smart Rotation")
                except Exception as e:
                    log.warning(f"   ⚠️ Error cerrando posiciones: {e}")
            results["smart_rotation"] = {"ok": True, "closed": len(symbols_to_close)}
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
                # Ejecutar cierre real de todas las posiciones
                try:
                    # Fix 3: Excluir posiciones recién abiertas del cierre por Daily Target
                    open_symbols = [p["symbol"] for p in portfolio_data.get("positions", [])
                                    if p.get("status") == "open" and p["symbol"] not in just_opened_symbols]
                    if open_symbols:
                        market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                        history = json.loads((DATA_DIR / "trade_history.json").read_text()) if (DATA_DIR / "trade_history.json").exists() else []
                        # FIX: Usar reason específica "DAILY_TARGET" en lugar de "EMERGENCY_CLOSE"
                        close_reason_label = f"DAILY_TARGET: {target_result['close_reason'][:60]}"
                        closed = ex.close_positions_emergency(portfolio_data, open_symbols, market_data, history, reason=close_reason_label)
                        _cycle_emergency_closes += len(closed)
                        ex.save_portfolio(portfolio_data)
                        (DATA_DIR / "trade_history.json").write_text(json.dumps(history, indent=2))
                        log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Daily Target")
                    elif just_opened_symbols:
                        log.info(f"   🛡️ Skipped cierre — todas las posiciones son recién abiertas")
                except Exception as e:
                    log.warning(f"   ⚠️ Error cerrando posiciones: {e}")
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
            # Fix 3: Excluir posiciones recién abiertas de recomendaciones de cierre
            # OPTIMIZADO 2026-03-24: umbral 0.70→0.80 — reducir false CLOSE signals
            # La tasa de cierre por POSITION_DECISION tenía WR=100% (solo 2 casos)
            # pero la causa raíz del EC excesivo era confidence demasiado bajo
            close_recs  = [d for d in decisions if d["action"] == "CLOSE"  and d["confidence"] >= 0.80
                           and d["symbol"] not in just_opened_symbols]
            reduce_recs = [d for d in decisions if d["action"] == "REDUCE" and d["confidence"] >= 0.70]

            if close_recs:
                log.info(f"   🔴 CERRAR ({len(close_recs)}): {', '.join(d['symbol'] for d in close_recs)}")
                # Ejecutar cierre real
                try:
                    close_symbols = [d["symbol"] for d in close_recs]
                    market_data = json.loads((DATA_DIR / "market_latest.json").read_text()) if (DATA_DIR / "market_latest.json").exists() else {}
                    history = json.loads((DATA_DIR / "trade_history.json").read_text()) if (DATA_DIR / "trade_history.json").exists() else []
                    # FIX: Usar reason específica para Position Decision
                    closed = ex.close_positions_emergency(portfolio_data, close_symbols, market_data, history, reason="POSITION_DECISION")
                    ex.save_portfolio(portfolio_data)
                    (DATA_DIR / "trade_history.json").write_text(json.dumps(history, indent=2))
                    log.info(f"   ✅ Cerradas {len(closed)} posición(es) por Position Decision")
                except Exception as e:
                    log.warning(f"   ⚠️ Error cerrando posiciones: {e}")
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

    # ── Circuit Breaker Check ─────────────────────────────────────────────────
    if _cycle_emergency_closes >= CIRCUIT_BREAKER_LIMIT:
        if _circuit_breaker_check_and_record(_cycle_emergency_closes):
            log.error("🛑 CIRCUIT BREAKER: Bot detenido. Revisar causa y eliminar STOP_TRADING para reanudar.")
            results["circuit_breaker"] = {"triggered": True, "closes": _cycle_emergency_closes}
            return results

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

    # ── HEALTH LINE ───────────────────────────────────────────────────────────
    # Extraer equity y posiciones del portfolio para la línea de salud
    try:
        pf = json.loads(portfolio_file.read_text()) if portfolio_file.exists() else {}
        equity = pf.get("equity", pf.get("balance", 0.0))
        n_positions = len([p for p in pf.get("positions", []) if p.get("status") == "open"])
    except Exception:
        equity = 0.0
        n_positions = 0
    exit_code = 0 if health_score >= 4 else 1
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    health_line = (
        f"[{ts}] [HEALTH] Ciclo completado — exit_code={exit_code} "
        f"— equity=${equity:.2f} — posiciones={n_positions}"
    )
    # Escribir directamente para garantizar el tag [HEALTH] literal
    print(health_line, flush=True)
    try:
        with open(LOG_FILE, "a") as _lf:
            _lf.write(health_line + "\n")
    except Exception:
        pass

    # Rotar log al finalizar si supera 50MB (doble check desde Python)
    _rotate_log_if_needed()

    return results


def run_token_scanner():
    """Ejecuta el scanner de tokens para encontrar nuevas oportunidades."""
    try:
        import token_scanner
        log.info("━" * 40)
        log.info("🔍 [SCANNER] Buscando nuevas oportunidades...")
        result = token_scanner.scan(debug=False)
        if result.get("tokens_added"):
            log.info(f"   ➕ Nuevos tokens: {result['tokens_added']}")
        log.info(f"   📊 Total oportunidades: {result.get('opportunities_found', 0)}")
        return result
    except Exception as e:
        log.warning(f"   ⚠️ Scanner error: {e}")
        return {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--live", action="store_true", help="Run in continuous loop (alias for default loop mode)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles in loop mode (default: 60)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--scan-interval", type=int, default=10, help="Run token scanner every N cycles (default: 10)")
    args = parser.parse_args()

    if args.once:
        run_cycle(debug=args.debug)
    else:
        # --live or bare invocation: run continuous loop
        interval = args.interval if args.interval else 60
        scan_interval = args.scan_interval if args.scan_interval else 10
        cycle_count = 0
        
        log.info(f"🔄 Modo continuo — intervalo: {interval}s")
        log.info(f"🔍 Token Scanner cada {scan_interval} ciclos (~{scan_interval}min)")
        
        while True:
            try:
                cycle_count += 1
                
                # Ejecutar scanner cada N ciclos
                if cycle_count % scan_interval == 0:
                    run_token_scanner()
                
                run_cycle(debug=args.debug)
                time.sleep(interval)
            except KeyboardInterrupt:
                log.info("🛑 Detenido por usuario")
                break
