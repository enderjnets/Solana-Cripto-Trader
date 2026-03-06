#!/usr/bin/env python3
"""
🎯 Agente 6: Orchestrator
Ejecuta el ciclo completo: MarketData → RiskManager → Strategy → Executor → Reporter.
Watchdog: reintenta agentes fallidos y reporta errores.

Uso:
    python3 orchestrator.py --once    # Un ciclo y termina
    python3 orchestrator.py --live    # Ciclo continuo cada 60s
    python3 orchestrator.py --report  # Solo ejecuta Reporter
    python3 orchestrator.py --debug   # Con output detallado

Modos de trading:
    --paper   → Paper trading (default, seguro)
    --live    → Trades reales (requiere wallet configurada)
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
log = logging.getLogger("orchestrator")

# ─── Importar agentes ─────────────────────────────────────────────────────────

def import_agents():
    """Importa todos los módulos de agentes."""
    import market_data
    import risk_manager
    import strategy
    import executor
    import reporter
    return market_data, risk_manager, strategy, executor, reporter


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


# ─── Ciclo Principal ─────────────────────────────────────────────────────────

def run_cycle(safe: bool = True, debug: bool = False, report_only: bool = False) -> dict:
    """
    Ejecuta un ciclo completo del sistema.
    Orden: MarketData → RiskManager → Strategy → Executor → Reporter
    """
    cycle_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info("=" * 60)
    log.info(f"🔄 CICLO INICIADO — {now}")
    mode_label = "📄 PAPER" if safe else "🔴 LIVE"
    log.info(f"   Modo: {mode_label}")
    log.info("=" * 60)

    # Importar agentes
    try:
        md, rm, st, ex, rp = import_agents()
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
        log.info("🌐 [1/5] Market Data")
        ok, result = run_agent("MarketData", md.run, debug=debug)
        results["market_data"] = {"ok": ok}
        if not ok:
            log.error("❌ MarketData falló — abortando ciclo")
            return {**results, "status": "ABORTED", "step": "market_data"}

        prices_ok = result.get("prices_ok", 0)
        log.info(f"   → {prices_ok} precios obtenidos")

        # ── Paso 2: Risk Manager ─────────────────────────────────────
        log.info("━" * 40)
        log.info("🛡️  [2/5] Risk Manager")
        ok, result = run_agent("RiskManager", rm.run, debug=debug)
        results["risk_manager"] = {"ok": ok}
        if not ok:
            log.warning("⚠️  RiskManager falló — continuando sin gestión de riesgo")
        else:
            portfolio_status = result.get("portfolio_status", "ACTIVE")
            approved = result.get("tokens_approved", 0)
            log.info(f"   → Status: {portfolio_status} | {approved} tokens aprobados")

        # ── Paso 3: Strategy ─────────────────────────────────────────
        log.info("━" * 40)
        log.info("🧠 [3/5] Strategy")
        ok, result = run_agent("Strategy", st.run, debug=debug)
        results["strategy"] = {"ok": ok}
        if not ok:
            log.warning("⚠️  Strategy falló — sin señales este ciclo")
            n_signals = 0
        else:
            n_signals = result.get("total_signals", 0)
            log.info(f"   → {n_signals} señal(es) generada(s)")

        # ── Paso 4: Executor ─────────────────────────────────────────
        log.info("━" * 40)
        log.info("⚡ [4/5] Executor")
        ok, result = run_agent("Executor", ex.run, safe=safe, debug=debug)
        results["executor"] = {"ok": ok}
        if not ok:
            log.warning("⚠️  Executor falló")
        else:
            opened = result.get("opened_this_cycle", 0)
            closed = result.get("closed_this_cycle", 0)
            capital = result.get("capital", 0)
            wr = result.get("win_rate", 0)
            log.info(f"   → Abiertos: {opened} | Cerrados: {closed} | Capital: ${capital:.2f} | WR: {wr:.1f}%")

        # ── Paso 5: Reporter ─────────────────────────────────────────
        log.info("━" * 40)
        log.info("📊 [5/5] Reporter")
        ok, result = run_agent("Reporter", rp.run)
        results["reporter"] = {"ok": ok}
        if result:
            alerts = result.get("alerts_sent", [])
            if alerts:
                log.info(f"   → 🔔 {len(alerts)} alerta(s) enviada(s)")

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

def run_live(safe: bool = True, debug: bool = False, interval: int = 60):
    """Ciclo continuo, una iteración cada `interval` segundos."""
    log.info(f"🚀 Modo LIVE — ciclo cada {interval}s")
    log.info(f"   Modo trading: {'PAPER 📄' if safe else 'REAL 🔴'}")
    log.info("   Ctrl+C para detener\n")

    cycle_num = 0
    while True:
        cycle_num += 1
        log.info(f"\n{'='*60}")
        log.info(f"📍 CICLO #{cycle_num}")

        try:
            run_cycle(safe=safe, debug=debug)
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
        description="🎯 Solana Trading Bot — Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python3 orchestrator.py --once          # Un ciclo, paper trading
  python3 orchestrator.py --live          # Ciclo continuo, paper trading
  python3 orchestrator.py --once --debug  # Un ciclo con output detallado
  python3 orchestrator.py --report        # Solo reporte
  python3 orchestrator.py --live --real   # Ciclo continuo REAL (¡cuidado!)
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

    args = parser.parse_args()

    # Safety check para modo real
    safe_mode = not args.real
    if not safe_mode:
        print("\n" + "🔴" * 30)
        print("ADVERTENCIA: Modo REAL activado — se ejecutarán TRADES REALES en Solana")
        print("Asegúrate de tener HOT_WALLET_PRIVATE_KEY configurada en .env")
        print("🔴" * 30 + "\n")
        confirm = input("Escribe 'CONFIRMO TRADES REALES' para continuar: ")
        if confirm.strip() != "CONFIRMO TRADES REALES":
            print("Cancelado.")
            sys.exit(0)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.once:
        result = run_cycle(safe=safe_mode, debug=args.debug)
        sys.exit(0 if result.get("status") == "OK" else 1)

    elif args.live:
        run_live(safe=safe_mode, debug=args.debug, interval=args.interval)
        sys.exit(0)

    elif args.report:
        try:
            rp = __import__("reporter")
            result = rp.run(daily=True)
            sys.exit(0)
        except Exception as e:
            log.error(f"❌ Reporter error: {e}")
            sys.exit(1)
