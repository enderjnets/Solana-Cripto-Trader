#!/usr/bin/env python3
"""
🎯 BitTrader Orchestrator — Orquestador Principal
Ejecuta el pipeline completo: Scout → Creator → Producer → Publisher
Ejecutar: python3 agents/orchestrator.py --full
"""
import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ensure agents/ is in sys.path for imports
AGENTS_DIR = BITTRADER / "agents"
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

MINIMAX_KEY = json.loads((BITTRADER / "keys/minimax.json").read_text())["minimax_api_key"]


# ══════════════════════════════════════════════════════════════════════════
# TELEGRAM REPORT
# ══════════════════════════════════════════════════════════════════════════

def send_telegram_report(message: str):
    """Intenta enviar reporte por Telegram si hay configuración disponible."""
    try:
        import requests
        # Busca config de Telegram en workspace
        tg_config = WORKSPACE / "memory/telegram_config.json"
        if not tg_config.exists():
            print(f"  📱 [Telegram no configurado]\n{message}")
            return
        cfg     = json.loads(tg_config.read_text())
        bot_tok = cfg.get("bot_token", "")
        chat_id = cfg.get("chat_id", "")
        if not bot_tok or not chat_id:
            return
        url = f"https://api.telegram.org/bot{bot_tok}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"  ⚠️ Telegram falló: {e}")


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════

def step_scout(args) -> dict:
    print("\n" + "═"*50)
    print("PASO 1/4: 🔭 SCOUT")
    print("═"*50)
    from scout import run_scout
    return run_scout(skip_youtube=args.no_youtube)


def step_creator(args, scout_result: dict = None) -> dict:
    print("\n" + "═"*50)
    print("PASO 2/4: 🎨 CREATOR")
    print("═"*50)
    from creator import run_creator
    return run_creator(dry_run=args.dry_run)


def step_producer(args, creator_result: dict = None) -> dict:
    print("\n" + "═"*50)
    print("PASO 3/4: 🎬 PRODUCER")
    print("═"*50)
    from producer import run_producer
    guiones_file = None
    if args.guiones:
        guiones_file = Path(args.guiones)
    return run_producer()


def step_publisher(args, producer_result: dict = None) -> dict:
    print("\n" + "═"*50)
    print("PASO 4/4: 📤 PUBLISHER")
    print("═"*50)
    from publisher import run_publisher
    prod_file = None
    if args.production:
        prod_file = Path(args.production)
    return run_publisher(production_file=prod_file, process_queue=False)


# ══════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════

def build_final_report(results: dict, started_at: datetime, args) -> str:
    now      = datetime.now(timezone.utc)
    duration = (now - started_at).total_seconds()
    mins     = int(duration // 60)
    secs     = int(duration % 60)

    lines = [
        f"🎯 <b>BitTrader Pipeline — Reporte Final</b>",
        f"📅 {now.strftime('%Y-%m-%d %H:%M')} UTC | ⏱️ {mins}m {secs}s",
        "",
    ]

    # Scout
    scout = results.get("scout", {})
    if scout and not scout.get("error"):
        btc    = scout.get("crypto", {}).get("bitcoin", {})
        trends = scout.get("crypto", {}).get("trending_coins", [])
        alert  = scout.get("alert")
        lines += [
            "🔭 <b>Scout</b>: ✅",
            f"  BTC: ${btc.get('price_usd',0):,} ({btc.get('change_24h',0):+.1f}%)",
            f"  Trending: {', '.join(c['symbol'] for c in trends[:5])}",
        ]
        if alert:
            lines += [f"  🚨 ALERTA: {alert['alerts'][0][:80]}"]
        vids = scout.get("youtube", {}).get("all_videos", [])
        if vids:
            lines += [f"  YouTube: {len(vids)} videos analizados"]
    elif "scout" in results:
        lines += [f"🔭 <b>Scout</b>: ❌ {results['scout'].get('error','?')[:60]}"]

    # Creator
    creator = results.get("creator", {})
    if creator and not creator.get("error"):
        stats = creator.get("stats", {})
        lines += [
            f"🎨 <b>Creator</b>: ✅",
            f"  Shorts: {stats.get('shorts',0)} | Largos: {stats.get('longs',0)} | Errores: {stats.get('errors',0)}",
        ]
        for s in creator.get("scripts", [])[:4]:
            emoji = "✅" if s.get("status") != "error" else "❌"
            lines += [f"  {emoji} [{s.get('type','?').upper()}] {s.get('title','?')[:50]}"]
    elif "creator" in results:
        lines += [f"🎨 <b>Creator</b>: ❌ {results['creator'].get('error','?')[:60]}"]

    # Producer
    producer = results.get("producer", {})
    if producer and not producer.get("error"):
        stats = producer.get("stats", {})
        lines += [
            f"🎬 <b>Producer</b>: ✅",
            f"  Producidos: {stats.get('success',0)} | Fallidos: {stats.get('failed',0)}",
        ]
    elif "producer" in results:
        lines += [f"🎬 <b>Producer</b>: ❌ {results['producer'].get('error','?')[:60]}"]

    # Publisher
    publisher = results.get("publisher", {})
    if publisher and not publisher.get("error"):
        stats = publisher.get("stats", {})
        lines += [
            f"📤 <b>Publisher</b>: ✅",
            f"  Subidos: {stats.get('uploaded',0)} | Fallidos: {stats.get('failed',0)} | Cola: {stats.get('queued',0)}",
        ]
        for v in publisher.get("uploaded", []):
            lines += [f"  📹 {v.get('title','?')[:45]} → {v.get('url','?')}"]
    elif "publisher" in results:
        lines += [f"📤 <b>Publisher</b>: ❌ {results['publisher'].get('error','?')[:60]}"]

    lines += ["", "🤖 BitTrader Agent System"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BitTrader Orchestrator — Pipeline completo de producción"
    )
    # Modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full",         action="store_true", help="Pipeline completo (default)")
    mode_group.add_argument("--scout-only",   action="store_true", help="Solo ejecutar Scout")
    mode_group.add_argument("--create-only",  action="store_true", help="Solo ejecutar Creator")
    mode_group.add_argument("--produce-only", action="store_true", help="Solo ejecutar Producer")
    mode_group.add_argument("--publish-only", action="store_true", help="Solo ejecutar Publisher")
    mode_group.add_argument("--process-queue",action="store_true", help="Procesar cola de uploads pendientes")

    # Options
    parser.add_argument("--no-youtube",  action="store_true", help="Scout omite YouTube (ahorra quota)")
    parser.add_argument("--dry-run",     action="store_true", help="Creator no llama al LLM")
    parser.add_argument("--guiones",     type=str, help="Archivo guiones para Producer")
    parser.add_argument("--production",  type=str, help="Archivo producción para Publisher")
    parser.add_argument("--no-telegram", action="store_true", help="No enviar reporte Telegram")

    args = parser.parse_args()

    # Default to --full if no mode specified
    if not any([args.scout_only, args.create_only, args.produce_only,
                args.publish_only, args.process_queue, args.full]):
        args.full = True

    started_at = datetime.now(timezone.utc)
    print(f"\n{'═'*50}")
    print(f"🎯 BitTrader Orchestrator — {started_at.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'═'*50}")

    results = {}
    error_occurred = False

    try:
        # ── Scout ────────────────────────────────────────────────────────
        if args.full or args.scout_only:
            try:
                results["scout"] = step_scout(args)
            except Exception as e:
                print(f"\n❌ Scout falló: {e}")
                traceback.print_exc()
                results["scout"] = {"error": str(e)}
                if not args.full:
                    error_occurred = True

        # ── Creator ──────────────────────────────────────────────────────
        if args.full or args.create_only:
            try:
                results["creator"] = step_creator(args, results.get("scout"))
            except Exception as e:
                print(f"\n❌ Creator falló: {e}")
                traceback.print_exc()
                results["creator"] = {"error": str(e)}
                if not args.full:
                    error_occurred = True

        # ── Producer ─────────────────────────────────────────────────────
        if args.full or args.produce_only:
            # Skip if Creator had a fatal error and no guiones file provided
            if results.get("creator", {}).get("error") and not args.guiones:
                print("\n⏭️  Producer omitido (Creator falló y no hay guiones alternativos)")
            else:
                try:
                    results["producer"] = step_producer(args, results.get("creator"))
                except Exception as e:
                    print(f"\n❌ Producer falló: {e}")
                    traceback.print_exc()
                    results["producer"] = {"error": str(e)}

        # ── Publisher ────────────────────────────────────────────────────
        if args.full or args.publish_only or args.process_queue:
            if results.get("producer", {}).get("error") and not args.production:
                print("\n⏭️  Publisher omitido (Producer falló y no hay producción alternativa)")
            else:
                try:
                    if args.process_queue:
                        from publisher import run_publisher
                        results["publisher"] = run_publisher(process_queue=True)
                    else:
                        results["publisher"] = step_publisher(args, results.get("producer"))
                except Exception as e:
                    print(f"\n❌ Publisher falló: {e}")
                    traceback.print_exc()
                    results["publisher"] = {"error": str(e)}

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrumpido por usuario")
        error_occurred = True
    except Exception as e:
        print(f"\n\n❌ Error crítico: {e}")
        traceback.print_exc()
        error_occurred = True

    # ── Final Report ─────────────────────────────────────────────────────
    print("\n" + "═"*50)
    print("REPORTE FINAL")
    print("═"*50)

    report_text = build_final_report(results, started_at, args)
    # Strip HTML for console
    import re
    console_text = re.sub(r'<[^>]+>', '', report_text)
    print(console_text)

    # Save orchestration log
    date_str = started_at.strftime("%Y-%m-%d")
    log_file = DATA_DIR / f"orchestration_{date_str}.json"
    log = {
        "started_at":  started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "args":        vars(args),
        "results_summary": {
            "scout":     "ok" if not results.get("scout", {}).get("error") else "error",
            "creator":   "ok" if not results.get("creator", {}).get("error") else "error",
            "producer":  "ok" if not results.get("producer", {}).get("error") else "error",
            "publisher": "ok" if not results.get("publisher", {}).get("error") else "error",
        }
    }
    log_file.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    # Telegram
    if not args.no_telegram:
        send_telegram_report(report_text)

    sys.exit(1 if error_occurred else 0)


if __name__ == "__main__":
    main()
