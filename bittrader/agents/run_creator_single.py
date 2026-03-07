#!/usr/bin/env python3
"""
Creator Single - Genera UN guion a la vez.
Uso: python3 run_creator_single.py <index>
Donde index es 0-9 (el número del guion en el plan)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data")

def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Cargar scout
    from creator import load_scout_data, build_content_plan, generate_short_script, generate_long_script

    scout = load_scout_data()
    plan = build_content_plan(scout)

    if idx >= len(plan):
        print(f"Index {idx} fuera de rango (plan tiene {len(plan)} items)")
        sys.exit(1)

    item = plan[idx]
    print(f"Generando guion {idx+1}/{len(plan)}: [{item['type']}] {item['topic'][:60]}")

    if item["type"] == "short":
        script = generate_short_script(item, scout)
    else:
        script = generate_long_script(item, scout)

    print(f"✅ '{script.get('title','?')[:55]}' — {len(script.get('script',''))} chars | {script.get('status')}")

    # Cargar guiones existentes o crear nuevo resultado
    guiones_file = DATA_DIR / "guiones_wip.json"
    if guiones_file.exists():
        data = json.loads(guiones_file.read_text())
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date": date_str,
            "plan": plan,
            "scripts": [],
            "stats": {"total": 0, "shorts": 0, "longs": 0, "errors": 0}
        }

    # Agregar script (evitar duplicados por idx)
    existing_ids = {s.get("plan_idx") for s in data["scripts"]}
    if idx not in existing_ids:
        script["plan_idx"] = idx
        data["scripts"].append(script)
        data["stats"]["total"] = len(data["scripts"])
        data["stats"]["shorts"] = sum(1 for s in data["scripts"] if s.get("type") == "short")
        data["stats"]["longs"] = sum(1 for s in data["scripts"] if s.get("type") == "long")
        data["stats"]["errors"] = sum(1 for s in data["scripts"] if s.get("status") in ("error","timeout"))
        data["generated_at"] = datetime.now(timezone.utc).isoformat()

    guiones_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Guardado en guiones_wip.json ({data['stats']['total']} scripts)")

    # Si ya tenemos todos los guiones del plan → copiar a latest
    if len(data["scripts"]) >= len(plan):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out = DATA_DIR / f"guiones_{date_str}.json"
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        (DATA_DIR / "guiones_latest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"✅ TODOS LOS GUIONES COMPLETOS → guiones_latest.json")

if __name__ == "__main__":
    main()
