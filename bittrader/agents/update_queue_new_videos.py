#!/usr/bin/env python3
"""
Update upload_queue.json:
1. Add 5 new videos (short_1774708351_945, short_1774708455_613, short_1774708563_481,
                      long_1774708225_575, long_1774708531_991)
2. Fix SOL video: change status from blocked → pending (thumbnail regenerated)
"""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

AGENTS = Path("/home/enderj/.openclaw/workspace/bittrader/agents")
QUEUE_FILE = AGENTS / "data/upload_queue.json"
THUMBS_DIR = AGENTS / "data/thumbnails"
OUTPUT_BASE = AGENTS / "output/2026-03-28"

queue = json.loads(QUEUE_FILE.read_text())

# ── Already in queue? Check ─────────────────────────────────────────────────
existing_ids = {item.get("script_id") for item in queue}
print(f"Queue has {len(queue)} items. Existing IDs for target videos:")
for sid in ["short_1774708351_945","short_1774708455_613","short_1774708563_481",
            "long_1774708225_575","long_1774708531_991","short_1774708280_155"]:
    print(f"  {sid}: {'EXISTS' if sid in existing_ids else 'NOT FOUND'}")

# ── New videos data ─────────────────────────────────────────────────────────
now = datetime.now(timezone.utc)
# Schedule starting ~3 hours from now, staggered every 2.5 hours
base_schedule = now + timedelta(hours=3)

new_videos = [
    {
        "script_id": "short_1774708351_945",
        "title": "TAO: La crypto IA que explota HOY",
        "description": "TAO esta rompiendo records. Descubre por que esta crypto de inteligencia artificial esta explotando y si vale la pena entrar ahora. #crypto #IA #TAO #trading",
        "type": "short",
        "tags": ["TAO","Bittensor","inteligencia artificial","crypto","altcoins","trading","Bitcoin","DeFi","inversion","criptomonedas"],
        "output_file": str(OUTPUT_BASE / "short_1774708351_945/short_1774708351_945.mp4"),
        "status": "ready",
        "thumbnail_path": str(THUMBS_DIR / "short_1774708351_945_thumbnail.jpg"),
        "thumbnail": str(OUTPUT_BASE / "short_1774708351_945/thumbnail.jpg"),
        "thumbnail_ready": True,
        "scheduled_date": (base_schedule + timedelta(hours=0)).isoformat(),
        "created_at": now.isoformat(),
        "qa_passed": False,
        "gate_passed": False,
    },
    {
        "script_id": "short_1774708455_613",
        "title": "SIREN y RAIN: ¿Próximos gainers?",
        "description": "SIREN y RAIN están mostrando señales interesantes en los charts. Analizo los niveles clave que debes vigilar. #crypto #trading #altcoins",
        "type": "short",
        "tags": ["crypto","trading","altcoins","SIREN","RAIN","criptomonedas"],
        "output_file": str(OUTPUT_BASE / "short_1774708455_613/short_1774708455_613.mp4"),
        "status": "ready",
        "thumbnail_path": str(THUMBS_DIR / "short_1774708455_613_thumbnail.jpg"),
        "thumbnail": str(OUTPUT_BASE / "short_1774708455_613/thumbnail.jpg"),
        "thumbnail_ready": True,
        "scheduled_date": (base_schedule + timedelta(hours=2.5)).isoformat(),
        "created_at": now.isoformat(),
        "qa_passed": False,
        "gate_passed": False,
    },
    {
        "script_id": "short_1774708563_481",
        "title": "SIREN: Que es y por que sube HOY",
        "description": "Descubre que es SIREN, el token de agentes de IA que esta rompiendo el mercado. Analisis de su tendencia y potencial en cripto. #SIREN #Crypto #IA",
        "type": "short",
        "tags": ["SIREN token","criptomonedas","agentes IA","trading","Bitcoin","altcoins","DeFi","tendencia cripto","tokens IA","BSC"],
        "output_file": str(OUTPUT_BASE / "short_1774708563_481/short_1774708563_481.mp4"),
        "status": "ready",
        "thumbnail_path": str(THUMBS_DIR / "short_1774708563_481_thumbnail.jpg"),
        "thumbnail": str(OUTPUT_BASE / "short_1774708563_481/thumbnail.jpg"),
        "thumbnail_ready": True,
        "scheduled_date": (base_schedule + timedelta(hours=5)).isoformat(),
        "created_at": now.isoformat(),
        "qa_passed": False,
        "gate_passed": False,
    },
    {
        "script_id": "long_1774708225_575",
        "title": "Claude IA: Automatiza tu Trading HOY",
        "description": "Descubre como usar Claude IA para automatizar tu analisis de trading. Paso a paso, con código real que puedes usar hoy mismo. #claude #trading #IA #automatizacion",
        "type": "long",
        "tags": ["claude","IA","trading","automatizacion","bitcoin","cripto","futuros","bots","analisis tecnico"],
        "output_file": str(OUTPUT_BASE / "long_1774708225_575/long_1774708225_575.mp4"),
        "status": "ready",
        "thumbnail_path": str(THUMBS_DIR / "long_1774708225_575_thumbnail.jpg"),
        "thumbnail": str(OUTPUT_BASE / "long_1774708225_575/thumbnail.jpg"),
        "thumbnail_ready": True,
        "scheduled_date": (base_schedule + timedelta(hours=7.5)).isoformat(),
        "created_at": now.isoformat(),
        "qa_passed": False,
        "gate_passed": False,
    },
    {
        "script_id": "long_1774708531_991",
        "title": "3 reglas que salvaron mi cuenta de trading",
        "description": "Descubre las 3 reglas de gestion de riesgo que pueden salvar tu cuenta de trading de la quiebra. Aprende a calcular el tamano de posicion correcto, usar stop loss efectivo y diversificar como los profesionales. #trading #gestionderiesgo #bitcoin",
        "type": "long",
        "tags": ["trading","gestion de riesgo","stop loss","posicion size","bitcoin","cripto","futuros","nas100","sp500","ftmo"],
        "output_file": str(OUTPUT_BASE / "long_1774708531_991/long_1774708531_991.mp4"),
        "status": "ready",
        "thumbnail_path": str(THUMBS_DIR / "long_1774708531_991_thumbnail.jpg"),
        "thumbnail": str(OUTPUT_BASE / "long_1774708531_991/thumbnail.jpg"),
        "thumbnail_ready": True,
        "scheduled_date": (base_schedule + timedelta(hours=10)).isoformat(),
        "created_at": now.isoformat(),
        "qa_passed": False,
        "gate_passed": False,
    },
]

# ── Add new videos (skip if already exists) ─────────────────────────────────
added = []
for v in new_videos:
    if v["script_id"] not in existing_ids:
        queue.append(v)
        added.append(v["script_id"])
        print(f"  ➕ Added: {v['script_id']} — {v['title']}")
    else:
        print(f"  ⚠️  Already exists: {v['script_id']} — skipping")

# ── Fix SOL video ────────────────────────────────────────────────────────────
sol_fixed = False
for item in queue:
    if item.get("script_id") == "short_1774708280_155":
        old_status = item.get("status")
        item["status"] = "ready"
        item["thumbnail_path"] = str(THUMBS_DIR / "short_1774708280_155_thumbnail.jpg")
        item["thumbnail"] = str(OUTPUT_BASE / "short_1774708280_155/thumbnail.jpg")
        item["thumbnail_ready"] = True
        item["gate_passed"] = False
        item["gate_issues"] = []
        item["thumbnail_regenerated_at"] = now.isoformat()
        item["scheduled_date"] = (base_schedule + timedelta(hours=1.5)).isoformat()
        item.pop("gate_checked_at", None)
        sol_fixed = True
        print(f"  🔧 SOL: status {old_status} → ready, thumbnail updated, scheduled {item['scheduled_date']}")
        break

if not sol_fixed:
    print("  ⚠️  SOL video not found in queue!")

# ── Save ─────────────────────────────────────────────────────────────────────
QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))
print(f"\n✅ Queue saved: {len(queue)} total entries, {len(added)} added, SOL fixed={sol_fixed}")
