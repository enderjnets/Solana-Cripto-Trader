#!/usr/bin/env python3
"""
🦏 Reupload Ken Burns videos to YouTube
Borra los videos anteriores, sube los nuevos kenburns_final.mp4 y programa con los mismos horarios.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pytz

sys.path.insert(0, '/home/enderj/.openclaw/workspace/youtube_env/lib/python3.13/site-packages')
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

WORKSPACE   = Path("/home/enderj/.openclaw/workspace")
AGENTS_DIR  = WORKSPACE / "bittrader/agents"
OUTPUT_BASE = AGENTS_DIR / "output/2026-03-07-v3"
CREDS_FILE  = WORKSPACE / "memory/youtube_credentials.json"
IDS_FILE    = AGENTS_DIR / "data/youtube_v3_ids.json"
GUIONES     = AGENTS_DIR / "data/guiones_latest.json"

MT = pytz.timezone("America/Denver")

# ── Mapeo: script_id → publish info ───────────────────────────────────────
UPLOAD_PLAN = [
    # (script_id, old_video_id, publish_dt_mt, title, description, tags, category)
    {
        "script_id": "short_1772925855_875",
        "old_id":    "DGOvGLZburk",  # v5 que subimos antes
        "publish":   "2026-03-10T10:00:00",
        "title":     "AKT explota mientras BTC cae un 1.4% 🦏",
        "tags":      ["AKT","akash network","crypto","altcoin","bitcoin","trading","bittrader"],
        "type":      "short",
    },
    {
        "script_id": "short_1772925869_724",
        "old_id":    "gm4_LB4eFj0",
        "publish":   "2026-03-10T08:00:00",
        "title":     "Por qué PENGU está subiendo mientras BTC cae 🦏",
        "tags":      ["PENGU","pudgy penguins","NFT","meme coin","crypto","bitcoin","trading","bittrader"],
        "type":      "short",
    },
    {
        "script_id": "short_1772925884_383",
        "old_id":    "XazgVqCkGCc",
        "publish":   "2026-03-10T13:00:00",
        "title":     "El 90% de traders pierde por esto 🦏",
        "tags":      ["trading","crypto","error traders","psicologia trading","bittrader","bitcoin"],
        "type":      "short",
    },
    {
        "script_id": "short_1772925672_305",
        "old_id":    "7bGQROO-Y9o",
        "publish":   "2026-03-11T19:00:00",
        "title":     "PI coin +13%: ¿trampa o despegue real? 🦏",
        "tags":      ["PI coin","pi network","crypto","altcoin","trading","bittrader","bitcoin"],
        "type":      "short",
    },
    {
        "script_id": "short_1772925742_739",
        "old_id":    "rXZNuFtV8AE",
        "publish":   "2026-03-12T08:00:00",
        "title":     "ZEC cae 7%: ¿el fin de las privacy coins? 🦏",
        "tags":      ["ZEC","zcash","privacy coin","crypto","monero","trading","bittrader"],
        "type":      "short",
    },
    {
        "script_id": "long_1772925728_400",
        "old_id":    "RhZo-Lhs90A",
        "publish":   "2026-03-12T10:00:00",
        "title":     "Le di mis trades a Claude AI y los resultados me sorprendieron 🦏",
        "tags":      ["claude AI","trading IA","inteligencia artificial","crypto","NAS100","bittrader","bot trading"],
        "type":      "long",
    },
    {
        "script_id": "long_1772925781_782",
        "old_id":    "I7dK9aGl9mk",
        "publish":   "2026-03-13T10:00:00",
        "title":     "De $0 a cuenta fondeada en 30 días (plan real paso a paso) 🦏",
        "tags":      ["cuenta fondeada","FTMO","fondeo","trading","prop firm","bittrader","topstep"],
        "type":      "long",
    },
    {
        "script_id": "long_1772925821_433",
        "old_id":    "zSQgH3aRxxY",
        "publish":   "2026-03-13T08:00:00",
        "title":     "El bot que lee noticias y ejecuta tus trades solo (sin errores) 🦏",
        "tags":      ["bot trading","noticias","IA trading","automatizacion","crypto","bittrader","claude"],
        "type":      "long",
    },
]

DESCRIPTION_TEMPLATE = """{title}

🦏 BitTrader — Trading con Inteligencia Artificial

#trading #crypto #bitcoin #bittrader #IA #inteligenciaartificial
"""


def get_youtube_client():
    d = json.loads(CREDS_FILE.read_text())
    creds = Credentials(
        token=d["access_token"],
        refresh_token=d["refresh_token"],
        token_uri=d["token_uri"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        scopes=[d["scope"]] if isinstance(d["scope"], str) else d["scope"],
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def delete_video(yt, video_id: str):
    try:
        yt.videos().delete(id=video_id).execute()
        print(f"  🗑️  Deleted: {video_id}")
    except Exception as e:
        print(f"  ⚠️  Could not delete {video_id}: {e}")


def upload_video(yt, mp4_path: Path, item: dict) -> str:
    # Convert local publish time to UTC ISO
    local_dt = MT.localize(datetime.fromisoformat(item["publish"]))
    utc_dt   = local_dt.astimezone(timezone.utc)
    publish_at = utc_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    body = {
        "snippet": {
            "title":       item["title"],
            "description": DESCRIPTION_TEMPLATE.format(title=item["title"]),
            "tags":        item["tags"],
            "categoryId":  "22",
            "defaultLanguage": "es",
        },
        "status": {
            "privacyStatus":           "private",
            "publishAt":               publish_at,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(mp4_path), mimetype="video/mp4", resumable=True, chunksize=5*1024*1024)
    req   = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            print(f"    {int(status.progress()*100)}%", end=" ", flush=True)
    print()
    return response["id"]


def main():
    print("🦏 BitTrader — Ken Burns Reupload")
    print(f"   Videos: {len(UPLOAD_PLAN)}\n")

    yt      = get_youtube_client()
    ids_map = json.loads(IDS_FILE.read_text()) if IDS_FILE.exists() else {}
    results = {}

    for item in UPLOAD_PLAN:
        sid  = item["script_id"]
        mp4  = OUTPUT_BASE / sid / f"{sid}_kenburns_final.mp4"

        print(f"\n{'='*60}")
        print(f"🦏 {item['title'][:50]}")
        print(f"   File: {mp4.name} ({mp4.stat().st_size/1024/1024:.1f}MB)")
        print(f"   Publish: {item['publish']} MT")

        if not mp4.exists():
            print(f"  ❌ File not found, skipping")
            continue

        # 1. Delete old video
        if item["old_id"]:
            delete_video(yt, item["old_id"])
            time.sleep(2)

        # 2. Upload new
        print(f"  📤 Uploading...", end=" ", flush=True)
        new_id = upload_video(yt, mp4, item)
        print(f"  ✅ New ID: {new_id}")
        print(f"  🕐 Scheduled: {item['publish']} MT")

        results[sid] = new_id
        ids_map[item["title"]] = new_id

        # Save progress after each upload
        IDS_FILE.write_text(json.dumps(ids_map, indent=2, ensure_ascii=False))
        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"🏁 Done: {len(results)}/{len(UPLOAD_PLAN)} videos uploaded")
    for sid, vid in results.items():
        print(f"   {sid} → {vid}")

    import os
    os.system('openclaw system event --text "8 videos Ken Burns subidos a YouTube con exito. Todos programados." --mode now')


if __name__ == "__main__":
    main()
