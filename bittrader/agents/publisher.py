#!/usr/bin/env python3
"""
📤 BitTrader Publisher — Agente Publicador
Sube videos a YouTube con horarios inteligentes basados en datos del Scout.
Ejecutar: python3 agents/publisher.py
"""
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YouTube env
sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))

YT_CREDS        = WORKSPACE / "memory/youtube_credentials.json"
PRODUCTION_FILE = DATA_DIR / "production_latest.json"
UPLOAD_QUEUE    = DATA_DIR / "upload_queue.json"
SCHEDULE_FILE   = DATA_DIR / "optimal_schedule.json"

# Category IDs
CAT_SHORTS     = "28"  # Science & Technology (best for crypto/trading shorts)
CAT_EDUCATION  = "27"  # Education (for long-form)

DEFAULT_TAGS_SHORTS = [
    "crypto", "trading", "bitcoin", "shorts", "finanzas",
    "criptomonedas", "bittrader", "inversion", "blockchain"
]
DEFAULT_TAGS_LONG = [
    "crypto", "trading", "bitcoin", "finanzas", "criptomonedas",
    "educacion financiera", "bittrader", "inversion", "blockchain",
    "NAS100", "futuros"
]


# ══════════════════════════════════════════════════════════════════════════
# YOUTUBE CLIENT
# ══════════════════════════════════════════════════════════════════════════

def get_youtube_client():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    d = json.loads(YT_CREDS.read_text())
    creds = Credentials(
        token=d["access_token"],
        refresh_token=d["refresh_token"],
        token_uri=d["token_uri"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        scopes=d.get("scope", "").split()
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


# ══════════════════════════════════════════════════════════════════════════
# SMART SCHEDULING — Data-Driven (from Schedule Optimizer)
# ══════════════════════════════════════════════════════════════════════════

# Track assigned slots to avoid collisions within a batch
_assigned_slots = []


def _load_optimal_schedule() -> dict:
    """Load schedule data from Schedule Optimizer."""
    if SCHEDULE_FILE.exists():
        try:
            return json.loads(SCHEDULE_FILE.read_text())
        except Exception:
            pass
    return {}


def get_optimal_schedule(video_type: str, scout_data: dict = None) -> datetime:
    """
    Determina el mejor horario de publicación basado en datos REALES
    del Schedule Optimizer (análisis de 250+ videos históricos).

    Lógica:
    1. Carga horarios óptimos del Schedule Optimizer
    2. Filtra los mejores slots (día + hora) ordenados por VPD esperado
    3. Asigna el próximo slot disponible que no colisione con otros del batch
    4. Máximo 1 video por slot de 2 horas (para no saturar)

    Videos se suben como PRIVADOS y se programan para publicarse
    en el horario óptimo automáticamente.

    Retorna datetime en UTC.
    """
    global _assigned_slots
    now = datetime.now(timezone.utc)

    # Load data-driven schedule
    schedule_data = _load_optimal_schedule()
    schedule = schedule_data.get("schedule", {})
    best_hours_data = schedule_data.get("best_hours", {})
    best_days_data = schedule_data.get("best_days", {})

    # Extract ranked slots from optimizer
    optimizer_slots = schedule.get("slots", [])

    # Build ranked hours from real data (fallback to optimizer slots)
    if best_hours_data.get("best_hours"):
        ranked_hours = [h["hour"] for h in best_hours_data["best_hours"]]
    elif optimizer_slots:
        ranked_hours = list(dict.fromkeys(s["hour"] for s in optimizer_slots))
    else:
        # Hardcoded fallback (MT to UTC: +7 in summer, +6 in winter)
        ranked_hours = [16, 14, 1, 19, 22]  # UTC — based on historical best

    # Build ranked days from real data
    if best_days_data.get("best_days"):
        ranked_days = [d["day"] for d in best_days_data["best_days"]]
    elif optimizer_slots:
        ranked_days = list(dict.fromkeys(s["day_number"] for s in optimizer_slots))
    else:
        ranked_days = [4, 3, 1]  # Fri, Thu, Tue

    # For shorts: use all ranked hours (more publishing slots)
    # For longs: use only top 2 hours (prime time only)
    if video_type == "long":
        hours_to_use = ranked_hours[:2]
    else:
        hours_to_use = ranked_hours[:4]

    # Find next available slot
    # Strategy: scan next 7 days, prefer ranked days, then fill any day
    for day_offset in range(1, 8):
        candidate_date = now + timedelta(days=day_offset)
        weekday = candidate_date.weekday()

        # Score this day (ranked days get priority)
        if weekday not in ranked_days and day_offset < 4:
            continue  # Skip non-optimal days in the first 3 days

        for h in hours_to_use:
            candidate = candidate_date.replace(
                hour=h, minute=0, second=0, microsecond=0
            )

            # Must be in the future
            if candidate <= now:
                continue

            # Check not already assigned in this batch
            # Also enforce 2-hour buffer between videos
            collision = False
            for assigned in _assigned_slots:
                if abs((candidate - assigned).total_seconds()) < 7200:  # 2 hours
                    collision = True
                    break

            if not collision:
                _assigned_slots.append(candidate)
                return candidate

    # If all optimal slots taken, fall back to any available slot
    for day_offset in range(1, 14):
        for h in ranked_hours:
            candidate = (now + timedelta(days=day_offset)).replace(
                hour=h, minute=0, second=0, microsecond=0
            )
            if candidate > now and candidate not in _assigned_slots:
                _assigned_slots.append(candidate)
                return candidate

    # Ultimate fallback: tomorrow at best hour
    fallback = (now + timedelta(days=1)).replace(
        hour=ranked_hours[0] if ranked_hours else 16,
        minute=0, second=0, microsecond=0
    )
    _assigned_slots.append(fallback)
    return fallback


def reset_schedule_slots():
    """Reset assigned slots for a new batch."""
    global _assigned_slots
    _assigned_slots = []


def build_description(script: dict, video_type: str) -> str:
    base_desc = script.get("description", "")
    if not base_desc:
        base_desc = script.get("title", "")

    hashtags_short = "\n\n#Crypto #Trading #Bitcoin #Shorts #Finanzas #Criptomonedas #BitTrader #Inversion"
    hashtags_long  = "\n\n#Crypto #Trading #Bitcoin #Finanzas #Criptomonedas #Educacion #BitTrader #NAS100 #Futuros"

    footer = (
        "\n\n━━━━━━━━━━━━━━━━━━━━━\n"
        "🔔 Suscríbete para más contenido de trading y crypto\n"
        "📈 Canal: BitTrader\n"
        "━━━━━━━━━━━━━━━━━━━━━"
    )

    hashtags = hashtags_short if video_type == "short" else hashtags_long
    return base_desc + hashtags + footer


# ══════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════

def set_thumbnail(yt, video_id: str, thumbnail_path: Path) -> bool:
    """Upload custom thumbnail for a video."""
    from googleapiclient.http import MediaFileUpload

    if not thumbnail_path.exists():
        print(f"       ⚠️ Thumbnail no encontrada: {thumbnail_path}")
        return False

    try:
        media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
        yt.thumbnails().set(videoId=video_id, media_body=media).execute()
        print(f"       🖼️ Thumbnail subida")
        return True
    except Exception as e:
        print(f"       ⚠️ Thumbnail error: {str(e)[:100]}")
        return False


def upload_video(yt, video: dict, scout_data: dict = None) -> dict:
    """Sube un video a YouTube. Retorna resultado con video_id."""
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    video_path = Path(video["output_file"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")

    vtype      = video.get("type", "short")
    script     = video.get("script_data", {})
    title      = video.get("title", video_path.stem)[:100]
    tags       = script.get("tags", []) or (DEFAULT_TAGS_SHORTS if vtype == "short" else DEFAULT_TAGS_LONG)
    cat_id     = CAT_SHORTS if vtype == "short" else CAT_EDUCATION
    description = build_description(script, vtype)

    # Schedule
    publish_at = get_optimal_schedule(vtype, scout_data)
    publish_at_str = publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    body = {
        "snippet": {
            "title":       title,
            "description": description,
            "tags":        tags[:15],
            "categoryId":  cat_id,
            "defaultLanguage": "es",
        },
        "status": {
            "privacyStatus":       "private",  # Will be published at scheduled time
            "publishAt":           publish_at_str,
            "selfDeclaredMadeForKids": False,
        }
    }

    # Show schedule in Mountain Time for user
    mt_offset = timedelta(hours=-7)  # MDT (summer) or -6 (MST winter)
    publish_mt = publish_at + mt_offset
    day_name = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][publish_mt.weekday()]

    print(f"    📤 Subiendo: {title[:55]}")
    print(f"       Tipo: {vtype} | Cat: {cat_id}")
    print(f"       📅 Programado: {day_name} {publish_mt.strftime('%d/%m %I:%M %p')} MT")

    media = MediaFileUpload(str(video_path), mimetype="video/mp4",
                            resumable=True, chunksize=1024*1024*5)

    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"       ⬆️  {pct}%...", end="\r")

    video_id = response["id"]
    url = f"https://youtube.com/{'shorts/' if vtype=='short' else 'watch?v='}{video_id}"
    print(f"\n    ✅ Subido: {url}")
    print(f"       🔒 Privado hasta {day_name} {publish_mt.strftime('%d/%m %I:%M %p')} MT")

    # Upload thumbnail if available
    thumb_uploaded = False
    thumb_path = None

    # Check for thumbnail in video data or directory
    if video.get("thumbnail"):
        thumb_path = Path(video["thumbnail"])
    else:
        # Look in video directory
        video_dir = video_path.parent
        for candidate in ["thumbnail.jpg", "thumbnail.png", "thumbnail_final.jpg"]:
            p = video_dir / candidate
            if p.exists():
                thumb_path = p
                break

    if thumb_path and thumb_path.exists():
        thumb_uploaded = set_thumbnail(yt, video_id, thumb_path)

    return {
        "video_id":    video_id,
        "url":         url,
        "scheduled":   publish_at_str,
        "scheduled_mt": f"{day_name} {publish_mt.strftime('%d/%m %I:%M %p')} MT",
        "thumbnail":   thumb_uploaded,
        "status":      "uploaded",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════
# QUEUE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def load_queue() -> list:
    if not UPLOAD_QUEUE.exists():
        return []
    return json.loads(UPLOAD_QUEUE.read_text())


def save_queue(queue: list):
    UPLOAD_QUEUE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))


def check_duplicate_topic(title: str, uploaded_today: list) -> bool:
    """Verifica si ya se publicó un video de tema similar hoy."""
    import re
    # Extract key words
    words = set(re.findall(r'\b\w{5,}\b', title.lower()))
    for prev in uploaded_today:
        prev_words = set(re.findall(r'\b\w{5,}\b', prev.lower()))
        overlap = words & prev_words
        if len(overlap) >= 2:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run_publisher(production_file: Path = None, video_ids: list = None,
                  process_queue: bool = False) -> dict:
    print("\n📤 BitTrader Publisher iniciando...")
    reset_schedule_slots()  # Clean slate for this batch

    src = production_file or PRODUCTION_FILE
    production = None
    videos_to_upload = []

    # Load from queue if requested
    if process_queue:
        queue = load_queue()
        if queue:
            print(f"  📋 Procesando cola: {len(queue)} videos pendientes")
            videos_to_upload = queue
        else:
            print("  ℹ️  Cola vacía")

    elif src.exists():
        production = json.loads(src.read_text())
        all_vids   = production.get("videos", [])
        videos_to_upload = [
            v for v in all_vids
            if v.get("status") == "success"
            and v.get("output_file")
            and (not video_ids or v.get("script_id") in video_ids)
        ]
        print(f"  📋 {len(videos_to_upload)} videos listos para subir")
    else:
        raise FileNotFoundError(f"No hay producción en {src}. Ejecuta producer.py primero.")

    # Load scout data for smart scheduling
    scout_latest = DATA_DIR / "scout_latest.json"
    scout_data   = json.loads(scout_latest.read_text()) if scout_latest.exists() else None

    # Connect YouTube
    print("  🔗 Conectando con YouTube API...")
    try:
        yt = get_youtube_client()
    except Exception as e:
        print(f"  ❌ Error YouTube auth: {e}")
        # Save to queue and exit
        queue = load_queue()
        queue.extend(videos_to_upload)
        save_queue(queue)
        print(f"  💾 {len(videos_to_upload)} videos guardados en cola")
        return {"status": "auth_failed", "queued": len(videos_to_upload)}

    results        = []
    failed         = []
    uploaded_today = []
    queue_pending  = []

    for video in videos_to_upload:
        title = video.get("title", "")

        # Check duplicate topic today
        if check_duplicate_topic(title, uploaded_today):
            print(f"  ⏭️  Omitiendo (tema similar ya publicado hoy): {title[:50]}")
            queue_pending.append({**video, "skip_reason": "duplicate_topic"})
            continue

        try:
            # Attach script data if available
            if production:
                scripts = {s["id"]: s for s in production.get("scripts", [])}
                video["script_data"] = scripts.get(video.get("script_id"), {})

            upload_result = upload_video(yt, video, scout_data)
            results.append({**video, **upload_result})
            uploaded_today.append(title)
            time.sleep(3)  # Rate limiting

        except Exception as e:
            err_str = str(e)
            print(f"    ❌ Error subiendo '{title[:40]}': {err_str[:100]}")

            if "quotaExceeded" in err_str or "quota" in err_str.lower():
                print("    ⚠️  Quota de YouTube agotada — guardando en cola")
                queue_pending.extend(videos_to_upload[videos_to_upload.index(video):])
                break
            else:
                failed.append({**video, "error": err_str})

    # Save remaining to queue
    if queue_pending:
        existing_queue = load_queue()
        existing_queue.extend(queue_pending)
        save_queue(existing_queue)
        print(f"  💾 {len(queue_pending)} videos en cola para siguiente ciclo")

    # Save report
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "date":         date_str,
        "uploaded":     results,
        "failed":       failed,
        "queued":       len(queue_pending),
        "stats": {
            "total_attempted": len(videos_to_upload),
            "uploaded":        len(results),
            "failed":          len(failed),
            "queued":          len(queue_pending),
        }
    }
    report_file = DATA_DIR / f"publish_report_{date_str}.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n✅ Publisher completado → {report_file.name}")
    print(f"   📤 {len(results)} subidos | ❌ {len(failed)} fallidos | ⏳ {len(queue_pending)} en cola")
    for r in results:
        print(f"   📹 [{r['type'].upper()}] {r['title'][:50]} → {r.get('url','?')}")
    return report


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BitTrader Publisher — Agente Publicador")
    parser.add_argument("--production",   type=str, help="Ruta a production JSON")
    parser.add_argument("--ids",          type=str, help="IDs de videos a subir, separados por coma")
    parser.add_argument("--queue",        action="store_true", help="Procesar cola de pendientes")
    parser.add_argument("--show-queue",   action="store_true", help="Ver cola actual")
    args = parser.parse_args()

    if args.show_queue:
        queue = load_queue()
        print(f"\n📋 Cola de upload ({len(queue)} videos):")
        for v in queue:
            print(f"  - [{v.get('type','?').upper()}] {v.get('title','?')[:60]}")
        print()
        sys.exit(0)

    prod_path  = Path(args.production) if args.production else None
    script_ids = [s.strip() for s in args.ids.split(",")] if args.ids else None

    report = run_publisher(
        production_file=prod_path,
        video_ids=script_ids,
        process_queue=args.queue
    )

    print("\n── Resultado ─────────────────────────────")
    stats = report.get("stats", {})
    print(f"  Subidos:  {stats.get('uploaded', 0)}")
    print(f"  Fallidos: {stats.get('failed', 0)}")
    print(f"  En cola:  {stats.get('queued', 0)}")
    print("─────────────────────────────────────────\n")
