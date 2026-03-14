#!/usr/bin/env python3
"""
📤 Queue Processor — Procesa upload_queue automáticamente según scheduled_date

Cron job recomendado: Cada hora en la hora (0 * * * *)
  */30 * * * * cd /home/enderj/.openclaw/workspace/bittrader && python3 agents/queue_processor.py

Lógica:
1. Carga upload_queue.json
2. Filtra videos con scheduled_date <= AHORA y status='ready'
3. Sube videos usando publisher.py (respetando rate limits)
4. Actualiza status a 'uploaded' con video_id
5. Guarda queue actualizado

Videos que fallan por quota quedan en cola para el siguiente ciclo.
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YouTube env
sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))

UPLOAD_QUEUE = DATA_DIR / "upload_queue.json"
PROCESSED_LOG = DATA_DIR / "queue_processor_log.json"

# ── Logging ────────────────────────────────────────────────────────────────
def log(message: str, level: str = "INFO"):
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = {"timestamp": timestamp, "level": level, "message": message}
    
    # Append to log file
    if PROCESSED_LOG.exists():
        logs = json.loads(PROCESSED_LOG.read_text())
        logs.append(log_entry)
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
    else:
        logs = [log_entry]
    
    PROCESSED_LOG.write_text(json.dumps(logs, indent=2))
    print(f"[{level}] {message}")


def load_queue() -> list:
    if not UPLOAD_QUEUE.exists():
        log("upload_queue.json not found", "ERROR")
        return []
    return json.loads(UPLOAD_QUEUE.read_text())


def save_queue(queue: list):
    UPLOAD_QUEUE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))


def get_youtube_client():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    YT_CREDS = WORKSPACE / "memory/youtube_credentials.json"
    d = json.loads(YT_CREDS.read_text())
    
    # Handle scopes: can be list or string
    scopes_value = d.get("scopes") or d.get("scope", "")
    if isinstance(scopes_value, str):
        scopes_value = scopes_value.split()
    
    creds = Credentials(
        token=d.get("token") or d.get("access_token"),
        refresh_token=d["refresh_token"],
        token_uri=d["token_uri"],
        client_id=d["client_id"],
        client_secret=d["client_secret"],
        scopes=scopes_value
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        d["access_token"] = creds.token
        YT_CREDS.write_text(json.dumps(d, indent=2))
    return build("youtube", "v3", credentials=creds)


def upload_video(yt, video_data: dict) -> dict:
    """Upload a single video to YouTube."""
    from googleapiclient.http import MediaFileUpload
    
    video_path = Path(video_data["output_file"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Video metadata
    body = {
        "snippet": {
            "title": video_data["title"],
            "description": video_data.get("description", ""),
            "tags": video_data.get("tags", []),
            "categoryId": "28" if video_data.get("type") == "short" else "27"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }

    # For shorts: use short keyword
    if video_data.get("type") == "short":
        if "description" not in body["snippet"]:
            body["snippet"]["description"] = ""
        body["snippet"]["description"] = "#shorts " + body["snippet"]["description"]

    # Upload
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    
    insert_request = yt.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )
    
    response = insert_request.execute()
    return {"video_id": response["id"], "status": "uploaded"}


def upload_thumbnail(yt, video_id: str, thumbnail_path: str):
    """Upload thumbnail to video."""
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    
    if not thumbnail_path or not Path(thumbnail_path).exists():
        log(f"Thumbnail not found: {thumbnail_path}", "WARNING")
        return
    
    try:
        media = MediaFileUpload(thumbnail_path, resumable=True)
        yt.thumbnails().set(videoId=video_id, media_body=media).execute()
        log(f"Thumbnail uploaded for {video_id}", "INFO")
    except HttpError as e:
        log(f"Failed to upload thumbnail: {e}", "WARNING")


def schedule_video(yt, video_id: str, publish_at: str):
    """Schedule video for future publication."""
    try:
        yt.videos().update(
            part="status",
            body={
                "id": video_id,
                "status": {
                    "privacyStatus": "scheduled",
                    "publishAt": publish_at
                }
            }
        ).execute()
        log(f"Video {video_id} scheduled for {publish_at}", "INFO")
    except Exception as e:
        log(f"Failed to schedule video {video_id}: {e}", "WARNING")


def main():
    log("=" * 70, "INFO")
    log("📤 Queue Processor iniciando...", "INFO")
    log("=" * 70, "INFO")

    now = datetime.now(timezone.utc)
    log(f"Hora actual UTC: {now.isoformat()}", "INFO")
    
    # Load queue
    queue = load_queue()
    if not queue:
        log("Cola vacía, nada que procesar", "INFO")
        return
    
    log(f"Videos en cola: {len(queue)}", "INFO")

    # Filter videos ready to upload
    ready_to_upload = []
    for i, item in enumerate(queue):
        status = item.get("status", "ready")
        sched_date = item.get("scheduled_date")
        
        # Skip if already uploaded or not ready
        if status in ["uploaded", "published", "uploading"]:
            continue
        if status != "ready":
            log(f"Item {i}: status='{status}' - skipping", "DEBUG")
            continue
        
        # Check scheduled date
        if sched_date:
            sched_dt = datetime.fromisoformat(sched_date)
            # Convert to UTC for comparison if needed
            if sched_dt.tzinfo is None:
                sched_dt = sched_dt.replace(tzinfo=timezone.utc)
            
            if sched_dt <= now:
                ready_to_upload.append((i, item))
                log(f"Item {i}: Ready - scheduled={sched_date}, now={now.isoformat()}", "DEBUG")
            else:
                log(f"Item {i}: Future - scheduled={sched_date}", "DEBUG")
        else:
            # No scheduled date - upload now
            ready_to_upload.append((i, item))
            log(f"Item {i}: Ready (no scheduled date)", "DEBUG")

    log(f"Videos listos para subir: {len(ready_to_upload)}", "INFO")

    if not ready_to_upload:
        log("Nada que subir ahora", "INFO")
        return

    # Connect YouTube
    log("Conectando a YouTube API...", "INFO")
    try:
        yt = get_youtube_client()
        log("Conexión exitosa", "INFO")
    except Exception as e:
        log(f"Error conectando a YouTube: {e}", "ERROR")
        return

    # Process videos
    uploaded = []
    failed = []
    quota_hit = False

    for idx, item in ready_to_upload:
        if quota_hit:
            log("Quota alcanzada, deteniendo", "WARNING")
            break

        i, video_data = idx, item
        title = video_data.get("title", "Unknown")
        log(f"\n{'─' * 70}", "INFO")
        log(f"Procesando: {title}", "INFO")

        # Mark as uploading
        queue[i]["status"] = "uploading"
        save_queue(queue)

        try:
            # Upload video
            log(f"Subiendo video...", "INFO")
            upload_result = upload_video(yt, video_data)
            video_id = upload_result["video_id"]
            
            log(f"✅ Video subido: {video_id}", "INFO")
            
            # Upload thumbnail
            thumb_path = video_data.get("thumbnail")
            if thumb_path:
                upload_thumbnail(yt, video_id, thumb_path)
            
            # Update status
            queue[i].update({
                "status": "uploaded",
                "video_id": video_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            })
            
            uploaded.append(video_id)
            log(f"Estado actualizado a 'uploaded'", "INFO")

        except Exception as e:
            err_str = str(e)
            log(f"❌ Error: {err_str[:200]}", "ERROR")
            
            if "quota" in err_str.lower():
                quota_hit = True
                queue[i]["status"] = "pending_quota"  # Try again later
            else:
                queue[i]["status"] = "failed"
                queue[i]["error"] = err_str[:500]
                failed.append({"title": title, "error": err_str[:200]})
        
        # Save after each video
        save_queue(queue)

    # Final summary
    log("\n" + "=" * 70, "INFO")
    log("📊 RESUMEN DEL CICLO", "INFO")
    log("=" * 70, "INFO")
    log(f"Videos subidos exitosamente: {len(uploaded)}", "INFO")
    log(f"Videos fallidos: {len(failed)}", "INFO")
    log(f"Quota alcanzada: {'SÍ' if quota_hit else 'NO'}", "INFO")
    
    if uploaded:
        log("\n📹 IDs de videos subidos:", "INFO")
        for vid_id in uploaded:
            log(f"   • {vid_id}", "INFO")
    
    if failed:
        log("\n❌ Videos fallidos:", "WARNING")
        for f in failed:
            log(f"   • {f['title']}: {f['error']}", "WARNING")


if __name__ == "__main__":
    main()
