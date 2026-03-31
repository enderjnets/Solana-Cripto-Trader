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
QUEUE_FILE = DATA_DIR / "upload_queue.json"

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
        return False
    
    try:
        media = MediaFileUpload(thumbnail_path, resumable=True)
        yt.thumbnails().set(videoId=video_id, media_body=media).execute()
        log(f"Thumbnail uploaded for {video_id}", "INFO")
        return True
    except HttpError as e:
        log(f"Failed to upload thumbnail: {e}", "WARNING")
        return False


def verify_thumbnail_on_youtube(yt, video_id: str) -> bool:
    """Verifica si el video tiene una thumbnail custom en YouTube (no auto-generated).
    
    YouTube auto-generated thumbnails tienen URLs de tipo /vi/<id>/default.jpg (baja res).
    Las custom thumbnails tienen /vi/<id>/maxresdefault.jpg.
    Retorna True si tiene thumbnail custom, False si no.
    """
    try:
        response = yt.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        items = response.get("items", [])
        if not items:
            log(f"Video {video_id} no encontrado en YouTube", "WARNING")
            return False
        thumbnails = items[0].get("snippet", {}).get("thumbnails", {})
        # Si tiene 'maxres' o 'standard', tiene thumbnail de alta calidad (custom o no)
        # La forma más confiable: verificar si el video tiene la thumbnail a través del
        # campo thumbnails().get() — si la API devuelve la thumbnail, es custom.
        has_maxres = "maxres" in thumbnails
        has_standard = "standard" in thumbnails
        log(f"Thumbnail check {video_id}: maxres={has_maxres}, standard={has_standard}", "DEBUG")
        return has_maxres or has_standard
    except Exception as e:
        log(f"Error verificando thumbnail {video_id}: {e}", "WARNING")
        return True  # Asumir OK si falla la verificación para no bloquear


def retry_failed_thumbnails(yt, queue: list) -> list:
    """Busca videos 'uploaded' sin thumbnail confirmada y reintenta subirla.
    
    Detecta dos casos:
    1. thumbnail_uploaded=False en el JSON
    2. status='thumbnail_failed'
    
    Retorna lista de video_ids corregidos.
    """
    fixed = []
    log("🔍 Verificando thumbnails de videos ya subidos...", "INFO")
    
    for i, item in enumerate(queue):
        status = item.get("status", "")
        video_id = item.get("video_id", "")
        
        # Solo procesar videos subidos sin thumbnail confirmada
        needs_retry = (
            status in ("uploaded", "published") and not item.get("thumbnail_uploaded", True)
        ) or status == "thumbnail_failed"
        
        if not needs_retry or not video_id:
            continue
        
        sid = item.get("script_id", "")
        log(f"⚠️ Video {video_id} ({item.get('title','?')[:40]}) — thumbnail_uploaded={item.get('thumbnail_uploaded')} — buscando miniatura...", "WARNING")
        
        # Buscar thumbnail en múltiples ubicaciones
        thumb_candidates = [
            item.get("thumbnail_path", ""),
            item.get("thumbnail", ""),
            str(DATA_DIR / "thumbnails" / f"{sid}_thumbnail.jpg"),
            str(DATA_DIR / "thumbnails" / f"{sid}_thumbnail.png"),
        ]
        # También buscar en output/<fecha>/<sid>/thumbnail.jpg
        output_base = Path(item.get("output_file", "")).parent
        if output_base.exists():
            thumb_candidates.extend([
                str(output_base / "thumbnail.jpg"),
                str(output_base / "thumbnail.png"),
                str(output_base / f"{sid}_thumbnail.jpg"),
            ])
        
        thumb_found = None
        for tc in thumb_candidates:
            if tc and Path(tc).exists():
                thumb_found = tc
                break
        
        if not thumb_found:
            log(f"❌ No se encontró thumbnail local para {video_id} — no se puede subir", "ERROR")
            continue
        
        log(f"🖼️ Reintentando thumbnail para {video_id}: {Path(thumb_found).name}", "INFO")
        success = upload_thumbnail(yt, video_id, thumb_found)
        
        if success:
            queue[i]["thumbnail_uploaded"] = True
            queue[i]["status"] = "uploaded"
            queue[i]["thumbnail_retry_at"] = datetime.now(timezone.utc).isoformat()
            fixed.append(video_id)
            log(f"✅ Thumbnail resubida exitosamente para {video_id}", "INFO")
        else:
            log(f"❌ Retry de thumbnail falló para {video_id} — se dejará pendiente", "ERROR")
    
    return fixed


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

    # ── PIPELINE GUARDIAN: Gate check before any upload ───────────────────
    try:
        sys.path.insert(0, str(BITTRADER / "agents"))
        from pipeline_guardian import audit_queue, verify_uploaded
        log("🛡️ Pipeline Guardian: auditando queue...", "INFO")
        summary = audit_queue(auto_fix=True, verbose=False)
        # Reload queue after fixes
        queue = json.loads(QUEUE_FILE.read_text())
        log(f"   Guardian: {summary['ready']} ready | {summary['fixed']} fixed | {summary['blocked']} blocked", "INFO")
        _guardian_available = True
    except Exception as e:
        log(f"⚠️ Pipeline Guardian no disponible: {e}", "WARNING")
        _guardian_available = False
        def verify_uploaded(vid_id, title, thumb_ok): return {}

    # ── QA AGENT: pre-load full gate (includes narration + visual OCR checks) ──
    try:
        from qa_agent import QAAgent as _QAAgentClass
        _qa_agent_instance = _QAAgentClass()
        _qa_available = True
        log("🔬 QA Agent: disponible (checks 1-9 incluyendo narración y OCR)", "INFO")
    except Exception as e:
        log(f"⚠️ QA Agent no disponible: {e}", "WARNING")
        _qa_agent_instance = None
        _qa_available = False

    # ── QUALITY AGENT (legacy gate — kept as secondary layer) ────────────
    try:
        from quality_agent import gate_before_upload
        _legacy_qa_available = True
        log("🎬 Quality Agent (legacy): también disponible", "INFO")
    except Exception as e:
        log(f"⚠️ Quality Agent legacy no disponible: {e}", "WARNING")
        _legacy_qa_available = False
        def gate_before_upload(script_id, entry): return True

    # Load set of already-uploaded script_ids (persist across runs to prevent race-condition duplicates)
    uploaded_scripts_file = DATA_DIR / "uploaded_scripts.json"
    try:
        uploaded_scripts = set(json.loads(uploaded_scripts_file.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        uploaded_scripts = set()

    # Filter videos ready to upload
    ready_to_upload = []
    for i, item in enumerate(queue):
        status = item.get("status", "ready")
        sched_date = item.get("scheduled_date")
        sid = item.get("script_id", "")

        # Skip if already in our persistent uploaded set (race-condition dedupe)
        if sid in uploaded_scripts:
            log(f"Item {i}: script_id={sid} already uploaded — skipping", "DEBUG")
            continue

        # Skip if already uploaded or not ready
        if status in ["uploaded", "published", "uploading"]:
            continue
        if status in ("blocked", "held_by_quality_agent", "deleted_by_quality_agent",
                      "needs_regeneration", "deleted_contaminated", "deleted_header_bug",
                      "duplicate_skipped"):
            log(f"Item {i}: status='{status}' — skipping", "DEBUG")
            continue
        if status != "ready":
            log(f"Item {i}: status='{status}' - skipping", "DEBUG")
            continue
        
        # Check scheduled date
        if sched_date:
            sched_dt = datetime.fromisoformat(sched_date)
            if sched_dt.tzinfo is None:
                sched_dt = sched_dt.replace(tzinfo=timezone.utc)
            
            if sched_dt <= now:
                # ── QA Agent gate (full 9-check suite) ───────────────
                script_id = item.get("script_id","")
                # Skip QA if already manually approved
                if item.get("quality_decision") == "manual-approved":
                    ready_to_upload.append((i, item))
                    log(f"Item {i}: ✅ QA bypassed (manual-approved) — {item.get('title','?')[:40]}", "DEBUG")
                elif _qa_available and _qa_agent_instance:
                    # Construir descripción real para QA (misma lógica que publisher)
                    _sd = item.get("script_data") or {}
                    _qa_desc = _sd.get("description") or item.get("description", "")
                    _qa_tags = _sd.get("tags") or item.get("tags", [])
                    qa_result = _qa_agent_instance.run_all_checks(
                        video_path=item.get("output_file",""),
                        thumb_path=item.get("thumbnail_path", item.get("thumbnail","")),
                        title=item.get("title",""),
                        description=_qa_desc,
                        script_text=item.get("script",""),
                        video_type=item.get("type","long"),
                        tags=_qa_tags,
                    )
                    if not qa_result["passed"]:
                        log(f"Item {i}: ⛔ BLOCKED by QA Agent — {item.get('title','?')[:40]} — issues: {qa_result['issues']}", "WARNING")
                        item["status"] = "qa_failed"
                        item["qa_issues"] = qa_result["issues"]
                        item["qa_checked_at"] = qa_result["timestamp"]
                        continue
                elif _legacy_qa_available and not gate_before_upload(script_id, item):
                    log(f"Item {i}: ⛔ BLOCKED by Quality Agent — {item.get('title','?')[:40]}", "WARNING")
                    item["status"] = "held_by_quality_agent"
                    continue
                ready_to_upload.append((i, item))
                log(f"Item {i}: Ready - scheduled={sched_date}, now={now.isoformat()}", "DEBUG")
            else:
                log(f"Item {i}: Future - scheduled={sched_date}", "DEBUG")
        else:
            # ── QA Agent gate (no scheduled date) ────────────────────
            script_id = item.get("script_id","")
            # Skip QA if already manually approved
            if item.get("quality_decision") == "manual-approved":
                ready_to_upload.append((i, item))
                log(f"Item {i}: ✅ QA bypassed (manual-approved) — {item.get('title','?')[:40]}", "DEBUG")
            elif _qa_available and _qa_agent_instance:
                _sd2 = item.get("script_data") or {}
                _qa_desc2 = _sd2.get("description") or item.get("description", "")
                _qa_tags2 = _sd2.get("tags") or item.get("tags", [])
                qa_result = _qa_agent_instance.run_all_checks(
                    video_path=item.get("output_file",""),
                    thumb_path=item.get("thumbnail_path", item.get("thumbnail","")),
                    title=item.get("title",""),
                    description=_qa_desc2,
                    script_text=item.get("script",""),
                    video_type=item.get("type","long"),
                    tags=_qa_tags2,
                )
                if not qa_result["passed"]:
                    log(f"Item {i}: ⛔ BLOCKED by QA Agent — {item.get('title','?')[:40]} — issues: {qa_result['issues']}", "WARNING")
                    item["status"] = "qa_failed"
                    item["qa_issues"] = qa_result["issues"]
                    item["qa_checked_at"] = qa_result["timestamp"]
                    continue
            elif _legacy_qa_available and not gate_before_upload(script_id, item):
                log(f"Item {i}: ⛔ BLOCKED by Quality Agent — {item.get('title','?')[:40]}", "WARNING")
                item["status"] = "held_by_quality_agent"
                continue
            ready_to_upload.append((i, item))
            log(f"Item {i}: Ready (no scheduled date)", "DEBUG")

    log(f"Videos listos para subir: {len(ready_to_upload)}", "INFO")

    # Connect YouTube (needed for thumbnail retries AND new uploads)
    log("Conectando a YouTube API...", "INFO")
    try:
        yt = get_youtube_client()
        log("Conexión exitosa", "INFO")
    except Exception as e:
        log(f"Error conectando a YouTube: {e}", "ERROR")
        return

    # ── THUMBNAIL RETRY: Fix videos uploaded without thumbnail ────────────
    fixed_thumbs = retry_failed_thumbnails(yt, queue)
    if fixed_thumbs:
        log(f"✅ Thumbnails recuperadas: {len(fixed_thumbs)} video(s) — {fixed_thumbs}", "INFO")
        save_queue(queue)
    else:
        log("📷 Sin thumbnails pendientes de retry", "INFO")

    if not ready_to_upload:
        log("Nada que subir ahora", "INFO")
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
            
            # Upload thumbnail — busca en múltiples ubicaciones automáticamente
            sid = video_data.get("script_id", "")
            thumb_path = None
            thumb_candidates = [
                video_data.get("thumbnail_path",""),
                video_data.get("thumbnail",""),
                str(DATA_DIR / "thumbnails" / f"{sid}_thumbnail.jpg"),
                str(DATA_DIR / "thumbnails" / f"{sid}_thumbnail.png"),
            ]
            for tc in thumb_candidates:
                if tc and Path(tc).exists():
                    thumb_path = tc
                    break

            if thumb_path:
                log(f"🖼️ Subiendo thumbnail: {Path(thumb_path).name}", "INFO")
                upload_thumbnail(yt, video_id, thumb_path)
                queue[i]["thumbnail_uploaded"] = True
            else:
                log(f"⚠️ Sin thumbnail para {sid} — video sin miniatura personalizada", "WARNING")
                queue[i]["thumbnail_uploaded"] = False

            # FIX 5: If thumbnail_uploaded=False, attempt auto-fix before marking complete
            if not queue[i].get("thumbnail_uploaded", False):
                log(f"🔧 FIX 5: thumbnail_uploaded=False para {video_id} — intentando auto-fix...", "WARNING")
                # Re-scan for thumbnail in all known locations
                thumb_candidates_retry = [
                    video_data.get("thumbnail_path", ""),
                    video_data.get("thumbnail", ""),
                    str(DATA_DIR / "thumbnails" / f"{sid}_thumbnail.jpg"),
                    str(DATA_DIR / "thumbnails" / f"{sid}_thumbnail.png"),
                ]
                for tc in thumb_candidates_retry:
                    if tc and Path(tc).exists():
                        log(f"🔧 Reintentando thumbnail: {Path(tc).name}", "INFO")
                        try:
                            from googleapiclient.http import MediaFileUpload
                            media = MediaFileUpload(tc, resumable=True)
                            yt.thumbnails().set(videoId=video_id, media_body=media).execute()
                            queue[i]["thumbnail_uploaded"] = True
                            log(f"✅ Auto-fix exitoso: thumbnail subida para {video_id}", "INFO")
                            break
                        except Exception as retry_err:
                            log(f"❌ Auto-fix falló: {retry_err}", "WARNING")

            # FIX 5: Only mark as 'uploaded' if thumbnail was successfully uploaded
            # Otherwise mark as 'thumbnail_failed' so it can be retried
            thumb_ok = queue[i].get("thumbnail_uploaded", False)
            final_status = "uploaded" if thumb_ok else "thumbnail_failed"

            queue[i].update({
                "status": final_status,
                "video_id": video_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            })

            # Track uploaded script_id to prevent race-condition duplicates
            if sid:
                uploaded_scripts.add(sid)
                uploaded_scripts_file.write_text(json.dumps(list(uploaded_scripts)))

            uploaded.append(video_id)
            log(f"Estado actualizado a '{final_status}'", "INFO")
            if not thumb_ok:
                log(f"⚠️ Video {video_id} marcado como 'thumbnail_failed' — requiere retry de thumbnail", "WARNING")

            # ── POST-UPLOAD VERIFICATION (FIX 1: pass youtube client for real API check) ──
            verify_uploaded(video_id, video_data.get("title", "?"), thumb_ok, youtube_client=yt)
            if not thumb_ok:
                log(f"⚠️ POST-UPLOAD: {video_id} subió SIN thumbnail — acción requerida", "WARNING")

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
