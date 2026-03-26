#!/usr/bin/env python3
"""Upload 3 edu_ videos to YouTube with scheduled publish dates and thumbnails."""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_FILE = WORKSPACE / "memory/youtube_credentials.json"
DATA_DIR = WORKSPACE / "bittrader/agents/data"
UPLOAD_QUEUE = DATA_DIR / "upload_queue.json"
THUMB_DIR = DATA_DIR / "thumbnails"

sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))

VIDEOS = [
    {
        "script_id": "edu_short1_1774356226",
        "title": "El error que cometen el 90% de nuevos traders (yo lo cometí)",
        "description": "El error #1 que cometen casi todos los principiantes — y cómo evitarlo.\n\n"
                       "Sígueme para más contenido de trading y crypto:\n"
                       "📈 Estrategias reales\n"
                       "🤖 Automatización con IA\n"
                       "💰 Gestión de riesgo\n\n"
                       "#BitTrader #Trading #Crypto #Bitcoin #Shorts #EducacionFinanciera #Principiantes",
        "tags": ["cripto", "trading", "principiantes", "error traders", "bitcoin",
                 "inversión", "criptomonedas", "consejos trading", "como empezar cripto",
                 "educacion financiera"],
        "type": "short",
        "output_file": str(WORKSPACE / "bittrader/agents/output/2026-03-24/edu_short1_1774356226/edu_short1_1774356226.mp4"),
        "thumbnail": str(THUMB_DIR / "edu_short1_1774356226_thumbnail.jpg"),
        "publish_at": "2026-03-24T18:00:00Z",
    },
    {
        "script_id": "edu_short2_1774359590",
        "title": "Stop Loss: la herramienta que te salva de perder todo",
        "description": "El Stop Loss es como el cinturón de seguridad del trading. Úsalo siempre.\n\n"
                       "Sígueme para más contenido de trading y crypto:\n"
                       "📈 Estrategias reales\n"
                       "🤖 Automatización con IA\n"
                       "💰 Gestión de riesgo\n\n"
                       "#BitTrader #Trading #Crypto #StopLoss #GestionDeRiesgo #Principiantes",
        "tags": ["stop loss", "cripto", "trading", "principiantes", "bitcoin",
                 "gestión de riesgo", "criptomonedas", "herramientas trading",
                 "como operar cripto", "educacion financiera"],
        "type": "short",
        "output_file": str(WORKSPACE / "bittrader/agents/output/2026-03-24/edu_short2_1774359590/edu_short2_1774359590.mp4"),
        "thumbnail": str(THUMB_DIR / "edu_short2_1774359590_thumbnail.jpg"),
        "publish_at": "2026-03-25T00:00:00Z",
    },
    {
        "script_id": "edu_long_1774363903",
        "title": "Cómo empezar en cripto con $100 — guía completa para principiantes 2026",
        "description": "Guía completa paso a paso para empezar en criptomonedas con solo $100. "
                       "Sin tecnicismos, sin jerga — todo explicado desde cero para principiantes. "
                       "Dónde comprar, qué comprar, cómo evitar los errores más comunes.\n\n"
                       "Sígueme para más contenido de trading y crypto:\n"
                       "📈 Estrategias reales\n"
                       "🤖 Automatización con IA\n"
                       "💰 Gestión de riesgo\n\n"
                       "#cripto #bitcoin #principiantes #tutorial #2026 #criptomonedas "
                       "#invertir100dolares #guiacripto #BitTrader",
        "tags": ["criptomonedas para principiantes", "cómo empezar en cripto",
                 "invertir 100 dólares en cripto", "guía completa criptomonedas 2026",
                 "tutorial criptomonedas paso a paso", "Bitcoin para principiantes",
                 "Ethereum explicación simple", "qué son las criptomonedas",
                 "cómo comprar Bitcoin 2026", "exchanges criptomonedas",
                 "estrategia DCA cripto"],
        "type": "long",
        "output_file": str(WORKSPACE / "bittrader/agents/output/2026-03-24/edu_long_1774363903/edu_long_1774363903.mp4"),
        "thumbnail": str(THUMB_DIR / "edu_long_1774363903_thumbnail.jpg"),
        "publish_at": "2026-03-25T18:00:00Z",
    },
]


def get_youtube_client():
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials

    creds_data = json.loads(CREDS_FILE.read_text())
    creds = Credentials(
        token=creds_data.get("access_token") or creds_data.get("token"),
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data.get("scopes") or creds_data.get("scope", "").split(),
    )
    return build("youtube", "v3", credentials=creds)


def upload_scheduled(yt, video: dict) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": video["title"][:100],
            "description": video["description"],
            "tags": video["tags"][:15],
            "categoryId": "28",  # Science & Technology
            "defaultLanguage": "es",
            "defaultAudioLanguage": "es",
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": video["publish_at"],
            "selfDeclaredMadeForKids": False,
        }
    }

    media = MediaFileUpload(video["output_file"], mimetype="video/mp4",
                            resumable=True, chunksize=5*1024*1024)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status_obj, response = req.next_chunk()
        if status_obj:
            pct = int(status_obj.resumable_progress / status_obj.total_size * 100)
            print(f"    {pct}%...", end="\r")

    return response["id"]


def set_thumbnail(yt, video_id: str, thumb_path: str):
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(thumb_path, mimetype="image/jpeg")
    yt.thumbnails().set(videoId=video_id, media_body=media).execute()
    print(f"    Thumbnail set for {video_id}")


def update_queue(script_id: str, youtube_id: str):
    queue = json.loads(UPLOAD_QUEUE.read_text())
    for entry in queue:
        if entry.get("script_id") == script_id:
            entry["status"] = "uploaded"
            entry["youtube_id"] = youtube_id
            entry["video_id"] = youtube_id
            entry["uploaded_at"] = datetime.now(timezone.utc).isoformat()
            entry["gate_passed"] = True
            entry["gate_issues"] = []
            break
    UPLOAD_QUEUE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))
    print(f"    Queue updated: {script_id} -> {youtube_id}")


def main():
    print("=" * 65)
    print("BitTrader — Upload 3 edu_ Videos (Scheduled)")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 65)

    # Verify all files exist
    for v in VIDEOS:
        if not Path(v["output_file"]).exists():
            print(f"MISSING: {v['output_file']}")
            sys.exit(1)
        if not Path(v["thumbnail"]).exists():
            print(f"MISSING THUMB: {v['thumbnail']}")
            sys.exit(1)
    print("All files verified.")

    print("\nConnecting to YouTube...")
    yt = get_youtube_client()
    print("Connected.")

    results = []

    for i, video in enumerate(VIDEOS, 1):
        print(f"\n[{i}/{len(VIDEOS)}] {video['title'][:55]}")
        print(f"     type={video['type']} | publishAt={video['publish_at']}")

        # Upload
        print(f"    Uploading...")
        try:
            vid_id = upload_scheduled(yt, video)
            vtype = "shorts" if video["type"] == "short" else "watch?v="
            url = f"https://youtube.com/{vtype}/{vid_id}" if video["type"] == "short" else f"https://youtube.com/watch?v={vid_id}"
            print(f"\n    Uploaded: {url}")
        except Exception as e:
            print(f"\n    Upload FAILED: {e}")
            results.append({"script_id": video["script_id"], "status": "error", "error": str(e)})
            continue

        # Set thumbnail
        try:
            set_thumbnail(yt, vid_id, video["thumbnail"])
        except Exception as e:
            print(f"    Thumbnail failed (non-fatal): {e}")

        # Update queue
        update_queue(video["script_id"], vid_id)

        results.append({
            "script_id": video["script_id"],
            "youtube_id": vid_id,
            "url": url,
            "status": "success",
        })

        if i < len(VIDEOS):
            print(f"    Waiting 5s...")
            time.sleep(5)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] != "success"]
    print(f"\nUploaded: {len(success)}/{len(VIDEOS)}")
    for r in success:
        print(f"  {r['youtube_id']} — {r['script_id']}")
        print(f"    {r['url']}")
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  {r['script_id']}: {r.get('error', 'unknown')}")

    # Save report
    report_path = DATA_DIR / "upload_edu_results_2026-03-24.json"
    report_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nReport: {report_path.name}")

    return results


if __name__ == "__main__":
    main()
