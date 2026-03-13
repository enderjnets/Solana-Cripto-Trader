#!/usr/bin/env python3
"""
🔧 Fix Thumbnails Script
Re-sube miniaturas personalizadas a videos que están mostrando miniaturas automáticas
"""
from pathlib import Path
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
import json

# ── Config ─────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_PATH = WORKSPACE / "memory/youtube_credentials.json"
DATA_DIR = WORKSPACE / "bittrader/agents/data"
PUBLISH_REPORT = DATA_DIR / "publish_report_2026-03-13.json"

# YouTube setup
creds = Credentials.from_authorized_user_file(str(CREDS_PATH))
youtube = build('youtube', 'v3', credentials=creds)

# ── Load Publish Report ─────────────────────────────────────────────────
with open(PUBLISH_REPORT) as f:
    report = json.load(f)

videos_to_fix = [
    {
        "video_id": "Bb2H6nChYwQ",
        "title": "De $0 a cuenta fondeada en 30 días",
        "thumbnail_path": WORKSPACE / "bittrader/agents/output/rhino_v1/fondeada/thumbnail.jpg"
    },
    {
        "video_id": "2RzldSPc2Ck",
        "title": "El bot que lee noticias y ejecuta trades",
        "thumbnail_path": WORKSPACE / "bittrader/agents/output/rhino_v1/bot_noticias/thumbnail.jpg"
    },
    {
        "video_id": "2VEzHeGgDG8",
        "title": "Le di mis trades a Claude",
        "thumbnail_path": WORKSPACE / "bittrader/agents/output/rhino_v1/claude_trades/thumbnail.jpg"
    },
]

# ── Re-upload Thumbnails ────────────────────────────────────────────────
print("🔧 Re-subiendo miniaturas personalizadas...")
print()

for i, video in enumerate(videos_to_fix, 1):
    thumb_path = Path(video["thumbnail_path"])

    if not thumb_path.exists():
        print(f"⚠️  {i}. Miniatura no encontrada: {thumb_path}")
        continue

    print(f"📋 {i}. {video['title']}")
    print(f"   Video ID: {video['video_id']}")
    print(f"   Thumbnail: {thumb_path}")

    try:
        media = MediaFileUpload(str(thumb_path), mimetype='image/jpeg')
        response = youtube.thumbnails().set(
            videoId=video['video_id'],
            media_body=media
        ).execute()

        print(f"   ✅ Miniatura re-subida")
        print()
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
        print()

print("=" * 60)
print("✅ Proceso completado")
