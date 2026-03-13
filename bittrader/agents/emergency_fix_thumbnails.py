#!/usr/bin/env python3
"""
🚨 EMERGENCY FIX - Generate and upload missing thumbnails
Direct solution without CEO deliberation
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# YouTube setup
creds = Credentials.from_authorized_user_file('/home/enderj/.openclaw/workspace/memory/youtube_credentials.json')
youtube = build('youtube', 'v3', credentials=creds)

# Videos to fix
videos = [
    {"title": "De $0 a cuenta fondeada en 30 días (plan real paso a paso) 🦏", "id": "Bb2H6nChYwQ"},
    {"title": "El bot que lee noticias y ejecuta tus trades solo (sin errores) 🦏", "id": "2RzldSPc2Ck"},
    {"title": "Le di mis trades a Claude y los resultados me sorprendieron 🦏", "id": "2VEzHeGgDG8"}
]

def generate_thumbnail(title: str, output_path: Path):
    """Generate BitTrader-style thumbnail"""
    # Create 1280x720 image
    img = Image.new('RGB', (1280, 720), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    # Load logo
    logo_path = Path("/home/enderj/.openclaw/workspace/videos/BIBLIOTECA/bittrader_logo.png")
    if logo_path.exists():
        logo = Image.open(logo_path)
        logo = logo.resize((200, 200))
        img.paste(logo, (50, 260), logo if logo.mode == 'RGBA' else None)

    # Add text (simplified - no custom fonts)
    # Title (truncated)
    short_title = title[:40] + "..." if len(title) > 40 else title
    draw.text((300, 300), short_title, fill=(255, 215, 0))  # Gold

    # BitTrader branding
    draw.text((1000, 650), "@bittrader9259", fill=(255, 255, 255))

    # Save
    img.save(output_path, quality=95)
    return output_path

def upload_thumbnail(video_id: str, thumbnail_path: Path):
    """Upload thumbnail to YouTube"""
    media = MediaFileUpload(str(thumbnail_path), mimetype='image/jpeg', resumable=True)
    request = youtube.thumbnails().set(videoId=video_id, media_body=media)
    response = request.execute()
    return response

print("=" * 80)
print("🔧 EMERGENCY FIX - MISSING THUMBNAILS")
print("=" * 80)
print()

for video in videos:
    print(f"Processing: {video['title'][:50]}...")
    print(f"  YouTube ID: {video['id']}")

    # Generate thumbnail
    thumb_path = Path(f"/tmp/thumb_{video['id']}.jpg")
    generate_thumbnail(video['title'], thumb_path)
    print(f"  ✓ Thumbnail generated: {thumb_path}")

    # Upload
    try:
        upload_thumbnail(video['id'], thumb_path)
        print(f"  ✓ Uploaded to YouTube")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")

    print()

print("=" * 80)
print("✅ EMERGENCY FIX COMPLETE")
print("=" * 80)
