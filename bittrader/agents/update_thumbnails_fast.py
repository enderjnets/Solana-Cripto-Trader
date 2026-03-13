#!/usr/bin/env python3
"""
🎨 Update Thumbnails - Fast Version (no external API)
Genera thumbnails estilo MrBeast SIN API externa (PIL directo)
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# ── Config ─────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_PATH = WORKSPACE / "memory/youtube_credentials.json"

creds = Credentials.from_authorized_user_file(str(CREDS_PATH))
youtube = build('youtube', 'v3', credentials=creds)

videos = [
    {"title": "De $0 a cuenta fondeada en 30 días", "id": "Bb2H6nChYwQ", "hook": "DE $0 A\nCUENTA\nFONDEADA"},
    {"title": "El bot que lee noticias y ejecuta trades", "id": "2RzldSPc2Ck", "hook": "BOT que\nTRADEA\nSOLO"},
    {"title": "Le di mis trades a Claude", "id": "2VEzHeGgDG8", "hook": "CLAUDE\nANALIZÓ\nmis TRADES"}
]


def get_font(size: int, bold: bool = True):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_text_border(draw, pos, text, font, fill, border=(0,0,0), width=8):
    x, y = pos
    for ox in range(-width, width+1):
        for oy in range(-width, width+1):
            if ox != 0 or oy != 0:
                draw.text((x+ox, y+oy), text, font=font, fill=border)
    draw.text(pos, text, font=font, fill=fill)


def generate_thumb(hook: str, output_path: Path):
    # Create base with gradient
    img = Image.new('RGB', (1280, 720), color=(25, 15, 35))
    draw = ImageDraw.Draw(img)
    
    # Gradient overlay (darker left)
    for i in range(500):
        r = int(25 + (i/500) * 40)
        g = int(15 + (i/500) * 20)
        b = int(35 + (i/500) * 25)
        draw.line([(i, 0), (i, 720)], fill=(r, g, b))
    
    # Right side: warm accent
    for i in range(780, 1280):
        intensity = (i - 780) / 500
        r = int(65 + intensity * 80)
        g = int(35 + intensity * 30)
        b = int(60 + intensity * 20)
        draw.line([(i, 0), (i, 720)], fill=(r, g, b))
    
    # Main hook text (GOLD with THICK black border)
    lines = hook.split('\n')
    font_big = get_font(90, bold=True)
    y = 200
    
    for line in lines:
        draw_text_border(draw, (60, y), line, font_big, (255, 215, 0), width=10)
        y += 110
    
    # Red arrow pointing right
    arrow_points = [(550, 500), (480, 450), (480, 480), (350, 480), (350, 520), (480, 520), (480, 550)]
    draw.polygon(arrow_points, fill=(220, 50, 50))
    
    # BitTrader branding
    font_small = get_font(32)
    draw_text_border(draw, (900, 665), "@bittrader9259", font_small, (255, 255, 255), width=3)
    
    img.save(output_path, quality=95)
    return output_path


def upload_thumb(video_id: str, path: Path):
    media = MediaFileUpload(str(path), mimetype='image/jpeg', resumable=True)
    return youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


print("=" * 70)
print("🎨 ACTUALIZANDO THUMBNAILS - MRBEAST STYLE")
print("=" * 70)

for v in videos:
    print(f"\n📺 {v['title']}")
    thumb = Path(f"/tmp/thumb_{v['id']}.jpg")
    generate_thumb(v['hook'], thumb)
    print(f"   ✓ Generado")
    
    try:
        upload_thumb(v['id'], thumb)
        print(f"   ✅ Subido a YouTube")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("✅ COMPLETADO")
print("=" * 70)
for v in videos:
    print(f"  • https://youtube.com/watch?v={v['id']}")
