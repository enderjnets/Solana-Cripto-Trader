#!/usr/bin/env python3
"""
🎨 BitTrader Thumbnail Agent - Update Published Videos
Usa el estilo oficial de thumbnail_agent.py + mejoras MrBeast
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# ── Config ─────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_PATH = WORKSPACE / "memory/youtube_credentials.json"
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

creds = Credentials.from_authorized_user_file(str(CREDS_PATH))
youtube = build('youtube', 'v3', credentials=creds)

# Videos to update
videos = [
    {"title": "De $0 a cuenta fondeada en 30 días", "id": "Bb2H6nChYwQ", "headline": "DE $0 A\nCUENTA\nFONDEADA", "subtitle": "Plan REAL paso a paso"},
    {"title": "El bot que lee noticias y ejecuta trades", "id": "2RzldSPc2Ck", "headline": "BOT que\nTRADEA\nSOLO", "subtitle": "Sin errores"},
    {"title": "Le di mis trades a Claude", "id": "2VEzHeGgDG8", "headline": "CLAUDE\nANALIZÓ\nmis TRADES", "subtitle": "Resultados reales"}
]

# ── BitTrader Brand Colors ────────────────────────────────────────────────
GOLD = (245, 166, 35)
WHITE = (255, 255, 255)
BLACK = (15, 15, 15)
RED = (220, 40, 40)
GREEN = (40, 200, 80)
CYAN = (0, 200, 255)


def get_font(size: int, bold: bool = True):
    """Get BitTrader font."""
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def create_bittrader_thumbnail(headline: str, subtitle: str, output_path: Path, title_color=GOLD) -> Path:
    """Create thumbnail using BitTrader official style + MrBeast improvements."""
    
    # ── Base background (dark gradient) ──
    img = Image.new('RGB', (1280, 720), color=(20, 15, 30))
    draw = ImageDraw.Draw(img)
    
    # Gradient left-to-center (darker for text)
    for i in range(600):
        intensity = i / 600
        r = int(20 + intensity * 50)
        g = int(15 + intensity * 25)
        b = int(30 + intensity * 40)
        draw.line([(i, 0), (i, 720)], fill=(r, g, b))
    
    # Right side accent (warmer)
    for i in range(600, 1280):
        intensity = (i - 600) / 680
        r = int(70 + intensity * 60)
        g = int(40 + intensity * 20)
        b = int(70 + intensity * 30)
        draw.line([(i, 0), (i, 720)], fill=(r, g, b))
    
    # ── Logo (top-left) ──
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert('RGBA')
        logo = logo.resize((180, 180))
        img.paste(logo, (40, 30), logo)
    
    # ── Main headline (THICK black border - MrBeast style) ──
    lines = headline.split('\n')
    font_big = get_font(85, bold=True)
    y = 200
    line_height = 105
    
    for line in lines:
        # MrBeast-style: 10px black border
        for ox in range(-10, 11):
            for oy in range(-10, 11):
                if ox != 0 or oy != 0:
                    draw.text((40 + ox, y + oy), line, font=font_big, fill=BLACK)
        draw.text((40, y), line, font=font_big, fill=title_color)
        y += line_height
    
    # ── Subtitle ──
    if subtitle:
        font_sub = get_font(36)
        # Border 4px
        for ox in range(-4, 5):
            for oy in range(-4, 5):
                if ox != 0 or oy != 0:
                    draw.text((40, y + 15 + oy), subtitle, font=font_sub, fill=BLACK)
        draw.text((40, y + 15), subtitle, font=font_sub, fill=WHITE)
    
    # ── @bittrader9259 handle (bottom-left) ──
    font_handle = get_font(28)
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            if ox != 0 or oy != 0:
                draw.text((40, 610 + oy), "@bittrader9259", font=font_handle, fill=BLACK)
    draw.text((40, 610), "@bittrader9259", font=font_handle, fill=WHITE)
    
    # ── WATCH NOW button (BitTrader style) ──
    btn_y = 660
    draw.rectangle([40, btn_y, 260, btn_y + 50], fill=GOLD)
    draw.rectangle([40, btn_y, 80, btn_y + 50], fill=RED)
    font_btn = get_font(30)
    draw.text((47, btn_y + 10), "▶", font=font_btn, fill=WHITE)
    draw.text((87, btn_y + 12), "WATCH NOW", font=font_btn, fill=BLACK)
    
    # ── Red arrow (MrBeast element - pointing to headline) ──
    arrow_points = [(580, 350), (520, 310), (520, 330), (420, 330), (420, 370), (520, 370), (520, 390)]
    draw.polygon(arrow_points, fill=RED)
    
    # Save
    img.save(output_path, quality=95)
    return output_path


def upload_thumbnail(video_id: str, path: Path):
    """Upload thumbnail to YouTube."""
    media = MediaFileUpload(str(path), mimetype='image/jpeg', resumable=True)
    return youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def main():
    print("=" * 70)
    print("🎨 BITTRADER THUMBNAIL AGENT")
    print("   Estilo oficial + mejoras MrBeast")
    print("=" * 70)
    
    colors = [GOLD, GREEN, CYAN, GOLD, GOLD]
    
    for i, v in enumerate(videos):
        print(f"\n📺 {v['title']}")
        print(f"   ID: {v['id']}")
        
        # Generate thumbnail
        thumb_path = Path(f"/tmp/thumb_bittrader_{v['id']}.jpg")
        create_bittrader_thumbnail(
            headline=v['headline'],
            subtitle=v['subtitle'],
            output_path=thumb_path,
            title_color=colors[i % len(colors)]
        )
        print(f"   ✓ Generado: {thumb_path.stat().st_size / 1024:.0f}KB")
        
        # Upload
        try:
            upload_thumbnail(v['id'], thumb_path)
            print(f"   ✅ Subido a YouTube")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETADO - Revisa en YouTube:")
    print("=" * 70)
    for v in videos:
        print(f"  • https://youtube.com/watch?v={v['id']}")


if __name__ == "__main__":
    main()
