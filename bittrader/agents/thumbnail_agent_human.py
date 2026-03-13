#!/usr/bin/env python3
"""
🎨 BitTrader Thumbnail Agent - Human Style
Genera thumbnails con personas reales + expresiones dramáticas (estilo viral)
"""
import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
from io import BytesIO

# ── Config ─────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_PATH = WORKSPACE / "memory/youtube_credentials.json"
POLLINATIONS_CONFIG = WORKSPACE / "memory/pollinations_config.json"
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# Load Pollinations API
pollinations_data = json.loads(POLLINATIONS_CONFIG.read_text())
POLLINATIONS_API_KEY = pollinations_data["api_key"]

# YouTube setup
creds = Credentials.from_authorized_user_file(str(CREDS_PATH))
youtube = build('youtube', 'v3', credentials=creds)

# Videos to update
videos = [
    {
        "title": "De $0 a cuenta fondeada en 30 días",
        "id": "Bb2H6nChYwQ",
        "headline": "DE $0 A\nCUENTA FONDEADA",
        "subtitle": "Plan REAL paso a paso",
        "expression": "shocked surprised man face dramatic expression mouth open looking at camera professional lighting 4K"
    },
    {
        "title": "El bot que lee noticias y ejecuta trades",
        "id": "2RzldSPc2Ck",
        "headline": "BOT que\nTRADEA SOLO",
        "subtitle": "Sin errores",
        "expression": "excited happy man face thumbs up confident smile looking at camera professional lighting 4K"
    },
    {
        "title": "Le di mis trades a Claude",
        "id": "2VEzHeGgDG8",
        "headline": "CLAUDE\nANALIZÓ mis TRADES",
        "subtitle": "Resultados reales",
        "expression": "curious intrigued man face looking at screen surprised expression professional lighting 4K"
    }
]

# ── Colors ─────────────────────────────────────────────────────────────────
GOLD = (245, 166, 35)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 40, 40)


def get_font(size: int, bold: bool = True):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def generate_human_thumbnail(expression_prompt: str, headline: str, subtitle: str, output_path: Path, title_color=GOLD) -> Path:
    """Generate thumbnail with human face + dramatic expression."""
    
    print(f"    🎨 Generando imagen con Pollinations AI...")
    
    # Step 1: Generate human face background with Pollinations
    full_prompt = f"YouTube thumbnail background, {expression_prompt}, trading theme, financial charts in background, professional studio lighting, high quality, 4K, dramatic composition"
    
    try:
        response = requests.post(
            "https://api.pollinations.ai/v1/images/generate",
            headers={"Authorization": f"Bearer {POLLINATIONS_API_KEY}"},
            json={
                "prompt": full_prompt,
                "width": 1280,
                "height": 720,
                "model": "flux",
                "nologo": True
            },
            timeout=90
        )
        
        if response.status_code == 200:
            data = response.json()
            image_url = data.get("url") or data.get("image_url")
            
            if image_url:
                img_response = requests.get(image_url, timeout=30)
                img = Image.open(BytesIO(img_response.content))
                
                # Ensure correct size
                if img.size != (1280, 720):
                    img = img.resize((1280, 720), Image.LANCZOS)
                
                # Enhance contrast and saturation
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.2)
            else:
                raise Exception("No image URL in response")
        else:
            raise Exception(f"API error: {response.status_code}")
            
    except Exception as e:
        print(f"    ⚠️ Pollinations falló ({e}), usando fallback")
        # Fallback: create dramatic gradient
        img = Image.new('RGB', (1280, 720), color=(30, 20, 40))
    
    # Convert to RGBA for overlay
    img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    
    # Step 2: Add dark gradient overlay at bottom (for text readability)
    overlay = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Bottom 40% gets darker gradient
    for y in range(430, 720):
        alpha = int(200 * (y - 430) / 290)  # Gradual fade from 0 to 200
        overlay_draw.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha))
    
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    
    # Step 3: Add logo (top-left, small)
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert('RGBA')
        logo = logo.resize((140, 140))
        img.paste(logo, (30, 20), logo)
    
    # Step 4: Main headline (BOTTOM - MrBeast style, THICK border)
    lines = headline.split('\n')
    font_big = get_font(75, bold=True)
    y = 480  # Bottom positioning
    line_height = 90
    
    for line in lines:
        # Extra thick black border (12px) for maximum contrast
        for ox in range(-12, 13):
            for oy in range(-12, 13):
                if abs(ox) + abs(oy) <= 16:  # Rounded corners on border
                    draw.text((40 + ox, y + oy), line, font=font_big, fill=BLACK)
        draw.text((40, y), line, font=font_big, fill=title_color)
        y += line_height
    
    # Step 5: Subtitle (below headline)
    if subtitle:
        font_sub = get_font(32)
        for ox in range(-5, 6):
            for oy in range(-5, 6):
                if ox != 0 or oy != 0:
                    draw.text((40, y + 10 + oy), subtitle, font=font_sub, fill=BLACK)
        draw.text((40, y + 10), subtitle, font=font_sub, fill=WHITE)
    
    # Step 6: @bittrader9259 handle (top-right)
    font_handle = get_font(26)
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            if ox != 0 or oy != 0:
                draw.text((980 + ox, 20 + oy), "@bittrader9259", font=font_handle, fill=BLACK)
    draw.text((980, 20), "@bittrader9259", font=font_handle, fill=WHITE)
    
    # Convert back to RGB and save
    img = img.convert('RGB')
    img.save(output_path, quality=95)
    
    return output_path


def upload_thumbnail(video_id: str, path: Path):
    media = MediaFileUpload(str(path), mimetype='image/jpeg', resumable=True)
    return youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def main():
    print("=" * 70)
    print("🎨 BITTRADER THUMBNAIL AGENT - HUMAN STYLE")
    print("   Personas reales + expresiones dramáticas")
    print("=" * 70)
    
    colors = [GOLD, (100, 255, 100), (100, 200, 255), GOLD]
    
    for i, v in enumerate(videos):
        print(f"\n📺 {v['title']}")
        print(f"   ID: {v['id']}")
        
        # Generate thumbnail with human face
        thumb_path = Path(f"/tmp/thumb_human_{v['id']}.jpg")
        
        try:
            generate_human_thumbnail(
                expression_prompt=v['expression'],
                headline=v['headline'],
                subtitle=v['subtitle'],
                output_path=thumb_path,
                title_color=colors[i % len(colors)]
            )
            print(f"   ✓ Generado: {thumb_path.stat().st_size / 1024:.0f}KB")
            
            # Upload
            upload_thumbnail(v['id'], thumb_path)
            print(f"   ✅ Subido a YouTube")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETADO")
    print("=" * 70)
    for v in videos:
        print(f"  • https://youtube.com/watch?v={v['id']}")


if __name__ == "__main__":
    main()
