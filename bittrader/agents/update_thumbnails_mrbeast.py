#!/usr/bin/env python3
"""
🎨 Update Thumbnails with MrBeast Improvements
Regenera los últimos 3 thumbnails con mejoras:
1. Texto con borde negro grueso/sombra
2. Gráfico más grande o eliminar
3. Simplificar - cara + texto + UN elemento
4. Flecha apuntando al error/gráfico
"""
import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
import textwrap

# ── Config ─────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_PATH = WORKSPACE / "memory/youtube_credentials.json"
POLLINATIONS_CONFIG = WORKSPACE / "memory/pollinations_config.json"

# Load Pollinations API key
pollinations_data = json.loads(POLLINATIONS_CONFIG.read_text())
POLLINATIONS_API_KEY = pollinations_data["api_key"]

# YouTube setup
creds = Credentials.from_authorized_user_file(str(CREDS_PATH))
youtube = build('youtube', 'v3', credentials=creds)

# Videos to update
videos = [
    {"title": "De $0 a cuenta fondeada en 30 días (plan real paso a paso)", "id": "Bb2H6nChYwQ", "hook": "De $0 a CUENTA FONDEADA"},
    {"title": "El bot que lee noticias y ejecuta tus trades solo (sin errores)", "id": "2RzldSPc2Ck", "hook": "BOT que TRADEA SOLO"},
    {"title": "Le di mis trades a Claude y los resultados me sorprendieron", "id": "2VEzHeGgDG8", "hook": "Claude ANALIZÓ mis TRADES"}
]


def get_font(size: int, bold: bool = True):
    """Get best available font."""
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf" if bold else "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_text_with_border(draw, pos, text, font, fill_color, border_color=(0, 0, 0), border_width=6):
    """Draw text with thick black border (MrBeast style)."""
    x, y = pos
    
    # Draw border (multiple passes for thickness)
    for offset in range(-border_width, border_width + 1):
        for offset2 in range(-border_width, border_width + 1):
            if offset != 0 or offset2 != 0:
                draw.text((x + offset, y + offset2), text, font=font, fill=border_color)
    
    # Draw main text
    draw.text(pos, text, font=font, fill=fill_color)


def generate_mrbeast_thumbnail(title: str, hook: str, output_path: Path) -> Path:
    """Generate thumbnail with MrBeast-style improvements using Pollinations AI."""
    
    # Step 1: Generate background with Pollinations AI
    print(f"  🎨 Generando imagen base con Pollinations AI...")
    
    prompt = f"YouTube thumbnail background, trading theme, dramatic lighting, red and gold colors, financial charts, dramatic expression, professional, 4K quality, high contrast"
    
    response = requests.post(
        "https://api.pollinations.ai/v1/images/generate",
        headers={"Authorization": f"Bearer {POLLINATIONS_API_KEY}"},
        json={
            "prompt": prompt,
            "width": 1280,
            "height": 720,
            "model": "flux",
            "nologo": True
        },
        timeout=60
    )
    
    if response.status_code != 200:
        print(f"    ⚠️ Pollinations error, usando fallback")
        # Fallback: create solid background
        img = Image.new('RGB', (1280, 720), color=(30, 20, 40))
    else:
        data = response.json()
        image_url = data.get("url") or data.get("image_url")
        if image_url:
            img_response = requests.get(image_url, timeout=30)
            img = Image.open(BytesIO(img_response.content))
        else:
            img = Image.new('RGB', (1280, 720), color=(30, 20, 40))
    
    draw = ImageDraw.Draw(img)
    
    # Step 2: Add dark gradient overlay (improves text contrast)
    overlay = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Left side darker (for text)
    for i in range(400):
        alpha = int(180 * (1 - i / 400))
        overlay_draw.line([(i, 0), (i, 720)], fill=(0, 0, 0, alpha))
    
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Step 3: Add hook text with THICK black border (MrBeast style)
    print(f"  ✍️ Añadiendo texto con borde grueso...")
    
    # Split hook into 2 lines if needed
    words = hook.split()
    if len(words) > 3:
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
    else:
        line1 = hook
        line2 = ""
    
    # Draw main hook (LARGE, gold/yellow)
    font_large = get_font(80, bold=True)
    y_pos = 280
    
    draw_text_with_border(draw, (50, y_pos), line1, font_large, (255, 215, 0), border_width=8)
    
    if line2:
        draw_text_with_border(draw, (50, y_pos + 100), line2, font_large, (255, 215, 0), border_width=8)
    
    # Step 4: Add simplified visual element (arrow pointing to hook)
    print(f"  ➡️ Añadiendo flecha...")
    
    # Draw simple arrow
    arrow_color = (255, 50, 50)  # Red arrow
    draw.polygon([(450, 550), (380, 520), (380, 530), (300, 530), (300, 570), (380, 570), (380, 580)], fill=arrow_color)
    
    # Step 5: Add BitTrader branding (bottom right)
    font_small = get_font(36, bold=False)
    draw_text_with_border(draw, (900, 660), "@bittrader9259", font_small, (255, 255, 255), border_width=3)
    
    # Step 6: Save
    img.save(output_path, quality=95, optimize=True)
    print(f"  ✓ Thumbnail guardado: {output_path}")
    
    return output_path


def upload_thumbnail(video_id: str, thumbnail_path: Path):
    """Upload thumbnail to YouTube."""
    media = MediaFileUpload(str(thumbnail_path), mimetype='image/jpeg', resumable=True)
    request = youtube.thumbnails().set(videoId=video_id, media_body=media)
    response = request.execute()
    return response


def main():
    print("=" * 80)
    print("🎨 ACTUALIZANDO THUMBNAILS - ESTILO MRBEAST")
    print("=" * 80)
    print()
    print("Mejoras aplicadas:")
    print("  ✓ Texto con borde negro grueso (8px)")
    print("  ✓ Fondo oscuro gradiente para mejor contraste")
    print("  ✓ Simplificado: solo hook + flecha + branding")
    print("  ✓ Flecha roja apuntando al texto")
    print()
    
    for video in videos:
        print(f"📺 Procesando: {video['title'][:50]}...")
        print(f"  YouTube ID: {video['id']}")
        
        # Generate thumbnail
        thumb_path = Path(f"/tmp/thumb_mrbeast_{video['id']}.jpg")
        generate_mrbeast_thumbnail(video['title'], video['hook'], thumb_path)
        
        # Upload
        try:
            upload_thumbnail(video['id'], thumb_path)
            print(f"  ✅ Subido a YouTube")
        except Exception as e:
            print(f"  ❌ Error subiendo: {e}")
        
        print()
    
    print("=" * 80)
    print("✅ TODOS LOS THUMBNAILS ACTUALIZADOS")
    print("=" * 80)
    print()
    print("🔍 Revisa los videos:")
    for v in videos:
        print(f"  • https://youtube.com/watch?v={v['id']}")


if __name__ == "__main__":
    from io import BytesIO
    main()
