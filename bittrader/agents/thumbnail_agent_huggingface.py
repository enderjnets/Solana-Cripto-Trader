#!/usr/bin/env python3
"""
🎨 BitTrader Thumbnail Agent - Hugging Face Version
Usa Hugging Face Inference API (FREE, sin API key)
Modelos: Stable Diffusion XL, Flux
"""
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
from io import BytesIO

# ── Config ─────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
CREDS_PATH = WORKSPACE / "memory/youtube_credentials.json"
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# YouTube setup
creds = Credentials.from_authorized_user_file(str(CREDS_PATH))
youtube = build('youtube', 'v3', credentials=creds)

# Hugging Face Inference API
# Load API key from config
HF_CONFIG_PATH = WORKSPACE / "memory/huggingface_config.json"
if HF_CONFIG_PATH.exists():
    import json
    hf_config = json.loads(HF_CONFIG_PATH.read_text())
    HF_API_KEY = hf_config.get("api_key", "")
else:
    HF_API_KEY = ""

HF_API_URL = "https://router.huggingface.co/hf-inference/models/"
HF_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL - high quality
    "black-forest-labs/FLUX.1-schnell",          # Flux - fast
    "runwayml/stable-diffusion-v1-5",            # SD 1.5 - fast backup
]

# Videos to update
videos = [
    {
        "title": "De $0 a cuenta fondeada en 30 días",
        "id": "Bb2H6nChYwQ",
        "headline": "DE $0 A\nCUENTA FONDEADA",
        "subtitle": "Plan REAL paso a paso",
        "prompt": "shocked surprised man face dramatic expression looking at camera, trading charts in background, professional studio lighting, 4K, cinematic, high contrast"
    },
    {
        "title": "El bot que lee noticias y ejecuta trades",
        "id": "2RzldSPc2Ck",
        "headline": "BOT que\nTRADEA SOLO",
        "subtitle": "Sin errores",
        "prompt": "excited confident man thumbs up smiling at camera, futuristic tech background, green and gold colors, professional lighting, 4K"
    },
    {
        "title": "Le di mis trades a Claude",
        "id": "2VEzHeGgDG8",
        "headline": "CLAUDE\nANALIZÓ mis TRADES",
        "subtitle": "Resultados reales",
        "prompt": "curious intrigued man looking at computer screen surprised expression, trading dashboard background, blue and gold colors, 4K"
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


def generate_with_huggingface(prompt: str) -> Image.Image:
    """Generate image using Hugging Face Inference API (FREE)."""
    
    full_prompt = f"YouTube thumbnail, {prompt}, highly detailed, sharp focus, professional photography"
    
    for model in HF_MODELS:
        try:
            print(f"    🤗 Probando modelo: {model.split('/')[-1]}...")
            
            response = requests.post(
                HF_API_URL + model,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": full_prompt},
                timeout=60
            )
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                print(f"    ✅ Imagen generada exitosamente")
                return img
            elif response.status_code == 503:
                # Model loading, wait and retry
                print(f"    ⏳ Modelo cargando, esperando 20s...")
                import time
                time.sleep(20)
                response = requests.post(
                    HF_API_URL + model,
                    headers={"Authorization": f"Bearer {HF_API_KEY}"},
                    json={"inputs": full_prompt},
                    timeout=60
                )
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    return img
            else:
                print(f"    ⚠️ Error {response.status_code}: {response.text[:100]}")
                continue
                
        except Exception as e:
            print(f"    ⚠️ Error con {model}: {e}")
            continue
    
    raise Exception("Todos los modelos fallaron")


def create_thumbnail(image: Image.Image, headline: str, subtitle: str, output_path: Path, title_color=GOLD) -> Path:
    """Add BitTrader branding and text to generated image."""
    
    # Resize to 1280x720 with CROP (not stretch)
    img = image
    
    # Calculate crop dimensions (center crop to 16:9)
    target_w, target_h = 1280, 720
    target_ratio = target_w / target_h  # 16:9 = 1.78
    
    img_w, img_h = img.size
    img_ratio = img_w / img_h
    
    if img_ratio > target_ratio:
        # Image is wider - crop sides
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, img_h))
    else:
        # Image is taller - crop top/bottom
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        img = img.crop((0, top, img_w, top + new_h))
    
    # Now resize to exact dimensions
    img = img.resize((target_w, target_h), Image.LANCZOS)
    
    # Enhance image
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    
    # Convert to RGBA for overlay
    img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    
    # Add dark gradient at bottom (for text)
    overlay = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    for y in range(450, 720):
        alpha = int(180 * (y - 450) / 270)
        overlay_draw.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha))
    
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    
    # Logo (top-left)
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert('RGBA')
        logo = logo.resize((140, 140))
        img.paste(logo, (30, 20), logo)
    
    # Main headline (BOTTOM - MrBeast style)
    lines = headline.split('\n')
    font_big = get_font(75, bold=True)
    y = 480
    line_height = 90
    
    for line in lines:
        # 12px black border
        for ox in range(-12, 13):
            for oy in range(-12, 13):
                if abs(ox) + abs(oy) <= 16:
                    draw.text((40 + ox, y + oy), line, font=font_big, fill=BLACK)
        draw.text((40, y), line, font=font_big, fill=title_color)
        y += line_height
    
    # Subtitle
    if subtitle:
        font_sub = get_font(32)
        for ox in range(-5, 6):
            for oy in range(-5, 6):
                if ox != 0 or oy != 0:
                    draw.text((40, y + 10 + oy), subtitle, font=font_sub, fill=BLACK)
        draw.text((40, y + 10), subtitle, font=font_sub, fill=WHITE)
    
    # @bittrader9259 handle (top-right)
    font_handle = get_font(26)
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            if ox != 0 or oy != 0:
                draw.text((980 + ox, 20 + oy), "@bittrader9259", font=font_handle, fill=BLACK)
    draw.text((980, 20), "@bittrader9259", font=font_handle, fill=WHITE)
    
    # Convert to RGB and save
    img = img.convert('RGB')
    img.save(output_path, quality=95)
    
    return output_path


def upload_thumbnail(video_id: str, path: Path):
    media = MediaFileUpload(str(path), mimetype='image/jpeg', resumable=True)
    return youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def main():
    print("=" * 70)
    print("🎨 BITTRADER THUMBNAIL AGENT - HUGGING FACE (FREE)")
    print("   Modelos: SDXL, Flux | Sin API key necesaria")
    print("=" * 70)
    
    colors = [GOLD, (100, 255, 100), (100, 200, 255)]
    
    for i, v in enumerate(videos):
        print(f"\n📺 {v['title']}")
        print(f"   ID: {v['id']}")
        
        thumb_path = Path(f"/tmp/thumb_hf_{v['id']}.jpg")
        
        try:
            # Generate with Hugging Face
            print(f"    🎨 Generando con Hugging Face...")
            img = generate_with_huggingface(v['prompt'])
            
            # Add text and branding
            create_thumbnail(
                image=img,
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
