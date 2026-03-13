#!/usr/bin/env python3
"""
🎨 Generar Miniaturas CORRECTAS con prompts específicos al contenido
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import requests
import json

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
HF_CONFIG_PATH = WORKSPACE / "memory/huggingface_config.json"
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# Colors
GOLD = (245, 166, 35)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 40, 40)

# Hugging Face
if HF_CONFIG_PATH.exists():
    hf_config = json.loads(HF_CONFIG_PATH.read_text())
    HF_API_KEY = hf_config.get("api_key", "")
else:
    HF_API_KEY = ""

HF_API_URL = "https://router.huggingface.co/hf-inference/models/"

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
    """Generate image using Hugging Face Inference API."""
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "black-forest-labs/FLUX.1-schnell",
        "runwayml/stable-diffusion-v1-5",
    ]

    for model in models:
        try:
            response = requests.post(
                HF_API_URL + model,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": prompt},
                timeout=60
            )
            if response.status_code == 200:
                image_bytes = response.content
                return Image.open(BytesIO(image_bytes))
        except Exception as e:
            print(f"      ⚠️ {model.split('/')[-1]}: {e}")
    return None

def create_thumbnail(image: Image.Image, headline: str, subtitle: str, output_path: Path, title_color=GOLD):
    """Create final thumbnail with text overlay."""
    thumb = image.resize((1920, 1080), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(thumb)

    # Logo
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo = logo.resize((80, 80), Image.Resampling.LANCZOS)
        thumb.paste(logo, (50, 50), logo)

    # Main headline (BOTTOM - MrBeast style)
    lines = headline.split('\n')
    font_main = get_font(120, bold=True)
    font_sub = get_font(50, bold=True)

    total_height = len(lines) * 140 + (font_sub.size + 10 if subtitle else 0)
    y = 1080 - total_height - 80

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_main)
        text_width = bbox[2] - bbox[0]
        x = (1920 - text_width) // 2

        # Outline
        for ox, oy in [(-3,-3), (3,3), (-3,3), (3,-3), (0,-3), (0,3), (-3,0), (3,0)]:
            draw.text((x+ox, y+oy), line, font=font_main, fill=BLACK)

        # Main text
        draw.text((x, y), line, font=font_main, fill=title_color)
        y += 140

    # Subtitle
    if subtitle:
        bbox = draw.textbbox((0, 0), subtitle, font=font_sub)
        text_width = bbox[2] - bbox[0]
        x = 40
        y += 10

        # Outline
        for ox, oy in [(-2,-2), (2,2), (-2,2), (2,-2), (0,-2), (0,2), (-2,0), (2,0)]:
            draw.text((x+ox, y+oy), subtitle, font=font_sub, fill=BLACK)

        draw.text((x, y), subtitle, font=font_sub, fill=WHITE)

    # Save
    thumb.save(output_path, "JPEG", quality=95)
    return output_path

from io import BytesIO

# Videos with SPECIFIC prompts based on actual content
videos = [
    {
        "video_id": "Bb2H6nChYwQ",
        "title": "De $0 a cuenta fondeada en 30 días",
        "output_dir": WORKSPACE / "bittrader/agents/output/rhino_v1/fondeada",
        "headline": "DE $0 A\nCUENTA FONDEADA",
        "subtitle": "Plan REAL paso a paso",
        "prompt": "professional trading chart screen showing green upward candlesticks and profit curve, prop firm challenge dashboard with $100,000 account balance, clean dark trading desk setup, blue and green neon lighting, 4K, high contrast, trading education style"
    },
    {
        "video_id": "2RzldSPc2Ck",
        "title": "El bot que lee noticias y ejecuta trades",
        "output_dir": WORKSPACE / "bittrader/agents/output/rhino_v1/bot_noticias",
        "headline": "BOT que\nTRADEA SOLO",
        "subtitle": "Sin errores",
        "prompt": "futuristic AI trading dashboard with news feed scrolling in background, automated trading signals appearing, Bitcoin price chart with buy/sell indicators, glowing blue and green neon lights, high-tech interface, 4K, trading bot automation style"
    },
    {
        "video_id": "2VEzHeGgDG8",
        "title": "Le di mis trades a Claude",
        "output_dir": WORKSPACE / "bittrader/agents/output/rhino_v1/claude_trades",
        "headline": "CLAUDE\nANALIZÓ mis TRADES",
        "subtitle": "Resultados reales",
        "prompt": "split screen showing trading spreadsheet data on left transforming into AI chat interface on right, Claude Anthropic logo visible, data flowing between screens, professional trading analysis setup, blue and gold colors, 4K, clean modern tech aesthetic"
    }
]

print("=" * 70)
print("🎨 Generando Miniaturas CORRECTAS con prompts específicos")
print("=" * 70)

for v in videos:
    print(f"\n📋 {v['title']}")
    print(f"   Video ID: {v['video_id']}")
    print(f"   Output: {v['output_dir']}")

    thumb_path = v['output_dir'] / "thumbnail.jpg"
    backup_path = v['output_dir'] / "thumbnail_old.jpg"

    # Backup old thumbnail
    if thumb_path.exists():
        import shutil
        shutil.copy(thumb_path, backup_path)
        print(f"   📦 Backup guardado: {backup_path}")

    try:
        print(f"    🎨 Generando con Hugging Face...")
        img = generate_with_huggingface(v['prompt'])

        if img:
            print(f"    ✅ Imagen generada: {img.size}")

            # Add text overlay
            create_thumbnail(
                image=img,
                headline=v['headline'],
                subtitle=v['subtitle'],
                output_path=thumb_path,
                title_color=GOLD
            )
            print(f"    ✅ Thumbnail guardada: {thumb_path.stat().st_size / 1024:.0f}KB")
        else:
            print(f"    ❌ No se pudo generar imagen, usando fallback...")

            # Fallback: Create thumbnail with solid background
            from PIL import Image, ImageDraw
            bg = Image.new('RGB', (1920, 1080), color=(20, 30, 50))
            create_thumbnail(
                image=bg,
                headline=v['headline'],
                subtitle=v['subtitle'],
                output_path=thumb_path,
                title_color=GOLD
            )
            print(f"    ⚠️ Fallback thumbnail creada")

    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("✅ COMPLETADO")
print("=" * 70)
print("📁 Thumbnails generadas en:")
for v in videos:
    print(f"  • {v['output_dir']}/thumbnail.jpg")
