#!/usr/bin/env python3
"""Generate missing thumbnails for existing videos"""

import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import PIL

# Config
HF_API_KEY = Path("/home/enderj/.openclaw/workspace/memory/hf_api_key.txt").read_text().strip()
HF_ENDPOINT = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
LOGO_PATH = Path("/home/enderj/.openclaw/workspace/videos/BIBLIOTECA/bittrader_logo.png")
OUTPUT_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Videos needing thumbnails
videos = [
    {
        "id": "long_1773470276_498",
        "title": "Cómo pasar FTMO",
        "full_title": "Cómo pasar el eval de FTMO",
        "prompt": "FTMO trading challenge success, trader celebrating, forex trading, professional trading setup, gold and green colors, cinematic lighting, motivational",
        "colors": ["#FFD700", "#4CAF50"]
    },
    {
        "id": "short_1773470119_255",
        "title": "Bitcoin News",
        "full_title": "Análisis Bitcoin hoy",
        "prompt": "Bitcoin BTC trending up, cryptocurrency market, golden Bitcoin coin, professional trading, orange and gold background, dramatic lighting",
        "colors": ["#F39C12", "#E67E22"]
    },
    {
        "id": "short_1773470200_200",
        "title": "TRUMP Coin",
        "full_title": "TRUMP Coin: ¿Estafa o MILLONES?",
        "prompt": "TRUMP cryptocurrency coin, political meme coin trading, dramatic market movement, red and blue colors, professional trading, explosive growth",
        "colors": ["#E74C3C", "#3498DB"]
    }
]

def generate_image(prompt: str) -> Image.Image | None:
    """Generate image using Hugging Face API"""
    response = requests.post(
        HF_ENDPOINT,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": prompt, "parameters": {"num_inference_steps": 20}},
        timeout=30
    )
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    return None

def add_branding(img: Image.Image, title: str, colors: list) -> Image.Image:
    """Add BitTrader branding to thumbnail"""
    # Resize to 16:9
    img = img.resize((1024, 576), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(img)

    # Logo
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo = logo.resize((80, 80), Image.Resampling.LANCZOS)
        img.paste(logo, (20, 20), logo)

    # Handle
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((img.width - 180, 20), "@bittrader9259", fill="white", font=font, stroke_width=2, stroke_fill="black")

    # Title at bottom (MrBeast style)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        title_font = ImageFont.load_default()

    # Draw gradient overlay
    overlay = PIL.Image.new('RGBA', (img.width, 180), (0, 0, 0, 150))
    img.paste(overlay, (0, img.height - 180), overlay)

    # Title text
    text_bbox = draw.textbbox((0, 0), title, font=title_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (img.width - text_width) // 2
    text_y = img.height - 120

    # Draw black outline
    for offset in [-2, -1, 1, 2]:
        draw.text((text_x + offset, text_y), title, fill="black", font=title_font)
        draw.text((text_x, text_y + offset), title, fill="black", font=title_font)
    draw.text((text_x, text_y), title, fill=colors[0], font=title_font)

    return img

# Generate thumbnails
for video in videos:
    print(f"🎨 Generating thumbnail for {video['id']}...")

    # Check if already exists
    thumb_path = OUTPUT_DIR / f"{video['id']}_thumbnail.jpg"
    if thumb_path.exists():
        print(f"   ⏭️  Already exists, skipping")
        continue

    img = generate_image(video["prompt"])
    if img:
        thumb = add_branding(img, video["title"], video["colors"])
        thumb_path = OUTPUT_DIR / f"{video['id']}_thumbnail.jpg"
        thumb.save(thumb_path, "JPEG", quality=95)
        print(f"✅ Saved: {thumb_path}")
    else:
        print(f"❌ Failed to generate image for {video['id']}")

print("\n✅ Done generating thumbnails")
