#!/usr/bin/env python3
"""Generate thumbnails for new videos using Hugging Face API"""

import json
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Config
HF_API_KEY = Path("/home/enderj/.openclaw/workspace/memory/hf_api_key.txt").read_text().strip()
HF_ENDPOINT = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
LOGO_PATH = Path("/home/enderj/.openclaw/workspace/videos/BIBLIOTECA/bittrader_logo.png")
OUTPUT_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/thumbnails")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

videos = [
    {
        "id": "short_1773470306_372",
        "title": "NEIRO explota",
        "full_title": "NEIRO explota: Qué está pasando",
        "prompt": "cryptocurrency NEIRO coin exploding to the moon, dramatic upward chart, fiery background, professional trading visualization, cinematic lighting",
        "colors": ["#FF6B6B", "#FFE66D"]
    },
    {
        "id": "long_1773470394_349",
        "title": "SUI y DOT",
        "full_title": "SUI y DOT caerán más? Análisis",
        "prompt": "cryptocurrency market crash analysis, SUI and DOT coins falling downward, bearish trend, professional trading analysis, dark moody background",
        "colors": ["#4ECDC4", "#6C5CE7"]
    },
    {
        "id": "short_1773470428_706",
        "title": "Bitcoin BTC",
        "full_title": "¿Qué es BTC (Bitcoin)",
        "prompt": "Bitcoin BTC golden coin, trending upward, orange gold background, cryptocurrency explosion, professional trading visualization, cinematic",
        "colors": ["#F39C12", "#E67E22"]
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
    for y in range(img.height - 180, img.height):
        alpha = int((y - (img.height - 180)) / 180 * 200)
        draw.rectangle([(0, y), (img.width, y)], fill=(0, 0, 0, alpha))

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

    img = generate_image(video["prompt"])
    if img:
        thumb = add_branding(img, video["title"], video["colors"])
        thumb_path = OUTPUT_DIR / f"{video['id']}_thumbnail.jpg"
        thumb.save(thumb_path, "JPEG", quality=95)
        print(f"✅ Saved: {thumb_path}")

        # Update upload queue
        queue_file = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data/upload_queue.json")
        queue = json.loads(queue_file.read_text())
        for item in queue:
            if item.get("script_id") == video["id"]:
                item["thumbnail_path"] = str(thumb_path)
                item["thumbnail_ready"] = True
        queue_file.write_text(json.dumps(queue, indent=2))
    else:
        print(f"❌ Failed to generate image for {video['id']}")

print("\n✅ All thumbnails generated")
