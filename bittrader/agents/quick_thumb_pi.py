#!/usr/bin/env python3
"""Quick thumbnail for PI se hunde -29%"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/enderj/.openclaw/workspace/bittrader/agents")

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import requests
from io import BytesIO

# Generate with Hugging Face
HF_API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_API_KEY = "hf_kVtFqUyDvBzCjXfKqYHvXvHJxPcUqQhVZ"

prompt = "YouTube thumbnail, dramatic cryptocurrency crash, PI coin falling chart with red arrow down, worried trader reaction, red and dark colors, dramatic lighting, cinematic 4K"

print("🎨 Generando thumbnail...")
response = requests.post(
    HF_API_URL,
    headers={"Authorization": f"Bearer {HF_API_KEY}"},
    json={"inputs": prompt},
    timeout=60
)

if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    print(f"✅ Imagen generada: {img.size}")
    
    # Crop to 16:9
    target_w, target_h = 1280, 720
    target_ratio = target_w / target_h
    
    img_w, img_h = img.size
    img_ratio = img_w / img_h
    
    if img_ratio > target_ratio:
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        img = img.crop((0, top, img_w, top + new_h))
    
    img = img.resize((target_w, target_h), Image.LANCZOS)
    
    # Enhance
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Convert to RGBA
    img = img.convert('RGBA')
    
    # Dark gradient overlay
    overlay = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    for y in range(450, 720):
        alpha = int(180 * (y - 450) / 270)
        overlay_draw.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha))
    
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    
    # Fonts
    font_big = ImageFont.load_default()  # Fallback
    font_sub = ImageFont.load_default()
    font_handle = ImageFont.load_default()
    
    # Try real fonts
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in font_paths:
        if Path(p).exists():
            try:
                font_big = ImageFont.truetype(p, 75)
                font_sub = ImageFont.truetype(p, 32)
                font_handle = ImageFont.truetype(p, 26)
                break
            except:
                pass
    
    # Headline
    RED = (220, 40, 40)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
    headline = "PI SE HUNDE\n-29%"
    y = 480
    line_height = 90
    
    for line in headline.split('\n'):
        for ox in range(-12, 13):
            for oy in range(-12, 13):
                if abs(ox) + abs(oy) <= 16:
                    draw.text((40 + ox, y + oy), line, font=font_big, fill=BLACK)
        draw.text((40, y), line, font=font_big, fill=RED)
        y += line_height
    
    # Subtitle
    subtitle = "En 24 horas"
    for ox in range(-5, 6):
        for oy in range(-5, 6):
            if ox != 0 or oy != 0:
                draw.text((40, y + 10 + oy), subtitle, font=font_sub, fill=BLACK)
    draw.text((40, y + 10), subtitle, font=font_sub, fill=WHITE)
    
    # Handle
    for ox in range(-3, 4):
        for oy in range(-3, 4):
            if ox != 0 or oy != 0:
                draw.text((980 + ox, 20 + oy), "@bittrader9259", font=font_handle, fill=BLACK)
    draw.text((980, 20), "@bittrader9259", font=font_handle, fill=WHITE)
    
    # Save
    thumb_path = Path("/home/enderj/.openclaw/workspace/bittrader/agents/output/2026-03-14/short_1773470150_254/short_1773470150_254_thumbnail.jpg")
    img = img.convert('RGB')
    img.save(thumb_path, quality=95)
    
    print(f"✅ Thumbnail guardado: {thumb_path}")
    print(f"📏 Tamaño: 1280x720")
    
else:
    print(f"❌ Error: {response.status_code}")
