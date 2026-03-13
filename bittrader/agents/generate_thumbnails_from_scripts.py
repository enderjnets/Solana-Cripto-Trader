#!/usr/bin/env python3
"""
🎨 Generar miniaturas usando los VIDEO_PROMPT de los guiones generados

Lee los guiones (guiones_YYYY-MM-DD.json), extrae los VIDEO_PROMPT,
y genera miniaturas específicas al contenido.
"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
HF_CONFIG_PATH = WORKSPACE / "memory/huggingface_config.json"
GUIONES_DIR = WORKSPACE / "bittrader/agents/data"
LOGO_PATH = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# Colors
GOLD = (245, 166, 35)
WHITE = (255,255,255)
BLACK = (0, 0, 0)

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

def main():
    # Find latest guiones file
    guiones_files = sorted(GUIONES_DIR.glob("guiones_*.json"))
    if not guiones_files:
        print("❌ No se encontraron archivos de guiones")
        return

    latest_file = guiones_files[-1]
    print(f"📋 Leyendo guiones de: {latest_file}")

    with open(latest_file) as f:
        guiones = json.load(f)

    scripts = guiones.get("scripts", [])
    pending = [s for s in scripts if s.get("status") in ["produced", "published"]]

    print(f"   {len(pending)} guiones procesados de {len(scripts)} total")
    print()

    for script in pending:
        script_id = script.get("id")
        title = script.get("title", "")
        video_prompts = script.get("video_prompts", [])

        if not video_prompts:
            print(f"⏭️  {title[:50]}... - Sin VIDEO_PROMPT")
            continue

        print(f"📺 {title}")
        print(f"   ID: {script_id}")
        print(f"   VIDEO_PROMPTs: {len(video_prompts)}")

        # Use first video prompt for thumbnail
        video_prompt = video_prompts[0]
        print(f"   Prompt: {video_prompt[:60]}...")

        # Generate headline from title (max 2 lines)
        words = title.split()
        if len(words) > 6:
            # Split roughly in half
            mid = len(words) // 2
            headline = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        else:
            headline = title

        # Output directory
        output_dir = WORKSPACE / f"bittrader/agents/output/{script_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = output_dir / "thumbnail.jpg"

        try:
            # Generate image with Hugging Face using VIDEO_PROMPT
            print(f"    🎨 Generando con Hugging Face...")
            img = generate_with_huggingface(video_prompt)

            if img:
                # Add text overlay
                create_thumbnail(
                    image=img,
                    headline=headline,
                    subtitle="",  # No subtitle from script
                    output_path=thumb_path,
                    title_color=GOLD
                )
                print(f"    ✅ Thumbnail guardada: {thumb_path.stat().st_size / 1024:.0f}KB")
            else:
                print(f"    ❌ No se pudo generar imagen")
        except Exception as e:
            print(f"    ❌ Error: {e}")

        print()

    print("=" * 70)
    print("✅ COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
