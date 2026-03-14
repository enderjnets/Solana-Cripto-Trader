#!/usr/bin/env python3
"""
🦏 Fix All Rhino Thumbnails - Use proper rhino images
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

# ── Colors ─────────────────────────────────────────────────────────────────
GOLD = (245, 166, 35)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 40, 40)
BLUE = (100, 150, 255)
GREEN = (100, 255, 100)

# ALL Rhino videos to fix (proper rhino imagery)
all_rhino_videos = [
    {
        "title": "Le di mis trades a Claude y los resultados me sorprendieron 🦏",
        "id": "2VEzHeGgDG8",
        "headline": "CLAUDE\nANALIZÓ\nmis TRADES",
        "subtitle": "Resultados reales",
        "prompt": "impressive rhino wearing reading glasses looking at trading charts on computer screen, professional office setting, dramatic lighting, 4K, high quality"
    },
    {
        "title": "El bot que lee noticias y ejecuta tus trades solo (sin errores) 🦏",
        "id": "2RzldSPc2Ck",
        "headline": "BOT que\nTRADEA\nSOLO",
        "subtitle": "Sin errores",
        "prompt": "confident rhino pointing finger at screen showing trading algorithm, futuristic tech background, professional lighting, 4K, cinematic"
    },
    {
        "title": "De $0 a cuenta fondeada en 30 días (plan real paso a paso) 🦏",
        "id": "Bb2H6nChYwQ",
        "headline": "DE $0 A\nCUENTA\nFONDEADA",
        "subtitle": "Plan REAL paso a paso",
        "prompt": "determined rhino climbing mountain of money, success journey metaphor, dramatic lighting, epic cinematic, 4K"
    },
    {
        "title": "El 90% de traders pierde por esto 🦏",
        "id": "6SK7R4tbsDQ",
        "headline": "EL 90% DE\nTRADERS\nPIERDE",
        "subtitle": "Por esto",
        "prompt": "shocked rhino face, dramatic expression, warning style, red and gold colors, professional photography, 4K"
    },
    {
        "title": "Inversor vs Ahorrador: ¿Cuál gana en 10 años? 🦏",
        "id": "OfUGi0YWKzc",
        "headline": "INVERSOR vs\nAHORRADOR",
        "subtitle": "¿Cuál gana en 10 años?",
        "prompt": "two rhinos facing each other in epic battle stance, representing investor vs saver, dramatic lighting, cinematic, 4K"
    },
    {
        "title": "ZEC cae 7%: ¿el fin de las privacy coins? 🦏",
        "id": "k7sR93HNlFs",
        "headline": "ZEC CAE 7%",
        "subtitle": "¿El fin de privacy coins?",
        "prompt": "rhino looking concerned at downward chart, dramatic lighting with red accents, cinematic, 4K"
    },
    {
        "title": "PI coin +13%: ¿trampa o despegue real? 🦏",
        "id": "-DX5oxnJnco",
        "headline": "PI COIN +13%",
        "subtitle": "¿Trampa o despegue real?",
        "prompt": "suspicious rhino examining rising chart, questioning expression, dramatic lighting, cinematic, 4K"
    },
    {
        "title": "AKT explota mientras BTC cae un 1.4% 🦏",
        "id": "ciiiE0klMBg",
        "headline": "AKT EXPLOTA",
        "subtitle": "Mientras BTC cae",
        "prompt": "excited rhino celebrating with fireworks in background, rising chart, vibrant colors, 4K"
    },
    {
        "title": "Por qué PENGU está subiendo mientras BTC cae 🦏",
        "id": "lRKLIFJkmmg",
        "headline": "PENGU\nSUBIENDO",
        "subtitle": "Mientras BTC cae",
        "prompt": "curious rhino watching penguin and chart, mixed expression, dramatic lighting, 4K"
    },
    {
        "title": "Una IA hizo +12% mientras tú dormías",
        "id": "WQAlMFodrrI",
        "headline": "IA HIZO +12%",
        "subtitle": "Mientras tú dormías",
        "prompt": "rhino sleeping peacefully with AI robot working in background, peaceful yet productive, 4K"
    }
]


# Hugging Face API
HF_CONFIG_PATH = WORKSPACE / "memory/huggingface_config.json"
if HF_CONFIG_PATH.exists():
    import json
    hf_config = json.loads(HF_CONFIG_PATH.read_text())
    HF_API_KEY = hf_config.get("api_key", "")
else:
    HF_API_KEY = ""

HF_API_URL = "https://router.huggingface.co/hf-inference/models/"
HF_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-schnell",
    "runwayml/stable-diffusion-v1-5",
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
                print(f"    ✅ Imagen generada: {img.size}")
                return img
            elif response.status_code == 503:
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
                print(f"    ⚠️ Error {response.status_code}")
                continue

        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            continue

    raise Exception("Todos los modelos fallaron")


def create_horizontal_thumbnail(image: Image.Image, headline: str, subtitle: str, output_path: Path, title_color=GOLD) -> Path:
    """Add BitTrader branding and ensure horizontal 16:9 aspect ratio."""

    target_w, target_h = 1280, 720
    target_ratio = target_w / target_h

    img = image
    img_w, img_h = img.size
    img_ratio = img_w / img_h

    print(f"    📐 Imagen original: {img_w}x{img_h} (ratio: {img_ratio:.2f})")

    if img_ratio > target_ratio:
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        img = img.crop((0, top, img_w, top + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)

    img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)

    # Dark gradient
    overlay = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for y in range(450, 720):
        alpha = int(180 * (y - 450) / 270)
        overlay_draw.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Logo
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert('RGBA')
        logo = logo.resize((140, 140))
        img.paste(logo, (30, 20), logo)

    # Headline
    lines = headline.split('\n')
    font_big = get_font(70, bold=True)
    y = 480
    line_height = 85

    for line in lines:
        for ox in range(-10, 11):
            for oy in range(-10, 11):
                if abs(ox) + abs(oy) <= 14:
                    draw.text((40 + ox, y + oy), line, font=font_big, fill=BLACK)
        draw.text((40, y), line, font=font_big, fill=title_color)
        y += line_height

    # Subtitle
    if subtitle:
        font_sub = get_font(30)
        for ox in range(-4, 5):
            for oy in range(-4, 5):
                if ox != 0 or oy != 0:
                    draw.text((40, y + 5 + oy), subtitle, font=font_sub, fill=BLACK)
        draw.text((40, y + 5), subtitle, font=font_sub, fill=WHITE)

    # Handle
    font_handle = get_font(24)
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            if ox != 0 or oy != 0:
                draw.text((980 + ox, 20 + oy), "@bittrader9259", font=font_handle, fill=BLACK)
    draw.text((980, 20), "@bittrader9259", font=font_handle, fill=WHITE)

    img = img.convert('RGB')
    img.save(output_path, quality=95)
    return output_path


def upload_thumbnail(video_id: str, path: Path):
    media = MediaFileUpload(str(path), mimetype='image/jpeg', resumable=True)
    return youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def main():
    print("=" * 70)
    print("🦏 FIX ALL RHINO THUMBNAILS - CEO COORDINATED")
    print("   RINOCERONTES reales + 16:9 horizontal + branding")
    print("=" * 70)

    colors = [GOLD, RED, BLUE, GREEN, (200, 100, 255)]

    for i, v in enumerate(all_rhino_videos):
        print(f"\n{'─' * 70}")
        print(f"🦏 {v['title']}")
        print(f"   ID: {v['id']}")
        print(f"{'─' * 70}")

        thumb_path = Path(f"/tmp/thumb_rhino_all_{v['id']}.jpg")

        try:
            print(f"\n  1️⃣ Generando rhino...")
            img = generate_with_huggingface(v['prompt'])

            print(f"\n  2️⃣ Creando thumbnail 16:9...")
            create_horizontal_thumbnail(
                image=img,
                headline=v['headline'],
                subtitle=v['subtitle'],
                output_path=thumb_path,
                title_color=colors[i % len(colors)]
            )

            print(f"\n  3️⃣ Subiendo...")
            upload_thumbnail(v['id'], thumb_path)
            print(f"   ✅ Actualizado!")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ TODOS LOS VIDEOS DE RHINO CORREGIDOS")
    print("=" * 70)


if __name__ == "__main__":
    main()
