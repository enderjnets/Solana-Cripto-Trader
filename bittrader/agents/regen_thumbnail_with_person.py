#!/usr/bin/env python3
"""
Regenera el thumbnail con persona para "3 reglas que salvaron mi cuenta de trading"
Video ID: fMm8uwLMtUQ
"""
import requests
import json
import sys
import time
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
OUTPUT_PATH = WORKSPACE / "bittrader/agents/data/thumbnails/long_1774708531_991_thumbnail.jpg"
LOGO_PATH = WORKSPACE / "bittrader/assets/bittrader_logo_transparent.png"
CREDS_PATH = WORKSPACE / "memory/youtube_credentials.json"

# ── HuggingFace Config ─────────────────────────────────────────────────────
HF_API_KEY = "HF_TOKEN_REDACTED"
HF_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-schnell",
    "runwayml/stable-diffusion-v1-5",
]
PROMPT = "excited young latin trader man looking at camera with serious dramatic expression, pointing finger at camera, trading charts in background, dark blue background, professional studio lighting, 4K cinematic, high contrast, sharp focus"

# ── Colors ─────────────────────────────────────────────────────────────────
GOLD = (245, 166, 35)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 40, 40)
GREEN = (0, 200, 80)
DARK_BLUE = (10, 10, 60)

# ── Canvas ─────────────────────────────────────────────────────────────────
W, H = 1280, 720


def get_font(size: int, bold: bool = True):
    paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_text_with_stroke(draw, text, x, y, font, fill, stroke_width=0, stroke_fill=BLACK, anchor="la"):
    if stroke_width > 0:
        for dx in range(-stroke_width, stroke_width + 1, max(1, stroke_width // 3)):
            for dy in range(-stroke_width, stroke_width + 1, max(1, stroke_width // 3)):
                if dx * dx + dy * dy <= stroke_width * stroke_width:
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill, anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def generate_person_image():
    """Generate person image via HuggingFace API"""
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    
    for model in HF_MODELS:
        print(f"  Trying model: {model}")
        url = f"https://router.huggingface.co/hf-inference/models/{model}"
        
        payload = {
            "inputs": PROMPT,
            "parameters": {
                "width": 768,
                "height": 1024,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
            }
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "image" in content_type or len(resp.content) > 10000:
                    img = Image.open(BytesIO(resp.content))
                    print(f"  ✅ Generated with {model}: {img.size}")
                    return img
                else:
                    print(f"  ⚠️ Non-image response: {resp.content[:200]}")
            elif resp.status_code == 503:
                # Model loading, wait and retry
                try:
                    data = resp.json()
                    wait_time = data.get("estimated_time", 20)
                    print(f"  Model loading, waiting {wait_time:.1f}s...")
                    time.sleep(min(wait_time + 2, 40))
                    # Retry once
                    resp2 = requests.post(url, headers=headers, json=payload, timeout=120)
                    if resp2.status_code == 200:
                        img = Image.open(BytesIO(resp2.content))
                        print(f"  ✅ Generated with {model} (retry): {img.size}")
                        return img
                except Exception as e:
                    print(f"  Retry failed: {e}")
            elif resp.status_code == 422:
                # Try simpler payload
                simple_payload = {"inputs": PROMPT}
                resp3 = requests.post(url, headers=headers, json=simple_payload, timeout=120)
                if resp3.status_code == 200:
                    img = Image.open(BytesIO(resp3.content))
                    print(f"  ✅ Generated with {model} (simple): {img.size}")
                    return img
                print(f"  ❌ {model}: {resp.status_code} - {resp.text[:200]}")
            else:
                print(f"  ❌ {model}: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"  ❌ {model} exception: {e}")
    
    return None


def remove_background_simple(img):
    """Simple background removal - darken edges and create mask for person"""
    # Convert to RGBA
    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size
    
    # Create a mask that keeps center/foreground and removes background
    # Use luminosity-based approach: blue-ish backgrounds get removed
    data = np.array(img_rgba)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # Detect dark blue background pixels (typical HF generated backgrounds)
    # Background tends to be dark blue: low R, low G, medium-high B
    is_background = (
        (b.astype(int) - r.astype(int) > 30) &  # More blue than red
        (b.astype(int) - g.astype(int) > 20) &  # More blue than green
        (r.astype(int) < 80) & (g.astype(int) < 80)  # Dark overall
    )
    
    # Also remove edges (gradual fade)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Make background transparent
    mask[is_background] = 0
    
    # Smooth the mask
    from PIL import ImageFilter
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(2))
    mask_arr = np.array(mask_img)
    
    data[:,:,3] = mask_arr
    return Image.fromarray(data)


def build_thumbnail(person_img):
    """Build the final 1280x720 thumbnail"""
    # Create base canvas with dark blue gradient
    canvas = Image.new("RGB", (W, H), DARK_BLUE)
    draw = ImageDraw.Draw(canvas)
    
    # Background gradient (dark blue to purple-blue)
    for y in range(H):
        for x in range(W):
            # Left side darker, right side slightly lighter blue
            t_x = x / W
            t_y = y / H
            r_val = int(8 + t_x * 15 + t_y * 5)
            g_val = int(8 + t_x * 10)
            b_val = int(50 + t_x * 30 + t_y * 10)
            draw.point((x, y), fill=(r_val, g_val, b_val))
    
    # ── Place person on RIGHT side ──────────────────────────────────────────
    person_w = int(W * 0.52)  # 52% width
    person_h = H  # Full height
    
    # Resize person maintaining aspect ratio, crop to fit
    orig_w, orig_h = person_img.size
    scale = max(person_w / orig_w, person_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    person_resized = person_img.resize((new_w, new_h), Image.LANCZOS)
    
    # Crop center
    crop_x = (new_w - person_w) // 2
    crop_y = max(0, new_h - person_h)  # From bottom (feet cut is ok, keep head)
    if crop_y > 0:
        crop_y = (new_h - person_h) // 4  # Slightly from top to keep face
    person_cropped = person_resized.crop((crop_x, crop_y, crop_x + person_w, crop_y + person_h))
    
    # Position on right side
    person_x = W - person_w
    
    # Paste person
    if person_cropped.mode == "RGBA":
        canvas.paste(person_cropped, (person_x, 0), person_cropped)
    else:
        canvas.paste(person_cropped, (person_x, 0))
    
    # Redraw for overlays
    draw = ImageDraw.Draw(canvas)
    
    # ── Gradient overlay on LEFT for text readability ───────────────────────
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    # Dark gradient from left: full opacity at left edge, fades to transparent
    text_zone_end = int(W * 0.65)
    for x in range(text_zone_end + 100):
        if x < text_zone_end:
            alpha = 200
        else:
            alpha = max(0, int(200 * (1 - (x - text_zone_end) / 100)))
        ov_draw.line([(x, 0), (x, H)], fill=(5, 5, 40, alpha))
    
    canvas = canvas.convert("RGBA")
    canvas = Image.alpha_composite(canvas, overlay)
    canvas = canvas.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    # ── Badge "⚠️ MUST KNOW" top left ──────────────────────────────────────
    badge_font = get_font(28, bold=True)
    badge_text = "⚠ MUST KNOW"
    badge_x, badge_y = 18, 18
    # Badge background
    draw.rectangle([badge_x - 5, badge_y - 5, badge_x + 185, badge_y + 42], fill=RED)
    draw.text((badge_x + 5, badge_y + 2), badge_text, font=badge_font, fill=WHITE)
    
    # ── @bittrader9259 top right ────────────────────────────────────────────
    handle_font = get_font(30, bold=True)
    handle_text = "@bittrader9259"
    # Position: top right
    handle_bbox = draw.textbbox((0, 0), handle_text, font=handle_font)
    handle_w = handle_bbox[2] - handle_bbox[0]
    handle_x = W - handle_w - 18
    handle_y = 18
    draw_text_with_stroke(draw, handle_text, handle_x, handle_y, handle_font, WHITE, stroke_width=4, stroke_fill=BLACK, anchor="la")
    
    # ── Main text LEFT side ─────────────────────────────────────────────────
    text_left = 45
    
    # "3 REGLAS" — gold, ~110px bold
    font_3reglas = get_font(108, bold=True)
    y_pos = 90
    draw_text_with_stroke(draw, "3 REGLAS", text_left, y_pos, font_3reglas, GOLD, stroke_width=10, stroke_fill=BLACK)
    
    # "QUE SALVARON" — white, ~90px bold
    font_main = get_font(88, bold=True)
    y_pos += 118
    draw_text_with_stroke(draw, "QUE SALVARON", text_left, y_pos, font_main, WHITE, stroke_width=8, stroke_fill=BLACK)
    
    # "MI CUENTA" — white, ~90px bold
    y_pos += 98
    draw_text_with_stroke(draw, "MI CUENTA", text_left, y_pos, font_main, WHITE, stroke_width=8, stroke_fill=BLACK)
    
    # "de trading" — green, ~55px
    font_sub = get_font(55, bold=True)
    y_pos += 98
    draw_text_with_stroke(draw, "de trading", text_left, y_pos, font_sub, GREEN, stroke_width=5, stroke_fill=BLACK)
    
    # "Aplica ESTAS reglas o pierde todo" — white smaller
    font_small = get_font(36, bold=True)
    y_pos += 68
    draw_text_with_stroke(draw, "Aplica ESTAS reglas o pierde todo", text_left, y_pos, font_small, WHITE, stroke_width=4, stroke_fill=BLACK)
    
    # ── Logo BitTrader bottom left ──────────────────────────────────────────
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo_size = (100, 100)
        logo = logo.resize(logo_size, Image.LANCZOS)
        logo_x = 20
        logo_y = H - logo_size[1] - 15
        canvas.paste(logo, (logo_x, logo_y), logo)
        
        # "BitTrader" text next to logo
        bt_font = get_font(42, bold=True)
        canvas_draw = ImageDraw.Draw(canvas)
        draw_text_with_stroke(canvas_draw, "BitTrader", logo_x + 110, logo_y + 30, bt_font, GOLD, stroke_width=4, stroke_fill=BLACK)
        draw = canvas_draw
    else:
        # Just text if no logo
        bt_font = get_font(42, bold=True)
        draw_text_with_stroke(draw, "BitTrader", text_left, H - 60, bt_font, GOLD, stroke_width=4, stroke_fill=BLACK)
    
    # ── Gold border ─────────────────────────────────────────────────────────
    border_width = 8
    draw.rectangle([0, 0, W - 1, H - 1], outline=GOLD, width=border_width)
    
    return canvas


def upload_thumbnail_to_youtube(thumbnail_path: str, video_id: str):
    """Upload thumbnail to YouTube using saved credentials"""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    
    creds_data = json.loads(Path(CREDS_PATH).read_text())
    
    creds = Credentials(
        token=creds_data.get("token"),
        refresh_token=creds_data.get("refresh_token"),
        token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=creds_data.get("client_id"),
        client_secret=creds_data.get("client_secret"),
        scopes=creds_data.get("scopes"),
    )
    
    # Refresh token if needed
    if creds.expired or not creds.valid:
        print("  Refreshing YouTube credentials...")
        creds.refresh(Request())
        # Save updated token
        creds_data["token"] = creds.token
        Path(CREDS_PATH).write_text(json.dumps(creds_data, indent=2))
    
    youtube = build("youtube", "v3", credentials=creds)
    
    media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
    response = youtube.thumbnails().set(
        videoId=video_id,
        media_body=media
    ).execute()
    
    return response


def main():
    print("=" * 60)
    print("BitTrader Thumbnail Generator — With Person")
    print("=" * 60)
    
    # Step 1: Generate person image
    print("\n[1/4] Generating person image via HuggingFace...")
    person_img = generate_person_image()
    
    if person_img is None:
        print("❌ Failed to generate person image from HuggingFace")
        print("  Will create thumbnail without person image as fallback...")
        # Create a placeholder person (silhouette)
        person_img = create_placeholder_person()
    
    print(f"  Person image: {person_img.size}, mode={person_img.mode}")
    
    # Step 2: Build thumbnail
    print("\n[2/4] Building thumbnail...")
    thumbnail = build_thumbnail(person_img)
    print(f"  Canvas: {thumbnail.size}")
    
    # Step 3: Save thumbnail
    print(f"\n[3/4] Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.save(str(OUTPUT_PATH), "JPEG", quality=95, optimize=True)
    file_size = OUTPUT_PATH.stat().st_size
    print(f"  ✅ Saved: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Step 4: Upload to YouTube
    print(f"\n[4/4] Uploading to YouTube (video: fMm8uwLMtUQ)...")
    try:
        sys.path.insert(0, "/home/enderj/.openclaw/workspace/youtube_env/lib/python3.13/site-packages")
        response = upload_thumbnail_to_youtube(str(OUTPUT_PATH), "fMm8uwLMtUQ")
        print(f"  ✅ Uploaded! Response keys: {list(response.keys()) if response else 'empty'}")
        print("\n✅ SUCCESS — Thumbnail with person generated and uploaded to YouTube!")
    except Exception as e:
        print(f"  ❌ YouTube upload failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Thumbnail saved locally but NOT uploaded to YouTube")
        sys.exit(1)


def create_placeholder_person():
    """Create a simple placeholder if API fails"""
    img = Image.new("RGB", (768, 1024), (5, 10, 50))
    draw = ImageDraw.Draw(img)
    # Simple silhouette shape
    draw.ellipse([234, 80, 534, 380], fill=(60, 60, 80))  # Head
    draw.rectangle([200, 380, 568, 900], fill=(50, 50, 70))  # Body
    return img


if __name__ == "__main__":
    main()
