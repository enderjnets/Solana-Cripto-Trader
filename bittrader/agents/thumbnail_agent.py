#!/usr/bin/env python3
"""
🎨 BitTrader Thumbnail Agent — Generador de Miniaturas
Genera thumbnails profesionales estilo BitTrader para cada video producido.

Pipeline: Producer → Thumbnail Agent → Quality Checker → Publisher

Estilo BitTrader:
- Fondo: frame extraído del clip IA (oscurecido a la izquierda)
- Texto: título grande dorado/verde con borde negro
- Subtítulo descriptivo blanco
- Logo BitTrader arriba izquierda
- Handle @bittrader9259 abajo izquierda
- Botón "WATCH NOW" dorado
- Diagonal dorada decorativa

Ejecutar: python3 agents/thumbnail_agent.py [--limit N]
"""
import json
import os
import subprocess
import random
import requests
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("❌ Pillow no instalado. Ejecuta: pip3 install Pillow")
    exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
OUTPUT_DIR = BITTRADER / "agents/output"

PRODUCTION_FILE = DATA_DIR / "production_latest.json"
LOGO_PATH       = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"

# ── Thumbnail Specs ────────────────────────────────────────────────────────
THUMB_W, THUMB_H = 1280, 720  # YouTube standard
THUMB_MAX_KB     = 2048        # YouTube max 2MB

# ── Colors (BitTrader brand) ──────────────────────────────────────────────
GOLD      = (245, 166, 35)
WHITE     = (255, 255, 255)
BLACK     = (15, 15, 15)
RED       = (220, 40, 40)
GREEN     = (40, 200, 80)
CYAN      = (0, 200, 255)

TITLE_COLORS = [GOLD, GREEN, CYAN, GOLD, GOLD]  # Rotate for variety

# ── LLM Config (for generating catchy thumbnail text) ─────────────────────
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL    = "claude-sonnet-4-6"


# ════════════════════════════════════════════════════════════════════════
# FONT HELPERS
# ════════════════════════════════════════════════════════════════════════

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Get best available bold font."""
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap_text(text: str, max_chars: int = 16) -> list:
    """Split text into lines that fit the thumbnail."""
    words = text.upper().split()
    lines = []
    current = ""
    for word in words:
        if len(current + " " + word) <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]  # Max 4 lines


# ════════════════════════════════════════════════════════════════════════
# LLM — Generate catchy thumbnail text
# ════════════════════════════════════════════════════════════════════════

def generate_thumb_text(title: str, description: str = "") -> dict:
    """Ask LLM for catchy thumbnail headline + subtitle."""
    try:
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        prompt = (
            f"Para un video de YouTube sobre trading/crypto con título: \"{title}\"\n"
            f"Genera un texto CORTO para la miniatura (thumbnail) del video.\n\n"
            f"Responde en JSON exacto:\n"
            f'{{"headline": "TEXTO GRANDE (max 3 palabras por línea, max 3 líneas)", '
            f'"subtitle": "frase corta descriptiva (max 8 palabras)"}}\n\n'
            f"REGLAS:\n"
            f"- Headline: impactante, MAYÚSCULAS, 2-3 líneas cortas\n"
            f"- Subtitle: complementa el headline, genera curiosidad\n"
            f"- Estilo YouTube clickbait pero informativo\n"
            f"- En español"
        )
        data = {
            "model": CLAUDE_MODEL, "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post(CLAUDE_BASE_URL, headers=headers, json=data, timeout=30)
        if r.status_code == 200:
            text = r.json().get("content", [{}])[0].get("text", "")
            import re
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    except Exception as e:
        print(f"      ⚠️ LLM thumbnail text error: {e}")
    
    # Fallback: extract from title
    return {
        "headline": title[:40].upper(),
        "subtitle": ""
    }


# ════════════════════════════════════════════════════════════════════════
# THUMBNAIL GENERATION
# ════════════════════════════════════════════════════════════════════════

def extract_frame(video_path: Path, output_path: Path, time_sec: float = 2.0) -> bool:
    """Extract a frame from video at given timestamp."""
    result = subprocess.run([
        "ffmpeg", "-y", "-ss", str(time_sec), "-i", str(video_path),
        "-vframes", "1", "-q:v", "2", str(output_path)
    ], capture_output=True, text=True, timeout=10)
    return result.returncode == 0 and output_path.exists()


def is_frame_usable(frame_path: Path) -> bool:
    """
    FIX 3: Validate that a video frame is not a generic black/blue frame.
    Returns True only if the frame has meaningful color variety.
    Criteria: avg_b < 150 AND color variance > 30
    """
    try:
        img = Image.open(frame_path).convert("RGB")
        img_small = img.resize((64, 36), Image.LANCZOS)  # Fast check on small sample
        pixels = list(img_small.getdata())
        if not pixels:
            return False

        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_g = sum(p[1] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)

        # Check if too blue (narration video blue background)
        if avg_b > 150 and avg_b > avg_r * 1.8:
            print(f"    ⚠️ Frame rejazado: fondo azul genérico (avg_b={avg_b:.0f}, avg_r={avg_r:.0f})")
            return False

        # Check color variance — monocromatic frames are unusable
        import statistics
        all_channels = [p[0] for p in pixels] + [p[1] for p in pixels] + [p[2] for p in pixels]
        variance = statistics.variance(all_channels)
        if variance < 30:
            print(f"    ⚠️ Frame rechazado: imagen monocromática (varianza={variance:.1f})")
            return False

        avg_brightness = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
        if avg_brightness < 15:
            print(f"    ⚠️ Frame rechazado: demasiado oscuro (brightness={avg_brightness:.1f})")
            return False

        return True
    except Exception as e:
        print(f"    ⚠️ Error validando frame: {e}")
        return False


def generate_hf_person_image(output_path: Path, prompt: str = None) -> bool:
    """
    FIX 3: Generate a person image via HuggingFace SDXL API.
    Used as the right-side visual in thumbnails instead of a video frame.
    """
    if prompt is None:
        prompt = (
            "shocked surprised businessman looking at phone screen, dramatic expression, "
            "stock market charts green going up, gold and green colors, professional studio "
            "lighting, 4K cinematic, high contrast, sharp focus"
        )

    HF_API_KEY = os.environ.get(
        "HF_API_KEY",
        "HF_TOKEN_REMOVED"
    )
    HF_ENDPOINT = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "width": 512,
                "height": 512,
                "num_inference_steps": 30,
            }
        }
        print(f"    🤖 Generando imagen HF SDXL...")
        resp = requests.post(HF_ENDPOINT, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            output_path.write_bytes(resp.content)
            size_kb = output_path.stat().st_size / 1024
            print(f"    ✅ Imagen HF generada: {size_kb:.0f}KB")
            return True
        else:
            print(f"    ⚠️ HF API error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"    ⚠️ HF SDXL error: {e}")
    return False


def create_gradient_background() -> Image.Image:
    """
    FIX 3: Create a solid gradient background for thumbnails.
    Dark navy-to-black gradient — avoids any dependency on video frames.
    """
    bg = Image.new("RGB", (THUMB_W, THUMB_H), (10, 10, 20))
    draw = ImageDraw.Draw(bg)

    # Vertical gradient: dark navy top → black bottom
    for y in range(THUMB_H):
        ratio = y / THUMB_H
        r = int(5 + 15 * (1 - ratio))
        g = int(5 + 20 * (1 - ratio))
        b = int(20 + 60 * (1 - ratio))
        draw.line([(0, y), (THUMB_W, y)], fill=(r, g, b))

    return bg


def create_thumbnail(
    title: str,
    subtitle: str,
    bg_image_path: Path,
    output_path: Path,
    title_color: tuple = GOLD,
    video_type: str = "long",
    person_image_path: Path = None,
) -> bool:
    """
    Create a professional BitTrader thumbnail.
    FIX 3: Uses gradient background + HF-generated person image.
    Video frames are only used if they pass usability validation.
    """

    # ── Background ──
    # FIX 3: Prefer gradient background over video frame
    # Only use bg_image_path if it passes the usability check
    use_frame = (
        bg_image_path is not None
        and bg_image_path.exists()
        and is_frame_usable(bg_image_path)
    )

    if use_frame:
        bg = Image.open(bg_image_path).convert("RGB")
        bg = bg.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        # Darken left side for text readability (gradient)
        overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        gradient_width = THUMB_W * 2 // 3
        for x in range(gradient_width):
            alpha = int(210 * (1 - x / gradient_width))
            ov_draw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    else:
        # FIX 3: Use gradient background (no generic video frames)
        bg = create_gradient_background()

    # ── Person image (right side) — HF generated or provided ──
    person_img = None
    if person_image_path and person_image_path.exists():
        try:
            person_img = Image.open(person_image_path).convert("RGBA")
        except Exception:
            person_img = None

    if person_img:
        # Resize to fit right half of thumbnail
        person_h = THUMB_H
        person_w = int(person_img.width * (person_h / person_img.height))
        person_img = person_img.resize((person_w, person_h), Image.LANCZOS)
        # Paste on right side (no mask if no alpha)
        paste_x = THUMB_W - min(person_w, THUMB_W // 2 + 80)
        if person_img.mode == "RGBA":
            bg.paste(person_img, (paste_x, 0), person_img)
        else:
            bg_rgba = bg.convert("RGBA")
            bg_rgba.paste(person_img.convert("RGBA"), (paste_x, 0))
            bg = bg_rgba.convert("RGB")
    
    draw = ImageDraw.Draw(bg)
    
    # ── Diagonal gold accent ──
    overlay2 = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov2 = ImageDraw.Draw(overlay2)
    ov2.polygon(
        [(THUMB_W * 0.55, THUMB_H), (THUMB_W, THUMB_H * 0.25), (THUMB_W, THUMB_H)],
        fill=(*GOLD, 40)
    )
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay2).convert("RGB")
    draw = ImageDraw.Draw(bg)
    
    # ── Logo (top left) ──
    if LOGO_PATH.exists():
        try:
            logo = Image.open(LOGO_PATH).convert("RGBA")
            logo_h = 70
            logo_w = int(logo.width * (logo_h / logo.height))
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
            bg.paste(logo, (20, 15), logo)
            draw = ImageDraw.Draw(bg)
        except Exception:
            pass
    
    # ── Main Title (big text, left side) ──
    lines = wrap_text(title)
    font_size = 95 if len(lines) <= 3 else 80
    font_big = get_font(font_size)
    
    y = 100
    line_height = font_size + 10
    for line in lines:
        draw.text((35, y), line, font=font_big, fill=title_color,
                  stroke_width=4, stroke_fill=BLACK)
        y += line_height
    
    # ── Subtitle ──
    if subtitle:
        font_sub = get_font(32)
        draw.text((35, y + 15), subtitle, font=font_sub, fill=WHITE,
                  stroke_width=2, stroke_fill=BLACK)
    
    # ── Handle @bittrader9259 ──
    font_handle = get_font(26)
    draw.text((35, THUMB_H - 100), "@bittrader9259", font=font_handle,
              fill=WHITE, stroke_width=2, stroke_fill=BLACK)
    
    # ── WATCH NOW button ──
    btn_y = THUMB_H - 70
    draw.rectangle([35, btn_y, 255, btn_y + 45], fill=GOLD)
    draw.rectangle([35, btn_y, 75, btn_y + 45], fill=RED)
    font_btn = get_font(28)
    draw.text((42, btn_y + 8), "▶", font=font_btn, fill=WHITE)
    draw.text((82, btn_y + 10), "WATCH NOW", font=font_btn, fill=BLACK)
    
    # ── Save ──
    bg.save(str(output_path), "JPEG", quality=95)
    
    # Check size (YouTube max 2MB)
    size_kb = output_path.stat().st_size / 1024
    if size_kb > THUMB_MAX_KB:
        # Re-save with lower quality
        bg.save(str(output_path), "JPEG", quality=80)
    
    return output_path.exists()


# ════════════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ════════════════════════════════════════════════════════════════════════

def run_thumbnail_agent(limit: int = 10, types: str = "all") -> dict:
    """Generate thumbnails for all produced videos."""
    print("\n🎨 BitTrader Thumbnail Agent v1.0")
    
    if not PRODUCTION_FILE.exists():
        print("  ⚠️ No production file found. Run producer first.")
        return {"generated": 0, "errors": 0}
    
    production = json.loads(PRODUCTION_FILE.read_text())
    videos = production.get("videos", [])
    
    # Filter by type if specified
    if types == "long":
        videos = [v for v in videos if v.get("type") == "long"]
    elif types == "short":
        videos = [v for v in videos if v.get("type") == "short"]
    
    print(f"  📋 {len(videos)} videos to process")
    
    generated = 0
    errors = 0
    results = []
    
    for i, video in enumerate(videos[:limit], 1):
        title = video.get("title", "BitTrader")
        vtype = video.get("type", "short")
        video_path = Path(video.get("output_file", ""))
        script_dir = video_path.parent if video_path.exists() else None
        
        thumb_path = script_dir / "thumbnail.jpg" if script_dir else None
        
        print(f"\n  [{i}/{len(videos)}] [{vtype}] {title[:50]}")
        
        # Skip if thumbnail already exists
        if thumb_path and thumb_path.exists():
            size_kb = thumb_path.stat().st_size / 1024
            print(f"    ⏭️ Ya existe ({size_kb:.0f}KB)")
            results.append({
                "title": title, "type": vtype,
                "thumbnail": str(thumb_path), "status": "exists"
            })
            continue
        
        if not script_dir:
            print(f"    ❌ No output directory")
            errors += 1
            continue
        
        # ── Step 1: Get catchy text from LLM ──
        print(f"    📝 Generando texto...")
        thumb_data = generate_thumb_text(title, video.get("description", ""))
        headline = thumb_data.get("headline", title[:40].upper())
        subtitle = thumb_data.get("subtitle", "")
        print(f"    📝 \"{headline}\" | \"{subtitle}\"")
        
        # ── Step 2: Try to extract background frame from video (FIX 3: validate before use) ──
        bg_frame = script_dir / "thumb_bg.jpg"
        person_image = script_dir / "thumb_person.jpg"

        if video_path.exists():
            # Extract frame at 3 seconds (after intro)
            frame_ok = extract_frame(video_path, bg_frame, time_sec=3.0)
            if frame_ok and not is_frame_usable(bg_frame):
                # FIX 3: Frame is a generic blue/black frame — discard it
                print(f"    ⚠️ Frame del video rechazado — usando fondo gradiente")
                bg_frame.unlink(missing_ok=True)

        # ── Step 2b: Generate person image via HF SDXL (FIX 3) ──
        if not person_image.exists():
            generate_hf_person_image(person_image)

        # ── Step 3: Generate thumbnail ──
        print(f"    🎨 Generando miniatura...")
        color_idx = i % len(TITLE_COLORS)
        success = create_thumbnail(
            title=headline,
            subtitle=subtitle,
            bg_image_path=bg_frame if bg_frame.exists() else None,
            output_path=thumb_path,
            title_color=TITLE_COLORS[color_idx],
            video_type=vtype,
            person_image_path=person_image if person_image.exists() else None,
        )

        # Clean up temp files
        bg_frame.unlink(missing_ok=True)
        person_image.unlink(missing_ok=True)
        
        if success:
            size_kb = thumb_path.stat().st_size / 1024
            print(f"    ✅ Thumbnail: {size_kb:.0f}KB → {thumb_path.name}")
            generated += 1
            
            # Update video entry
            video["thumbnail"] = str(thumb_path)
            
            results.append({
                "title": title, "type": vtype,
                "thumbnail": str(thumb_path),
                "size_kb": round(size_kb),
                "headline": headline,
                "subtitle": subtitle,
                "status": "generated"
            })
        else:
            print(f"    ❌ Falló generación")
            errors += 1
            results.append({
                "title": title, "type": vtype,
                "status": "error"
            })
    
    # Save updated production file with thumbnail paths
    production["videos"] = [v for v in production["videos"]]  # Keep all
    production["thumbnails"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": generated,
        "errors": errors,
    }
    PRODUCTION_FILE.write_text(json.dumps(production, indent=2, ensure_ascii=False))
    
    print(f"\n✅ Thumbnail Agent completado")
    print(f"   🎨 {generated} thumbnails generados")
    if errors:
        print(f"   ⚠️  {errors} errores")
    
    return {
        "generated": generated,
        "errors": errors,
        "results": results
    }


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BitTrader Thumbnail Agent v1.0")
    parser.add_argument("--limit", type=int, default=10, help="Max thumbnails to generate")
    parser.add_argument("--types", choices=["all", "long", "short"], default="all",
                       help="Generate for which video types")
    args = parser.parse_args()
    
    result = run_thumbnail_agent(limit=args.limit, types=args.types)
    
    print("\n── Thumbnails ──────────────────────")
    for r in result.get("results", []):
        status = "✅" if r["status"] in ("generated", "exists") else "❌"
        print(f"  {status} [{r.get('type','?')}] {r.get('title','?')[:45]} | "
              f"{r.get('size_kb', '?')}KB")
    print("─────────────────────────────────────────\n")
