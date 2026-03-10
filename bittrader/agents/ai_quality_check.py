#!/usr/bin/env python3
"""
🤖 BitTrader AI Quality Check — Sonnet 4.6 Visual Analysis
Sends video frames + thumbnail to Claude Sonnet 4.6 for subjective quality assessment.

Checks:
- Logo watermark visible and correctly positioned
- Subtitles legible on mobile (size, contrast, position)
- Visual composition (professional, engaging, not cluttered)
- Thumbnail attractiveness (click-worthy, clear text, good contrast)
- Audio/visual coherence (does the visual match the topic?)
- Overall professional quality rating

Returns a structured JSON with scores and observations.
"""
import base64
import json
import subprocess
import requests
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8443"
API_KEY = "oauth-proxy"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500

SYSTEM_PROMPT = """Eres un experto en producción de contenido para YouTube/TikTok/Shorts.
Tu trabajo es evaluar la calidad visual de videos del canal BitTrader (trading + IA).

El canal usa un rinoceronte con traje como mascota/personaje.
Los videos tienen:
- Logo de BitTrader (círculos naranjas con rinoceronte y texto "BITTRADER by EnderJ") en esquina superior derecha
- Subtítulos estilo karaoke (amarillo, bold, centrados)
- Imágenes/clips del personaje rinoceronte en diferentes escenarios

Evalúa como si fueras un viewer en su teléfono móvil scrolleando YouTube Shorts o TikTok.
Responde SIEMPRE en JSON válido. Sin markdown, sin explicaciones fuera del JSON."""

EVAL_PROMPT = """Analiza estos frames de un video de YouTube ({video_type}) titulado: "{title}"

Evalúa cada aspecto del 1 al 10 y agrega observaciones específicas.

Responde en este JSON exacto:
{{
  "scores": {{
    "logo_watermark": <1-10>,
    "subtitle_legibility": <1-10>,
    "visual_composition": <1-10>,
    "thumbnail_appeal": <1-10>,
    "professional_quality": <1-10>,
    "mobile_readability": <1-10>
  }},
  "overall_score": <1-10>,
  "grade": "<A|B|C|F>",
  "issues": ["lista de problemas críticos que impiden publicar"],
  "warnings": ["lista de mejoras recomendadas pero no bloqueantes"],
  "observations": "resumen de 1-2 frases de la evaluación general"
}}

Criterios de grade:
- A (9-10): Perfecto, listo para publicar inmediatamente
- B (7-8): Bueno, publicable con mejoras menores opcionales  
- C (5-6): Aceptable pero con problemas notables
- F (1-4): No publicar, necesita correcciones

IMPORTANTE:
- Si NO ves el logo de BitTrader (círculos naranjas, esquina superior derecha), eso es un issue CRÍTICO
- Subtítulos deben ser legibles en pantalla de teléfono (texto grande, contraste alto)
- Un thumbnail sin texto claro o sin contraste = issue"""

THUMBNAIL_PROMPT = """Analiza esta thumbnail para YouTube. ¿Es atractiva y click-worthy?

Responde en JSON:
{{
  "thumbnail_score": <1-10>,
  "issues": ["problemas críticos"],
  "warnings": ["mejoras sugeridas"],
  "observations": "evaluación en 1 frase"
}}

Criterios: texto legible en miniatura, contraste alto, imagen llamativa, no genérica."""


def encode_image(path: Path) -> str:
    """Encode image to base64 for API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_frames(video_path: Path, count: int = 4) -> list[Path]:
    """Extract evenly-spaced frames from video."""
    frames = []
    
    # Get duration
    r = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(video_path)
    ], capture_output=True, text=True, timeout=10)
    duration = float(r.stdout.strip() or "0")
    
    if duration <= 0:
        return frames
    
    # Extract frames at evenly spaced intervals
    # Skip first 0.5s and last 0.5s to avoid transitions
    start = 0.5
    end = max(duration - 0.5, start + 1)
    interval = (end - start) / max(count, 1)
    
    for i in range(count):
        timestamp = start + (i * interval)
        frame_path = video_path.parent / f"_ai_qc_frame_{i}.jpg"
        
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
            "-vframes", "1", "-q:v", "3", "-vf", "scale=540:-1",  # Downscale for API efficiency
            str(frame_path)
        ], capture_output=True, timeout=10)
        
        if frame_path.exists() and frame_path.stat().st_size > 1000:
            frames.append(frame_path)
    
    return frames


def call_sonnet_vision(images: list[Path], prompt: str, system: str = SYSTEM_PROMPT) -> dict:
    """Call Claude Sonnet 4.6 with images for visual analysis."""
    
    # Build content with images
    content = []
    for img_path in images:
        b64 = encode_image(img_path)
        # Detect media type
        suffix = img_path.suffix.lower()
        media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64
            }
        })
    
    content.append({"type": "text", "text": prompt})
    
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": content}]
    }
    
    try:
        r = requests.post(
            f"{API_BASE}/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        
        text = data.get("content", [{}])[0].get("text", "")
        
        # Parse JSON from response
        # Handle potential markdown wrapping
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        
        return json.loads(text)
    
    except json.JSONDecodeError as e:
        print(f"    ⚠️ AI QC: Could not parse JSON response: {e}")
        print(f"    Raw: {text[:300]}")
        return {"error": f"JSON parse error: {e}", "raw": text[:300]}
    except requests.exceptions.ConnectionError:
        print(f"    ⚠️ AI QC: Cannot connect to Claude proxy at {API_BASE}")
        return {"error": "Connection refused — Claude proxy not running"}
    except Exception as e:
        print(f"    ⚠️ AI QC error: {e}")
        return {"error": str(e)}


def check_video_ai(video_path: Path, title: str, video_type: str, 
                    thumbnail_path: Path = None) -> dict:
    """
    Run AI visual quality check on a video.
    
    Returns dict with:
      - scores: dict of category scores (1-10)
      - overall_score: int (1-10)
      - grade: str (A/B/C/F)
      - issues: list of critical problems
      - warnings: list of suggestions
      - observations: str summary
      - cost_estimate: str
    """
    result = {
        "ai_check": True,
        "model": MODEL,
        "scores": {},
        "overall_score": 0,
        "grade": "?",
        "issues": [],
        "warnings": [],
        "observations": "",
    }
    
    # Extract frames
    frames = extract_frames(video_path, count=4)
    if not frames:
        result["error"] = "Could not extract frames"
        return result
    
    # Add thumbnail if available
    images = list(frames)
    if thumbnail_path and thumbnail_path.exists():
        images.append(thumbnail_path)
    
    prompt = EVAL_PROMPT.format(
        video_type="Short vertical" if video_type == "short" else "Video largo horizontal",
        title=title
    )
    
    if thumbnail_path and thumbnail_path.exists():
        prompt += "\n\nLa última imagen es la thumbnail del video. Evalúa también su atractivo."
    
    # Call AI
    print(f"    🤖 AI QC: Sending {len(images)} images to {MODEL}...")
    ai_result = call_sonnet_vision(images, prompt)
    
    # Cleanup temp frames
    for f in frames:
        f.unlink(missing_ok=True)
    
    if "error" in ai_result:
        result["error"] = ai_result["error"]
        return result
    
    # Merge AI results
    result["scores"] = ai_result.get("scores", {})
    result["overall_score"] = ai_result.get("overall_score", 0)
    result["grade"] = ai_result.get("grade", "?")
    result["issues"] = ai_result.get("issues", [])
    result["warnings"] = ai_result.get("warnings", [])
    result["observations"] = ai_result.get("observations", "")
    
    return result


# ════════════════════════════════════════════════════════════════════════
# CLI TEST
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Test with production_latest.json
        DATA_DIR = Path("/home/enderj/.openclaw/workspace/bittrader/agents/data")
        prod_file = DATA_DIR / "production_latest.json"
        
        if not prod_file.exists():
            print("Usage: python3 ai_quality_check.py <video_path> [title] [type]")
            print("   or: python3 ai_quality_check.py   (tests all from production_latest.json)")
            sys.exit(1)
        
        prod = json.loads(prod_file.read_text())
        videos = prod.get("videos", [])
        
        print(f"\n🤖 BitTrader AI Quality Check — {MODEL}")
        print(f"  📋 Testing {len(videos)} videos...\n")
        
        for v in videos:
            title = v.get("title", "?")[:50]
            vtype = v.get("type", "short")
            path = Path(v.get("output_file", ""))
            if not path.exists():
                path = Path("/home/enderj/.openclaw/workspace/bittrader/agents") / v.get("output_file", "")
            
            thumb = Path(v.get("thumbnail", "")) if v.get("thumbnail") else None
            if thumb and not thumb.exists():
                thumb = Path("/home/enderj/.openclaw/workspace/bittrader/agents") / v.get("thumbnail", "")
            
            if not path.exists():
                print(f"  ⚠️ NOT FOUND: {title}")
                continue
            
            print(f"  [{vtype.upper()}] {title}")
            result = check_video_ai(path, v.get("title", ""), vtype, thumb)
            
            grade = result.get("grade", "?")
            score = result.get("overall_score", 0)
            icon = {"A": "✅", "B": "🟡", "C": "🟠", "F": "❌"}.get(grade, "❓")
            
            print(f"    {icon} Grade {grade} ({score}/10)")
            if result.get("observations"):
                print(f"    📝 {result['observations']}")
            for issue in result.get("issues", []):
                print(f"    ❌ {issue}")
            for warning in result.get("warnings", []):
                print(f"    ⚠️ {warning}")
            print()
    
    else:
        video_path = Path(sys.argv[1])
        title = sys.argv[2] if len(sys.argv) > 2 else "Test Video"
        vtype = sys.argv[3] if len(sys.argv) > 3 else "short"
        
        result = check_video_ai(video_path, title, vtype)
        print(json.dumps(result, indent=2, ensure_ascii=False))
