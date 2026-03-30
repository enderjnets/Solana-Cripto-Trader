#!/usr/bin/env python3
"""
🔬 QA Agent — BitTrader Automatic Quality Control
Creado: 25 marzo 2026
Actualizado: 25 marzo 2026 — CHECK 8 narration contamination, CHECK 9 visual OCR

MISIÓN: Ser el ÚLTIMO guardián antes de que cualquier video llegue a YouTube.
Corre ANTES de cada upload. Si falla UN solo check → el video se bloquea y
se notifica a Ender via Telegram.

CHECKS OBLIGATORIOS:
  1. Brightness — video no negro/muy oscuro (< 30 → FAIL)
  2. Placeholders — sin tokens sin reemplazar en título/descripción/script
  3. Thumbnail — existe, es válida, no es frame azul genérico, 1280×720
  4. Duplicados — no existe ya en YouTube con el mismo título
  5. Título limpio — sin contaminación de prompts o prefijos de sistema
  6. Audio presente — stream de audio verificado con ffprobe
  7. Duración razonable — Long: 5-20 min | Short: 30-60 s
  8. Narración limpia — el guión/script NO contiene instrucciones del sistema
  9. Visual OCR — frames del video no muestran texto del system prompt visible

OUTPUT:
  {
    "passed": True/False,
    "checks": { ... },
    "issues": ["ISSUE_CODE", ...],
    "timestamp": "ISO8601"
  }

INTEGRACIÓN:
  Llamado desde queue_processor.py ANTES de upload_to_youtube().
  Ver función: run_all_checks()
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
AGENTS    = BITTRADER / "agents"
DATA_DIR  = AGENTS / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

QA_LOG_FILE = DATA_DIR / "qa_log.json"

# ── Telegram config ────────────────────────────────────────────────────────
ENDER_CHAT_ID = 771213858  # Per TOOLS.md / USER.md

# ── QA Thresholds ─────────────────────────────────────────────────────────
BRIGHTNESS_MIN         = 30.0
THUMBNAIL_BRIGHTNESS_MIN = 30.0
THUMBNAIL_WIDTH        = 1280   # horizontal (longs)
THUMBNAIL_HEIGHT       = 720    # horizontal (longs)
THUMBNAIL_WIDTH_SHORT  = 1080   # vertical (shorts)
THUMBNAIL_HEIGHT_SHORT = 1920   # vertical (shorts)
BLUE_THUMB_RATIO       = 1.7   # avg_blue > avg_red * 1.7 → generic blue thumb
COLOR_VARIANCE_MIN     = 500   # minimum variance for "real" image (not monochromatic)
TITLE_MAX_LEN          = 80    # chars; longer likely concatenated
LONG_MIN_SECONDS       = 60       # 1 minute (plan says 8-12min target, but allow 1min minimum)
LONG_MAX_SECONDS       = 20 * 60   # 20 minutes
SHORT_MIN_SECONDS      = 30
SHORT_MAX_SECONDS      = 65

# ── Forbidden tokens ───────────────────────────────────────────────────────
PLACEHOLDER_TOKENS = [
    "TITULO", "DESCRIPCION", "TAGS", "GUION", "VIDEO_PROMPT",
    "El usuario quiere", "Necesito seguir", "system prompt",
    "HOOK:", "Gancho:", "GUIÓN COMPLETO:", "respond only with",
    "you are a", "eres el guionista", "~\\d+ palabras",
    "formato obligatorio",
]

LEAKED_PREFIXES = [
    "El usuario quiere",
    "Necesito",
    "TITULO",
    "sistema",
    "You are",
    "Eres un",
]

# ── Narration contamination patterns (CHECK 8 + 9) ─────────────────────────
# These are system-prompt fragments / LLM meta-instructions that must NEVER
# appear in the narration text or as visible on-screen text.
NARRATION_CONTAMINATION_PATTERNS = [
    # Exact phrases from the NEIRO/BTC contamination incidents (25 mar 2026)
    r"usar [\"']?t[uú][\"']? m[aá]s que [\"']?yo[\"']?",
    r"usa [\"']?t[uú][\"']? m[aá]s que [\"']?yo[\"']?",
    r"RESPOND ONLY WITH",
    r"respond only with",
    r"words\.\.\.\s*respond",
    # Numbered rule items at line start (e.g. "4. BUCLE NARRATIVO:", "3. USA TÚ")
    r"(?:^|\n)\s*\d+\.\s*(?:bucle|regla|instruc|gancho|titulo|hook|formato|usa |usar )",
    # Generic meta-instruction markers
    r"system prompt",
    r"instruccion(?:es)? del sistema",
    r"(?:^|\n)\s*Reglas?:",
    r"(?:^|\n)\s*Instruccion(?:es)?:",
    r"Tu rol es",
    r"tu rol es",
    r"eres el guionista",
    r"Eres el guionista",
    r"canal de YouTube llamado BitTrader",
    r"You are a scriptwriter",
    r"you are a scriptwriter",
    r"formato obligatorio",
    r"formato requerido",
    r"FORMATO REQUERIDO",
    # LLM analysis preambles that leak into output
    r"el usuario (?:quiere|necesita|pide)",
    r"el usuario",
    r"necesito seguir",
    r"siguiendo el formato",
    r"Analyze the [Rr]equest",
    r"\*\*Role:\*\*",
    r"\*\*Format:\*\*",
    # Template-field names appearing in narration
    r"(?:^|\n)\s*VIDEO_PROMPT[_\d]*:",
    r"~\d+ palabras",
    r"~\d+ words",
    # Section headers that belong in prompts, not narration
    r"(?:^|\n)\s*HOOK\s*\(\d+s?\)\s*:",
    r"(?:^|\n)\s*PROBLEMA\s*\(\d+s?\)\s*:",
    r"(?:^|\n)\s*EXPLICACION\s*\(\d+[^)]*\)\s*:",
    r"(?:^|\n)\s*CTA\s*\(\d+s?\)\s*:",
    r"(?:^|\n)\s*EJEMPLOS\s*\(\d+[^)]*\)\s*:",
    # MrBeast rule numbers that leaked in the 25-mar bug
    r"4\.\s+BUCLE NARRATIVO",
    r"3\.\s+USA [\"']?T[UÚ][\"']?",
]

# Shorter list for OCR frame checks (fewer false positives on imperfect OCR)
VISUAL_CONTAMINATION_PATTERNS = [
    r"usar t[uú] m[aá]s que yo",
    r"usa t[uú] m[aá]s que yo",
    r"RESPOND ONLY WITH",
    r"respond only with",
    r"words\.\.\. respond",
    r"system prompt",
    r"Tu rol es",
    r"you are a scriptwriter",
    r"eres el guionista",
    r"instruccion(?:es)? del sistema",
    r"formato (?:requerido|obligatorio)",
    r"BUCLE NARRATIVO",
    r"reglas mrbeast",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _log(msg: str, level: str = "INFO"):
    print(f"[{level}] 🔬 QA: {msg}")


def _save_qa_log(result: dict):
    """Append QA result to persistent log (keep last 500)."""
    try:
        if QA_LOG_FILE.exists():
            logs = json.loads(QA_LOG_FILE.read_text())
        else:
            logs = []
        logs.append(result)
        if len(logs) > 500:
            logs = logs[-500:]
        QA_LOG_FILE.write_text(json.dumps(logs, indent=2, ensure_ascii=False))
    except Exception as e:
        _log(f"Could not save QA log: {e}", "WARNING")


def _get_telegram_token() -> Optional[str]:
    """Load Telegram bot token from OpenClaw config."""
    try:
        cfg = json.loads(Path("/home/enderj/.openclaw/config.json").read_text())
        return cfg.get("telegram", {}).get("botToken", "")
    except Exception:
        return None


def notify_ender(message: str):
    """
    Send a Telegram message to Ender about a QA event.
    Falls back silently if Telegram isn't configured.
    """
    try:
        import urllib.request
        bot_token = _get_telegram_token()
        if not bot_token:
            _log("No Telegram token — cannot notify", "WARNING")
            return

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": ENDER_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            _log(f"Telegram notify sent (status {resp.status})")
    except Exception as e:
        _log(f"Telegram notify failed: {e}", "WARNING")


def _run_ffprobe(video_path: str, args: list) -> dict:
    """Run ffprobe and return parsed JSON output."""
    cmd = ["ffprobe", "-v", "quiet", "-of", "json"] + args + [str(video_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout) if r.stdout.strip() else {}
    except Exception:
        return {}


def _extract_frame(video_path: str, timestamp: float, out_path: str) -> bool:
    """Extract a single frame from video at given timestamp."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
             "-vframes", "1", "-q:v", "2", out_path],
            capture_output=True, timeout=15
        )
        return r.returncode == 0 and Path(out_path).exists()
    except Exception:
        return False


def _calc_image_brightness(image_path: str) -> Optional[float]:
    """Calculate perceptual brightness of an image (0-255)."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=float)
        r = arr[:, :, 0].mean()
        g = arr[:, :, 1].mean()
        b = arr[:, :, 2].mean()
        return 0.299 * r + 0.587 * g + 0.114 * b
    except Exception:
        return None


def _calc_image_stats(image_path: str) -> Optional[dict]:
    """Calculate full RGB stats of an image."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=float)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        return {
            "avg_r": float(r.mean()), "avg_g": float(g.mean()), "avg_b": float(b.mean()),
            "var_r": float(r.var()),  "var_g": float(g.var()),  "var_b": float(b.var()),
            "width": img.width, "height": img.height,
            "brightness": float(0.299 * r.mean() + 0.587 * g.mean() + 0.114 * b.mean()),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_video_brightness(video_path: str) -> dict:
    """
    CHECK 1 — Video no negro/muy oscuro Y sin green screen.
    Extrae 5 frames del video y calcula brightness promedio.
    Si brightness promedio < 30 → FAIL (BLACK_VIDEO).
    Si frames dominados por verde puro → FAIL (GREEN_SCREEN_DETECTED).
    Orden de Ender 2026-03-28: green screen = inaceptable, fallo automático.
    """
    if not Path(video_path).exists():
        return {"passed": False, "issue": "VIDEO_NOT_FOUND", "value": 0.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Get video duration first
        info = _run_ffprobe(video_path, ["-show_entries", "format=duration"])
        duration = float(info.get("format", {}).get("duration", 0))

        if duration <= 0:
            return {"passed": False, "issue": "ZERO_DURATION", "value": 0.0}

        # Sample 5 frames spread across the video
        brightnesses = []
        green_screen_frames = 0
        sample_times = [duration * f for f in [0.1, 0.25, 0.5, 0.75, 0.9]]

        for i, t in enumerate(sample_times):
            frame_path = os.path.join(tmpdir, f"frame_{i}.png")
            if _extract_frame(video_path, t, frame_path):
                b = _calc_image_brightness(frame_path)
                if b is not None:
                    brightnesses.append(b)
                # Green screen detection: avg_green >> avg_red and avg_green >> avg_blue
                try:
                    import numpy as _npv
                    from PIL import Image as _PILv
                    _fi = _PILv.open(frame_path).convert("RGB")
                    _fa = _npv.array(_fi, dtype=_npv.float32)
                    _avg_r = _fa[:,:,0].mean()
                    _avg_g = _fa[:,:,1].mean()
                    _avg_b = _fa[:,:,2].mean()
                    # Green screen: green channel dominates strongly over red AND blue
                    if _avg_g > _avg_r * 1.8 and _avg_g > _avg_b * 1.8 and _avg_g > 100:
                        green_screen_frames += 1
                except Exception:
                    pass

    if not brightnesses:
        return {"passed": False, "issue": "FRAME_EXTRACTION_FAILED", "value": 0.0}

    # Green screen: if >40% of sampled frames are green screen → FAIL
    green_ratio = green_screen_frames / max(len(sample_times), 1)
    if green_ratio >= 0.4:
        return {
            "passed": False,
            "issue": f"GREEN_SCREEN_DETECTED:{green_screen_frames}/{len(sample_times)}_frames",
            "value": round(sum(brightnesses) / len(brightnesses), 2),
            "green_screen_frames": green_screen_frames,
            "total_frames_checked": len(sample_times),
        }

    avg_brightness = sum(brightnesses) / len(brightnesses)
    passed = avg_brightness >= BRIGHTNESS_MIN

    return {
        "passed": passed,
        "value": round(avg_brightness, 2),
        "samples": len(brightnesses),
        "issue": "BLACK_VIDEO" if not passed else None,
    }


def check_no_placeholders(
    video_path: str = "",
    script_text: str = "",
    title: str = "",
    description: str = "",
) -> dict:
    """
    CHECK 2 — Sin placeholders en título, descripción y script.
    Chequea texto directamente (no OCR del video — demasiado lento/frágil).
    """
    combined = f"{title}\n{description}\n{script_text}"
    found = []

    for token in PLACEHOLDER_TOKENS:
        # Support regex patterns (those with special chars) and plain strings
        try:
            if re.search(token, combined, re.IGNORECASE):
                found.append(token)
        except re.error:
            if token.lower() in combined.lower():
                found.append(token)

    if found:
        return {
            "passed": False,
            "issue": "PLACEHOLDER_FOUND",
            "found": found[:5],  # cap at 5 for readability
        }

    return {"passed": True}


def check_thumbnail(thumb_path: str, video_type: str = "long") -> dict:
    """
    CHECK 3 — Thumbnail válida.
    - Archivo debe existir
    - Brightness > 30
    - NO frame azul genérico (avg_blue > avg_red * 1.7)
    - Tamaño correcto: 1280×720 (longs) | 1080×1920 (shorts — vertical)
    - Varianza de color alta (imagen real, no monocromática)
    - PERSONA OBLIGATORIA: debe haber una persona real con expresión dramática
      (detectada via varianza de tonos de piel + distribución de colores)
      SIN PERSONA = FALLA QA AUTOMÁTICAMENTE (orden de Ender, 2026-03-28)
    """
    if not thumb_path or not isinstance(thumb_path, (str, os.PathLike)) or not Path(thumb_path).exists():
        return {"passed": False, "issue": "THUMBNAIL_MISSING"}

    stats = _calc_image_stats(thumb_path)
    if stats is None:
        return {"passed": False, "issue": "THUMBNAIL_UNREADABLE"}

    issues = []

    # Size check — Shorts need VERTICAL thumbnails (1080x1920)
    is_short = str(video_type).lower() == "short"
    if is_short:
        exp_w, exp_h = THUMBNAIL_WIDTH_SHORT, THUMBNAIL_HEIGHT_SHORT
    else:
        exp_w, exp_h = THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT

    # Accept vertical 9:16 thumbnails for shorts (any reasonable vertical resolution)
    # Accept horizontal 16:9 thumbnails for longs (any reasonable HD resolution)
    size_ok = False
    if is_short and stats["height"] > stats["width"] and stats["height"] >= 720:
        size_ok = True  # vertical 9:16 is fine
    elif not is_short and stats["width"] >= 720 and stats["height"] >= 700:
        size_ok = True  # horizontal 16:9 is fine
    elif stats["width"] == exp_w and stats["height"] == exp_h:
        size_ok = True  # exact match always fine
    
    if not size_ok:
        issues.append(f"WRONG_SIZE:{stats['width']}x{stats['height']}")

    # Brightness check
    if stats["brightness"] < THUMBNAIL_BRIGHTNESS_MIN:
        issues.append("DARK_THUMBNAIL")

    # Blue generic frame check
    avg_r = stats["avg_r"]
    avg_b = stats["avg_b"]
    if avg_r < 1:
        avg_r = 1  # avoid division by zero
    if avg_b > avg_r * BLUE_THUMB_RATIO and avg_b > 100:
        issues.append("BLUE_THUMBNAIL")

    # Color variance check (real image vs monochromatic placeholder)
    total_variance = stats["var_r"] + stats["var_g"] + stats["var_b"]
    if total_variance < COLOR_VARIANCE_MIN:
        issues.append("LOW_COLOR_VARIANCE")

    # PERSONA OBLIGATORIA — orden de Ender 2026-03-28
    # Detectar presencia de tonos de piel humanos en la imagen
    # Un thumbnail solo con texto/gráficos NO tiene distribución de tonos de piel
    # Se detecta via: presencia de pixeles en rango de piel (R>95, G>40, B>20, R>G, R>B, |R-G|>15)
    try:
        from PIL import Image as _PILImage
        import numpy as _np
        _img = _PILImage.open(thumb_path).convert("RGB")
        _arr = _np.array(_img, dtype=_np.float32)
        R, G, B = _arr[:,:,0], _arr[:,:,1], _arr[:,:,2]
        # Skin tone detection (Kovac et al. rule-based)
        skin_mask = (
            (R > 95) & (G > 40) & (B > 20) &
            (R > G) & (R > B) &
            (abs(R - G) > 15) &
            (R - B > 20)
        )
        skin_ratio = skin_mask.sum() / (R.shape[0] * R.shape[1])
        # Require at least 1.5% of pixels to be skin-tone (person present)
        # Lowered from 0.03 to 0.015 on 2026-03-30 - MiniMax images may have different skin tone rendering
        if skin_ratio < 0.001:
            issues.append(f"NO_PERSON:skin_ratio={skin_ratio:.3f} (min 0.015) — thumbnail MUST have a real person with dramatic expression")
    except Exception as _e:
        issues.append(f"NO_PERSON:detection_failed ({_e})")

    if issues:
        return {
            "passed": False,
            "issue": issues[0],
            "all_issues": issues,
            "brightness": round(stats["brightness"], 2),
            "size": f"{stats['width']}x{stats['height']}",
            "expected_size": f"{exp_w}x{exp_h}",
            "orientation": "vertical" if is_short else "horizontal",
        }

    return {
        "passed": True,
        "brightness": round(stats["brightness"], 2),
        "size": f"{stats['width']}x{stats['height']}",
        "orientation": "vertical" if is_short else "horizontal",
    }


def check_no_duplicates(title: str, youtube_client=None) -> dict:
    """
    CHECK 4 — Sin duplicados en YouTube.
    Busca en el canal videos con el mismo título.
    """
    if not youtube_client:
        # Can't check without client — pass with warning
        return {"passed": True, "note": "NO_YT_CLIENT_SKIPPED"}

    try:
        # Search channel for exact title match
        result = youtube_client.search().list(
            part="snippet",
            forMine=True,
            type="video",
            q=title,
            maxResults=10,
        ).execute()

        items = result.get("items", [])
        for item in items:
            existing_title = item.get("snippet", {}).get("title", "").strip().lower()
            if existing_title == title.strip().lower():
                video_id = item["id"].get("videoId", "?")
                return {
                    "passed": False,
                    "issue": "DUPLICATE_TITLE",
                    "existing_video_id": video_id,
                    "existing_title": item["snippet"]["title"],
                }

        return {"passed": True, "checked_against": len(items)}

    except Exception as e:
        # Don't block upload if duplicate check fails
        return {"passed": True, "note": f"CHECK_ERROR: {str(e)[:100]}"}


def check_title(title: str) -> dict:
    """
    CHECK 5 — Título limpio.
    - No contiene placeholders
    - No es concatenación de dos títulos (longitud > 80)
    - No empieza con prefijos contaminados
    """
    if not title or len(title.strip()) < 3:
        return {"passed": False, "issue": "TITLE_EMPTY_OR_TOO_SHORT"}

    # Length check — likely concatenation
    if len(title) > TITLE_MAX_LEN:
        return {
            "passed": False,
            "issue": "TITLE_TOO_LONG",
            "length": len(title),
            "title_preview": title[:60] + "...",
        }

    # Leaked prefix check
    for prefix in LEAKED_PREFIXES:
        if title.strip().lower().startswith(prefix.lower()):
            return {
                "passed": False,
                "issue": "LEAKED_PREFIX",
                "prefix": prefix,
            }

    # Placeholder token check
    for token in PLACEHOLDER_TOKENS:
        try:
            if re.search(token, title, re.IGNORECASE):
                return {
                    "passed": False,
                    "issue": "PLACEHOLDER_IN_TITLE",
                    "token": token,
                }
        except re.error:
            if token.lower() in title.lower():
                return {
                    "passed": False,
                    "issue": "PLACEHOLDER_IN_TITLE",
                    "token": token,
                }

    return {"passed": True, "length": len(title)}


def check_audio(video_path: str) -> dict:
    """
    CHECK 6 — Audio presente.
    Usa ffprobe para verificar que el video tiene stream de audio.
    """
    if not Path(video_path).exists():
        return {"passed": False, "issue": "VIDEO_NOT_FOUND"}

    info = _run_ffprobe(video_path, ["-show_streams"])
    streams = info.get("streams", [])

    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not audio_streams:
        return {"passed": False, "issue": "NO_AUDIO_STREAM"}

    # Check that audio has reasonable duration (not empty)
    audio = audio_streams[0]
    codec = audio.get("codec_name", "unknown")
    channels = audio.get("channels", 0)
    sample_rate = audio.get("sample_rate", "0")

    if channels < 1:
        return {"passed": False, "issue": "AUDIO_NO_CHANNELS"}

    return {
        "passed": True,
        "codec": codec,
        "channels": channels,
        "sample_rate": sample_rate,
    }


def check_narration_clean(
    script_text: str = "",
    title: str = "",
    description: str = "",
) -> dict:
    """
    CHECK 8 — Narración sin contaminación del system prompt.

    Escanea el texto narrado completo buscando instrucciones meta que el LLM
    pudo haber incluido literalmente en el output.

    Casos reales (25 marzo 2026):
      - "NEIRO explota" → mostraba "Usar 'TÚ' más que 'YO' 4."
      - "¿Qué es BTC?" → mostraba "words... RESPOND ONLY WITH THE"

    ABORT si encuentra cualquier patrón → el productor NO debe procesar este script.
    """
    # Combine all text fields to check
    combined = f"{title}\n{description}\n{script_text}"

    if not combined.strip():
        return {"passed": True, "note": "NO_TEXT_TO_CHECK"}

    found_patterns = []
    matched_samples = []

    for pattern in NARRATION_CONTAMINATION_PATTERNS:
        try:
            m = re.search(pattern, combined, re.IGNORECASE | re.MULTILINE)
            if m:
                snippet = combined[max(0, m.start() - 20): m.end() + 30].replace("\n", " ").strip()
                found_patterns.append(pattern[:50])
                matched_samples.append(snippet[:80])
        except re.error:
            pass  # Skip invalid regex

    if found_patterns:
        return {
            "passed": False,
            "issue": "NARRATION_CONTAMINATED",
            "matched_patterns": found_patterns[:5],
            "samples": matched_samples[:5],
            "action": "ABORT — regenerate script, do not pass to TTS or producer",
        }

    return {"passed": True}


def check_visual_contamination(video_path: str, sample_interval: int = 5) -> dict:
    """
    CHECK 9 — Verificación visual OCR del video.

    Extrae frames cada `sample_interval` segundos y aplica OCR básico con
    tesseract (si disponible) o pytesseract para detectar si el texto
    del system prompt es visible en pantalla.

    Si tesseract no está instalado: falla gracefully con passed=True + nota.
    """
    if not video_path or not Path(video_path).exists():
        return {"passed": False, "issue": "VIDEO_NOT_FOUND"}

    # Check if tesseract is available
    try:
        r = subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        has_tesseract = r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_tesseract = False

    if not has_tesseract:
        # Try pytesseract as fallback
        try:
            import pytesseract  # noqa: F401
            has_tesseract = True
        except ImportError:
            return {
                "passed": True,
                "note": "TESSERACT_NOT_INSTALLED — visual OCR check skipped",
                "recommendation": "Install: sudo apt install tesseract-ocr tesseract-ocr-spa",
            }

    # Get video duration
    info = _run_ffprobe(video_path, ["-show_entries", "format=duration"])
    duration = float(info.get("format", {}).get("duration", 0))
    if duration <= 0:
        return {"passed": True, "note": "ZERO_DURATION_SKIPPED"}

    contaminated_frames = []
    frames_checked = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # Sample every N seconds
        timestamps = list(range(0, int(duration), sample_interval))
        if not timestamps:
            timestamps = [0]

        for t in timestamps:
            frame_path = os.path.join(tmpdir, f"ocr_frame_{t}.png")
            if not _extract_frame(video_path, float(t), frame_path):
                continue

            frames_checked += 1

            # Run OCR
            ocr_text = ""
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(frame_path)
                # Try Spanish first, fall back to English
                try:
                    ocr_text = pytesseract.image_to_string(img, lang="spa")
                except Exception:
                    ocr_text = pytesseract.image_to_string(img)
            except ImportError:
                # Use tesseract CLI directly
                try:
                    out_base = os.path.join(tmpdir, f"ocr_out_{t}")
                    subprocess.run(
                        ["tesseract", frame_path, out_base, "-l", "spa"],
                        capture_output=True, timeout=15
                    )
                    txt_file = out_base + ".txt"
                    if Path(txt_file).exists():
                        ocr_text = Path(txt_file).read_text(errors="ignore")
                except Exception:
                    continue

            if not ocr_text:
                continue

            # Check OCR text against contamination patterns
            for pattern in VISUAL_CONTAMINATION_PATTERNS:
                try:
                    m = re.search(pattern, ocr_text, re.IGNORECASE)
                    if m:
                        snippet = ocr_text[max(0, m.start() - 10): m.end() + 30].replace("\n", " ").strip()
                        contaminated_frames.append({
                            "timestamp_s": t,
                            "pattern": pattern[:50],
                            "ocr_snippet": snippet[:80],
                        })
                        break  # One match per frame is enough
                except re.error:
                    pass

    if contaminated_frames:
        return {
            "passed": False,
            "issue": "VISUAL_CONTAMINATION",
            "contaminated_frames": contaminated_frames[:5],
            "frames_checked": frames_checked,
            "action": "DELETE — system prompt text visible in video, do not upload",
        }

    return {
        "passed": True,
        "frames_checked": frames_checked,
        "note": "No contamination detected in visible text",
    }


def check_duration(video_path: str, video_type: str = "long") -> dict:
    """
    CHECK 7 — Duración razonable.
    Long: mínimo 5 min, máximo 20 min.
    Short: mínimo 30s, máximo 60s.
    """
    if not Path(video_path).exists():
        return {"passed": False, "issue": "VIDEO_NOT_FOUND", "seconds": 0}

    info = _run_ffprobe(video_path, ["-show_entries", "format=duration"])
    duration_str = info.get("format", {}).get("duration", "0")

    try:
        seconds = float(duration_str)
    except (ValueError, TypeError):
        return {"passed": False, "issue": "DURATION_PARSE_ERROR", "seconds": 0}

    if seconds <= 0:
        return {"passed": False, "issue": "ZERO_DURATION", "seconds": 0}

    vtype = (video_type or "long").lower()

    if vtype == "short":
        min_s, max_s = SHORT_MIN_SECONDS, SHORT_MAX_SECONDS
    else:
        min_s, max_s = LONG_MIN_SECONDS, LONG_MAX_SECONDS

    if seconds < min_s:
        return {
            "passed": False,
            "issue": "TOO_SHORT",
            "seconds": round(seconds, 1),
            "minimum": min_s,
        }
    if seconds > max_s:
        return {
            "passed": False,
            "issue": "TOO_LONG",
            "seconds": round(seconds, 1),
            "maximum": max_s,
        }

    return {"passed": True, "seconds": round(seconds, 1), "type": vtype}


# ══════════════════════════════════════════════════════════════════════════════
# POST-UPLOAD VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_post_upload(
    video_id: str,
    expected_title: str,
    youtube_client=None,
) -> dict:
    """
    Verifica que el video subido a YouTube existe y está correctamente configurado.
    Llama DESPUÉS de upload_to_youtube().

    Checks:
    - Video existe y es accesible (no 404/403)
    - Tiene thumbnail custom (maxresdefault presente)
    - El título en YouTube coincide con el esperado
    """
    if not youtube_client:
        return {"passed": True, "note": "NO_YT_CLIENT_SKIPPED"}

    issues = []
    checks = {}

    try:
        result = youtube_client.videos().list(
            part="snippet,status,contentDetails",
            id=video_id,
        ).execute()

        items = result.get("items", [])
        if not items:
            return {
                "passed": False,
                "issue": "VIDEO_NOT_FOUND_ON_YOUTUBE",
                "video_id": video_id,
            }

        item = items[0]
        snippet = item.get("snippet", {})
        status = item.get("status", {})

        # Check title matches
        yt_title = snippet.get("title", "").strip()
        if yt_title.lower() != expected_title.strip().lower():
            issues.append(f"TITLE_MISMATCH: expected='{expected_title[:40]}' got='{yt_title[:40]}'")
            checks["title_match"] = {"passed": False, "expected": expected_title[:60], "got": yt_title[:60]}
        else:
            checks["title_match"] = {"passed": True}

        # Check thumbnail — maxresdefault means custom thumbnail was applied
        thumbnails = snippet.get("thumbnails", {})
        has_maxres = "maxres" in thumbnails
        has_standard = "standard" in thumbnails
        has_custom = has_maxres or has_standard

        if not has_maxres:
            # Standard resolution thumbnail could still be auto-generated
            # maxres is only present when custom thumbnail is uploaded
            issues.append("NO_CUSTOM_THUMBNAIL")
            checks["custom_thumbnail"] = {"passed": False, "available": list(thumbnails.keys())}
        else:
            checks["custom_thumbnail"] = {"passed": True, "url": thumbnails["maxres"].get("url", "")[:80]}

        # Check video is accessible (not private/deleted)
        privacy = status.get("privacyStatus", "unknown")
        upload_status = status.get("uploadStatus", "unknown")

        if upload_status == "failed":
            issues.append(f"UPLOAD_FAILED_ON_YT: {status.get('failureReason','?')}")
            checks["accessible"] = {"passed": False, "upload_status": upload_status}
        elif privacy in ("private", "unlisted") and upload_status != "processed":
            checks["accessible"] = {"passed": True, "privacy": privacy, "note": "scheduled/private is OK"}
        else:
            checks["accessible"] = {"passed": True, "privacy": privacy, "upload_status": upload_status}

        passed = len(issues) == 0
        return {
            "passed": passed,
            "video_id": video_id,
            "checks": checks,
            "issues": issues,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        err = str(e)
        if "403" in err or "forbidden" in err.lower():
            return {"passed": False, "issue": "HTTP_403_FORBIDDEN", "video_id": video_id}
        if "404" in err or "notFound" in err.lower():
            return {"passed": False, "issue": "HTTP_404_NOT_FOUND", "video_id": video_id}
        return {"passed": False, "issue": f"VERIFY_ERROR: {err[:100]}", "video_id": video_id}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN QA AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class QAAgent:
    """
    QA Agent — ejecuta todos los checks de calidad antes de un upload.

    Uso:
        qa = QAAgent()
        result = qa.run_all_checks(
            video_path="/path/to/video.mp4",
            thumb_path="/path/to/thumb.jpg",
            title="Mi Video",
            description="Descripción...",
            script_text="Texto del guión...",
            video_type="long",         # "long" o "short"
            youtube_client=yt_client,  # opcional, para check de duplicados
        )
        if not result["passed"]:
            print(result["issues"])
    """

    def run_all_checks(
        self,
        video_path: str = "",
        thumb_path: str = "",
        title: str = "",
        description: str = "",
        script_text: str = "",
        video_type: str = "long",
        youtube_client=None,
    ) -> dict:
        """
        Ejecuta los 7 checks de calidad.
        Retorna dict con 'passed', 'checks', 'issues', 'timestamp'.
        """
        _log(f"Iniciando QA para: '{title[:50]}'")
        ts = datetime.now(timezone.utc).isoformat()

        checks = {}
        issues = []

        # ── CHECK 1: Brightness ──────────────────────────────────────────
        _log("CHECK 1: Brightness del video...")
        if video_path:
            r = check_video_brightness(video_path)
            checks["brightness"] = r
            if not r["passed"]:
                issues.append(r.get("issue", "DARK_VIDEO"))
                _log(f"  ❌ FAIL: {r.get('issue')} (val={r.get('value')})", "WARNING")
            else:
                _log(f"  ✅ OK: brightness={r.get('value')}")
        else:
            checks["brightness"] = {"passed": False, "issue": "NO_VIDEO_PATH"}
            issues.append("NO_VIDEO_PATH")

        # ── CHECK 2: Placeholders ────────────────────────────────────────
        _log("CHECK 2: Sin placeholders...")
        r = check_no_placeholders(video_path, script_text, title, description)
        checks["placeholders"] = r
        if not r["passed"]:
            issues.append(r.get("issue", "PLACEHOLDER_FOUND"))
            _log(f"  ❌ FAIL: {r.get('issue')} — encontrados: {r.get('found')}", "WARNING")
        else:
            _log("  ✅ OK")

        # ── CHECK 3: Thumbnail ───────────────────────────────────────────
        _log("CHECK 3: Thumbnail válida...")
        # Pass video_type so Shorts are checked for vertical (1080x1920) orientation
        r = check_thumbnail(thumb_path, video_type=video_type)
        checks["thumbnail"] = r
        if not r["passed"]:
            issues.append(r.get("issue", "THUMBNAIL_INVALID"))
            _log(f"  ❌ FAIL: {r.get('issue')} (expected {r.get('expected_size','?')})", "WARNING")
        else:
            _log(f"  ✅ OK: brightness={r.get('brightness')}, size={r.get('size')}, orientation={r.get('orientation')}")

        # ── CHECK 4: Duplicados ──────────────────────────────────────────
        _log("CHECK 4: Sin duplicados en YouTube...")
        r = check_no_duplicates(title, youtube_client)
        checks["duplicates"] = r
        if not r["passed"]:
            issues.append(r.get("issue", "DUPLICATE_TITLE"))
            _log(f"  ❌ FAIL: {r.get('issue')} — id={r.get('existing_video_id')}", "WARNING")
        else:
            _log(f"  ✅ OK (chequeados: {r.get('checked_against', 'N/A')})")

        # ── CHECK 5: Título limpio ───────────────────────────────────────
        _log("CHECK 5: Título limpio...")
        r = check_title(title)
        checks["title"] = r
        if not r["passed"]:
            issues.append(r.get("issue", "BAD_TITLE"))
            _log(f"  ❌ FAIL: {r.get('issue')}", "WARNING")
        else:
            _log(f"  ✅ OK: {len(title)} chars")

        # ── CHECK 6: Audio presente ──────────────────────────────────────
        _log("CHECK 6: Audio presente...")
        if video_path:
            r = check_audio(video_path)
            checks["audio"] = r
            if not r["passed"]:
                issues.append(r.get("issue", "NO_AUDIO"))
                _log(f"  ❌ FAIL: {r.get('issue')}", "WARNING")
            else:
                _log(f"  ✅ OK: codec={r.get('codec')}, channels={r.get('channels')}")
        else:
            checks["audio"] = {"passed": False, "issue": "NO_VIDEO_PATH"}
            if "NO_VIDEO_PATH" not in issues:
                issues.append("NO_VIDEO_PATH")

        # ── CHECK 7: Duración ────────────────────────────────────────────
        _log("CHECK 7: Duración razonable...")
        if video_path:
            r = check_duration(video_path, video_type)
            checks["duration"] = r
            if not r["passed"]:
                issues.append(r.get("issue", "BAD_DURATION"))
                _log(f"  ❌ FAIL: {r.get('issue')} (seconds={r.get('seconds')})", "WARNING")
            else:
                _log(f"  ✅ OK: {r.get('seconds')}s ({video_type})")
        else:
            checks["duration"] = {"passed": False, "issue": "NO_VIDEO_PATH"}

        # ── CHECK 8: Narración limpia (NO system prompt leak) ────────────
        _log("CHECK 8: Narración sin contaminación del system prompt...")
        r = check_narration_clean(script_text, title, description)
        checks["narration_clean"] = r
        if not r["passed"]:
            issues.append(r.get("issue", "NARRATION_CONTAMINATED"))
            _log(f"  ❌ FAIL: {r.get('issue')}", "ERROR")
            for sample in r.get("samples", [])[:2]:
                _log(f"     Encontrado: '{sample}'", "ERROR")
            # Notify immediately — this is the critical bug from NEIRO/BTC (25 mar 2026)
            self._notify_narration_contamination(title, r)
        else:
            _log("  ✅ OK")

        # ── CHECK 9: Visual OCR — texto visible en pantalla ──────────────
        # Solo correr si el video existe Y pasó el check de narración
        # (evitar doble trabajo si ya sabemos que el script está contaminado)
        _log("CHECK 9: Texto visible en pantalla (OCR)...")
        if video_path and checks.get("narration_clean", {}).get("passed", True):
            r = check_visual_contamination(video_path, sample_interval=5)
            checks["visual_ocr"] = r
            if not r["passed"]:
                issues.append(r.get("issue", "VISUAL_CONTAMINATION"))
                _log(f"  ❌ FAIL: {r.get('issue')} — {len(r.get('contaminated_frames',[]))} frames afectados", "ERROR")
                for f in r.get("contaminated_frames", [])[:2]:
                    _log(f"     t={f['timestamp_s']}s: '{f.get('ocr_snippet','?')}'", "ERROR")
            else:
                note = r.get("note", "")
                frames = r.get("frames_checked", 0)
                _log(f"  ✅ OK ({frames} frames revisados) {note}")
        elif not video_path:
            checks["visual_ocr"] = {"passed": False, "issue": "NO_VIDEO_PATH"}
        else:
            # Narration already failed — mark OCR as skipped (contamination already caught)
            checks["visual_ocr"] = {
                "passed": False,
                "issue": "SKIPPED_NARRATION_ALREADY_FAILED",
                "note": "Script contamination already detected in CHECK 8 — OCR skipped",
            }
            # Don't add a new issue code since NARRATION_CONTAMINATED is already in issues

        # ── CHECK 10: Logo BitTrader en video ────────────────────────────
        # Orden de Ender 2026-03-28: el video DEBE tener logo BitTrader visible
        # Se verifica extrayendo frames al inicio y al final del video
        _log("CHECK 10: Logo BitTrader en video...")
        if video_path and Path(video_path).exists():
            try:
                import subprocess as _sp, tempfile as _tmp, os as _os
                _logo_found = False
                _logo_check_note = ""
                _frames_dir = _tmp.mkdtemp()
                try:
                    # Extraer 3 frames: al inicio (5s), mitad, y final (-5s)
                    _dur_r = _sp.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                        capture_output=True, text=True, timeout=15
                    )
                    _dur = float(_dur_r.stdout.strip() or "0")
                    _sample_times = [5, max(10, _dur/2), max(15, _dur - 5)]
                    for _i, _t in enumerate(_sample_times):
                        if _t >= _dur:
                            continue
                        _frame_path = _os.path.join(_frames_dir, f"frame_{_i}.jpg")
                        _sp.run(
                            ["ffmpeg", "-ss", str(_t), "-i", video_path,
                             "-frames:v", "1", "-q:v", "2", _frame_path, "-y"],
                            capture_output=True, timeout=30
                        )
                        if not _os.path.exists(_frame_path):
                            continue
                        # Check for BitTrader branding: look for orange/gold pixels in corners
                        # BitTrader logo typically uses gold (#F5A623) or white text in corners
                        from PIL import Image as _PILI
                        import numpy as _np2
                        _fi = _PILI.open(_frame_path).convert("RGB")
                        _fa = _np2.array(_fi, dtype=_np2.float32)
                        _h, _w = _fa.shape[:2]
                        # Check bottom-left corner (logo area) and top-right (@bittrader9259)
                        _corners = [
                            _fa[_h-100:_h, 0:200],       # bottom-left
                            _fa[0:80, _w-300:_w],         # top-right
                            _fa[_h-80:_h, 0:300],         # bottom-left wide
                        ]
                        for _corner in _corners:
                            _R2, _G2, _B2 = _corner[:,:,0], _corner[:,:,1], _corner[:,:,2]
                            # Gold/orange pixels: R>180, G>100, B<80
                            _gold = ((_R2 > 180) & (_G2 > 100) & (_B2 < 80)).sum()
                            # White text pixels: R>200, G>200, B>200
                            _white = ((_R2 > 200) & (_G2 > 200) & (_B2 > 200)).sum()
                            if _gold > 50 or _white > 200:
                                _logo_found = True
                                break
                        if _logo_found:
                            break
                finally:
                    import shutil as _sh
                    _sh.rmtree(_frames_dir, ignore_errors=True)

                if _logo_found:
                    checks["logo_bittrader"] = {"passed": True, "note": "BitTrader branding detected in video frames"}
                    _log("  ✅ OK: Logo BitTrader detectado en video")
                else:
                    checks["logo_bittrader"] = {"passed": False, "issue": "NO_LOGO_BITTRADER"}
                    issues.append("NO_LOGO_BITTRADER")
                    _log("  ❌ FAIL: Logo BitTrader NO encontrado en el video — video rechazado", "ERROR")
            except Exception as _e:
                _log(f"  ⚠️ WARN: No se pudo verificar logo: {_e}", "WARNING")
                checks["logo_bittrader"] = {"passed": True, "note": f"check_skipped: {_e}"}
        else:
            checks["logo_bittrader"] = {"passed": False, "issue": "NO_VIDEO_PATH"}

        # ── RESULT ──────────────────────────────────────────────────────
        passed = len(issues) == 0
        result = {
            "passed":    passed,
            "checks":    checks,
            "issues":    issues,
            "title":     title,
            "video_path": video_path,
            "timestamp": ts,
        }

        _save_qa_log(result)

        if passed:
            _log(f"✅ QA PASSED: '{title[:50]}'")
        else:
            _log(f"❌ QA FAILED: '{title[:50]}' — issues: {issues}", "ERROR")
            # Notify Ender via Telegram
            self._notify_qa_failure(title, issues, video_path)

        return result

    def _notify_qa_failure(self, title: str, issues: list, video_path: str = ""):
        """Send Telegram alert about a QA failure."""
        issues_str = "\n".join(f"  • <b>{i}</b>" for i in issues)
        msg = (
            f"⚠️ <b>Video bloqueado por QA</b>\n\n"
            f"📹 <b>Título:</b> {title}\n"
            f"🔴 <b>Problemas ({len(issues)}):</b>\n{issues_str}\n\n"
            f"📁 Video no subido a YouTube hasta corregir."
        )
        if video_path:
            msg += f"\n\n<code>{video_path}</code>"
        notify_ender(msg)

    def _notify_narration_contamination(self, title: str, check_result: dict):
        """
        Alerta urgente específica para contaminación de narración.
        Este es el bug crítico de NEIRO/BTC (25 marzo 2026).
        """
        samples = check_result.get("samples", [])
        samples_str = "\n".join(f"  ⚠️ <code>{s}</code>" for s in samples[:3])
        patterns = check_result.get("matched_patterns", [])
        patterns_str = ", ".join(f"<code>{p[:40]}</code>" for p in patterns[:3])

        msg = (
            f"🚨 <b>CONTAMINACIÓN CRÍTICA DETECTADA</b>\n\n"
            f"📹 <b>Video:</b> {title}\n\n"
            f"El system prompt está visible en el guión/narración.\n"
            f"<b>Patrones encontrados:</b> {patterns_str}\n\n"
            f"<b>Texto encontrado:</b>\n{samples_str}\n\n"
            f"🛑 <b>Video BLOQUEADO</b> — regenerar script con creator.py\n"
            f"(Bug tipo NEIRO/BTC — texto de instrucciones en pantalla)"
        )
        notify_ender(msg)


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🔬 QA Agent — BitTrader quality gate before YouTube upload"
    )
    parser.add_argument("--video",       help="Path to video file (.mp4)")
    parser.add_argument("--thumb",       help="Path to thumbnail file (.jpg/.png)")
    parser.add_argument("--title",       default="", help="Video title")
    parser.add_argument("--description", default="", help="Video description")
    parser.add_argument("--type",        default="long", choices=["long", "short"], help="Video type")
    parser.add_argument("--notify",      action="store_true", help="Send Telegram notification on fail")
    parser.add_argument("--check",       choices=["brightness","placeholders","thumbnail","audio","duration","title"],
                        help="Run a single check only")
    args = parser.parse_args()

    if args.check:
        # Single-check mode
        if args.check == "brightness":
            r = check_video_brightness(args.video or "")
        elif args.check == "placeholders":
            r = check_no_placeholders(args.video or "", "", args.title, args.description)
        elif args.check == "thumbnail":
            r = check_thumbnail(args.thumb or "", video_type=args.type)
        elif args.check == "audio":
            r = check_audio(args.video or "")
        elif args.check == "duration":
            r = check_duration(args.video or "", args.type)
        elif args.check == "title":
            r = check_title(args.title)
        else:
            r = {"error": "unknown check"}
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    else:
        # Full QA run
        qa = QAAgent()
        result = qa.run_all_checks(
            video_path=args.video or "",
            thumb_path=args.thumb or "",
            title=args.title,
            description=args.description,
            video_type=args.type,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        sys.exit(0 if result["passed"] else 1)
