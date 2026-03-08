#!/usr/bin/env python3
"""
🎤 BitTrader Karaoke Subtitles Generator
Genera subtítulos ASS estilo viral con efecto karaoke.
La palabra activa se agranda y resalta mientras el locutor habla.

Estilo:
- Texto amarillo (#FFFF00) con borde negro (Outline=3)
- Grupos de ~5-6 palabras por línea
- Palabra activa: más grande (escala 130%) con borde más grueso
- Posición: parte inferior del video
- Sin fondo/caja — solo texto con borde

Pipeline: Audio → Whisper (word timestamps) → ASS karaoke

Uso:
  from karaoke_subs import generate_karaoke_subs
  ass_path = generate_karaoke_subs(audio_path, script_text, output_dir, video_type)
"""
import json
import subprocess
import re
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
WHISPER_MODEL    = "small"
WHISPER_LANGUAGE = "es"
WORDS_PER_GROUP  = 6

# ── ASS Style ─────────────────────────────────────────────────────────────
# Colors in ASS format: &HAABBGGRR (hex, reversed BGR)
YELLOW      = "&H0000FFFF"   # #FFFF00 in BGR
WHITE       = "&H00FFFFFF"
BLACK       = "&H00000000"
GOLD        = "&H0000D4FF"   # #FFD400
HIGHLIGHT   = "&H0000FFFF"   # Yellow for active word

# Font sizes (must be large enough for mobile viewing!)
# For 1080x1920 vertical: minimum 60px, recommended 75-80px
# For 1920x1080 horizontal: minimum 45px, recommended 55px
FONT_SIZE_SHORT  = 78
FONT_SIZE_LONG   = 55
HIGHLIGHT_SCALE  = 140  # % scale for active word


# ════════════════════════════════════════════════════════════════════════
# WHISPER — Word-level timestamps
# ════════════════════════════════════════════════════════════════════════

def get_word_timestamps(audio_path: Path) -> list:
    """Run Whisper to get word-level timestamps."""
    output_dir = audio_path.parent
    
    cmd = [
        "whisper", str(audio_path),
        "--model", WHISPER_MODEL,
        "--language", WHISPER_LANGUAGE,
        "--word_timestamps", "True",
        "--output_format", "json",
        "--output_dir", str(output_dir),
        "--verbose", "False",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"      ⚠️ Whisper error: {result.stderr[:200]}")
        return []
    
    # Read JSON output
    json_path = audio_path.with_suffix(".json")
    if not json_path.exists():
        stem = audio_path.stem
        json_path = output_dir / f"{stem}.json"
    
    if not json_path.exists():
        print(f"      ⚠️ Whisper JSON not found")
        return []
    
    data = json.loads(json_path.read_text())
    
    words = []
    for segment in data.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info.get("word", "").strip()
            if not word:
                continue
            words.append({
                "word": word,
                "start": word_info.get("start", 0),
                "end": word_info.get("end", 0),
            })
    
    # Cleanup Whisper output files
    for ext in [".json", ".srt", ".vtt", ".tsv", ".txt"]:
        p = output_dir / f"{audio_path.stem}{ext}"
        if p.exists():
            p.unlink(missing_ok=True)
    
    return words


def get_word_timestamps_proportional(script_text: str, duration: float) -> list:
    """
    Generate word timestamps proportionally from text and audio duration.
    Used when Whisper fails (e.g. MiniMax TTS voices not recognized).
    
    Distributes words across the audio duration based on character length,
    with small pauses at sentence boundaries (periods, question marks, etc).
    """
    # Clean script text
    clean = re.sub(r'[\U00010000-\U0010ffff]', '', script_text)
    clean = re.sub(r'\[.*?\]|\(.*?\)', '', clean)
    
    # Split into words preserving punctuation
    raw_words = [w.strip() for w in clean.split() if w.strip()]
    if not raw_words:
        return []
    
    # Calculate weight per word based on character count
    # Longer words take more time to pronounce
    # Add pause weight after sentence-ending punctuation
    PAUSE_WEIGHT = 3  # Extra chars worth of pause after sentence end
    
    weights = []
    for w in raw_words:
        base = len(w)
        # Add pause after sentence enders
        if w.endswith(('.', '?', '!', '…')):
            base += PAUSE_WEIGHT
        elif w.endswith((',', ';', ':')):
            base += PAUSE_WEIGHT // 2
        weights.append(max(1, base))
    
    total_weight = sum(weights)
    
    # Leave small margin at start and end
    margin = 0.15
    usable_duration = duration - (margin * 2)
    if usable_duration < 1:
        usable_duration = duration
        margin = 0
    
    # Distribute time proportionally
    words = []
    current_time = margin
    
    for i, (word, weight) in enumerate(zip(raw_words, weights)):
        word_duration = (weight / total_weight) * usable_duration
        words.append({
            "word": word,
            "start": round(current_time, 3),
            "end": round(current_time + word_duration, 3),
        })
        current_time += word_duration
    
    return words


def align_words_to_script(whisper_words: list, script_text: str) -> list:
    """Align Whisper words to the original script text (correct spelling)."""
    # Clean script text
    script_clean = re.sub(r'[\U00010000-\U0010ffff]', '', script_text)
    script_clean = re.sub(r'\[.*?\]|\(.*?\)', '', script_clean)
    script_words = [w.strip() for w in script_clean.split() if w.strip()]
    
    # If counts match closely, use script words for spelling
    aligned = []
    for i, ww in enumerate(whisper_words):
        if i < len(script_words):
            # Use script word (correct spelling) with Whisper timing
            aligned.append({
                "word": script_words[i],
                "start": ww["start"],
                "end": ww["end"],
            })
        else:
            # Extra whisper words — use as-is
            aligned.append(ww)
    
    return aligned if aligned else whisper_words


# ════════════════════════════════════════════════════════════════════════
# ASS KARAOKE GENERATION
# ════════════════════════════════════════════════════════════════════════

def format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time: H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass_content(words: list, video_type: str = "short") -> str:
    """Generate ASS subtitle content with karaoke effect."""
    
    if video_type == "short":
        play_res_x, play_res_y = 1080, 1920
        font_size = FONT_SIZE_SHORT
        margin_v = 420  # Higher up on vertical video
    else:
        play_res_x, play_res_y = 1920, 1080
        font_size = FONT_SIZE_LONG
        margin_v = 80
    
    highlight_size = int(font_size * HIGHLIGHT_SCALE / 100)
    
    # ASS Header
    header = f"""[Script Info]
Title: BitTrader Karaoke Subtitles
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{font_size},{YELLOW},{YELLOW},{BLACK},&H80000000,1,0,0,0,100,100,0,0,1,3,0,2,20,20,{margin_v},1
Style: Highlight,DejaVu Sans,{highlight_size},{YELLOW},{YELLOW},{BLACK},&H80000000,1,0,0,0,{HIGHLIGHT_SCALE},{HIGHLIGHT_SCALE},0,0,1,4,0,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Group words into lines of WORDS_PER_GROUP
    groups = []
    for i in range(0, len(words), WORDS_PER_GROUP):
        group = words[i:i + WORDS_PER_GROUP]
        groups.append(group)
    
    events = []
    
    for group in groups:
        if not group:
            continue
        
        group_start = group[0]["start"]
        group_end = group[-1]["end"]
        
        start_str = format_ass_time(group_start)
        end_str = format_ass_time(group_end)
        
        # Build karaoke text with \kf tags
        # Each word gets a \kf duration (in centiseconds)
        # The active word gets highlighted via {\fscXXX} override
        parts = []
        for j, w in enumerate(group):
            word_dur_cs = max(1, int((w["end"] - w["start"]) * 100))
            
            # Use \kf for smooth fill effect
            # The word being spoken gets scaled up
            parts.append(
                f"{{\\kf{word_dur_cs}}}{w['word']}"
            )
        
        text = " ".join(parts)
        
        # Dialogue line
        events.append(
            f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
        )
    
    return header + "\n".join(events) + "\n"


def generate_ass_karaoke_word_highlight(words: list, video_type: str = "short") -> str:
    """
    Generate ASS with word-by-word highlight effect.
    Each word group shows the full line, with the active word enlarged.
    This is the viral TikTok/YouTube style.
    """
    
    if video_type == "short":
        play_res_x, play_res_y = 1080, 1920
        font_size = FONT_SIZE_SHORT
        margin_v = 420
    else:
        play_res_x, play_res_y = 1920, 1080
        font_size = FONT_SIZE_LONG
        margin_v = 80
    
    highlight_size = int(font_size * HIGHLIGHT_SCALE / 100)
    
    header = f"""[Script Info]
Title: BitTrader Viral Karaoke Subtitles
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{font_size},{YELLOW},{YELLOW},{BLACK},&H80000000,1,0,0,0,100,100,0,0,1,3,0,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Group words
    groups = []
    for i in range(0, len(words), WORDS_PER_GROUP):
        group = words[i:i + WORDS_PER_GROUP]
        groups.append(group)
    
    events = []
    
    for group in groups:
        if not group:
            continue
        
        group_start = group[0]["start"]
        group_end = group[-1]["end"]
        
        # For each word in the group, create a dialogue line where THAT word is highlighted
        for j, active_word in enumerate(group):
            w_start = active_word["start"]
            # End is either next word start or group end
            if j + 1 < len(group):
                w_end = group[j + 1]["start"]
            else:
                w_end = group_end
            
            start_str = format_ass_time(w_start)
            end_str = format_ass_time(w_end)
            
            # Build line: all words normal, active word enlarged
            parts = []
            for k, w in enumerate(group):
                if k == j:
                    # Active word: bigger, bolder outline
                    parts.append(
                        f"{{\\fscx{HIGHLIGHT_SCALE}\\fscy{HIGHLIGHT_SCALE}\\bord4}}"
                        f"{w['word']}"
                        f"{{\\fscx100\\fscy100\\bord3}}"
                    )
                else:
                    parts.append(w["word"])
            
            text = " ".join(parts)
            events.append(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
            )
    
    return header + "\n".join(events) + "\n"


# ════════════════════════════════════════════════════════════════════════
# FALLBACK — SRT sin Whisper
# ════════════════════════════════════════════════════════════════════════

def generate_srt_fallback(script_text: str, duration: float, output_path: Path):
    """Fallback: generate basic SRT if Whisper fails."""
    lines = script_text.strip().split('\n')
    display_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('('):
            continue
        line = re.sub(r'[\U00010000-\U0010ffff]', '', line).strip()
        if len(line) > 5:
            display_lines.append(line)
    
    chunks = []
    for line in display_lines:
        words = line.split()
        while len(words) > WORDS_PER_GROUP:
            chunks.append(' '.join(words[:WORDS_PER_GROUP]))
            words = words[WORDS_PER_GROUP:]
        if words:
            chunks.append(' '.join(words))
    
    if not chunks:
        return
    
    time_per = duration / len(chunks)
    srt_lines = []
    
    for i, chunk in enumerate(chunks):
        start = i * time_per
        end = min((i + 1) * time_per, duration - 0.05)
        
        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        
        srt_lines.extend([str(i+1), f"{fmt(start)} --> {fmt(end)}", chunk, ""])
    
    output_path.write_text('\n'.join(srt_lines), encoding='utf-8')


# ════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════

def generate_karaoke_subs(
    audio_path: Path,
    script_text: str,
    output_dir: Path,
    video_type: str = "short",
    style: str = "word_highlight",
    use_whisper: bool = False
) -> Path:
    """
    Generate karaoke subtitles from audio + script text.
    
    Uses proportional distribution by default (works with all TTS engines).
    Set use_whisper=True for edge-tts or natural speech audio.
    
    Args:
        audio_path: Path to audio file (mp3/wav)
        script_text: Original script text
        output_dir: Directory to save subtitle file
        video_type: "short" or "long"
        style: "karaoke" (smooth fill) or "word_highlight" (viral style)
        use_whisper: If True, try Whisper first (for edge-tts/natural audio)
    
    Returns:
        Path to generated subtitle file (.ass or .srt fallback)
    """
    print(f"      🎤 Generando subtítulos karaoke...")
    
    # Get audio duration
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-show_format", "-print_format", "json",
            str(audio_path)
        ], capture_output=True, text=True, timeout=10)
        duration = float(json.loads(result.stdout).get("format", {}).get("duration", 30))
    except:
        duration = 30
    
    words = []
    
    # Method 1: Whisper (optional — for edge-tts or natural audio)
    if use_whisper:
        words = get_word_timestamps(audio_path)
        if words and len(words) > 3:
            print(f"      📝 Whisper: {len(words)} palabras detectadas")
            words = align_words_to_script(words, script_text)
        else:
            print(f"      ⚠️ Whisper: pocas palabras ({len(words)}) — usando distribución proporcional")
            words = []
    
    # Method 2: Proportional distribution (default — works with any TTS)
    if not words:
        words = get_word_timestamps_proportional(script_text, duration)
        print(f"      📝 Distribución proporcional: {len(words)} palabras en {duration:.1f}s")
    
    if not words:
        print(f"      ⚠️ Sin palabras — usando SRT fallback")
        srt_path = output_dir / "subtitles.srt"
        generate_srt_fallback(script_text, duration, srt_path)
        return srt_path
    
    # Generate ASS
    ass_path = output_dir / "subtitles.ass"
    
    if style == "word_highlight":
        content = generate_ass_karaoke_word_highlight(words, video_type)
    else:
        content = generate_ass_content(words, video_type)
    
    ass_path.write_text(content, encoding='utf-8')
    
    n_groups = (len(words) + WORDS_PER_GROUP - 1) // WORDS_PER_GROUP
    print(f"      ✅ Subtítulos karaoke: {n_groups} grupos, {len(words)} palabras → {ass_path.name}")
    
    return ass_path


# ════════════════════════════════════════════════════════════════════════
# CLI TEST
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python3 karaoke_subs.py <audio.mp3> [script.txt] [short|long]")
        print("  Genera subtítulos ASS karaoke desde audio con Whisper")
        sys.exit(1)
    
    audio = Path(sys.argv[1])
    script = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else ""
    vtype = sys.argv[3] if len(sys.argv) > 3 else "short"
    
    result = generate_karaoke_subs(audio, script, audio.parent, vtype)
    print(f"\nGenerado: {result}")
