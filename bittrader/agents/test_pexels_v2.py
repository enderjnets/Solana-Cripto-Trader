"""
TEST v2: "3 reglas que salvaron mi cuenta de trading" con Pexels
Estrategia: muchos clips cortos (5-8s), cambio frecuente, sin loops, máxima variación.
"""
import json, os, requests, subprocess, tempfile, shutil, random
from pathlib import Path

PEXELS_KEY = json.loads(Path("keys/pexels.json").read_text())["api_key"]
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
AGENTS = Path(__file__).parent
OUTPUT_DIR = AGENTS / "output/pexels_test_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR = OUTPUT_DIR / "clips"
CLIPS_DIR.mkdir(exist_ok=True)

AUDIO_PATH = AGENTS / "output/2026-03-28/long_1774708531_991/long_1774708531_991.mp3"
LOGO_PATH = WORKSPACE / "bittrader/assets/bittrader_logo_transparent.png"
if not LOGO_PATH.exists():
    LOGO_PATH = WORKSPACE / "bittrader/assets/bittrader_logo.png"

OUTPUT_VIDEO = OUTPUT_DIR / "3reglas_pexels_v5.mp4"

CLIP_DURATION = 6  # segundos por clip en el video final
SUBS_PATH = AGENTS / "output/2026-03-28/long_1774708531_991/subtitles.ass"
HEADERS = {"Authorization": PEXELS_KEY}

# Keywords variadas — muchas categorías para máxima variación visual
KEYWORD_SETS = [
    # Trading / mercados
    "stock market trading", "bitcoin chart", "cryptocurrency trading",
    "financial charts", "trading terminal", "market analysis",
    "candlestick chart", "trading computer screens", "forex trading",
    # Riesgo / pérdidas
    "risk management business", "stop loss trading", "market crash",
    "red stock chart", "financial loss", "trading risk",
    # Éxito / ganancias
    "successful trader", "financial success", "profit growth",
    "green stock chart", "bitcoin profit", "financial freedom",
    # Conceptos
    "investment portfolio", "diversification finance", "money management",
    "trading strategy", "financial planning", "wealth management",
    # Ambiente
    "trading office", "financial district", "bank money",
    "laptop finance", "phone trading app", "crypto wallet",
]

def search_pexels(keyword: str, per_page: int = 3) -> list:
    """Buscar videos cortos en Pexels. Retorna lista de video_files."""
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=HEADERS,
            params={
                "query": keyword,
                "per_page": per_page,
                "orientation": "landscape",
                "size": "medium",
                "min_duration": 4,
                "max_duration": 20,  # clips cortos
            },
            timeout=15
        )
        if r.status_code != 200:
            return []
        videos = r.json().get("videos", [])
        results = []
        for v in videos:
            for vf in v.get("video_files", []):
                w = vf.get("width", 0)
                h = vf.get("height", 0)
                # Solo clips HORIZONTALES (width > height) y HD
                if w >= 1280 and w > h and vf.get("file_type") == "video/mp4":
                    results.append({
                        "url": vf["link"],
                        "width": w,
                        "height": h,
                        "duration": v.get("duration", 0),
                        "keyword": keyword,
                        "id": v["id"],
                    })
                    break
        return results
    except Exception as e:
        print(f"  ⚠️ '{keyword}': {e}")
        return []

def download_clip(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except:
        pass
    return False

def is_green_screen(video_path: Path) -> bool:
    try:
        import numpy as np
        from PIL import Image
        result = subprocess.run(
            ["ffmpeg", "-ss", "0.5", "-i", str(video_path), "-frames:v", "1", "-q:v", "2",
             "/tmp/gs_check.jpg", "-y"], capture_output=True, timeout=10
        )
        if not Path("/tmp/gs_check.jpg").exists():
            return False
        img = Image.open("/tmp/gs_check.jpg").convert("RGB")
        arr = np.array(img, dtype=np.float32)
        avg_r, avg_g, avg_b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        return avg_g > avg_r * 1.8 and avg_g > avg_b * 1.8 and avg_g > 100
    except:
        return False

def get_audio_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip() or "0")

def main():
    print("=" * 60)
    print("TEST v2: 3 reglas — Pexels clips cortos + variación máxima")
    print("=" * 60)

    audio_dur = get_audio_duration(AUDIO_PATH)
    clips_needed = int(audio_dur / CLIP_DURATION) + 2
    print(f"\n📢 Audio: {audio_dur:.1f}s → necesito ~{clips_needed} clips de {CLIP_DURATION}s")

    # 1. Buscar clips — muchas keywords, sin repetir IDs
    print(f"\n🔍 Buscando clips en Pexels...")
    all_clips = []
    seen_ids = set()
    
    random.shuffle(KEYWORD_SETS)  # variación aleatoria
    for keyword in KEYWORD_SETS:
        if len(all_clips) >= clips_needed:
            break
        results = search_pexels(keyword, per_page=3)
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_clips.append(r)
        if results:
            print(f"  ✅ '{keyword}' → {len(results)} clips (total: {len(all_clips)})")

    print(f"\n📥 Descargando {min(clips_needed, len(all_clips))} clips...")
    downloaded = []
    for i, clip_info in enumerate(all_clips[:clips_needed]):
        clip_path = CLIPS_DIR / f"clip_{i:03d}.mp4"
        if clip_path.exists():  # cache
            downloaded.append(clip_path)
            continue
        if download_clip(clip_info["url"], clip_path):
            if is_green_screen(clip_path):
                print(f"  🟢 Clip {i} green screen — descartado")
                clip_path.unlink(missing_ok=True)
                continue
            downloaded.append(clip_path)
            size_kb = clip_path.stat().st_size // 1024
            print(f"  💾 clip_{i:03d}: {clip_info['keyword'][:30]} ({clip_info['duration']}s, {size_kb}KB)")

    if len(downloaded) < 10:
        print(f"❌ Solo {len(downloaded)} clips — insuficiente")
        return False

    print(f"\n✅ {len(downloaded)} clips listos")

    # 2. Ensamblar — trim cada clip a CLIP_DURATION s, sin loop
    with tempfile.TemporaryDirectory() as tmpdir:
        trimmed = []
        for i, cp in enumerate(downloaded):
            out = Path(tmpdir) / f"t{i:03d}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(cp),
                "-t", str(CLIP_DURATION),
                # Forzar 16:9 1280x720 sin importar aspect ratio del clip fuente
                # zoompan + scale garantiza llenado completo sin barras negras
                # Escalar para que el lado más corto sea >= 720, luego crop centrado exacto a 1280x720
                "-vf", (
                    "scale='if(gt(iw/ih,16/9),trunc(oh*a/2)*2,1280)':'if(gt(iw/ih,16/9),720,trunc(ow/a/2)*2)',"
                    "crop=1280:720,setsar=1,fps=30"
                ),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
                "-pix_fmt", "yuv420p", "-an", str(out)
            ], capture_output=True, timeout=30)
            if out.exists():
                trimmed.append(out)

        print(f"  ✂️  {len(trimmed)} clips trimmed a {CLIP_DURATION}s c/u")

        # Concatenar
        concat_file = Path(tmpdir) / "list.txt"
        # Si faltan clips para cubrir el audio, repetir la lista (pero mezclada)
        all_trimmed = trimmed[:]
        while len(all_trimmed) * CLIP_DURATION < audio_dur + 10:
            shuffled = trimmed[:]
            random.shuffle(shuffled)
            all_trimmed += shuffled
        
        concat_file.write_text("\n".join(f"file '{p}'" for p in all_trimmed))
        
        intermediate = Path(tmpdir) / "concat.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", str(intermediate)
        ], capture_output=True, timeout=300)

        if not intermediate.exists():
            print("❌ Error concatenando")
            return False

        print(f"  🎞️  Concatenado: {intermediate.stat().st_size/1024/1024:.1f}MB")

        # Combinar con audio + logo + subtítulos karaoke
        sub_filter = ""
        if SUBS_PATH.exists():
            sub_esc = str(SUBS_PATH).replace("'", "'\\''").replace(":", "\\:")
            sub_filter = f",ass='{sub_esc}'"

        if LOGO_PATH.exists():
            fc = (f"[2:v]scale=120:-1,format=rgba,colorchannelmixer=aa=0.85[logo];"
                  f"[0:v][logo]overlay=W-w-25:25:format=auto{sub_filter}")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(intermediate), "-i", str(AUDIO_PATH), "-i", str(LOGO_PATH),
                "-filter_complex", fc,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                "-shortest", str(OUTPUT_VIDEO)
            ]
        else:
            vf = f"null{sub_filter}" if sub_filter else None
            cmd = [
                "ffmpeg", "-y",
                "-i", str(intermediate), "-i", str(AUDIO_PATH),
            ]
            if vf:
                cmd += ["-vf", sub_filter.lstrip(",")]
            cmd += [
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                "-shortest", str(OUTPUT_VIDEO)
            ]

        subprocess.run(cmd, capture_output=True, timeout=300)

    if OUTPUT_VIDEO.exists():
        size_mb = OUTPUT_VIDEO.stat().st_size / 1024 / 1024
        dur = get_audio_duration(OUTPUT_VIDEO)
        print(f"\n✅ VIDEO LISTO: {OUTPUT_VIDEO.name}")
        print(f"   Tamaño: {size_mb:.1f}MB | Duración: {dur:.1f}s")
        print(f"   Clips usados: {len(downloaded)} únicos, cambio cada {CLIP_DURATION}s")
        return True
    else:
        print("❌ Error generando video final")
        return False

if __name__ == "__main__":
    success = main()
    print("\n✅ COMPLETADO" if success else "\n❌ FALLIDO")
