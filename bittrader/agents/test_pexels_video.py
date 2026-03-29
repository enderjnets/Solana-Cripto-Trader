"""
TEST: Reconstruir "3 reglas que salvaron mi cuenta de trading" usando clips de Pexels
Compara calidad vs versión original con Hailuo.
"""
import json, os, requests, subprocess, tempfile, shutil
from pathlib import Path

PEXELS_KEY = json.loads(Path("keys/pexels.json").read_text())["api_key"]
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
AGENTS = Path(__file__).parent
OUTPUT_DIR = AGENTS / "output/pexels_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_PATH = AGENTS / "output/2026-03-28/long_1774708531_991/long_1774708531_991.mp3"
LOGO_PATH = WORKSPACE / "bittrader/assets/bittrader_logo_transparent.png"
if not LOGO_PATH.exists():
    LOGO_PATH = WORKSPACE / "bittrader/assets/bittrader_logo.png"

OUTPUT_VIDEO = OUTPUT_DIR / "3reglas_pexels_test.mp4"

# Segmentos del guion con keywords para búsqueda en Pexels
SEGMENTS = [
    {"text": "Introducción: reglas de trading que salvan cuentas", "keywords": ["trading rules", "stock market", "financial charts"]},
    {"text": "Regla 1: gestión de riesgo, tamaño de posición", "keywords": ["risk management", "trading position", "portfolio risk"]},
    {"text": "Regla 2: stop loss efectivo", "keywords": ["stop loss", "trading loss", "market crash"]},
    {"text": "Regla 3: diversificación como profesionales", "keywords": ["diversification", "investment portfolio", "cryptocurrency bitcoin"]},
    {"text": "Resultado: proteger capital y crecer consistentemente", "keywords": ["financial success", "trading profit", "bitcoin growth"]},
    {"text": "Llamada a la acción: aplicar las reglas ahora", "keywords": ["trading strategy", "financial freedom", "crypto trading"]},
]

def search_pexels_video(keywords: list, per_page: int = 5) -> list:
    """Buscar videos en Pexels por keywords. Retorna lista de URLs."""
    for keyword in keywords:
        try:
            r = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": PEXELS_KEY},
                params={"query": keyword, "per_page": per_page, "orientation": "landscape", "size": "medium"},
                timeout=15
            )
            if r.status_code == 200:
                data = r.json()
                videos = data.get("videos", [])
                if videos:
                    # Preferir HD (1280x720 o superior)
                    results = []
                    for v in videos:
                        for vf in v.get("video_files", []):
                            if vf.get("width", 0) >= 1280 and vf.get("file_type") == "video/mp4":
                                results.append({
                                    "url": vf["link"],
                                    "width": vf["width"],
                                    "height": vf["height"],
                                    "duration": v.get("duration", 0),
                                    "keyword": keyword
                                })
                                break
                    if results:
                        print(f"  ✅ '{keyword}' → {len(results)} clips encontrados")
                        return results
        except Exception as e:
            print(f"  ⚠️ Error buscando '{keyword}': {e}")
    return []

def download_clip(url: str, out_path: Path) -> bool:
    """Descargar clip de Pexels."""
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = out_path.stat().st_size / (1024*1024)
            print(f"    💾 Descargado: {out_path.name} ({size_mb:.1f}MB)")
            return True
    except Exception as e:
        print(f"    ❌ Error descargando: {e}")
    return False

def get_audio_duration(audio_path: Path) -> float:
    """Obtener duración del audio en segundos."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip() or "0")

def main():
    print("=" * 60)
    print("TEST: 3 reglas — Pexels vs Hailuo")
    print("=" * 60)

    # 1. Obtener duración del audio
    audio_dur = get_audio_duration(AUDIO_PATH)
    print(f"\n📢 Audio: {audio_dur:.1f}s ({audio_dur/60:.1f} min)")

    # 2. Buscar y descargar clips de Pexels
    clips_dir = OUTPUT_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    downloaded_clips = []
    print(f"\n🔍 Buscando {len(SEGMENTS)} clips en Pexels...")
    
    for i, seg in enumerate(SEGMENTS):
        print(f"\nSegmento {i+1}: {seg['text'][:50]}")
        results = search_pexels_video(seg["keywords"])
        
        if not results:
            print(f"  ⚠️ No encontrado, saltando segmento {i+1}")
            continue
        
        # Tomar el primer resultado
        clip_info = results[0]
        clip_path = clips_dir / f"clip_{i+1:02d}.mp4"
        
        print(f"  📥 Descargando: {clip_info['keyword']} ({clip_info['width']}x{clip_info['height']}, {clip_info['duration']}s)")
        if download_clip(clip_info["url"], clip_path):
            downloaded_clips.append(clip_path)

    if len(downloaded_clips) < 3:
        print(f"\n❌ Solo {len(downloaded_clips)} clips descargados — insuficiente")
        return False

    print(f"\n✅ {len(downloaded_clips)} clips descargados")

    # 3. Calcular duración por clip
    dur_per_clip = audio_dur / len(downloaded_clips)
    print(f"⏱️  Duración por clip: {dur_per_clip:.1f}s")

    # 4. Crear lista de clips con duración correcta (loop si son cortos)
    with tempfile.TemporaryDirectory() as tmpdir:
        trimmed_clips = []
        for i, clip_path in enumerate(downloaded_clips):
            trimmed = Path(tmpdir) / f"trimmed_{i:02d}.mp4"
            # Escalar a 1280x720 y trim/loop a dur_per_clip segundos
            subprocess.run([
                "ffmpeg", "-y",
                "-stream_loop", "-1",   # loop infinito para clips cortos
                "-i", str(clip_path),
                "-t", str(dur_per_clip),
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-an",
                str(trimmed)
            ], capture_output=True, timeout=60)
            if trimmed.exists():
                trimmed_clips.append(trimmed)
                print(f"  ✂️  Clip {i+1} trimmed: {trimmed.stat().st_size//1024}KB")

        # 5. Concatenar clips
        concat_list = Path(tmpdir) / "concat.txt"
        concat_list.write_text("\n".join(f"file '{p}'" for p in trimmed_clips))
        
        intermediate = Path(tmpdir) / "intermediate.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            str(intermediate)
        ], capture_output=True, timeout=120)

        if not intermediate.exists():
            print("❌ Error concatenando clips")
            return False

        # 6. Combinar con audio + logo
        logo_args = []
        filter_complex = None
        if LOGO_PATH.exists():
            filter_complex = "[2:v]scale=180:-1,format=rgba,colorchannelmixer=aa=0.85[logo];[0:v][logo]overlay=W-w-30:30:format=auto"
            logo_args = ["-i", str(LOGO_PATH)]

        # Encoding estándar compatible con YouTube Y Telegram/móvil
        VIDEO_FLAGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
                       "-profile:v", "baseline", "-level", "3.1", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart"]
        AUDIO_FLAGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
        cmd = ["ffmpeg", "-y", "-i", str(intermediate), "-i", str(AUDIO_PATH)]
        if filter_complex:
            cmd += logo_args + ["-filter_complex", filter_complex]
        cmd += [*VIDEO_FLAGS, *AUDIO_FLAGS, "-shortest", str(OUTPUT_VIDEO)]
        
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        
        if OUTPUT_VIDEO.exists():
            size_mb = OUTPUT_VIDEO.stat().st_size / (1024*1024)
            print(f"\n✅ VIDEO PEXELS LISTO: {OUTPUT_VIDEO}")
            print(f"   Tamaño: {size_mb:.1f}MB")
            print(f"\n📊 COMPARACIÓN:")
            orig = AGENTS / "output/2026-03-28/long_1774708531_991/long_1774708531_991.mp4"
            if orig.exists():
                orig_mb = orig.stat().st_size / (1024*1024)
                print(f"   Original (Hailuo): {orig_mb:.1f}MB — {AUDIO_PATH.stem}")
                print(f"   Pexels test:       {size_mb:.1f}MB — {OUTPUT_VIDEO.name}")
            return True
        else:
            print(f"❌ Error generando video final: {result.stderr.decode()[-200:]}")
            return False

if __name__ == "__main__":
    success = main()
    print("\n✅ PRUEBA COMPLETADA" if success else "\n❌ PRUEBA FALLIDA")
