#!/usr/bin/env python3
"""
Ensamblar los 4 videos del miércoles con clips de MiniMax
"""
import subprocess, json, os
from pathlib import Path

BASE = Path("/home/enderj/.openclaw/workspace/bittrader/miercoles_mar4")
CLIPS_DIR = BASE / "clips_minimax"

# Configuración de cada video
VIDEOS = [
    {
        "name": "short1_stoploss",
        "clips": [CLIPS_DIR / "s1_panic.mp4", CLIPS_DIR / "s1_phone.mp4"],
        "audio": BASE / "short1_stoploss" / "narration.mp3",
        "subs": BASE / "short1_stoploss" / "subs.ass",
        "thumb": BASE / "short1_stoploss" / "thumbnail_final_916.jpg",
        "aspect": "9:16",
        "resolution": "1080x1920"
    },
    {
        "name": "short2_btcvsoro",
        "clips": [CLIPS_DIR / "s2_gold.mp4", CLIPS_DIR / "s2_bitcoin.mp4"],
        "audio": BASE / "short2_btcvsoro" / "narration.mp3",
        "subs": BASE / "short2_btcvsoro" / "subs.ass",
        "thumb": BASE / "short2_btcvsoro" / "thumbnail_final_916.jpg",
        "aspect": "9:16",
        "resolution": "1080x1920"
    },
    {
        "name": "short3_regla1",
        "clips": [CLIPS_DIR / "s3_pro.mp4", CLIPS_DIR / "s3_screen.mp4"],
        "audio": BASE / "short3_regla1" / "narration.mp3",
        "subs": BASE / "short3_regla1" / "subs.ass",
        "thumb": BASE / "short3_regla1" / "thumbnail_final_916.jpg",
        "aspect": "9:16",
        "resolution": "1080x1920"
    },
    {
        "name": "video_principal",
        "clips": [CLIPS_DIR / "v1_growth.mp4", CLIPS_DIR / "v2_calendar.mp4",
                  CLIPS_DIR / "v3_chart.mp4", CLIPS_DIR / "v4_peaceful.mp4"],
        "audio": BASE / "video_principal" / "narration.mp3",
        "subs": BASE / "video_principal" / "subs.ass",
        "thumb": BASE / "video_principal" / "thumbnail_final.jpg",
        "aspect": "16:9",
        "resolution": "1920x1080"
    },
]

def get_duration(file):
    """Obtener duración de un archivo de audio/video"""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(file)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return float(json.loads(r.stdout)["format"]["duration"])

def assemble_video(video):
    """Ensamblar un video completo"""
    print(f"\n🎬 Ensamblando: {video['name']}")

    # 1. Crear lista de clips
    clips_list = BASE / f"{video['name']}_clips.txt"
    with open(clips_list, "w") as f:
        for clip in video["clips"]:
            f.write(f"file '{clip}'\n")

    # 2. Obtener duración del audio
    audio_duration = get_duration(video["audio"])
    print(f"   Audio: {audio_duration:.1f}s")

    # 3. Concatenar clips y escalar
    output_concat = BASE / f"{video['name']}_concat.mp4"
    width, height = video["resolution"].split("x")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(clips_list),
        "-t", str(audio_duration),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        str(output_concat)
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"   ✅ Clips concatenados")

    # 4. Añadir audio
    output_audio = BASE / f"{video['name']}_audio.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(output_concat),
        "-i", str(video["audio"]),
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(output_audio)
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"   ✅ Audio añadido")

    # 5. Quemar subtítulos
    output_final = BASE / f"{video['name']}_final.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(output_audio),
        "-vf", f"ass={video['subs']}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_final)
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"   ✅ Subtítulos quemados")

    # Limpiar temporales
    output_concat.unlink(missing_ok=True)
    output_audio.unlink(missing_ok=True)
    clips_list.unlink(missing_ok=True)

    # Verificar resultado
    size_mb = output_final.stat().st_size / 1024 / 1024
    print(f"   📹 Final: {output_final.name} ({size_mb:.1f} MB)")

    return output_final

# Ensamblar todos los videos
print("="*50)
print("ENSAMBlando 4 VIDEOS")
print("="*50)

results = []
for video in VIDEOS:
    try:
        final_path = assemble_video(video)
        results.append({"name": video["name"], "path": str(final_path), "status": "success"})
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({"name": video["name"], "status": "error", "error": str(e)})

# Resumen
print("\n" + "="*50)
print("RESUMEN")
print("="*50)

successful = [r for r in results if r["status"] == "success"]
print(f"✅ Videos ensamblados: {len(successful)}/4")

for r in successful:
    print(f"   - {r['name']}: {r['path']}")

# Guardar log
with open(BASE / "assembly_log.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n📋 Log: {BASE}/assembly_log.json")
