#!/usr/bin/env python3
"""
MiniMax Video Generation Script
Genera clips de video usando la API de MiniMax Hailuo 2.3
"""
import os
import time
import json
import requests
from pathlib import Path

# Configuración
API_KEY = json.load(open("keys/minimax.json"))["minimax_api_key"]
BASE_URL = "https://api.minimax.io/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Clips a generar para los 4 videos del miércoles
CLIPS = [
    # SHORT 1 - Stop Loss (2 clips 9:16)
    {
        "name": "short1_clip1_panic",
        "prompt": "Trader sitting at desk at night, dark room, multiple monitors showing red candlestick charts, numbers falling rapidly in red, trader's face lit by screen glow showing shock and panic, close-up of hands gripping desk tightly, cinematic dramatic lighting, high tension atmosphere",
        "aspect_ratio": "9:16",
        "duration": 6
    },
    {
        "name": "short1_clip2_phone",
        "prompt": "Close-up of hands holding smartphone showing Bitcoin price chart crashing with -47% in red, dramatic lighting, dark background, cinematic slow motion, financial crisis mood, fingers gripping phone tightly",
        "aspect_ratio": "9:16",
        "duration": 6
    },
    # SHORT 2 - Bitcoin vs Oro (2 clips 9:16)
    {
        "name": "short2_clip1_gold",
        "prompt": "Gold bars and coins glowing with warm golden light, slow dramatic camera pan over stacks of gold in luxury vault, cinematic lighting, wealth and stability atmosphere, metallic reflections",
        "aspect_ratio": "9:16",
        "duration": 6
    },
    {
        "name": "short2_clip2_bitcoin",
        "prompt": "Glowing Bitcoin symbol rotating slowly in digital space, orange light particles floating around, futuristic crypto atmosphere, dark background with subtle grid lines, cinematic 4K quality, blockchain aesthetic",
        "aspect_ratio": "9:16",
        "duration": 6
    },
    # SHORT 3 - Regla del 1% (2 clips 9:16)
    {
        "name": "short3_clip1_pro",
        "prompt": "Confident professional trader at modern desk, green profitable candlestick charts on multiple monitors, calm focused expression, dark elegant office, dramatic studio lighting, successful trading atmosphere, business casual attire",
        "aspect_ratio": "9:16",
        "duration": 6
    },
    {
        "name": "short3_clip2_screen",
        "prompt": "Close-up of hands typing on keyboard, trading platform on screen showing risk management panel, green percentage gauges, professional trading setup, dark room with monitor glow, financial technology aesthetic",
        "aspect_ratio": "9:16",
        "duration": 6
    },
    # VIDEO PRINCIPAL - $100/mes (4 clips 16:9)
    {
        "name": "video_clip1_growth",
        "prompt": "Person holding a small $100 bill with one hand, other hand showing large stacks of money growing upward like a chart, Bitcoin and crypto symbols floating around, dark dramatic background, golden light, cinematic wealth transformation",
        "aspect_ratio": "16:9",
        "duration": 6
    },
    {
        "name": "video_clip2_calendar",
        "prompt": "Person dropping coins into digital piggy bank every month, calendar pages flipping rapidly, crypto symbols appearing, time-lapse effect, warm lighting, financial growth journey, monthly investment concept",
        "aspect_ratio": "16:9",
        "duration": 6
    },
    {
        "name": "video_clip3_chart",
        "prompt": "Bitcoin price chart from 2020 to 2025 showing dramatic crash and recovery, red candles falling then green candles rising, financial crisis to bull market transformation, dark background with glowing chart lines, dramatic visualization",
        "aspect_ratio": "16:9",
        "duration": 6
    },
    {
        "name": "video_clip4_peaceful",
        "prompt": "Person sleeping peacefully at night while phone on nightstand shows automated trading notifications with green checkmarks, calm bedroom atmosphere, soft blue light, passive income concept, financial freedom",
        "aspect_ratio": "16:9",
        "duration": 6
    },
]

def create_video_task(prompt, aspect_ratio="9:16", duration=6):
    """Crear tarea de generación de video"""
    url = f"{BASE_URL}/video_generation"
    payload = {
        "prompt": prompt,
        "model": "MiniMax-Hailuo-2.3",
        "duration": duration,
        "resolution": "1080P"
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["task_id"]

def query_task_status(task_id):
    """Consultar estado de la tarea hasta que termine"""
    url = f"{BASE_URL}/query/video_generation"
    params = {"task_id": task_id}

    max_attempts = 60  # 10 minutos máximo
    attempt = 0

    while attempt < max_attempts:
        time.sleep(10)
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")

        print(f"     Estado: {status}")

        if status == "Success":
            return data["file_id"]
        elif status == "Fail":
            raise Exception(f"Error: {data.get('error_message', 'Unknown error')}")

        attempt += 1

    raise Exception("Timeout: La tarea no completó en 10 minutos")

def download_video(file_id, output_path):
    """Descargar video desde file_id"""
    url = f"{BASE_URL}/files/retrieve"
    params = {"file_id": file_id}

    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    download_url = response.json()["file"]["download_url"]

    video_response = requests.get(download_url)
    video_response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(video_response.content)

    return output_path

def main():
    output_dir = Path("miercoles_mar4/clips_minimax")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Generando {len(CLIPS)} clips con MiniMax Hailuo 2.3\n")

    results = []

    for i, clip in enumerate(CLIPS, 1):
        print(f"[{i}/{len(CLIPS)}] Generando: {clip['name']}")
        print(f"   Prompt: {clip['prompt'][:80]}...")

        try:
            # Crear tarea
            task_id = create_video_task(clip["prompt"], clip["aspect_ratio"], clip["duration"])
            print(f"   Task ID: {task_id}")

            # Esperar completado
            file_id = query_task_status(task_id)
            print(f"   File ID: {file_id}")

            # Descargar
            output_path = output_dir / f"{clip['name']}.mp4"
            download_video(file_id, output_path)
            print(f"   ✅ Guardado: {output_path}")

            results.append({"name": clip["name"], "status": "success", "path": str(output_path)})

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({"name": clip["name"], "status": "error", "error": str(e)})

        print()

    # Resumen final
    print("\n" + "="*50)
    print("RESUMEN DE GENERACIÓN")
    print("="*50)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"✅ Exitosos: {len(successful)}/{len(CLIPS)}")
    print(f"❌ Fallidos: {len(failed)}/{len(CLIPS)}")

    if failed:
        print("\nClips fallidos:")
        for f in failed:
            print(f"  - {f['name']}: {f['error']}")

    # Guardar log
    with open(output_dir / "generation_log.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📋 Log guardado en: {output_dir}/generation_log.json")

if __name__ == "__main__":
    os.chdir("/home/enderj/.openclaw/workspace/bittrader")
    main()
