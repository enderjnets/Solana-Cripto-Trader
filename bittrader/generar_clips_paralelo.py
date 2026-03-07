#!/usr/bin/env python3
"""
Generar todos los clips en paralelo y descargar cuando estén listos
"""
import requests, json, time, os
from pathlib import Path

API_KEY = json.load(open("keys/minimax.json"))["minimax_api_key"]
headers = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.minimax.io"

CLIPS = [
    {"name": "s1_panic", "prompt": "Trader at desk at night, dark room, red charts, panic, 9:16"},
    {"name": "s1_phone", "prompt": "Hands holding phone with Bitcoin crashing -47%, 9:16"},
    {"name": "s2_gold", "prompt": "Gold bars glowing, luxury vault, cinematic, 9:16"},
    {"name": "s2_bitcoin", "prompt": "Bitcoin symbol floating in digital space, orange glow, 9:16"},
    {"name": "s3_pro", "prompt": "Professional trader, green charts, calm, elegant office, 9:16"},
    {"name": "s3_screen", "prompt": "Hands typing, trading platform, risk management panel, 9:16"},
    {"name": "v1_growth", "prompt": "Person holding $100, money stacks growing, crypto symbols, 16:9"},
    {"name": "v2_calendar", "prompt": "Coins dropping into piggy bank, calendar flipping, 16:9"},
    {"name": "v3_chart", "prompt": "Bitcoin chart 2020-2025 crash and recovery, 16:9"},
    {"name": "v4_peaceful", "prompt": "Person sleeping, phone showing trading notifications, 16:9"},
]

output_dir = Path("miercoles_mar4/clips_minimax")
output_dir.mkdir(parents=True, exist_ok=True)

# Paso 1: Crear todas las tareas
print("🚀 Creando tareas de generación...")
tasks = []
for clip in CLIPS:
    r = requests.post(
        f"{BASE_URL}/v1/video_generation",
        headers=headers,
        json={
            "prompt": clip["prompt"],
            "model": "MiniMax-Hailuo-2.3",
            "duration": 6,
            "resolution": "1080P"
        },
        timeout=30
    )
    task_id = r.json()["task_id"]
    tasks.append({"name": clip["name"], "task_id": task_id})
    print(f"  ✅ {clip['name']}: {task_id}")

# Guardar task_ids para referencia
with open(output_dir / "tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

# Paso 2: Esperar y descargar
print("\n⏳ Esperando completado (esto tomará 3-5 minutos)...")
time.sleep(120)  # Esperar 2 minutos antes de empezar a consultar

completed = []
failed = []

for i in range(20):  # Intentar hasta 20 veces (200 segundos)
    print(f"\n[Intento {i+1}/20] Consultando estado...")

    all_done = True
    for task in tasks:
        if task["name"] in [c["name"] for c in completed] + [f["name"] for f in failed]:
            continue

        r = requests.get(
            f"{BASE_URL}/v1/query/video_generation",
            headers=headers,
            params={"task_id": task["task_id"]}
        )
        status = r.json().get("status")

        if status == "Success":
            file_id = r.json()["file_id"]
            # Descargar
            fr = requests.get(
                f"{BASE_URL}/v1/files/retrieve",
                headers=headers,
                params={"file_id": file_id}
            )
            download_url = fr.json()["file"]["download_url"]
            video_data = requests.get(download_url).content

            output_path = output_dir / f"{task['name']}.mp4"
            with open(output_path, "wb") as f:
                f.write(video_data)

            completed.append({"name": task["name"], "path": str(output_path)})
            print(f"  ✅ {task['name']} completado y descargado")

        elif status == "Fail":
            failed.append({"name": task["name"], "error": r.json().get("error_message")})
            print(f"  ❌ {task['name']} falló")

        else:
            all_done = False

    if all_done and len(completed) + len(failed) == len(tasks):
        break

    time.sleep(10)

# Resumen
print("\n" + "="*50)
print("RESUMEN")
print("="*50)
print(f"✅ Completados: {len(completed)}/{len(tasks)}")
print(f"❌ Fallidos: {len(failed)}/{len(tasks)}")

if failed:
    print("\nFallidos:")
    for f in failed:
        print(f"  - {f['name']}: {f['error']}")

# Guardar resultados
with open(output_dir / "results.json", "w") as f:
    json.dump({"completed": completed, "failed": failed}, f, indent=2)

print(f"\n✅ Clips guardados en: {output_dir}")
