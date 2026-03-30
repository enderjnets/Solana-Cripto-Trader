#!/usr/bin/env python3
"""
🎬 MiniMax Hailuo 2.3 — Text-to-Video Generator

Genera videos AI con MiniMax Hailuo 2.3.
Usa polling para verificar el estado (no requiere callback URL).

Uso:
  python3 hailuo_video.py "A man picks up a book [Pan left], then reads [Static shot]" --duration 6 --resolution 768P
  python3 hailuo_video.py "criptocurrency chart going up, green candles" --model MiniMax-Hailuo-2.3
"""

import argparse
import requests
import time
import sys
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "sk-cp-8tBIgoE2Vs8QE0AIoMjq4MTh8kiHtem3KWlOnNlAJZgKwAlYh_nt6oCq382Y0cmBi2buvch3nJJbMg7uqr_hIV6Z0ZqY3Q_qZ6AStHCUpKKT_IT-e0vEl4A"
BASE_URL = "https://api.minimax.io"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ── Generate Video ─────────────────────────────────────────────────────────────
def generate_video(prompt: str, model: str = "MiniMax-Hailuo-2.3",
                  duration: int = 6, resolution: str = "768P",
                  prompt_optimizer: bool = True) -> str:
    """Crea una tarea de video y devuelve el task_id."""
    url = f"{BASE_URL}/v1/video_generation"
    payload = {
        "model": model,
        "prompt": prompt,
        "prompt_optimizer": prompt_optimizer,
        "duration": duration,
        "resolution": resolution,
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    data = resp.json()
    if data.get("base_resp", {}).get("status_code") != 0:
        raise Exception(f"Error: {data.get('base_resp', {}).get('status_msg', data)}")
    return data["task_id"]


def query_video_status(task_id: str) -> dict:
    """Consulta el estado de una tarea de video."""
    url = f"{BASE_URL}/v1/video_generation"
    payload = {"task_id": task_id}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    return resp.json()


def poll_video(task_id: str, max_wait: int = 300, poll_interval: int = 10) -> dict:
    """Hace polling hasta que el video esté listo o falle."""
    print(f"Polling task {task_id} (max {max_wait}s)...")
    start = time.time()
    while time.time() - start < max_wait:
        result = query_video_status(task_id)
        status = result.get("status", "unknown")
        print(f"  [{int(time.time()-start)}s] Status: {status}")
        if status == "success":
            return result
        elif status == "failed":
            raise Exception(f"Video generation failed: {result.get('base_resp', {})}")
        time.sleep(poll_interval)
    raise Exception("Timeout waiting for video")


def download_video(file_id: str, output_path: Path) -> Path:
    """Descarga el video usando el file_id."""
    # Obtener URL de descarga
    url = f"{BASE_URL}/v1/files/{file_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    data = resp.json()
    download_url = data.get("data", {}).get("preview_url") or data.get("data", {}).get("download_url")
    
    if not download_url:
        raise Exception(f"No download URL found: {data}")
    
    # Descargar
    video_resp = requests.get(download_url, timeout=300, stream=True)
    video_resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in video_resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MiniMax Hailuo 2.3 Video Generator")
    parser.add_argument("prompt", help="Text description of the video")
    parser.add_argument("--model", default="MiniMax-Hailuo-2.3",
                        choices=["MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02", "T2V-01-Director", "T2V-01"])
    parser.add_argument("--duration", type=int, default=6, choices=[6, 10])
    parser.add_argument("--resolution", default="768P", choices=["720P", "768P", "1080P"])
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimizer")
    parser.add_argument("--output", "-o", default="output.mp4")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Generating video with {args.model}")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.duration}s | Resolution: {args.resolution}")
    
    # 1. Create task
    task_id = generate_video(
        prompt=args.prompt,
        model=args.model,
        duration=args.duration,
        resolution=args.resolution,
        prompt_optimizer=not args.no_optimize,
    )
    print(f"✅ Task created: {task_id}")

    # 2. Poll for completion
    result = poll_video(task_id)
    print(f"✅ Video ready!")

    # 3. Download
    file_id = result.get("file_id")
    if file_id:
        print(f"📥 Downloading...")
        path = download_video(file_id, output_path)
        print(f"✅ Saved to: {path}")
    else:
        print(f"⚠️ No file_id in response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
