#!/usr/bin/env python3
"""
🖼️ MiniMax image-01 — Image Generator

Genera imágenes AI con MiniMax image-01.

Uso:
  python3 minimax_image.py "a beautiful sunset over the ocean" --output sunset.png
  python3 minimax_image.py "crypto trading setup, green neon lights" --aspect-ratio 16:9 --resolution 1024x1024
"""

import argparse
import requests
import sys
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "sk-cp-8tBIgoE2Vs8QE0AIoMjq4MTh8kiHtem3KWlOnNlAJZgKwAlYh_nt6oCq382Y0cmBi2buvch3nJJbMg7uqr_hIV6Z0ZqY3Q_qZ6AStHCUpKKT_IT-e0vEl4A"
BASE_URL = "https://api.minimax.io"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ── Generate Image ─────────────────────────────────────────────────────────────
def generate_image(prompt: str, aspect_ratio: str = "1:1",
                   resolution: str = "1024x1024",
                   prompt_optimizer: bool = True) -> list[str]:
    """Genera imagen y devuelve lista de URLs."""
    url = f"{BASE_URL}/v1/image_generation"
    payload = {
        "model": "image-01",
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "prompt_optimizer": prompt_optimizer,
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
    data = resp.json()
    base_resp = data.get("base_resp", {})
    if base_resp.get("status_code") != 0:
        raise Exception(f"Error: {base_resp.get('status_msg', data)}")
    image_urls = data.get("data", {}).get("image_urls", [])
    return image_urls


def download_image(image_url: str, output_path: Path) -> Path:
    """Descarga la imagen desde la URL."""
    img_resp = requests.get(image_url, timeout=120)
    img_resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(img_resp.content)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MiniMax image-01 Generator")
    parser.add_argument("prompt", help="Text description of the image")
    parser.add_argument("--output", "-o", default="generated_image.png",
                        help="Output file path")
    parser.add_argument("--aspect-ratio", default="1:1",
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                        help="Aspect ratio")
    parser.add_argument("--resolution", default="1024x1024",
                        choices=["512x512", "768x768", "1024x1024", "1024x768", "768x1024"],
                        help="Resolution")
    parser.add_argument("--no-optimize", action="store_true", help="Disable prompt optimizer")
    parser.add_argument("--raw", action="store_true", help="Show raw JSON response")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🖼️ Generating image with image-01")
    print(f"   Prompt: {args.prompt}")
    print(f"   Aspect ratio: {args.aspect_ratio} | Resolution: {args.resolution}")

    image_urls = generate_image(
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        resolution=args.resolution,
        prompt_optimizer=not args.no_optimize,
    )

    if not image_urls:
        print(f"⚠️ No images generated")
        return

    print(f"✅ Generated {len(image_urls)} image(s)")

    for i, img_url in enumerate(image_urls):
        # Determinar extensión del archivo desde la URL
        parsed = urlparse(img_url)
        url_path = unquote(parsed.path)
        if ".png" in url_path.lower():
            ext = ".png"
        elif ".webp" in url_path.lower():
            ext = ".webp"
        else:
            ext = ".png"

        out = output_path if len(image_urls) == 1 else output_path.parent / f"{output_path.stem}_{i}{ext}"
        print(f"📥 Downloading {out}...")
        path = download_image(img_url, out)
        print(f"   ✅ Saved: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
