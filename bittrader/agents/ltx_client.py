"""
ltx_client.py — Cliente para generar clips de video via LTX-Video en Mac Pool.

Pool de Macs disponibles:
- MacBook Air M4 (24GB): primary, más RAM
- MacBook Pro M3 Pro (18GB): secondary

El cliente selecciona la Mac con menos trabajos activos (load balancing).
"""

import json
import subprocess
import tempfile
import time
import shutil
import random
from pathlib import Path
from typing import Optional

# ── Mac Pool Config ────────────────────────────────────────────────────────────
MAC_POOL = [
    {
        "name": "MacBook Air M4",
        "host": "enderj@100.118.215.73",
        "url": "http://100.118.215.73:8188",
        "local_url": "http://enders-macbook-air.local:8188",
        "priority": 1,  # lower = preferred
    },
    {
        "name": "MacBook Pro M3 Pro",
        "host": "enderj@enders-macbook-pro.local",
        "url": "http://enders-macbook-pro.local:8188",  # local network
        "local_url": "http://enders-macbook-pro.local:8188",
        "priority": 2,
    },
]

LTX_MODEL = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
T5_ENCODER = "t5xxl_fp8_e4m3fn.safetensors"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 97
DEFAULT_STEPS = 15
DEFAULT_CFG = 3.0
DEFAULT_FPS = 24


def _ping_mac(mac: dict) -> float:
    """Check Mac availability and response time. Returns latency in ms, or 9999 if unreachable."""
    try:
        start = time.time()
        # Try local URL first, then Tailscale
        for url in [mac["local_url"], mac["url"]]:
            try:
                import urllib.request
                req = urllib.request.Request(f"{url}/system_stats")
                urllib.request.urlopen(req, timeout=3)
                return (time.time() - start) * 1000
            except Exception:
                continue
        return 9999
    except Exception:
        return 9999


def _check_comfyui(mac: dict) -> bool:
    """Check if ComfyUI is running on Mac."""
    try:
        import urllib.request
        for url in [mac["local_url"], mac["url"]]:
            try:
                req = urllib.request.Request(f"{url}/system_stats")
                resp = urllib.request.urlopen(req, timeout=5)
                return resp.status == 200
            except Exception:
                continue
        return False
    except Exception:
        return False


def _select_mac() -> Optional[dict]:
    """Select best available Mac (lowest latency, ComfyUI running)."""
    candidates = []
    for mac in sorted(MAC_POOL, key=lambda m: m["priority"]):
        if _check_comfyui(mac):
            latency = _ping_mac(mac)
            candidates.append((latency, mac))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x[0])
    selected = candidates[0][1]
    print(f"      🎯 Selected: {selected['name']} ({candidates[0][0]:.0f}ms)")
    return selected


def _api_url(mac: dict) -> str:
    """Get working API URL for Mac."""
    for url in [mac["local_url"], mac["url"]]:
        try:
            import urllib.request
            req = urllib.request.Request(f"{url}/system_stats")
            urllib.request.urlopen(req, timeout=3)
            return url
        except Exception:
            continue
    return mac["url"]


def _build_workflow(prompt: str, negative: str = "blurry, low quality, distorted, watermark",
                    width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                    num_frames: int = DEFAULT_FRAMES, steps: int = DEFAULT_STEPS,
                    cfg: float = DEFAULT_CFG, seed: int = 42,
                    prefix: str = "ltx_clip") -> dict:
    """Build ComfyUI API workflow for LTX-Video generation."""
    return {
        "prompt": {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": LTX_MODEL}},
            "2": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": T5_ENCODER, "type": "ltxv"}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["2", 0]}},
            "4": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": negative, "clip": ["2", 0]}},
            "5": {"class_type": "CFGGuider",
                  "inputs": {"model": ["1", 0], "positive": ["3", 0],
                             "negative": ["4", 0], "cfg": cfg}},
            "6": {"class_type": "KSamplerSelect",
                  "inputs": {"sampler_name": "euler"}},
            "7": {"class_type": "BasicScheduler",
                  "inputs": {"model": ["1", 0], "scheduler": "simple",
                             "steps": steps, "denoise": 1.0}},
            "11": {"class_type": "RandomNoise",
                   "inputs": {"noise_seed": seed}},
            "8": {"class_type": "LTXVBaseSampler",
                  "inputs": {"model": ["1", 0], "vae": ["1", 2],
                             "width": width, "height": height,
                             "num_frames": num_frames,
                             "guider": ["5", 0], "sampler": ["6", 0],
                             "sigmas": ["7", 0], "noise": ["11", 0],
                             "seed": seed}},
            "9": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
            "10": {"class_type": "SaveImage",
                   "inputs": {"images": ["9", 0], "filename_prefix": prefix}}
        }
    }


def generate_clip(
    prompt: str,
    output_path: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_frames: int = DEFAULT_FRAMES,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    seed: int = None,
    fps: int = DEFAULT_FPS,
    negative: str = "blurry, low quality, distorted, watermark",
    timeout_seconds: int = 900,  # 15min — 1280x720 LTX clips can take 8-12min on Pro
) -> bool:
    """
    Generate a video clip using LTX-Video on the best available Mac.
    Returns True if successful, False otherwise.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    output_path = Path(output_path)
    prefix = f"ltx_{seed}"

    print(f"      🎥 LTX-Video: {prompt[:60]}...")
    print(f"         {width}x{height}, {num_frames} frames, {steps} steps")

    # 1. Select best Mac
    mac = _select_mac()
    if not mac:
        print("      ❌ No Mac available")
        return False

    api_url = _api_url(mac)

    # 2. Clean old output on Mac
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", mac["host"],
         f"rm -f ~/ComfyUI/output/{prefix}_*.png"],
        capture_output=True, timeout=15
    )

    # 3. Send workflow
    workflow = _build_workflow(prompt, negative, width, height, num_frames, steps, cfg, seed, prefix)
    import urllib.request
    data = json.dumps(workflow).encode()
    req = urllib.request.Request(
        f"{api_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        result = json.loads(resp.read().decode())
        if "error" in result:
            print(f"      ❌ Workflow error: {result['error']}")
            return False
        prompt_id = result.get("prompt_id", "")
        print(f"      ✅ Queued on {mac['name']}: {prompt_id[:8]}")
    except Exception as e:
        print(f"      ❌ API error: {e}")
        return False

    # 4. Wait for frames
    start = time.time()
    expected_frames = num_frames
    while time.time() - start < timeout_seconds:
        time.sleep(15)
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             mac["host"],
             f"ls ~/ComfyUI/output/{prefix}_*.png 2>/dev/null | wc -l"],
            capture_output=True, text=True, timeout=20
        )
        try:
            count = int(r.stdout.strip())
        except ValueError:
            count = 0

        elapsed = time.time() - start
        if count >= expected_frames:
            print(f"      ✅ {count} frames in {elapsed:.0f}s ({mac['name']})")
            break
        elif count > 0:
            print(f"         {count}/{expected_frames} frames ({elapsed:.0f}s)...")
    else:
        print(f"      ❌ Timeout after {timeout_seconds}s")
        return False

    # 5. Transfer frames
    tmp_dir = Path(tempfile.mkdtemp(prefix="ltx_frames_"))
    r = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no",
         f"{mac['host']}:~/ComfyUI/output/{prefix}_*.png", str(tmp_dir) + "/"],
        capture_output=True, timeout=120
    )
    local_frames = sorted(tmp_dir.glob(f"{prefix}_*.png"))
    if not local_frames:
        print("      ❌ No frames transferred")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    print(f"      📁 {len(local_frames)} frames transferred")

    # 6. Rename to sequential for ffmpeg
    for i, f in enumerate(local_frames):
        f.rename(tmp_dir / f"frame_{i:05d}.png")

    # 7. Assemble MP4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(tmp_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-profile:v", "high", "-level", "4.1",
         "-an", str(output_path)],
        capture_output=True, timeout=60
    )

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", mac["host"],
         f"rm -f ~/ComfyUI/output/{prefix}_*.png"],
        capture_output=True, timeout=10
    )

    if r.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"      ✅ LTX clip: {output_path.name} ({size_mb:.1f}MB)")
        return True
    else:
        print(f"      ❌ FFmpeg failed")
        return False


def is_available() -> bool:
    """Quick check if any Mac with LTX-Video is available."""
    return _select_mac() is not None


def list_macs() -> list:
    """List all Macs and their status."""
    results = []
    for mac in MAC_POOL:
        comfy = _check_comfyui(mac)
        latency = _ping_mac(mac) if comfy else 9999
        results.append({
            "name": mac["name"],
            "available": comfy,
            "latency_ms": round(latency, 0) if latency < 9999 else None
        })
    return results
