#!/usr/bin/env python3
"""
🎬 Execute Full Rebuild Pipeline
Convierte rebuild tasks a guiones y ejecuta producer + thumbnail + publisher
"""
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
AGENTS = BITTRADER / "agents"
DATA = AGENTS / "data"

def create_guiones_from_rebuild():
    """Convert rebuild tasks to guiones format for producer."""
    
    tasks_file = DATA / "rebuild_videos_tasks.json"
    if not tasks_file.exists():
        print("❌ No rebuild tasks found")
        return None
    
    tasks = json.loads(tasks_file.read_text())
    
    guiones = {"videos": []}
    
    for task in tasks:
        script = task.get("script", {})
        
        # Combine intro + sections + outro + CTA into full script
        full_script = f"{script.get('intro', '')}\n\n"
        for section in script.get('sections', []):
            full_script += f"{section}\n\n"
        full_script += f"{script.get('outro', '')}\n\n{script.get('cta', '')}"
        
        video = {
            "title": task["title"],
            "type": task["type"],
            "script": full_script.strip(),
            "hook": task["hook"],
            "topic": task["topic"],
            "original_id": task["original_video_id"],  # Fixed: was original_id
            "created_at": datetime.now().isoformat()
        }
        
        guiones["videos"].append(video)
    
    return guiones


def run_pipeline():
    """Execute full pipeline: producer → thumbnails → publisher"""
    
    print("=" * 70)
    print("🎬 EJECUTANDO PIPELINE COMPLETO")
    print("=" * 70)
    
    # Step 1: Create guiones
    print("\n[1/3] Convirtiendo tareas a guiones...")
    guiones = create_guiones_from_rebuild()
    
    if not guiones:
        print("❌ Error: No hay tareas para procesar")
        return
    
    guiones_file = DATA / "guiones_latest.json"
    guiones_file.write_text(json.dumps(guiones, indent=2, ensure_ascii=False))
    print(f"   ✓ Guiones creados: {len(guiones['videos'])} videos")
    
    # Step 2: Run producer
    print("\n[2/3] Ejecutando producer.py (esto tomará varios minutos)...")
    print("   Generando videos con Hailuo + Ken Burns...")
    
    try:
        result = subprocess.run(
            ["python3", str(AGENTS / "producer.py")],
            cwd=str(AGENTS),
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        if result.returncode == 0:
            print("   ✅ Producer completado")
        else:
            print(f"   ⚠️ Producer con errores: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        print("   ⏰ Producer timeout - continuando...")
    except Exception as e:
        print(f"   ❌ Error producer: {e}")
    
    # Step 3: Generate thumbnails
    print("\n[3/3] Generando thumbnails con Hugging Face SDXL...")
    
    try:
        result = subprocess.run(
            ["python3", str(AGENTS / "thumbnail_agent_huggingface.py")],
            cwd=str(AGENTS),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("   ✅ Thumbnails generados")
        else:
            print(f"   ⚠️ Thumbnails con errores: {result.stderr[-500:]}")
    except Exception as e:
        print(f"   ❌ Error thumbnails: {e}")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 70)
    print("\n📋 Revisa los videos generados en:")
    print(f"   {AGENTS / 'output'}")
    print("\n🔄 Para subir a YouTube, ejecuta:")
    print("   python3 publisher.py")


if __name__ == "__main__":
    run_pipeline()
