#!/usr/bin/env python3
"""
🔄 Recreate 3 BitTrader Videos - Enhanced Version
Crea versiones mejoradas de los 3 videos usando el nuevo sistema:
- Scripts estilo MrBeast
- Imágenes con Hugging Face SDXL
- Thumbnails profesionales
"""
import json
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
AGENTS = BITTRADER / "agents"
DATA = AGENTS / "data"
OUTPUT = AGENTS / "output"

# Videos to recreate
VIDEOS = [
    {
        "original_id": "Bb2H6nChYwQ",
        "title": "De $0 a cuenta fondeada en 30 días",
        "type": "long",
        "hook": "DE $0 A CUENTA FONDEADA",
        "topic": "Plan paso a paso para conseguir cuenta fondeada con trading algorítmico"
    },
    {
        "original_id": "2RzldSPc2Ck",
        "title": "El bot que lee noticias y ejecuta tus trades solo",
        "type": "long",
        "hook": "BOT que TRADEA SOLO",
        "topic": "Cómo crear un bot de trading que analiza noticias y ejecuta operaciones automáticamente"
    },
    {
        "original_id": "2VEzHeGgDG8",
        "title": "Le di mis trades a Claude",
        "type": "long",
        "hook": "CLAUDE ANALIZÓ MIS TRADES",
        "topic": "Cómo usar Claude AI para analizar operaciones de trading y mejorar resultados"
    }
]

def generate_mrbeast_script(video: dict) -> dict:
    """Generate MrBeast-style script for video."""
    
    prompts = {
        "Bb2H6nChYwQ": {
            "intro": "¿Quieres una cuenta de $100,000 para hacer trading PERO no tienes capital? Hoy te muestro el plan EXACTO que usé para conseguir 6 cuentas fondeadas en 6 meses.",
            "sections": [
                "PASO 1: Elige la prop firm correcta (TopStep vs FTMO vs Funding Pips)",
                "PASO 2: La estrategia de riesgo ultra conservador (0.5% por trade)",
                "PASO 3: Backtesting obligatorio - mínimo 100 operaciones",
                "PASO 4: Psicología - cómo sobrevivir los días rojos",
                "PASO 5: Escalado - de una cuenta a múltiples"
            ],
            "outro": "En 30 días puedes pasar el challenge. En 90 días puedes tener $300k en cuentas. ¿Vas a empezar hoy o seguir soñando?",
            "cta": "Sígueme para más estrategias de trading algorítmico"
        },
        "2RzldSPc2Ck": {
            "intro": "Imagina un bot que LEE las noticias por ti, analiza el sentimiento del mercado, y ejecuta trades automáticamente. Hoy te enseño a crearlo.",
            "sections": [
                "¿Qué es el news trading y por qué es rentable?",
                "Configurando el news fetcher (APIs gratuitas)",
                "Análisis de sentimiento con IA (Claude + GPT)",
                "Lógica de ejecución: cuándo entrar y cuándo salir",
                "Backtest del bot: +47% en 3 meses"
            ],
            "outro": "Este bot me ahorró 4 horas diarias. Y lo mejor: no necesita que estés frente a la pantalla.",
            "cta": "¿Quieres el código? Déjame un comentario"
        },
        "2VEzHeGgDG8": {
            "intro": "Le di acceso a Claude AI a mis 500 trades del último mes. Lo que descubrió me dejó SIN palabras.",
            "sections": [
                "Cómo exportar tu historial de trading de MT5",
                "Prompt mágico para Claude: análisis completo",
                "Lo que Claude encontró: 3 patrones de error",
                "Ajustes que hice basados en el análisis",
                "Resultados después de 2 semanas: winrate +12%"
            ],
            "outro": "La IA no reemplaza al trader. Pero el trader que usa IA le gana al que no.",
            "cta": "¿Quieres que analice tus trades? Sígueme y pregunta"
        }
    }
    
    return prompts.get(video["original_id"], {})


def create_rebuild_task(video: dict, script: dict) -> dict:
    """Create task for producer agent."""
    
    return {
        "original_video_id": video["original_id"],
        "title": video["title"],
        "type": video["type"],
        "hook": video["hook"],
        "topic": video["topic"],
        "script": script,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "steps": {
            "script_generated": False,
            "images_generated": False,
            "video_produced": False,
            "thumbnail_created": False,
            "uploaded": False
        }
    }


def main():
    print("=" * 70)
    print("🔄 RECONSTRUYENDO 3 VIDEOS BITTRADER")
    print("   Versión mejorada con sistema MrBeast + SDXL")
    print("=" * 70)
    print()
    
    tasks = []
    
    for i, video in enumerate(VIDEOS, 1):
        print(f"\n[{i}/3] Procesando: {video['title']}")
        print(f"   Original: https://youtube.com/watch?v={video['original_id']}")
        
        # Generate MrBeast script
        print("   📝 Generando script estilo MrBeast...")
        script = generate_mrbeast_script(video)
        
        # Create task
        task = create_rebuild_task(video, script)
        tasks.append(task)
        
        print(f"   ✓ Script generado:")
        print(f"     • Intro: {script.get('intro', '')[:60]}...")
        print(f"     • Secciones: {len(script.get('sections', []))}")
    
    # Save tasks
    tasks_file = DATA / "rebuild_videos_tasks.json"
    tasks_file.write_text(json.dumps(tasks, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("✅ TASKS GENERADOS")
    print("=" * 70)
    print(f"\n📄 Guardado: {tasks_file}")
    print("\n📋 Próximos pasos:")
    print("   1. python3 producer.py (producir videos)")
    print("   2. python3 thumbnail_agent_huggingface.py (thumbnails)")
    print("   3. python3 publisher.py (subir a YouTube)")
    print("\n¿Ejecuto el pipeline completo ahora? (responde 'adelante' para continuar)")


if __name__ == "__main__":
    main()
