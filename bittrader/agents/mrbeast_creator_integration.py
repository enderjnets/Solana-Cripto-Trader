#!/usr/bin/env python3
"""
Integration: MrBeast Optimizer + Creator Agent

Este módulo inyecta las tácticas de MrBeast en el pipeline de creación de contenido:
- Títulos optimizados (<50 chars, números, dinero)
- Estructura de retención con re-enganches
- Bucles narrativos abiertos
"""
import json
from pathlib import Path
from typing import Dict, Any

# Import MrBeast Optimizer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mrbeast_optimizer_agent import optimize_title, generate_retention_hooks

BITTRADER = Path("/home/enderj/.openclaw/workspace/bittrader")
DATA_DIR = BITTRADER / "agents/data"


# ════════════════════════════════════════════════════════════════════════
# MRBEAST-ENHANCED PROMPTS
# ════════════════════════════════════════════════════════════════════════

def inject_mrbeast_short_prompt(item: dict, scout: dict) -> str:
    """Generate MrBeast-optimized prompt for Shorts"""
    
    # Get MrBeast-optimized title suggestion
    title_suggestion = optimize_title(item['topic'], item.get('description', ''))
    
    # Build retention-focused prompt
    prompt = f"""Crea un guion COMPLETO para un YouTube Short del canal BitTrader.

TEMA: {item['topic']}
CATEGORIA: {item['theme']}

TITULO SUGERIDO (puedes ajustarlo): {title_suggestion.get('optimized_title', item['topic'][:50])}

Caracteristicas del guion:
- El titulo debe tener menos de 50 caracteres, incluye numero o cantidad de dinero si aplica
- La primera frase debe enganchar inmediatamente, sin intros
- Dirigete al espectador en segunda persona (habla de "ti", "tu", "vas a")
- Termina con una frase abierta que invite a ver mas

FORMATO DE RESPUESTA (responde exactamente con esta estructura, nada mas):

TITULO: [titulo optimizado, max 50 chars, sin emojis]
DESCRIPCION: [descripcion YouTube 100-150 chars con hashtags]
TAGS: [8-10 tags separados por coma]

GUION:
[Guion narrado de 100-140 palabras en espanol. Empieza con gancho. Termina con pregunta para comentarios. Solo el texto de la narracion, sin encabezados ni etiquetas de seccion.]

VIDEO_PROMPT: [Describe la escena con BitTrader rhino. Incluye: que hace el rinoceronte, el escenario, el ambiente. Hyper-realistic 3D, 9:16 vertical]"""

    return prompt


def inject_mrbeast_long_prompt(item: dict, scout: dict) -> str:
    """Generate MrBeast-optimized prompt for Long Videos"""
    
    # Get MrBeast-optimized title suggestion
    title_suggestion = optimize_title(item['topic'], item.get('description', ''))
    
    # Get retention hooks for this duration (assume 4 min = 240s for planning)
    hooks = generate_retention_hooks(240)  # 4 minutes
    
    # Build hooks string for prompt
    hooks_str = ""
    for i, hook in enumerate(hooks[:3], 1):  # First 3 hooks
        hooks_str += f"  - Minuto {hook['timestamp']//60}: {hook['suggestion']}\n"
    
    prompt = f"""Crea un guion COMPLETO para un video de YouTube de 3-5 minutos del canal BitTrader.

TEMA: {item['topic']}
CATEGORIA: {item['theme']}

TITULO SUGERIDO: {title_suggestion.get('optimized_title', item['topic'][:50])}

Caracteristicas del video:
- El titulo debe tener menos de 50 caracteres, incluye numero o cantidad de dinero si aplica
- Empieza directo con la accion prometida, sin intros lentas
- Inserta momentos de re-enganche naturales cada ~3 minutos ("Pero lo que descubri despues fue...")
- Las apuestas del video aumentan progresivamente hasta un climax al final
- Dirigete al espectador en segunda persona (habla de "ti", "tu", "vas a")
- Termina con una pregunta abierta para comentarios

Momentos de re-enganche sugeridos (incorpoalos de forma natural en el guion):
{hooks_str}

FORMATO DE RESPUESTA (responde exactamente con esta estructura, nada mas):

TITULO: [titulo optimizado, max 50 chars]
DESCRIPCION: [descripcion YouTube 200-300 chars, con hashtags]
TAGS: [10-15 tags separados por coma]

GUION:
[Guion narrado completo de 400-700 palabras en espanol. Solo el texto de la narracion, sin encabezados como HOOK:, PROBLEMA:, EXPLICACION:, CTA: ni ningun otro encabezado de seccion. Solo la narracion fluida del video.]

VIDEO_PROMPT_1: [Intro scene con BitTrader rhino. Describe que hace relacionado al hook. 16:9]
VIDEO_PROMPT_2: [Main scene con rhino. Accion relevante para la explicacion. 16:9]
VIDEO_PROMPT_3: [Closing scene con rhino. Confident, mirando al futuro. 16:9]"""

    return prompt


def enhance_creator_prompts():
    """
    Main integration function.
    Call this to enhance Creator Agent prompts with MrBeast tactics.
    
    Returns functions that can be used by Creator Agent:
    - inject_mrbeast_short_prompt()
    - inject_mrbeast_long_prompt()
    """
    return {
        "short_prompt_enhancer": inject_mrbeast_short_prompt,
        "long_prompt_enhancer": inject_mrbeast_long_prompt,
        "title_optimizer": optimize_title,
        "retention_hooks_generator": generate_retention_hooks
    }


# ════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("🔗 MRBEAST + CREATOR INTEGRATION")
    print("=" * 80)
    print()
    
    # Sample data
    sample_item = {
        "topic": "Bitcoin cae 1.4% mientras AKT explota al alza",
        "theme": "noticias",
        "description": "Análisis del mercado crypto de hoy"
    }
    
    sample_scout = {
        "crypto": {
            "bitcoin": {"price_usd": 67000, "change_24h": -1.4}
        }
    }
    
    print("📝 Sample Short Prompt (Enhanced with MrBeast):")
    print()
    short_prompt = inject_mrbeast_short_prompt(sample_item, sample_scout)
    print(short_prompt[:500] + "...")
    print()
    
    print("=" * 80)
    print("📝 Sample Long Video Prompt (Enhanced with MrBeast):")
    print()
    long_prompt = inject_mrbeast_long_prompt(sample_item, sample_scout)
    print(long_prompt[:700] + "...")
    print()
    
    print("=" * 80)
    print("✅ Integration ready!")
    print()
    print("Usage in creator.py:")
    print("  from mrbeast_creator_integration import inject_mrbeast_short_prompt")
    print("  enhanced_prompt = inject_mrbeast_short_prompt(item, scout)")
