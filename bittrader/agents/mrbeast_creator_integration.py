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
    prompt = f"""Crea un guión COMPLETO para un YouTube Short del canal BitTrader.

🎯 TEMA: {item['topic']}
📊 CATEGORÍA: {item['theme']}

⚡ REGLAS MRBEAST:
1. TÍTULO: Menos de 50 caracteres, incluye número o cantidad de dinero si aplica
2. GANCHO: Primera frase debe enganchar inmediatamente (sin intros)
3. USA "TÚ" más que "YO" para involucrar al espectador
4. BUCLE NARRATIVO: Termina con "...y lo que pasó después te sorprenderá"

📝 TÍTULO SUGERIDO (puedes ajustarlo): {title_suggestion.get('optimized_title', item['topic'][:50])}

FORMATO REQUERIDO:
TITULO: [título optimizado, máx 50 chars, sin emojis]
DESCRIPCION: [descripción YouTube 100-150 chars con hashtags]
TAGS: [8-10 tags separados por coma]

GUION:
[Guión de 100-140 palabras. Estructura:
- Gancho inmediato (primeras 5 palabras)
- Desarrollo del punto principal
- Dato impactante o ejemplo concreto
- CTA con pregunta ("¿Tú qué harías? Comenta")

BUCLE FINAL: Termina con frase abierta que invite a ver más]

VIDEO_PROMPT: [Describe la escena con BitTrader rhino. Incluye: qué hace el rinoceronte, el escenario, el ambiente. Hyper-realistic 3D, 9:16 vertical]"""

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
    
    prompt = f"""Crea un guión COMPLETO para un video de YouTube de 3-5 minutos del canal BitTrader.

🎯 TEMA: {item['topic']}
📊 CATEGORÍA: {item['theme']}

⚡ REGLAS MRBEAST:
1. TÍTULO: Menos de 50 caracteres, incluye número o cantidad de dinero
2. GANCHO: Sin intros lentas, empieza con la acción prometida
3. RE-ENGANCHES cada 3 minutos: giros, sorpresas, revelaciones
4. BUCLES ABIERTOS: Usa frases como "y lo que pasó después..." para mantener atención
5. STAIR-STEPPING: Las apuestas aumentan progresivamente, clímax al final

📝 TÍTULO SUGERIDO: {title_suggestion.get('optimized_title', item['topic'][:50])}

🎯 RE-ENGANCHES REQUERIDOS:
{hooks_str}

FORMATO REQUERIDO:
TITULO: [título optimizado, máx 50 chars]
DESCRIPCION: [descripción YouTube 200-300 chars, con hashtags]
TAGS: [10-15 tags separados por coma]

GUIÓN COMPLETO (400-700 palabras):
HOOK (30s): [Enganche poderoso, pregunta o dato impactante, entrega la promesa inmediatamente]
PROBLEMA (45s): [Por qué importa, el dolor del espectador, usa "TÚ"]
EXPLICACION (2-3min): [Contenido principal con ejemplos concretos, inserta re-enganche cada 3 min]
EJEMPLOS (1min): [Casos reales, números, comparaciones]
CTA (30s): [Llamada a acción con pregunta para comentarios, pide suscripción]

⚠️ IMPORTANTE:
- Inserta re-enganche en el guión cada ~3 minutos (ej: "Pero espera, esto se pone mejor...")
- Usa "TÚ" más que "YO" en toda la narración
- Termina con bucle narrativo abierto

VIDEO_PROMPT_1: [Intro scene con BitTrader rhino. Describe qué hace relacionado al hook. 16:9]
VIDEO_PROMPT_2: [Main scene con rhino. Acción relevante para la explicación. 16:9]
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
