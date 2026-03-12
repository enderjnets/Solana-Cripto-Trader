#!/usr/bin/env python3
"""
Simple CEO Agent decision for MrBeast tactics
"""
import json
from pathlib import Path
from ceo_agent import CEOAgent

ceo = CEOAgent()

# Simple question for CEO to answer
idea = """Tienes 2 o 2 táácticas de BitTrader: Quiero aplicar las tácticas del MrBeast Playbook.

Tácticas clave:
- Thumbnails con rostros expresivos + testing A/B
- Títulos <50 chars con números
- Subtítulos estilo MrBeast (2 palabra por línea

Opción A: Crear agente nuevo especializado
 B: Modificar agentes existentes. C: Ambos
Cuál prefieres?"""

result = ceo.evaluate_idea(idea)

print(f"\n{'='* 80}")
print(f"{'='* 60}")
print()
print(f"{'=' * 80}")
print(f"{'='* 80}\n{'=' * 80}\n{'=' * 80}")
print(f"{'=' * 80}")
print()

# Save result
result_file = Path(__file__).parent / "ceo_decision.json"
result_file.write_text(json.dumps(result, indent=2, default=str))
