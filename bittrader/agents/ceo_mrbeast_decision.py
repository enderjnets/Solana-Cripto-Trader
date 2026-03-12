#!/usr/bin/env python3
"""
Pass MrBeast Playbook analysis to CEO Agent for decision
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ceo_agent import CEOAgent

# Load analysis
analysis_file = Path(__file__).parent / "data" / "mrbeast_playbook_analysis.json"
analysis = json.loads(analysis_file.read_text())

print("=" * 80)
print("🎯 CEO AGENT - DECISIÓN ESTRATÉGICA")
print("=" * 80)
print()
print("📊 ANÁLISIS: MrBeast Playbook Implementation")
print()

# Pass to CEO
ceo = CEOAgent()

idea = f"""Implementar tácticas del MrBeast Playbook en BitTrader.

GAPS IDENTIFICADOS:
{json.dumps(analysis['current_bittrader_state']['gaps'], indent=2)}

TÁCTICAS PRIORITARIAS:
1. Thumbnails con rostros expresivos + testing A/B
2. Títulos <50 chars con números y dinero
3. Re-enganches cada 3 minutos
4. Subtítulos estilo Komika Axis con verde

OPCIONES:
A) Crear MrBeast Optimizer Agent nuevo
B) Modificar agentes existentes
C) Ambos

¿Qué opción recomendamos y por qué?"""

result = ceo.evaluate_idea(idea)

print("🧠 DECISIÓN DEL CEO:")
print()
print(f"  Decisión: {result['decision']}")
print(f"  Razón: {result['reasoning']}")
print()

if result.get('implementation_plan'):
    plan = result['implementation_plan']
    print("📋 PLAN DE IMPLEMENTACIÓN:")
    
    if plan.get('new_agent_name'):
        print(f"  Nuevo agente: {plan['new_agent_name']}")
    
    if plan.get('agents_involved'):
        print(f"  Agentes: {', '.join(plan['agents_involved'])}")
    
    if plan.get('steps'):
        print("  Pasos:")
        for i, step in enumerate(plan['steps'], 1):
            print(f"    {i}. {step}")

if result.get('questions'):
    print()
    print("❓ PREGUNTAS:")
    for q in result['questions']:
        print(f"  • {q}")

print()
print("=" * 80)
