#!/usr/bin/env python3
"""
💬 Pass an idea to CEO Agent for evaluation and delegation
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ceo_agent import CEOAgent

if len(sys.argv) < 2:
    print("Uso: python3 ceo_idea.py \"tu idea aquí\"")
    print("Ejemplo: python3 ceo_idea.py \"Quiero un agente que analice sentimiento de Twitter\"")
    sys.exit(1)

idea = " ".join(sys.argv[1:])

print("=" * 80)
print("💡 ENVIANDO IDEA AL CEO AGENT")
print("=" * 80)
print(f"Idea: {idea}")
print()

ceo = CEOAgent()
result = ceo.evaluate_idea(idea)

print("📊 DECISIÓN DEL CEO:")
print(f"  Decisión: {result['decision']}")
print(f"  Razón: {result['reasoning']}")
print()

if result.get('implementation_plan'):
    plan = result['implementation_plan']
    print("📋 PLAN DE IMPLEMENTACIÓN:")
    
    if plan.get('new_agent_name'):
        print(f"  Nuevo agente: {plan['new_agent_name']}")
    
    if plan.get('agents_involved'):
        print(f"  Agentes involucrados: {', '.join(plan['agents_involved'])}")
    
    if plan.get('steps'):
        print("  Pasos:")
        for i, step in enumerate(plan['steps'], 1):
            print(f"    {i}. {step}")
    
    if plan.get('estimated_time'):
        print(f"  Tiempo estimado: {plan['estimated_time']}")

if result.get('questions'):
    print()
    print("❓ PREGUNTAS DEL CEO:")
    for q in result['questions']:
        print(f"  • {q}")

# Save idea for reference
ideas_file = Path(__file__).parent / "data" / "ceo_ideas.json"
ideas = []
if ideas_file.exists():
    ideas = json.loads(ideas_file.read_text())

ideas.append({
    "idea": idea,
    "timestamp": result.get('timestamp', ''),
    "decision": result['decision'],
    "plan": result.get('implementation_plan', {})
})

ideas_file.write_text(json.dumps(ideas, indent=2))
print()
print(f"💾 Idea guardada: {ideas_file}")
