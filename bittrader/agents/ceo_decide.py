#!/usr/bin/env python3
"""CEO decision on MrBeast tactics"""
import json
from ceo_agent import CEOAgent

ceo = CEOAgent()

idea = """Implementar tácticas MrBeast en BitTrader. Opciones: A) Crear agente nuevo, B) Modificar existentes, C) Ambos. Recomienda UNA opción."""

result = ceo.evaluate_idea(idea)

print(json.dumps(result, indent=2, default=str))
