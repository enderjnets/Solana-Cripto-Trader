#!/usr/bin/env python3
"""
🎯 CEO Agent Issue Handler - Missing Thumbnails
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load issue
issue = json.loads(Path("data/issue_thumb_001.json").read_text())

print("=" * 80)
print("🚨 CEO AGENT - ISSUE HANDLER")
print("=" * 80)
print()
print(f"Issue ID: {issue['issue_id']}")
print(f"Severity: {issue['severity']}")
print(f"Title: {issue['title']}")
print()
print("AFFECTED VIDEOS:")
for video in issue['affected_videos']:
    print(f"  • {video['title']}")
    print(f"    YouTube ID: {video['youtube_id']}")
    print()

# CEO Decision
from ceo_agent import CEOAgent
from llm_config import call_llm

ceo = CEOAgent()

# Prepare context for CEO
issue_context = f"""
PROBLEMA URGENTE:
{issue['problem']}

Videos afectados:
{json.dumps(issue['affected_videos'], indent=2)}

Contexto:
- Tenemos thumbnail_agent.py que genera thumbnails
- Tenemos mrbeast_optimizer_agent.py con A/B testing
- Los thumbnails deberían tener: logo BitTrader, texto dorado, rinoceronte

¿Qué hacer? Responde con JSON:
{{
  "decision": "FIX_NOW" | "DELEGATE" | "MANUAL_REQUIRED",
  "reasoning": "explicación breve",
  "solution": {{
    "steps": [
      "paso 1",
      "paso 2",
      "paso 3"
    ],
    "agents_involved": ["thumbnail_agent", "mrbeast_optimizer"],
    "estimated_time": "tiempo estimado"
  }}
}}
"""

response = call_llm(
    prompt=issue_context,
    system="Eres el CEO Agent resolviendo problemas críticos del sistema BitTrader.",
    max_tokens=600
)

print("CEO DECISION:")
print(response)

# Try to parse JSON
try:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start != -1 and json_end > json_start:
        decision = json.loads(response[json_start:json_end])
        
        print()
        print("=" * 80)
        print("📋 SOLUTION PLAN:")
        print(f"Decision: {decision.get('decision')}")
        print(f"Reasoning: {decision.get('reasoning')}")
        print()
        
        if 'solution' in decision:
            print("Steps:")
            for i, step in enumerate(decision['solution'].get('steps', []), 1):
                print(f"  {i}. {step}")
            print()
            print(f"Agents: {', '.join(decision['solution'].get('agents_involved', []))}")
            print(f"Time: {decision['solution'].get('estimated_time')}")
        
        # Save decision
        decision_file = Path("data") / f"ceo_decision_{issue['issue_id']}.json"
        decision_file.write_text(json.dumps({
            "issue_id": issue['issue_id'],
            "decision": decision,
            "timestamp": "2026-03-11T21:24:00-06:00"
        }, indent=2))
        print()
        print(f"💾 Decision saved: {decision_file}")
        
except Exception as e:
    print(f"\n⚠️ Could not parse CEO response: {e}")
    print("Proceeding with default solution...")

print()
print("=" * 80)
