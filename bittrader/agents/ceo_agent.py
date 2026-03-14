#!/usr/bin/env python3
"""
🎯 BitTrader CEO Agent - Orquestador Principal

Responsabilidades:
- Monitorear todo el sistema BitTrader
- Revisar quality check reports
- Detectar errores y problemas
- Delegar tareas al Programmer Agent
- Tomar decisiones sobre creación de nuevos agentes
- Reportar estado al usuario

Modelo actual: GLM-4.7 (cambia a Claude Opus 4.6 el sábado 1PM)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Add workspace to path
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
sys.path.insert(0, str(BITTRADER / "agents"))

from llm_config import call_llm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = BITTRADER / "agents/data"
OUTPUT_DIR = BITTRADER / "agents/output"
PRODUCTION_FILE = DATA_DIR / "production_latest.json"
QUALITY_FILE = DATA_DIR / "quality_check_latest.json"
SCOUT_FILE = DATA_DIR / "scout_latest.json"
CEO_STATE_FILE = DATA_DIR / "ceo_state.json"
ORGANIZATION_FILE = DATA_DIR / "organization.json"

# ── Model Config ───────────────────────────────────────────────────────────
# Current: GLM-4.7, changes to Claude Opus 4.6 on Saturday 1PM
CURRENT_MODEL = "claude-sonnet-4-6"
FUTURE_MODEL = "claude-opus-4-6"
MODEL_SWITCH_TIME = "2026-03-14T13:00:00-06:00"  # Saturday 1PM Denver


# ════════════════════════════════════════════════════════════════════════
# CEO AGENT
# ════════════════════════════════════════════════════════════════════════

class CEOAgent:
    def __init__(self):
        self.state = self.load_state()
        self.model = self.get_current_model()

    def get_current_model(self) -> str:
        """Check if it's time to switch to Opus 4.6"""
        switch_time = datetime.fromisoformat(MODEL_SWITCH_TIME)
        now = datetime.now(timezone.utc)

        if now >= switch_time.replace(tzinfo=timezone.utc):
            return FUTURE_MODEL
        return CURRENT_MODEL

    def load_state(self) -> Dict[str, Any]:
        """Load CEO state from file"""
        if CEO_STATE_FILE.exists():
            return json.loads(CEO_STATE_FILE.read_text())
        return {
            "last_check": None,
            "issues_found": [],
            "tasks_delegated": [],
            "model": CURRENT_MODEL
        }

    def save_state(self):
        """Save CEO state to file"""
        self.state["last_check"] = datetime.now(timezone.utc).isoformat()
        self.state["model"] = self.model
        CEO_STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def check_production_status(self) -> Dict[str, Any]:
        """Check latest production status"""
        if not PRODUCTION_FILE.exists():
            return {"error": "No production file found"}

        data = json.loads(PRODUCTION_FILE.read_text())
        # Extract stats with fallbacks
        stats = data.get("stats", {})
        videos = data.get("videos", [])
        
        # Count shorts vs longs from videos list
        shorts_count = sum(1 for v in videos if v.get("type") == "short")
        longs_count = sum(1 for v in videos if v.get("type") == "long")
        
        return {
            "total_videos": stats.get("total", len(videos)),
            "shorts": stats.get("shorts", shorts_count),
            "longs": stats.get("longs", longs_count),
            "quality_pass_rate": data.get("quality_check", {}).get("pass_rate", "N/A"),
            "produced_at": data.get("produced_at", "N/A")
        }

    def check_quality_report(self) -> Dict[str, Any]:
        """Check latest quality check results"""
        if not QUALITY_FILE.exists():
            return {"error": "No quality check file found"}

        data = json.loads(QUALITY_FILE.read_text())
        return {
            "passed": data.get("passed", 0),
            "failed": data.get("failed", 0),
            "grades": data.get("grades", {}),
            "issues": data.get("issues", [])
        }

    def check_scout_status(self) -> Dict[str, Any]:
        """Check latest scout data"""
        if not SCOUT_FILE.exists():
            return {"error": "No scout file found"}

        data = json.loads(SCOUT_FILE.read_text())
        return {
            "btc_price": data.get("market_data", {}).get("btc_price", "N/A"),
            "news_count": len(data.get("news", [])),
            "breaking_count": len(data.get("breaking_news", [])),
            "collected_at": data.get("collected_at", "N/A")
        }

    def analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health"""
        production = self.check_production_status()
        quality = self.check_quality_report()
        scout = self.check_scout_status()

        issues = []

        # Check for production issues
        if "error" in production:
            issues.append({
                "severity": "HIGH",
                "area": "production",
                "issue": production["error"]
            })

        # Check for quality issues
        if quality.get("failed", 0) > 0:
            issues.append({
                "severity": "MEDIUM",
                "area": "quality",
                "issue": f"{quality['failed']} videos failed quality check"
            })

        # Check for scout issues
        if scout.get("news_count", 0) == 0:
            issues.append({
                "severity": "LOW",
                "area": "scout",
                "issue": "No news collected"
            })

        return {
            "health_score": max(0, 100 - len(issues) * 20),
            "issues": issues,
            "production": production,
            "quality": quality,
            "scout": scout
        }

    def make_decision(self, health_report: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to make decisions based on health report"""

        prompt = f"""Eres el CEO Agent de BitTrader. Tu trabajo es monitorear el sistema y tomar decisiones.

ESTADO ACTUAL DEL SISTEMA:
{json.dumps(health_report, indent=2)}

PROBLEMAS DETECTADOS:
{json.dumps(health_report['issues'], indent=2)}

Analiza el estado y responde con una decisión en formato JSON:
{{
  "decision": "CONTINUE" | "FIX_NEEDED" | "CRITICAL",
  "reasoning": "explicación breve",
  "actions": [
    {{"action": "delegate_to_programmer", "task": "descripción de la tarea"}},
    {{"action": "alert_user", "message": "mensaje para el usuario"}},
    {{"action": "wait", "reason": "por qué esperar"}}
  ]
}}

Solo responde con el JSON, sin explicaciones adicionales.
"""

        response = call_llm(
            prompt=prompt,
            system="Eres el CEO Agent de BitTrader, un orquestador de sistemas.",
            max_tokens=1000
        )

        # Check if LLM call failed
        if response is None:
            return {
                "decision": "CONTINUE",
                "reasoning": "All LLMs failed (GLM-4.7 + MiniMax)",
                "actions": []
            }

        try:
            # Extract JSON from response (handle markdown code blocks)
            # Try to find content between ```json and ```
            if "```json" in response and "```" in response[response.find("```json") + 7:]:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)

            # Try to find content between ``` and ``` (without json keyword)
            if "```" in response:
                first_code = response.find("```")
                first_code_end = response.find("\n", first_code)
                last_code = response.rfind("```")
                if first_code != -1 and last_code > first_code_end:
                    json_str = response[first_code_end + 1:last_code].strip()
                    # Try to find first { and last } within this block
                    json_start = json_str.find("{")
                    json_end = json_str.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        return json.loads(json_str[json_start:json_end])

            # Fallback: try to find first { and last } in entire response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except Exception as e:
            print(f"    ⚠️ Error parsing JSON: {e}")

        # Fallback decision
        return {
            "decision": "CONTINUE",
            "reasoning": "Could not parse LLM response",
            "actions": []
        }

    def delegate_to_programmer(self, task: str) -> bool:
        """Delegate a task to the Programmer Agent"""
        programmer_task_file = DATA_DIR / "programmer_tasks.json"

        tasks = []
        if programmer_task_file.exists():
            tasks = json.loads(programmer_task_file.read_text())

        tasks.append({
            "task": task,
            "assigned_at": datetime.now(timezone.utc).isoformat(),
            "assigned_by": "ceo_agent",
            "status": "pending"
        })

        programmer_task_file.write_text(json.dumps(tasks, indent=2))

        # Log delegation
        self.state["tasks_delegated"].append({
            "task": task,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        print(f"  📋 Delegado al Programador: {task}")
        return True

    def get_organization_info(self) -> Dict[str, Any]:
        """Load organization structure"""
        if ORGANIZATION_FILE.exists():
            return json.loads(ORGANIZATION_FILE.read_text())
        return {"agents": [], "pipelines": []}

    def verify_programmer_work(self, work_report: Dict[str, Any]) -> Dict[str, Any]:
        """Second verification point - review programmer's work before approval"""
        
        prompt = f"""Eres el CEO Agent verificando el trabajo del Programmer Agent.

REPORTE DE TRABAJO:
{json.dumps(work_report, indent=2)}

Como CEO, debes:
1. Verificar que el trabajo cumple con los requisitos
2. Revisar calidad del código
3. Identificar posibles problemas
4. Decidir si apruebas o necesitas cambios

Responde en JSON:
{{
  "approved": true|false,
  "verification_score": 0-100,
  "review_notes": "comentarios sobre el trabajo",
  "issues_found": ["problemas si hay"],
  "requested_changes": ["cambios necesarios si no apruebas"],
  "final_approval": true|false
}}
"""

        response = call_llm(
            prompt=prompt,
            system="Eres un CEO que hace control de calidad riguroso.",
            max_tokens=600
        )

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except:
            pass

        return {
            "approved": False,
            "verification_score": 0,
            "review_notes": "Could not verify work",
            "issues_found": ["Verification failed"],
            "requested_changes": ["Manual review needed"],
            "final_approval": False
        }

    def get_project_context(self, idea: str) -> str:
        """Determine which project an idea relates to"""
        idea_lower = idea.lower()
        
        solana_keywords = ['solana', 'trading', 'bot', 'crypto', 'trade', 'position', 'strategy', 'market', 'profit', 'loss']
        youtube_keywords = ['youtube', 'video', 'thumbnail', 'bittrader', 'short', 'long', 'content', 'publish']
        
        solana_score = sum(1 for kw in solana_keywords if kw in idea_lower)
        youtube_score = sum(1 for kw in youtube_keywords if kw in idea_lower)
        
        if solana_score > youtube_score:
            return "Solana Trading Bot"
        elif youtube_score > solana_score:
            return "BitTrader YouTube"
        else:
            return "general"

    def evaluate_idea(self, idea: str) -> Dict[str, Any]:
        """Evaluate a user idea and decide how to implement it"""
        
        # Get organization info
        org = self.get_organization_info()
        agents_list = [a['name'] for a in org.get('agents', [])]
        pipelines_list = [p['name'] for p in org.get('pipelines', [])]
        projects = org.get('projects', [])
        
        # Determine which project this relates to
        project_context = self.get_project_context(idea)
        
        prompt = f"""Eres el CEO Agent de BitTrader Media + Solana Trading. El usuario tiene una idea nueva.

IDEA DEL USUARIO:
{idea}

PROYECTO DETECTADO: {project_context}

PROYECTOS ACTIVOS:
{json.dumps([{'name': p['name'], 'type': p['type'], 'status': p['status']} for p in projects], indent=2)}

AGENTES ACTUALES EN LA ORGANIZACIÓN:
{json.dumps(agents_list, indent=2)}

PIPELINES ACTUALES:
{json.dumps(pipelines_list, indent=2)}

RECURSOS DISPONIBLES:
- Engineer Agent (crea/modifica código, inventa soluciones)
- Quality Checker (verifica videos)
- Scout (recolecta datos)
- Creator (genera contenido)
- Producer (ensambla videos)
- Publisher (sube a YouTube)
- AI Researcher (analiza mercado)
- Executor (ejecuta trades)
- Risk Manager (gestiona riesgo)

Analiza la idea y decide qué hacer. Responde en formato JSON:
{{
  "project": "{project_context}",
  "decision": "CREATE_AGENT" | "CREATE_PIPELINE" | "USE_ENGINEER" | "USE_EXISTING" | "NEED_MORE_INFO",
  "reasoning": "por qué tomaste esta decisión",
  "implementation_plan": {{
    "new_agent_name": "nombre si necesitas crear agente nuevo",
    "agents_involved": ["lista de agentes que participarán"],
    "steps": [
      "paso 1",
      "paso 2"
    ],
    "estimated_time": "tiempo estimado"
  }},
  "questions": ["preguntas si necesitas más información"]
}}

Solo responde con el JSON.
"""

        response = call_llm(
            prompt=prompt,
            system="Eres el CEO Agent, un orquestador de múltiples proyectos experto en delegación.",
            max_tokens=800
        )

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except:
            pass

        return {
            "project": project_context,
            "decision": "NEED_MORE_INFO",
            "reasoning": "Could not parse idea",
            "implementation_plan": {},
            "questions": ["¿Puedes dar más detalles sobre la idea?"]
        }

    def run(self) -> Dict[str, Any]:
        """Main CEO Agent execution"""
        print("=" * 80)
        print("🎯 CEO AGENT - MONITOREO DE SISTEMA")
        print("=" * 80)
        print(f"  Modelo: {self.model.upper()}")
        print(f"  Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Analyze system health
        print("📊 Analizando estado del sistema...")
        health = self.analyze_system_health()

        print(f"  Health Score: {health['health_score']}/100")
        print(f"  Issues: {len(health['issues'])}")
        print()

        # Make decision
        print("🧠 Tomando decisión...")
        decision = self.make_decision(health)

        print(f"  Decisión: {decision['decision']}")
        print(f"  Razón: {decision['reasoning']}")
        print()

        # Execute actions via LLM decision
        if decision['actions']:
            print("⚡ Ejecutando acciones:")
            for action in decision['actions']:
                action_type = action.get('action', '')

                if action_type == 'delegate_to_programmer':
                    self.delegate_to_programmer(action['task'])

                elif action_type == 'alert_user':
                    print(f"  🚨 ALERTA: {action['message']}")

                elif action_type == 'wait':
                    print(f"  ⏳ Esperando: {action['reason']}")

        # ── AUTONOMOUS BRAIN: runs regardless of LLM decision ─────────────
        print()
        print("🧠 Autonomous Brain — auto-diagnóstico y acciones autónomas...")
        try:
            from autonomous_brain import run_autonomous_cycle
            brain_report = run_autonomous_cycle(dry_run=False, verbose=True)
            print(f"  Brain: {brain_report['actions_taken']} acciones | {brain_report['issues_found']} issues")
        except Exception as e:
            print(f"  ⚠️ Autonomous Brain error: {e}")
            brain_report = {"actions_taken": 0, "issues_found": 0, "actions": []}

        # Save state
        self.save_state()

        print()
        print("=" * 80)
        print("✅ CEO AGENT COMPLETADO")
        print("=" * 80)

        return {
            "health":        health,
            "decision":      decision,
            "brain_report":  brain_report,
            "model":         self.model
        }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ceo = CEOAgent()
    result = ceo.run()

    # Save report
    report_file = DATA_DIR / f"ceo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n📄 Reporte guardado: {report_file}")
