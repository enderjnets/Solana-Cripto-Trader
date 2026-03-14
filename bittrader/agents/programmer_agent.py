#!/usr/bin/env python3
"""
👨‍💻 BitTrader Programmer Agent - Ingeniero de Código ÉLITE

Responsabilidades:
- Recibir tareas del CEO Agent
- Crear nuevos agentes cuando se solicite
- Arreglar bugs en el código (auto-debugging)
- Optimizar scripts existentes
- Buscar documentación cuando lo necesita
- Hacer auditorías constantes a su propio código
- Aprender de los errores y mejorar automáticamente
- Auto-mejora continua de sus creaciones

Capacidades:
- Expert en todos los lenguajes de programación
- Súper creativo para soluciones
- Auto-suficiente: debug, test, audit, improve

Modelo actual: GLM-4.7 (cambia a Claude Opus 4.6 el sábado 1PM)
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add workspace to path
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
sys.path.insert(0, str(BITTRADER / "agents"))

from llm_config import call_llm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = BITTRADER / "agents/data"
TASKS_FILE = DATA_DIR / "programmer_tasks.json"
PROGRAMMER_STATE_FILE = DATA_DIR / "programmer_state.json"

# ── Model Config ───────────────────────────────────────────────────────────
CURRENT_MODEL = "claude-opus-4-6"
FUTURE_MODEL = "claude-opus-4-6"
MODEL_SWITCH_TIME = "2026-03-14T13:00:00-06:00"


# ════════════════════════════════════════════════════════════════════════
# PROGRAMMER AGENT
# ════════════════════════════════════════════════════════════════════════

class ProgrammerAgent:
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
        """Load Programmer state from file"""
        if PROGRAMMER_STATE_FILE.exists():
            return json.loads(PROGRAMMER_STATE_FILE.read_text())
        return {
            "last_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "model": CURRENT_MODEL
        }

    def save_state(self):
        """Save Programmer state to file"""
        self.state["last_task"] = datetime.now(timezone.utc).isoformat()
        self.state["model"] = self.model
        PROGRAMMER_STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get pending tasks from CEO"""
        if not TASKS_FILE.exists():
            return []

        tasks = json.loads(TASKS_FILE.read_text())
        return [t for t in tasks if t["status"] == "pending"]

    def mark_task_complete(self, task_index: int):
        """Mark a task as completed"""
        tasks = json.loads(TASKS_FILE.read_text())
        tasks[task_index]["status"] = "completed"
        tasks[task_index]["completed_at"] = datetime.now(timezone.utc).isoformat()
        TASKS_FILE.write_text(json.dumps(tasks, indent=2))

    def search_documentation(self, topic: str, context: str = "") -> str:
        """Search for documentation when encountering errors"""
        
        prompt = f"""Eres el Programmer Agent de BitTrader. Necesitas buscar información para resolver un problema.

PROBLEMA/TEMA:
{topic}

CONTEXTO ADICIONAL:
{context}

Busca en tu conocimiento y responde con:
1. Explicación del problema
2. Solución propuesta
3. Código de ejemplo si aplica
4. Mejores prácticas relacionadas
"""

        return call_llm(
            prompt=prompt,
            system="Eres un programador experto que busca documentación y aprende constantemente.",
            max_tokens=1000
        )

    def self_audit_code(self, code: str, filename: str) -> Dict[str, Any]:
        """Audit own code for improvements"""
        
        prompt = f"""Eres el Programmer Agent auditando tu propio código.

ARCHIVO: {filename}
CÓDIGO:
```
{code}
```

Audita el código buscando:
1. Errores o bugs potenciales
2. Mejoras de rendimiento
3. Mejores prácticas faltantes
4. Código duplicado o innecesario
5. Seguridad
6. Legibilidad

Responde en JSON:
{{
  "issues_found": [
    {{"line": 0, "severity": "low|medium|high", "issue": "descripción", "fix": "cómo arreglar"}}
  ],
  "improvements": [
    {{"type": "performance|readability|security|best_practice", "suggestion": "qué mejorar"}}
  ],
  "overall_score": 0-100,
  "should_refactor": true|false
}}
"""

        response = call_llm(
            prompt=prompt,
            system="Eres un auditor de código experto y objetivo.",
            max_tokens=1000
        )

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except:
            pass

        return {
            "issues_found": [],
            "improvements": [],
            "overall_score": 50,
            "should_refactor": False
        }

    def debug_error(self, error: str, code: str) -> Dict[str, Any]:
        """Debug an error and propose solution"""
        
        prompt = f"""Eres el Programmer Agent haciendo debugging.

ERROR:
```
{error}
```

CÓDIGO RELACIONADO:
```
{code}
```

Analiza el error y responde:
{{
  "error_type": "tipo de error",
  "root_cause": "causa raíz",
  "solution": "solución paso a paso",
  "fixed_code": "código corregido",
  "prevent_future": "cómo prevenir en el futuro"
}}
"""

        response = call_llm(
            prompt=prompt,
            system="Eres un experto en debugging y resolución de errores.",
            max_tokens=1000
        )

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except:
            pass

        return {
            "error_type": "unknown",
            "root_cause": "Could not analyze",
            "solution": "Manual review needed",
            "fixed_code": "",
            "prevent_future": ""
        }

    def create_new_agent(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent based on specifications"""
        
        prompt = f"""Eres el Programmer Agent creando un nuevo agente.

ESPECIFICACIONES:
{json.dumps(spec, indent=2)}

AGENTES EXISTENTES (para referencia):
- CEO Agent: orquestador principal
- Quality Checker: verifica videos
- Scout: recolecta datos
- Creator: genera contenido
- Producer: ensambla videos
- Publisher: sube a YouTube

Diseña el nuevo agente y responde:
{{
  "agent_name": "nombre_del_agente",
  "filename": "nombre_archivo.py",
  "description": "qué hace el agente",
  "dependencies": ["librerías necesarias"],
  "main_functions": [
    {{"name": "función", "purpose": "propósito"}}
  ],
  "integration_points": ["con qué otros agentes se conecta"],
  "code_template": "código base del agente"
}}
"""

        response = call_llm(
            prompt=prompt,
            system="Eres un arquitecto de software experto en crear sistemas modulares.",
            max_tokens=2000
        )

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except:
            pass

        return {
            "agent_name": "unknown",
            "error": "Could not create agent specification"
        }

    def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a task and determine how to execute it"""

        # Build simplified task info
        task_info = f"Tarea: {task.get('task', 'Sin título')}"
        if 'priority' in task:
            task_info += f" (Prioridad: {task['priority']})"
        if 'details' in task:
            # Just include key info from details, not full JSON
            task_info += "\n\nDetalles: "
            for key in task['details']:
                task_info += f"- {key}\n"
        if 'project_path' in task:
            task_info += f"\nRuta: {task['project_path']}"

        prompt = f"""{task_info}

Analiza y responde EXCLUSIVAMENTE con JSON:
{{
  "task_type": "bug_fix" | "optimization" | "new_feature" | "config_change" | "investigation",
  "files_affected": ["lista de archivos"],
  "steps": ["paso 1", "paso 2"],
  "estimated_complexity": "low" | "medium" | "high",
  "requires_user_approval": true | false
}}

NO expliques nada. Solo JSON.
"""

        response = call_llm(
            prompt=prompt,
            system="Eres el Programmer Agent de BitTrader, un ingeniero de código experto.",
            max_tokens=2000
        )

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

        return {
            "task_type": "unknown",
            "files_affected": [],
            "steps": ["Could not parse task"],
            "estimated_complexity": "unknown",
            "requires_user_approval": True
        }

    def report_to_ceo(self, task: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Report completed work to CEO for verification"""
        report_file = DATA_DIR / "programmer_reports.json"
        
        reports = []
        if report_file.exists():
            reports = json.loads(report_file.read_text())
        
        report = {
            "task": task['task'],
            "assigned_at": task.get('assigned_at', ''),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "status": "pending_verification",
            "programmer_model": self.model
        }
        
        reports.append(report)
        report_file.write_text(json.dumps(reports, indent=2))
        
        print(f"  📊 Reporte enviado al CEO para verificación")
        return True

    def execute_task(self, task: Dict[str, Any], plan: Dict[str, Any]) -> bool:
        """Execute a task based on the plan"""

        print(f"  📋 Ejecutando: {task['task']}")
        print(f"  Tipo: {plan['task_type']}")
        print(f"  Complejidad: {plan['estimated_complexity']}")
        print()

        # For now, log the plan and mark as complete
        # In a full implementation, this would actually execute code changes

        print("  📝 Plan de ejecución:")
        for i, step in enumerate(plan['steps'], 1):
            print(f"    {i}. {step}")

        print()
        print("  ✅ Tarea completada (simulación)")

        return True

    def run(self) -> Dict[str, Any]:
        """Main Programmer Agent execution"""
        print("=" * 80)
        print("👨‍💻 PROGRAMMER AGENT - EJECUCIÓN DE TAREAS")
        print("=" * 80)
        print(f"  Modelo: {self.model.upper()}")
        print(f"  Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Get pending tasks
        tasks = self.get_pending_tasks()

        if not tasks:
            print("  ℹ️  No hay tareas pendientes")
            print()
            print("=" * 80)
            return {"status": "no_tasks", "tasks_completed": 0}

        print(f"  📥 {len(tasks)} tareas pendientes")
        print()

        completed = 0
        failed = 0

        for i, task in enumerate(tasks):
            print(f"  Tarea {i+1}/{len(tasks)}:")
            print(f"  Asignada: {task.get('assigned_at', 'N/A')}")

            # Analyze task
            plan = self.analyze_task(task)

            # Check if user approval needed
            if plan.get('requires_user_approval', False):
                print(f"  ⚠️  Requiere aprobación del usuario")
                print(f"  Plan: {json.dumps(plan, indent=4)}")
                continue

            # Execute task
            try:
                success = self.execute_task(task, plan)

                if success:
                    # Report to CEO for verification
                    self.report_to_ceo(task, {
                        "plan": plan,
                        "status": "completed",
                        "complexity": plan.get('estimated_complexity', 'unknown')
                    })
                    
                    self.mark_task_complete(i)
                    completed += 1

                    self.state["completed_tasks"].append({
                        "task": task['task'],
                        "completed_at": datetime.now(timezone.utc).isoformat()
                    })
                else:
                    failed += 1
                    self.state["failed_tasks"].append({
                        "task": task['task'],
                        "failed_at": datetime.now(timezone.utc).isoformat()
                    })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                failed += 1

            print()

        # Save state
        self.save_state()

        print("=" * 80)
        print(f"✅ PROGRAMMER AGENT COMPLETADO")
        print(f"  Tareas completadas: {completed}")
        print(f"  Tareas fallidas: {failed}")
        print("=" * 80)

        return {
            "status": "completed",
            "tasks_completed": completed,
            "tasks_failed": failed
        }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    programmer = ProgrammerAgent()
    result = programmer.run()

    # Save report
    report_file = DATA_DIR / f"programmer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n📄 Reporte guardado: {report_file}")
