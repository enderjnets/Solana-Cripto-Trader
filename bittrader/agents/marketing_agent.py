#!/usr/bin/env python3
"""
🎯 BitTrader Marketing Agent - Especialista en Marketing Digital

Responsabilidades:
- Análisis de mercado y tendencias
- Generación de estrategias de marketing
- Investigación de contenido y keywords
- Coordinación con CEO Agent para campañas
- Adquisición de clientes para Eco y BitTrader

Modelo actual: GLM-4.7 (cambia a Sonnet 4.6 el sábado 1PM)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Add workspace to path
WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
sys.path.insert(0, str(BITTRADER / "agents"))

from llm_config import call_llm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = BITTRADER / "agents/data"
OUTPUT_DIR = BITTRADER / "agents/output"
RESEARCH_DIR = WORKSPACE / "research"
MARKETING_STATE_FILE = DATA_DIR / "marketing_state.json"
ECO_TASK_FILE = DATA_DIR / "eco_task.json"
CAMPAIGNS_FILE = DATA_DIR / "marketing_campaigns.json"

# ── Model Config ───────────────────────────────────────────────────────────
# Current: GLM-4.7, changes to Sonnet 4.6 on Saturday 1PM
CURRENT_MODEL = "glm-4.7"
FUTURE_MODEL = "claude-sonnet-4-6"
MODEL_SWITCH_TIME = "2026-03-14T13:00:00-06:00"  # Saturday 1PM Denver


# ════════════════════════════════════════════════════════════════════════
# MARKETING AGENT
# ════════════════════════════════════════════════════════════════════════

class MarketingAgent:
    def __init__(self):
        self.state = self.load_state()
        self.model = CURRENT_MODEL
    
    def load_state(self) -> Dict[str, Any]:
        """Load Marketing state from file"""
        if MARKETING_STATE_FILE.exists():
            return json.loads(MARKETING_STATE_FILE.read_text())
        return {
            "last_campaign": None,
            "active_campaigns": [],
            "content_ideas": [],
            "prospects_analyzed": 0,
            "model": CURRENT_MODEL
        }
    
    def save_state(self):
        """Save Marketing state to file"""
        self.state["last_check"] = datetime.now(timezone.utc).isoformat()
        self.state["model"] = self.model
        MARKETING_STATE_FILE.write_text(json.dumps(self.state, indent=2))
    
    def analyze_market_trends(self, industry: str = "AI automation") -> Dict[str, Any]:
        """Analyze market trends for a specific industry"""
        # Use text format instead of JSON for better GLM-4.7 compatibility
        prompt = f"""Lista 3 tendencias de marketing para: {industry}

Formato de respuesta:
TENDENCIA: [tendencia1]
TENDENCIA: [tendencia2]
TENDENCIA: [tendencia3]

Sin explicaciones adicionales."""
        
        response = call_llm(prompt, "", max_tokens=500)
        
        if response is None:
            return {"error": "LLM failed"}
        
        # Parse text response into structured data
        trends = []
        for line in response.split('\n'):
            if 'TENDENCIA:' in line or 'tendencia' in line.lower():
                # Extract text after colon or number
                if ':' in line:
                    trend = line.split(':', 1)[1].strip()
                else:
                    trend = line.strip()
                if trend:
                    trends.append(trend)
        
        return {
            "trends": trends[:3] if trends else ["No trends extracted"],
            "raw_response": response
        }
    
    def generate_campaign_strategy(self, product: str, target: str, budget: str = "$500-1000/month") -> Dict[str, Any]:
        """Generate a complete marketing campaign strategy"""
        prompt = f"""Eres el Marketing Agent de BitTrader. Genera una estrategia de campaña completa.

PRODUCTO/SERVICIO: {product}
PÚBLICO OBJETIVO: {target}
PRESUPUESTO: {budget}

Genera una estrategia en formato JSON:
{{
  "campaign_name": "nombre de la campaña",
  "objectives": ["objetivo 1", "objetivo 2"],
  "channels": [
    {{"channel": "LinkedIn", "strategy": "descripción", "budget_pct": 40}},
    {{"channel": "Twitter/X", "strategy": "descripción", "budget_pct": 30}},
    {{"channel": "Email", "strategy": "descripción", "budget_pct": 30}}
  ],
  "content_calendar": [
    {{"day": "Monday", "content_type": "post", "topic": "tema"}},
    {{"day": "Wednesday", "content_type": "video", "topic": "tema"}}
  ],
  "kpis": ["KPI 1", "KPI 2"],
  "timeline_weeks": 4,
  "expected_results": "descripción de resultados esperados"
}}

Solo responde con el JSON.
"""
        
        response = call_llm(prompt, "Eres un estratega de marketing experto.", max_tokens=800)
        
        if response is None:
            return {"error": "LLM failed"}
        
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_start != -1 and json_end > json_start:
                    return json.loads(response[json_start:json_end].strip())
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except Exception as e:
            print(f"    ⚠️ Error parsing JSON: {e}")
        
        return {"error": "Could not parse response"}
    
    def analyze_prospect(self, company: str, industry: str) -> Dict[str, Any]:
        """Analyze a potential client/prospect"""
        prompt = f"""Eres el Marketing Agent de BitTrader. Analiza un prospecto potencial.

EMPRESA: {company}
INDUSTRIA: {industry}

Genera análisis en formato JSON:
{{
  "company_overview": "breve descripción",
  "pain_points": ["pain point 1", "pain point 2"],
  "automation_opportunities": ["oportunidad 1", "oportunidad 2"],
  "recommended_services": ["servicio 1", "servicio 2"],
  "approach_strategy": "cómo abordar a esta empresa",
  "estimated_budget": "rango de presupuesto estimado",
  "priority": "HIGH | MEDIUM | LOW",
  "next_steps": ["paso 1", "paso 2"]
}}

Solo responde con el JSON.
"""
        
        response = call_llm(prompt, "Eres un experto en B2B marketing y ventas.", max_tokens=800)
        
        if response is None:
            return {"error": "LLM failed"}
        
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_start != -1 and json_end > json_start:
                    return json.loads(response[json_start:json_end].strip())
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except Exception as e:
            print(f"    ⚠️ Error parsing JSON: {e}")
        
        return {"error": "Could not parse response"}
    
    def generate_content_ideas(self, topic: str, platform: str = "LinkedIn", count: int = 5) -> List[str]:
        """Generate content ideas for a specific platform"""
        prompt = f"""Dame {count} ideas de contenido para {platform} sobre: {topic}

Formato:
1. [primera idea]
2. [segunda idea]

Solo la lista."""
        
        response = call_llm(prompt, "", max_tokens=500)
        
        if response is None:
            return []
        
        # Extract ideas from numbered list
        ideas = []
        for line in response.split('\n'):
            line = line.strip()
            if line and any(line.startswith(f'{i}.') for i in range(1, 10)):
                idea = line.split('.', 1)[1].strip() if '.' in line else line
                if idea:
                    ideas.append(idea)
        
        return ideas[:count] if ideas else []
    
    def run(self) -> Dict[str, Any]:
        """Execute marketing analysis cycle"""
        print("=" * 80)
        print("🎯 MARKETING AGENT - INICIANDO")
        print("=" * 80)
        print(f"  Modelo: {self.model}")
        print(f"  Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        results = {
            "market_trends": None,
            "content_ideas": None,
            "video_review": None,
            "status": "ok"
        }
        
        # Check if there's a pending video review task
        marketing_task_file = DATA_DIR / "marketing_task.json"
        if marketing_task_file.exists():
            task = json.loads(marketing_task_file.read_text())
            if task.get("status") == "pending" and "video" in task:
                print("🎬 Revisando video pendiente...")
                video_info = task["video"]
                video_title = video_info.get("title", "Video sin título")
                video_url = video_info.get("url", "")
                
                # Generate review prompt
                if video_url:
                    prompt = f"""Analiza este video de BitTrader:

URL: {video_url}
CANAL: BitTrader (@bittrader9259)

Revisa y lista problemas/correcciones necesarias:
1. Problemas técnicos (audio, video, subtítulos)
2. Problemas de contenido
3. Problemas de calidad
4. Recomendaciones de mejora

Responde en formato simple:
PROBLEMAS: [lista de problemas]
CORRECCIONES: [correcciones sugeridas]
PRIORIDAD: [Alta/Media/Baja]"""
                else:
                    prompt = f"""Analiza este video para el canal BitTrader:

VIDEO: "{video_title}"

Evalúa problemas y correcciones necesarias.

Responde:
PROBLEMAS: [lista]
CORRECCIONES: [sugerencias]
PRIORIDAD: [Alta/Media/Baja]"""
                
                response = call_llm(prompt, "", max_tokens=600)
                
                if response:
                    print(f"  ✅ Video analizado")
                    results["video_review"] = {
                        "video": video_title,
                        "url": video_url,
                        "analysis": response
                    }
                    # Mark task as completed
                    task["status"] = "completed"
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    marketing_task_file.write_text(json.dumps(task, indent=2))
                else:
                    print("  ⚠️ Error analizando video")
                print()
        
        # 1. Analyze market trends for Eco
        print("📊 Analizando tendencias de mercado...")
        trends = self.analyze_market_trends("AI automation for small businesses")
        results["market_trends"] = trends
        
        if "error" not in trends:
            print(f"  ✅ {len(trends.get('trends', []))} tendencias identificadas")
            print(f"  ✅ {len(trends.get('opportunities', []))} oportunidades encontradas")
        else:
            print(f"  ⚠️ Error: {trends.get('error')}")
        
        print()
        
        # 2. Generate content ideas for BitTrader
        print("💡 Generando ideas de contenido...")
        ideas = self.generate_content_ideas("IA para automatización de negocios", "LinkedIn", 5)
        results["content_ideas"] = ideas
        
        if ideas:
            print(f"  ✅ {len(ideas)} ideas generadas")
            self.state["content_ideas"] = ideas
        else:
            print("  ⚠️ No se pudieron generar ideas")
        
        print()
        
        # Save state
        self.save_state()
        
        # Save report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
            "model": self.model
        }
        
        report_file = DATA_DIR / f"marketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.write_text(json.dumps(report, indent=2))
        print(f"📄 Reporte guardado: {report_file}")
        
        print()
        print("=" * 80)
        print("✅ MARKETING AGENT COMPLETADO")
        print("=" * 80)
        
        return report


# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    agent = MarketingAgent()
    result = agent.run()
