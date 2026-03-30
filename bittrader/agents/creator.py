#!/usr/bin/env python3
"""
🎨 BitTrader Creator — Agente Estratega Creativo
Lee datos del Scout y genera guiones completos con Claude Sonnet 4.6.
Ejecutar: python3 agents/creator.py

Fallback: Claude Sonnet → GLM-5 → MiniMax → Qwen
"""
import json
import random
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).parent))
import qwen_client

# MrBeast Integration (ENABLED with Claude Sonnet 4.6 - respects format, no visible reasoning)
try:
    from mrbeast_creator_integration import inject_mrbeast_short_prompt, inject_mrbeast_long_prompt
    MRBEAST_ENABLED = True  # Enabled - Claude Sonnet respects format without visible reasoning
    print("  ✅ MrBeast integration enabled (Claude Sonnet 4.6 primary)")
except ImportError:
    MRBEAST_ENABLED = False
    print("  ⚠️ MrBeast integration not available (mrbeast_creator_integration.py not found)")

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Claude Sonnet 4.6 (local API) - PRIMARY
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

# GLM-5 - FALLBACK #1 (cuando Claude está rate-limited)
ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-5"

# MiniMax M2.5 - FALLBACK #2 (Anthropic API compatible)
MINIMAX_KEY   = json.loads((BITTRADER / "keys/minimax.json").read_text())["minimax_coding_key"]
MINIMAX_URL   = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"

SCOUT_LATEST  = DATA_DIR / "scout_latest.json"


# ══════════════════════════════════════════════════════════════════════
# LLM CALLS (Claude Sonnet PRIMARY -> GLM-5 FALLBACK -> MiniMax FALLBACK)
# ══════════════════════════════════════════════════════════════════════

def call_claude(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama a Claude Sonnet 4.6 (PRIMARY)"""
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": []
    }

    if system:
        data["messages"].append({"role": "user", "content": system})
    data["messages"].append({"role": "user", "content": prompt})

    try:
        r = requests.post(CLAUDE_BASE_URL, headers=headers, json=data, timeout=300)  # 5 min para videos largos
        r.raise_for_status()
        response = r.json()
        return response.get("content", [{}])[0].get("text", "")
    except Exception as e:
        print(f"    ⚠️ Claude error: {e}")
        return None


def call_glm_5(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama a GLM-5 (FALLBACK #1)"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZAI_CODING_KEY}"
    }

    # GLM-5 soporta system role correctamente
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": ZAI_CODING_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        r = requests.post(ZAI_CODING_BASE_URL, headers=headers, json=data, timeout=120)
        r.raise_for_status()
        response = r.json()

        # Usar content (no reasoning_content que puede tener meta-texto)
        if "choices" in response and len(response["choices"]) > 0:
            msg = response["choices"][0]["message"]
            content = msg.get("content", "")
            # Post-process: eliminar contaminación común de GLM
            content = _clean_llm_output(content)
            return content if content else None
        return None
    except Exception as e:
        print(f"    ⚠️ GLM-5 error: {e}")
        return None


def _clean_llm_output(text: str) -> str:
    """Elimina contaminación de prompt del output del LLM"""
    if not text:
        return text
    
    import re
    
    # Patrones de contaminación a eliminar
    contamination_patterns = [
        r"^.*?(?=TITULO:)",  # Todo antes de TITULO:
        r"(?:el usuario quiere|el usuario necesita|el usuario pide).*?\n",
        r"(?:REGLA|INSTRUCCION|BUCLE|NOTA)[\s\d]*:.*?\n",
        r"(?:Responde|Genera|Crea).*?(?:guión|script|video).*?\n",
        r"\*\*Instrucciones.*?\*\*.*?\n",
        r"---+\s*\n",  # Separadores
    ]
    
    for pattern in contamination_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # Limpiar líneas vacías múltiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def call_minimax_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama a MiniMax usando Anthropic API compatible"""
    headers = {
        "x-api-key": MINIMAX_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    messages = []
    if system:
        messages.append({"role": "user", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MINIMAX_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    try:
        r = requests.post(MINIMAX_URL, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        # Parsear respuesta Anthropic API compatible
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "").strip()
        return None
    except Exception as e:
        print(f"    ⚠️ MiniMax error: {e}")
        return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama al LLM con fallback: Claude Sonnet -> GLM-5 -> MiniMax

    Fallback directo: Claude Sonnet -> GLM-5 -> MiniMax -> Qwen
    MrBeast puede usar GLM-5 (a diferencia de GLM-4.7 que generaba razonamiento visible)
    """
    # 1. Intentar Claude Sonnet 4.6
    result = call_claude(prompt, system, max_tokens)
    if result:
        return result

    # 2. Intentar GLM-5
    print("    ⚠️ Claude Sonnet falló, intentando GLM-5 fallback...")
    result = call_glm_5(prompt, system, max_tokens)
    if result:
        return result

    # 3. Intentar MiniMax
    print("    ⚠️ GLM-5 falló, intentando MiniMax fallback...")
    result = call_minimax_llm(prompt, system, max_tokens)
    if result:
        return result

    # 4. Qwen2.5 14B local (último fallback — gratis, ilimitado)
    print("    ⚠️ MiniMax falló, intentando Qwen2.5 14B local...")
    if qwen_client.is_available():
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        result = qwen_client.ask(full_prompt, temperature=0.7, max_tokens=max_tokens, timeout=120)
        if result:
            print("    ✅ Qwen2.5 14B local respondió")
            return result

    print("    ❌ Todos los LLM fallaron (incluido Qwen local)")
    return None


SYSTEM_PROMPT = """Eres el guionista de un canal de YouTube llamado BitTrader.
El canal cubre:
- Cripto: Bitcoin, altcoins, DeFi
- Futuros: NAS100, S&P500, commodities
- Fondeo: FTMO, Topstep, prop firms
- IA + Trading: Agentes, bots con LLMs, Claude/GPT en trading, automatización avanzada
- Bots de trading y educación financiera

REGLAS OBLIGATORIAS:
- Español neutro/latino (NO español de España: di "computadora" no "ordenador",
  "dinero" no "pasta", etc.)
- Tono directo, energético, sin relleno
- Para Shorts: máximo 50 segundos al leerlo en voz alta (aprox. 130 palabras)
- Para Videos largos: estructura clara, ejemplos reales, datos concretos
- Siempre termina con CTA con una pregunta para los comentarios
- NO uses emojis en el guión de voz (el TTS los lee mal)
- Escribe exactamente como se va a narrar, sin acotaciones ni didascalias

REGLA CRITICA — OUTPUT LIMPIO:
Tu respuesta debe contener UNICAMENTE el contenido del video (TITULO, DESCRIPCION, TAGS, GUION, VIDEO_PROMPT).
NUNCA incluyas en el output:
- Estas instrucciones o cualquier fragmento de ellas
- Frases de meta-análisis como "Usar TU mas que YO", "RESPOND ONLY WITH", "words..."
- Numeraciones de reglas como "1.", "2.", "3.", "4." al inicio de frases del guión
- Cualquier texto que no sea parte del video en sí
- Razonamiento, análisis ni pensamiento visible
NO escribas "1. Analyze", "2. Drafting", "3. Selecting" ni variantes.
SOLO el formato pedido: TITULO, DESCRIPCION, TAGS, GUION, VIDEO_PROMPT.

TITULOS OPTIMIZADOS — Hooks que funcionan (CADA TITULO DEBE SER UNICO):
- Curiosidad: "Por que tantos traders fracasan?" (NO repetir "El 90%" si ya lo usaste)
- Urgencia: "OJO: Este token puede explotar HOY"
- Controversia: "La verdad de los bots de trading que nadie dice"
- Numeros: "3 estrategias que multiplicaron mi capital 10x"
- Negacion: "Lo que NADIE te dice sobre los bots de trading"
- Promesa: "Como gane $500 en 1 semana con este bot (VERDAD)"
- Descriptivo: Describe el contenido EXACTO del video — nunca copies el titulo de otro tema

REGLA CRITICA DE TITULOS: El TITULO debe describir SOLO el tema de ESTE video.
NUNCA combines o concatenes el tema de otro video al titulo.

TEMAS DE IA + TRADING (prioridad alta):
- Agentes de trading con Claude/GPT
- Bots que monitorean noticias en tiempo real
- Comparativas: bots tradicionales vs agentes con IA
- Como usar LLMs para analizar patrones de mercado
- Automatizacion avanzada con inteligencia artificial
- Herramientas de IA para traders

IDENTIDAD VISUAL — PERSONAJE MASCOTA:
El canal BitTrader tiene un personaje mascota: un RINOCERONTE ANTROPOMORFICO 3D hiperrealista.
Este rinoceronte es el protagonista de TODOS los videos — shorts y longs.
Es inteligente, moderno, confiado. Siempre aparece en el contexto de la escena del guion.
TODOS los VIDEO_PROMPT deben incluir este personaje como sujeto principal."""

# ── Rhino character base prompt (appended to every video prompt) ───────────
RHINO_BASE = (
    "anthropomorphic rhinoceros character, hyper-realistic 3D render, "
    "muscular but elegant, wearing modern casual trading clothes, "
    "expressive face, dramatic cinematic lighting, "
    "ultra HD, photorealistic textures, "
    "9:16 vertical aspect ratio, dark moody background"
)

RHINO_BASE_LONG = (
    "anthropomorphic rhinoceros character, hyper-realistic 3D render, "
    "muscular but elegant, wearing modern casual trading clothes, "
    "expressive face, dramatic cinematic lighting, "
    "ultra HD, photorealistic textures, "
    "16:9 widescreen, dark moody background"
)


# ══════════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════════

def load_scout_data() -> dict:
    if not SCOUT_LATEST.exists():
        print("  ⚠️ No hay datos del Scout. Ejecuta scout.py primero.")
        return {}
    return json.loads(SCOUT_LATEST.read_text())


def build_content_plan(scout: dict) -> list:
    """Decide qué contenido crear basado en análisis del Scout."""
    plan = []
    topics  = scout.get("analysis", {}).get("recommended_topics", [])
    winners = scout.get("analysis", {}).get("top_winners", [])
    avoid   = scout.get("analysis", {}).get("saturated_avoid", [])
    gaps    = scout.get("analysis", {}).get("content_gaps", [])
    crypto  = scout.get("crypto", {})
    alert   = scout.get("alert")

    # Si hay alerta urgente → short de noticias inmediato
    if alert and alert.get("urgent"):
        plan.append({
            "type": "short", "priority": "urgente",
            "theme": "noticias",
            "topic": alert["alerts"][0],
            "crypto_data": crypto.get("bitcoin", {}),
        })

    # Priorizar temas de alta prioridad del Scout
    alta = [t for t in topics if t.get("priority") == "alta"]
    media = [t for t in topics if t.get("priority") == "media"]
    baja  = [t for t in topics if t.get("priority") == "baja"]

    added = set()
    for t in alta[:3] + media[:3] + baja[:1]:
        key = t["title"][:30]
        if key not in added:
            # Verificar que no sea tema saturado
            title_lower = t["title"].lower()
            if not any(sat in title_lower for sat in avoid):
                plan.append({
                    "type":     t.get("format", "short"),
                    "priority": t.get("priority", "media"),
                    "theme":    t.get("type", "educativo"),
                    "topic":    t["title"],
                    "crypto_data": crypto,
                })
                added.add(key)

    # Gaps de contenido → oportunidades únicas
    for gap in gaps[:2]:
        plan.append({
            "type": "short", "priority": "media",
            "theme": "educativo",
            "topic": f"¿Qué es {gap} y por qué está en tendencia ahora?",
            "crypto_data": crypto,
        })

    # 2 wildcards experimentales
    # ⚠️ BUG FIX (2026-03-25): Los wildcards con "El 90% pierde..." como prefijo
    # estaban causando que el LLM concatenase ese texto al título de OTRO video.
    # Solución: Usar temas que NO empiecen con el mismo prefijo repetitivo.
    # El prefijo "El 90% pierde" solo debe usarse UNA VEZ como máximo por ciclo.
    has_90_topic = any("90%" in item.get("topic", "").lower() or "pierde" in item.get("topic", "").lower() for item in plan)
    wildcards = [
        {"type": "short", "priority": "wildcard", "theme": "controversial",
         # Solo incluir el "90%" si no hay ya uno en el plan
         "topic": "¿Por qué el 90% de traders fracasa? La verdad que nadie dice" if not has_90_topic
                  else "Lo que diferencia a un trader rentable del resto"},
        {"type": "long",  "priority": "wildcard", "theme": "educativo",
         "topic": "Cómo usar un bot de trading sin saber programar (guía 2025)"},
    ]
    plan += wildcards

    # Calcular mix: 60% educativo, 20% controversial, 20% noticias
    # (ya está mezclado por los temas del scout + wildcards)
    return plan[:10]  # Máximo 10 piezas de contenido por ciclo


def winner_patterns_summary(scout: dict) -> str:
    patterns = scout.get("youtube", {}).get("title_patterns", {})
    if not patterns:
        return "Usa preguntas y números en los títulos."
    lines = []
    for pat, data in list(patterns.items())[:5]:
        lines.append(f"- {pat}: promedio {data.get('avg_views',0):.0f} vistas (n={data.get('count',0)})")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# SCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_short_script(item: dict, scout: dict) -> dict:
    """Genera guión completo para un Short."""
    btc   = scout.get("crypto", {}).get("bitcoin", {})
    trend = [c["symbol"] for c in scout.get("crypto", {}).get("trending_coins", [])[:5]]
    patterns = winner_patterns_summary(scout)

    # Use MrBeast-enhanced prompt if available
    if MRBEAST_ENABLED:
        prompt = inject_mrbeast_short_prompt(item, scout)
    else:
        prompt = f"""Crea un guión para un YouTube Short.

TEMA: {item['topic']}
CATEGORÍA: {item['theme']}
BTC: ${btc.get('price_usd',0):,} ({btc.get('change_24h',0):+.1f}% 24h)

FORMATO (responde así):

TITULO: [título corto, max 50 chars]
DESCRIPCION: [descripción 100-150 chars]
TAGS: [8-10 tags]
GUION:
[Guión 100-130 palabras en español. Empieza con gancho. Termina con pregunta.]

VIDEO_PROMPT: [Escena con BitTrader rhino. Describe qué hace el rinoceronte relacionado al tema "{item['topic']}", el escenario, el ambiente. Hyper-realistic 3D, 9:16 vertical, cinematic lighting]"""

    print(f"    🤖 Generando short: {item['topic'][:50]}...")
    raw = call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=2000)  # Aumentado de 800 a 2000
    return parse_script_response(raw, "short", item)


def generate_long_script(item: dict, scout: dict) -> dict:
    """Genera guión completo para un Video Largo."""
    btc   = scout.get("crypto", {}).get("bitcoin", {})
    patterns = winner_patterns_summary(scout)

    # Use MrBeast-enhanced prompt if available
    if MRBEAST_ENABLED:
        prompt = inject_mrbeast_long_prompt(item, scout)
    else:
        prompt = f"""Crea un guión para un video de YouTube (3-8 minutos).

TEMA: {item['topic']}
CATEGORÍA: {item['theme']}
BTC: ${btc.get('price_usd',0):,} ({btc.get('change_24h',0):+.1f}% 24h)

FORMATO (responde así):

TITULO: [título corto, max 70 chars]
DESCRIPCION: [descripción 200-300 chars]
TAGS: [10-15 tags]

GUIÓN COMPLETO:
[Guión 400-700 palabras en español. Estructura: HOOK + PROBLEMA + EXPLICACION + EJEMPLOS + CTA]

VIDEO_PROMPT_1: [Escena intro con BitTrader rhino. Describe qué hace relacionado al hook "{item['topic']}". 16:9 horizontal]
VIDEO_PROMPT_2: [Escena principal con rhino mostrando la explicación del tema. 16:9 horizontal]
VIDEO_PROMPT_3: [Escena final CTA con rhino confiado mirando al futuro. 16:9 horizontal]"""

    print(f"    🤖 Generando video largo: {item['topic'][:50]}...")
    raw = call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=2000)
    return parse_script_response(raw, "long", item)


def _validate_script_content(guion: str, title: str, topic: str) -> tuple:
    """
    Valida que el guión generado sea contenido real y no basura del LLM.
    Retorna (is_valid: bool, reason: str)
    
    DETECTA:
    - Prompt leakage: respuesta cruda del LLM con instrucciones del sistema
    - Template leakage: estructura del prompt en vez del contenido
    - Respuesta vacía o demasiado corta
    - Guión en idioma incorrecto (inglés cuando debería ser español)
    """
    import re

    if not guion or len(guion.strip()) < 50:
        return False, "EMPTY_OR_TOO_SHORT"

    text = guion.strip()

    # ── Señales de prompt leakage ──────────────────────────────────────────
    CONTAMINATION_PATTERNS = [
        # LLM responding to the prompt instead of writing the script
        r"el usuario quiere",
        r"el usuario necesita",
        r"el usuario pide",
        r"RESPOND ONLY WITH",
        r"respond only with",
        r"Analyze the Request",
        r"analyze the request",
        r"\*\*Role:\*\*",
        r"\*\*Format:\*\*",
        r"You are a",
        r"you are a scriptwriter",
        # Template structure leaked
        r"~\d+ palabras",
        r"~\d+ words",
        r"Gancho inmediato",
        r"especifica: TITULO",
        r"especific[ao]: TITULO",
        r"TITULO,\s*DESCRIPCION,\s*TAGS",
        # Markdown instruction headers
        r"^\d+\.\s+\*\*Analyze",
        r"^\d+\.\s+\*\*Write",
        # System prompt fragments
        r"Eres el guionista",
        r"Eres un guionista",
        r"canal de YouTube llamado BitTrader",
        r"formato obligatorio",
        # Patterns that caused the SUI/DOT video bug (Mar 2026)
        r"necesito seguir todas las reglas",
        r"guión completo para un video",
        r"siguiendo el formato",
        r"sin texto adicional",
        r"para un video de YouTube de",
        r"3-5 minutos sobre",
        r"específico:\s*TITULO",
        r"VIDEO_PROMPT_\d+:",
        # ── NEW: Patterns from NEIRO/BTC contamination bug (25 marzo 2026) ──
        # MrBeast numbered rules leaking into narration
        r"usar [\"']?t[uú][\"']? m[aá]s que [\"']?yo[\"']?",
        r"usa [\"']?t[uú][\"']? m[aá]s que [\"']?yo[\"']?",
        # Isolated rule numbers at start of lines (e.g. "4." standing alone in narration)
        r"(?:^|\n)\s*\d+\.\s*(?:BUCLE|REGLA|INSTRUC|GANCHO|TITULO|HOOK|FORMATO)\b",
        # "RESPOND ONLY WITH THE" (exact phrase from BTC short bug)
        r"RESPOND ONLY WITH THE",
        r"respond only with the",
        r"words\.\.\.\s*RESPOND",
        r"words\s*\.\s*\.\s*\.\s*respond",
        # Any instruction-like numbered list within narration
        r"(?:^|\n)\s*\d+\.\s+(?:GANCHO|TITULO|DESCRIPCION|REGLA|INSTRUC[CK]ION|FORMATO)\b",
        # Explicit system/meta references
        r"system prompt",
        r"instruccion(?:es)? del sistema",
        r"Regla[s]?:",
        r"Instruccion[es]?:",
        r"Tu rol es",
        r"tu rol es",
        r"este es tu rol",
    ]

    for pattern in CONTAMINATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return False, f"PROMPT_CONTAMINATION: matched '{pattern[:40]}'"

    # ── Debe estar en español ──────────────────────────────────────────────
    spanish_words = ["que", "con", "para", "como", "este", "pero", "por", "una",
                     "los", "las", "del", "más", "también", "cuando", "todo"]
    english_only  = ["the ", "this ", "that ", "with ", "from ", "will ", "about "]

    spanish_count = sum(1 for w in spanish_words if f" {w} " in text.lower())
    english_count = sum(1 for w in english_only if w in text.lower())

    if english_count > 5 and spanish_count < 3:
        return False, f"WRONG_LANGUAGE: english_count={english_count}, spanish_count={spanish_count}"

    # ── Debe mencionar el tema o ser razonablemente largo ─────────────────
    topic_words = [w.lower() for w in topic.split() if len(w) > 3]
    topic_hits   = sum(1 for w in topic_words if w in text.lower())

    # If very long and no prompt contamination, consider valid even without topic match
    if len(text) > 500:
        return True, "OK"

    if topic_hits == 0 and len(text) < 200:
        return False, f"OFF_TOPIC: none of {topic_words[:3]} found in short script"

    return True, "OK"


def parse_script_response(raw: str, vtype: str, item: dict) -> dict:
    """Parsea la respuesta del LLM en estructura de datos."""
    import re

    # Paso 1: Extraer solo la sección de formato (última ocurrencia de TITULO hacia adelante)
    # Esto ignora el razonamiento visible que GLM-5 puede generar (si el modelo lo produce)
    formato_match = re.search(
        r'(?:^|\n)\s*TITULO\s*:.*$',
        raw,
        re.MULTILINE | re.DOTALL
    )

    if formato_match:
        # Extraer desde la última ocurrencia de TITULO hasta el final
        last_titulo_pos = raw.rfind('TITULO')
        if last_titulo_pos != -1:
            raw = raw[last_titulo_pos:]

    def extract_field(key: str, text: str) -> str:
        # Multiple patterns to find the field value
        patterns = [
            # Pattern 1: KEY: value (next line until next KEY)
            rf"{key}\s*:\s*(.+?)(?=\n[A-Z_]+\s*:|\Z)",
            # Pattern 2: **KEY:** value (with asterisks)
            rf"\*?{key}\*?\*?:\s*(.+?)(?=\n\*?[A-Z_]+\*?\*?:|\Z)",
            # Pattern 3: KEY\nvalue (colon on separate line)
            rf"{key}\s*\n\s*(.+?)(?=\n[A-Z_]+\s*:|\n\Z)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if m:
                result = m.group(1).strip()
                # Remove markdown bold
                result = re.sub(r'\*+', '', result)
                return result
        return ""

    title    = extract_field("TITULO", raw)
    desc     = extract_field("DESCRIPCION", raw)
    tags_raw = extract_field("TAGS", raw)
    tags     = [t.strip().strip("*") for t in tags_raw.split(",") if t.strip()]

    # Extract VIDEO_PROMPT(s) — do this BEFORE extracting guion so we can exclude them
    vp_matches = re.findall(r"VIDEO_PROMPT[_\d]*:\s*(.+?)(?=\nVIDEO_PROMPT|\n\*?\*?[A-Z_]+\*?\*?:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    rhino_base = RHINO_BASE if vtype == "short" else RHINO_BASE_LONG
    video_prompts = []
    for v in vp_matches:
        p = v.strip().strip("*").strip()
        # Inject rhino base style if the LLM didn't include the character
        if "rhinoceros" not in p.lower() and "rhino" not in p.lower():
            p = f"Anthropomorphic rhinoceros — {p}, {rhino_base}"
        elif "hyper-realistic" not in p.lower() and "3d render" not in p.lower():
            p = f"{p}, {rhino_base}"
        video_prompts.append(p)

    def clean_script_text(text: str) -> str:
        """Remove structural headers and video prompts from script text (TTS-ready).
        
        Strips: GUIÓN COMPLETO:, HOOK (30s):, PROBLEMA (45s):, EXPLICACION (2-3min):,
        CTA (30s):, VIDEO_PROMPT_N:, **bold headers**, etc.
        The output should be pure narration — ready to pass to TTS.
        """
        # Remove GUION/GUIÓN COMPLETO: header line
        text = re.sub(
            r'^GU[IÍ](?:O|Ó)N(?:\s+COMPLETO)?\s*:\s*\n?',
            '', text, flags=re.MULTILINE | re.IGNORECASE
        )
        # Remove section timing headers: HOOK (30s):, PROBLEMA (45s):, CTA (30s):, etc.
        text = re.sub(
            r'^[ \t]*\*?\*?\s*(?:HOOK|INTRO|INTRODUCTION|PROBLEMA|PROBLEM|EXPLICACION|EXPLANATION|'
            r'EJEMPLOS|EXAMPLES|CTA|CONCLUSION|OUTRO|CUERPO|BODY|DESARROLLO|SECTION)\s*'
            r'(?:\([^)]*\))?\s*\*?\*?\s*:[ \t]*$',
            '', text, flags=re.MULTILINE | re.IGNORECASE
        )
        # Remove VIDEO_PROMPT lines (single-line and multi-line inline)
        text = re.sub(r'^VIDEO_PROMPT[_\d]*:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        # Remove markdown bold header lines **SOMETHING** (standalone lines)
        text = re.sub(r'^\*\*[^*\n]+\*\*\s*$', '', text, flags=re.MULTILINE)
        # Collapse multiple blank lines into one
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # Extract GUION block — handle both "GUIÓN:" and "GUIÓN COMPLETO:" variants
    guion_m = re.search(
        r"GU[IÍ](?:O|Ó)N(?:\s+COMPLETO)?\s*:\s*\n(.*?)(?=\nVIDEO_PROMPT|\Z)",
        raw, re.DOTALL | re.IGNORECASE
    )
    if guion_m:
        guion = clean_script_text(guion_m.group(1))
    else:
        # Fallback: strip all known metadata fields and clean
        guion = re.sub(
            r'^\*?\*?(?:TITULO|DESCRIPCION|TAGS|VIDEO_PROMPT[_\d]*|'
            r'GU[IÍ](?:O|Ó)N(?:\s+COMPLETO)?)\*?\*?:.*\n?',
            '', raw, flags=re.MULTILINE | re.IGNORECASE
        )
        guion = clean_script_text(guion)

    if not title:
        title = item["topic"][:60]

    # ── BUG FIX (2026-03-25): Strip leaked prefixes from title ───────────
    # The LLM sometimes concatenates a previous topic's hook as a prefix.
    # Strip known repetitive prefixes that don't belong in every video title.
    LEAKED_PREFIXES = [
        "El 90% pierde con ",
        "El 90% pierde — ",
        "El 90% de traders pierde con ",
        "El 90% de traders pierde — ",
        "El 90% de traders pierde por ",
        "El 90% pierde por ",
        "Por qué el 90% pierde con ",
        "90% pierde con ",
    ]
    for prefix in LEAKED_PREFIXES:
        if title.lower().startswith(prefix.lower()):
            title = title[len(prefix):].strip()
            print(f"    ⚠️ Título con prefijo filtrado → '{title}'")
            break

    if not tags:
        tags = ["crypto", "trading", "bitcoin", "shorts", "finanzas"]

    # ── CONTENT VALIDATION: reject contaminated/empty scripts ────────────
    is_valid, reason = _validate_script_content(guion, title, item.get("topic",""))
    if not is_valid:
        print(f"    ❌ SCRIPT INVÁLIDO ({reason}) — marcando como error para re-generación")
        return {
            "id":             f"{vtype}_{int(time.time())}_{random.randint(100,999)}",
            "type":           vtype,
            "title":          item.get("topic", "unknown")[:60],
            "status":         "invalid_content",
            "error":          f"Script validation failed: {reason}",
            "raw_response":   raw[:500],  # Keep for debugging
            "original_topic": item["topic"],
            "created_at":     datetime.now(timezone.utc).isoformat(),
        }

    return {
        "id":            f"{vtype}_{int(time.time())}_{random.randint(100,999)}",
        "type":          vtype,
        "theme":         item.get("theme", "educativo"),
        "priority":      item.get("priority", "media"),
        "title":         title,
        "description":   desc,
        "tags":          tags,
        "script":        guion,
        "video_prompts": video_prompts,
        "original_topic": item["topic"],
        "status":        "pending",
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "content_validated": True,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_creator(dry_run: bool = False) -> dict:
    print("\n🎨 BitTrader Creator iniciando (Claude Sonnet 4.6 primary)...")

    scout = load_scout_data()
    if not scout and not dry_run:
        # Run without scout data using defaults
        print("  📋 Ejecutando sin datos de Scout (modo standalone)...")

    plan = build_content_plan(scout)
    print(f"  📋 Plan: {len(plan)} piezas de contenido")
    for i, item in enumerate(plan,1):
        print(f"    {i}. [{item['type'].upper()}] [{item['priority']}] {item['topic'][:60]}")

    if dry_run:
        print("\n  🔍 Modo dry-run: plan generado, no se llama al LLM")
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_file = DATA_DIR / f"guiones_{date_str}.json"
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date":         date_str,
            "plan":         plan,
            "scripts":      [],
            "dry_run":      True,
        }
        out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  ✅ Plan guardado → {out_file.name}")
        return result

    scripts = []
    import concurrent.futures
    for item in plan:
        try:
            if item["type"] == "short":
                gen_fn = lambda i=item: generate_short_script(i, scout)
            else:
                gen_fn = lambda i=item: generate_long_script(i, scout)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(gen_fn)
                try:
                    # Timeout dinámico: shorts 3 min, videos largos 5 min
                    timeout_seconds = 300 if item["type"] == "long" else 180
                    script = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    print(f"    ⏱️ Timeout generando '{item['topic'][:40]}' ({timeout_seconds}s) — saltando")
                    scripts.append({
                        "id":     f"timeout_{int(time.time())}",
                        "type":   item["type"],
                        "title":  item["topic"],
                        "status": "timeout",
                        "error":  f"LLM timeout after {timeout_seconds}s",
                    })
                    continue
                except Exception as e:
                    print(f"    ❌ Error generando '{item['topic'][:40]}': {e}")
                    scripts.append({
                        "id":     f"error_{int(time.time())}",
                        "type":   item["type"],
                        "title":  item["topic"],
                        "status": "error",
                        "error":  str(e)[:200],
                    })
                    continue

            scripts.append(script)
            print(f"    ✅ '{script['title'][:55]}'")
            
            # Save partial progress (crash protection) - ACCUMULATE instead of overwrite
            if scripts:
                _valid = [s for s in scripts if s.get("status") not in ("timeout","error")]
                if _valid:
                    _date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    # Read existing partial to accumulate
                    existing = {"scripts": []}
                    existing_file = DATA_DIR / "guiones_latest.json"
                    if existing_file.exists():
                        try:
                            existing = json.loads(existing_file.read_text())
                        except:
                            pass
                    # Merge: add new scripts that don't already exist (by title)
                    existing_titles = {s.get("title", "") for s in existing.get("scripts", [])}
                    for s in _valid:
                        if s.get("title", "") not in existing_titles:
                            existing["scripts"].append(s)
                    # Save accumulated
                    partial = {"generated_at": datetime.now(timezone.utc).isoformat(), "date": _date, "plan": plan_id if 'plan_id' in dir() else "unknown", "scripts": existing["scripts"], "partial": True}
                    existing_file.write_text(json.dumps(partial, indent=2, ensure_ascii=False))
            
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"    ❌ Error generando '{item['topic'][:40]}': {e}")
            scripts.append({
                "id":     f"error_{int(time.time())}",
                "type":   item["type"],
                "title":  item["topic"],
                "status": "error",
                "error":  str(e),
            })

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_file = DATA_DIR / f"guiones_{date_str}.json"
    latest   = DATA_DIR / "guiones_latest.json"

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date":         date_str,
        "plan":         plan,
        "scripts":      scripts,
        "stats": {
            "total":   len(scripts),
            "shorts":  sum(1 for s in scripts if s.get("type") == "short"),
            "longs":   sum(1 for s in scripts if s.get("type") == "long"),
            "errors":  sum(1 for s in scripts if s.get("status") == "error"),
        }
    }
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    latest.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n✅ Creator completado → {out_file.name}")
    print(f"   📝 {result['stats']['shorts']} shorts + {result['stats']['longs']} largos generados")
    if result["stats"]["errors"]:
        print(f"   ⚠️  {result['stats']['errors']} errores")
    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BitTrader Creator — Agente Estratega Creativo")
    parser.add_argument("--dry-run", action="store_true", help="Solo genera el plan, no llama al LLM")
    args = parser.parse_args()

    result = run_creator(dry_run=args.dry_run)

    print("\n── Scripts generados ─────────────────────")
    for s in result.get("scripts", []):
        status = "✅" if s.get("status") != "error" else "❌"
        print(f"  {status} [{s.get('type','?').upper()}] {s.get('title','?')[:60]}")
    print("─────────────────────────────────────────\n")
