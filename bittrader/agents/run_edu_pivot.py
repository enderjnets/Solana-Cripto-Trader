#!/usr/bin/env python3
"""
🎓 BitTrader Educational Pivot — 3 videos para principiantes hispanohablantes
Genera guiones y produce 3 videos educativos:
  - Short 1: "El error que cometen el 90% de nuevos traders (yo lo cometí)"
  - Short 2: "Stop Loss: la herramienta que te salva de perder todo"
  - Long:    "Cómo empezar en cripto con $100 — guía completa para principiantes 2026"

Y los agrega a upload_queue.json con horarios programados.
"""
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
OUTPUT_DIR = BITTRADER / "agents/output"
QUEUE_FILE = DATA_DIR / "upload_queue.json"

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
OUTPUT_TODAY = OUTPUT_DIR / TODAY
OUTPUT_TODAY.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════
# LLM — mismo setup que creator.py
# ══════════════════════════════════════════════════════
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL    = "claude-sonnet-4-6"
ZAI_KEY         = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_URL         = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_MODEL       = "glm-4.7"  # Cambiado de GLM-5 por rate limits (12 mar 2026)

try:
    MINIMAX_KEY = json.loads((BITTRADER / "keys/minimax.json").read_text())["minimax_coding_key"]
    MINIMAX_URL  = "https://api.minimax.io/anthropic/v1/messages"
    MINIMAX_MODEL = "MiniMax-M2.5"
    HAS_MINIMAX = True
except Exception:
    HAS_MINIMAX = False


def call_claude(prompt: str, system: str = "", max_tokens: int = 3000) -> str | None:
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    msgs = []
    if system:
        msgs.append({"role": "user", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        r = requests.post(CLAUDE_BASE_URL, headers=headers,
                          json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": msgs},
                          timeout=300)
        r.raise_for_status()
        return r.json().get("content", [{}])[0].get("text", "") or None
    except Exception as e:
        print(f"  ⚠️ Claude error: {e}")
        return None


def call_glm5(prompt: str, system: str = "", max_tokens: int = 3000) -> str | None:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ZAI_KEY}"}
    msgs = []
    if system:
        msgs.append({"role": "user", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        r = requests.post(ZAI_URL, headers=headers,
                          json={"model": ZAI_MODEL, "messages": msgs,
                                "max_tokens": max_tokens, "temperature": 0.8},
                          timeout=180)
        r.raise_for_status()
        resp = r.json()
        if "choices" in resp:
            msg = resp["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or None
        return None
    except Exception as e:
        print(f"  ⚠️ GLM-5 error: {e}")
        return None


def call_minimax(prompt: str, system: str = "", max_tokens: int = 3000) -> str | None:
    if not HAS_MINIMAX:
        return None
    headers = {"x-api-key": MINIMAX_KEY, "Content-Type": "application/json",
               "anthropic-version": "2023-06-01"}
    msgs = []
    if system:
        msgs.append({"role": "user", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        r = requests.post(MINIMAX_URL, headers=headers,
                          json={"model": MINIMAX_MODEL, "max_tokens": max_tokens, "messages": msgs},
                          timeout=180)
        r.raise_for_status()
        for block in r.json().get("content", []):
            if block.get("type") == "text":
                return block.get("text", "").strip() or None
        return None
    except Exception as e:
        print(f"  ⚠️ MiniMax error: {e}")
        return None


def call_qwen(prompt: str, system: str = "", max_tokens: int = 3000) -> str | None:
    try:
        import qwen_client
        if qwen_client.is_available():
            full = f"{system}\n\n{prompt}" if system else prompt
            return qwen_client.ask(full, temperature=0.8, max_tokens=max_tokens, timeout=180)
    except Exception as e:
        print(f"  ⚠️ Qwen error: {e}")
    return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 3000) -> str | None:
    r = call_claude(prompt, system, max_tokens)
    if r: return r
    print("  ↩️ Claude falló → GLM-5...")
    r = call_glm5(prompt, system, max_tokens)
    if r: return r
    print("  ↩️ GLM-5 falló → MiniMax...")
    r = call_minimax(prompt, system, max_tokens)
    if r: return r
    print("  ↩️ MiniMax falló → Qwen...")
    r = call_qwen(prompt, system, max_tokens)
    return r


# ══════════════════════════════════════════════════════
# SYSTEM PROMPT (adaptado para educación principiantes)
# ══════════════════════════════════════════════════════
SYSTEM_EDU = """Eres el guionista del canal de YouTube BitTrader (@bittrader9259).
Tu especialidad hoy: contenido educativo para principiantes en criptomonedas.

REGLAS:
- Español neutro latino (NO España — di "computadora" no "ordenador", "dinero" no "pasta")
- CERO jerga técnica — explica todo como si el espectador nunca oyó de cripto
- Tono amigable, cercano, como un amigo que ya sabe y te explica
- Usa analogías de la vida cotidiana para explicar conceptos difíciles
- CTA obligatorio al final: "Si estás aprendiendo cripto desde cero, suscríbete — subimos videos todos los días"
- NO uses emojis (el TTS los lee mal)
- Escribe SOLO el texto narrado — sin headers, sin acotaciones, sin "GANCHO:", "SECCIÓN:", etc.
- Responde DIRECTAMENTE con el guión — sin análisis previo ni "Voy a escribir..."

PERSONAJE MASCOTA:
El canal tiene un rinoceronte antropomórfico 3D hiperrealista como mascota.
Aparece en todos los videos. Incluye descripciones de escenas para el VIDEO_PROMPT."""


# ══════════════════════════════════════════════════════
# GENERADORES DE GUIONES
# ══════════════════════════════════════════════════════

def generate_short_1() -> dict:
    """Short: El error que cometen el 90% de nuevos traders"""
    prompt = """Escribe el guión COMPLETO para un YouTube Short de 45-58 segundos (máx 130 palabras).

TÍTULO: "El error que cometen el 90% de nuevos traders (yo lo cometí)"
AUDIENCIA: Personas que nunca han hecho trading, total principiantes
TEMA: El error de operar con dinero que no puedes perder (miedo → malas decisiones)

ESTRUCTURA (sin poner los headers — solo el texto):
- Hook 3s: frase que impacta de inmediato, habla de perder dinero
- Problema (15s): describe la situación (el principiante pone sus ahorros, tiene miedo, vende en pánico)
- Solución (25s): explica qué hacer — solo invertir lo que estás dispuesto a perder completamente, sin presión emocional
- CTA (5s): "Si estás aprendiendo cripto desde cero, suscríbete — subimos videos todos los días"

IMPORTANTE: Solo el texto narrado. Sin "HOOK:", sin "PROBLEMA:", nada de eso. Solo el guión natural.

Después del guión, escribe en una línea:
VIDEO_PROMPT: [Descripción en inglés de una escena con el rinoceronte BitTrader 3D, hiperrealista, ambiente cripto/trading, 9:16 vertical]"""

    print("  📝 Generando guión Short 1...")
    raw = call_llm(prompt, system=SYSTEM_EDU, max_tokens=1000)
    if not raw:
        print("  ❌ LLM falló para Short 1")
        return None

    # Parse response
    parts = raw.split("VIDEO_PROMPT:")
    script_text = parts[0].strip()
    video_prompt = parts[1].strip() if len(parts) > 1 else (
        "Anthropomorphic rhinoceros trader looking distressed at phone showing red crypto chart, "
        "then looking relieved and confident, 3D hyper-realistic, 9:16 vertical, dark studio"
    )

    return {
        "id": f"edu_short1_{int(time.time())}",
        "title": "El error que cometen el 90% de nuevos traders (yo lo cometí)",
        "type": "short",
        "script": script_text,
        "video_prompts": [video_prompt],
        "tags": ["cripto", "trading", "principiantes", "error traders", "bitcoin", "inversión",
                 "criptomonedas", "consejos trading", "como empezar cripto", "educacion financiera"],
        "description": "El error #1 que cometen casi todos los principiantes — y cómo evitarlo.",
        "status": "ready",
        "format": "short_9:16",
        "audience": "principiantes",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_short_2() -> dict:
    """Short: Stop Loss explicado para principiantes"""
    prompt = """Escribe el guión COMPLETO para un YouTube Short de 45-58 segundos (máx 130 palabras).

TÍTULO: "Stop Loss: la herramienta que te salva de perder todo"
AUDIENCIA: Personas que nunca han hecho trading, total principiantes
TEMA: Qué es un Stop Loss y por qué es tu mejor aliado

ESTRUCTURA (sin poner los headers — solo el texto):
- Hook 3s: pregunta impactante o dato impresionante sobre perder dinero sin stop loss
- Qué es (10s): explica stop loss con una analogía de la vida real (ej: cinturón de seguridad, seguro del coche)
- Por qué importa (15s): qué pasa si no lo usas — ejemplo concreto con números simples
- Ejemplo simple (15s): "Si compras Bitcoin a $50, pones un stop loss a $45 — si cae, vendes automáticamente y solo pierdes $5, no todo"
- CTA (5s): "Si estás aprendiendo cripto desde cero, suscríbete — subimos videos todos los días"

IMPORTANTE: Solo el texto narrado. Sin "HOOK:", sin secciones marcadas. Solo el guión natural y fluido.

Después del guión, escribe en una línea:
VIDEO_PROMPT: [Descripción en inglés de una escena con el rinoceronte BitTrader 3D, hiperrealista, ambiente cripto/trading, 9:16 vertical]"""

    print("  📝 Generando guión Short 2...")
    raw = call_llm(prompt, system=SYSTEM_EDU, max_tokens=1000)
    if not raw:
        print("  ❌ LLM falló para Short 2")
        return None

    parts = raw.split("VIDEO_PROMPT:")
    script_text = parts[0].strip()
    video_prompt = parts[1].strip() if len(parts) > 1 else (
        "Anthropomorphic rhinoceros trader confidently setting a stop loss on trading app, "
        "screen shows Bitcoin chart with protective barrier line, 3D hyper-realistic, 9:16 vertical, dark studio"
    )

    return {
        "id": f"edu_short2_{int(time.time())}",
        "title": "Stop Loss: la herramienta que te salva de perder todo",
        "type": "short",
        "script": script_text,
        "video_prompts": [video_prompt],
        "tags": ["stop loss", "cripto", "trading", "principiantes", "bitcoin", "gestión de riesgo",
                 "criptomonedas", "herramientas trading", "como operar cripto", "educacion financiera"],
        "description": "El Stop Loss es como el cinturón de seguridad del trading. Úsalo siempre.",
        "status": "ready",
        "format": "short_9:16",
        "audience": "principiantes",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_long() -> dict:
    """Long: Cómo empezar en cripto con $100"""
    prompt = """Escribe el guión COMPLETO para un video de YouTube de 8-12 minutos (1,400-1,800 palabras).

TÍTULO: "Cómo empezar en cripto con $100 — guía completa para principiantes 2026"
AUDIENCIA: Total principiantes — nunca han comprado cripto, no saben nada
OBJETIVO: Guiarlos paso a paso para dar sus primeros pasos seguros con solo $100

ESTRUCTURA (sin poner headers en el guión — flujo continuo natural):

1. INTRO (60s): Engancha con una historia relatable — alguien que tuvo $100 y no sabía nada de cripto. Presenta que al final del video van a saber exactamente cómo empezar.

2. SECCIÓN 1 — ¿Qué es cripto realmente? (120s): Explicación simple con analogías. Sin blockchain, sin "tecnología descentralizada". Explica como si fuera dinero digital que nadie controla, que puede subir y bajar.

3. SECCIÓN 2 — ¿Dónde comprar cripto? (120s): Los exchanges (Coinbase, Binance). Explica como una tienda donde compras cripto. Proceso simple: registro, verificación, depósito. Sin tecnicismos.

4. SECCIÓN 3 — ¿Qué comprar con $100? (120s): Bitcoin y Ethereum como los más seguros para principiantes. NO altcoins especulativas. Explica la estrategia DCA (Dollar Cost Averaging) como "comprar $10 cada semana en vez de $100 de un golpe".

5. SECCIÓN 4 — Errores que debes evitar (120s): No invertir más de lo que puedes perder, no vender por pánico, no seguir consejos de redes sociales sin investigar, usar stop loss.

6. OUTRO (60s): Resumen de los 4 pasos. CTA: "Si estás aprendiendo cripto desde cero, suscríbete — subimos videos todos los días. Y cuéntame en los comentarios: ¿ya tienes cripto o apenas vas a empezar?"

IMPORTANTE: 
- Solo el texto narrado — sin "SECCIÓN 1:", sin headers, sin acotaciones
- Tono amigable, como un amigo que ya sabe y te explica tomando un café
- Datos concretos pero simples (precios referenciales de 2026)
- Usa analogías cotidianas constantemente

Después del guión, escribe:
VIDEO_PROMPT: [Descripción en inglés de escena principal con rinoceronte BitTrader 3D, hiperrealista, 16:9 horizontal]
TAGS: [lista de 12-15 tags separados por comas, incluir: tutorial, completo, paso a paso, guía, 2026, principiantes]"""

    print("  📝 Generando guión Long (puede tardar hasta 3 min)...")
    raw = call_llm(prompt, system=SYSTEM_EDU, max_tokens=3500)
    if not raw:
        print("  ❌ LLM falló para Long")
        return None

    # Parse
    parts_vp = raw.split("VIDEO_PROMPT:")
    script_text = parts_vp[0].strip()

    video_prompt = (
        "Anthropomorphic rhinoceros trader at a professional desk with laptop showing crypto portfolio "
        "and $100 bill, confident welcoming expression, pointing at viewer, modern office, 16:9 widescreen, "
        "hyper-realistic 3D render, cinematic lighting"
    )
    tags = ["cripto principiantes", "tutorial cripto", "guía completa cripto", "cómo empezar cripto",
            "bitcoin principiantes", "paso a paso cripto", "cripto 2026", "inverisón cripto",
            "comprar bitcoin", "criptomonedas para principiantes", "tutorial completo",
            "educacion financiera", "trading principiantes", "cripto desde cero", "guía 2026"]

    if len(parts_vp) > 1:
        remainder = parts_vp[1]
        parts_tags = remainder.split("TAGS:")
        video_prompt = parts_tags[0].strip() or video_prompt
        if len(parts_tags) > 1:
            raw_tags = parts_tags[1].strip()
            parsed = [t.strip() for t in raw_tags.replace("\n", ",").split(",") if t.strip()]
            if len(parsed) >= 5:
                tags = parsed[:15]

    return {
        "id": f"edu_long_{int(time.time())}",
        "title": "Cómo empezar en cripto con $100 — guía completa para principiantes 2026",
        "type": "long",
        "script": script_text,
        "video_prompts": [video_prompt],
        "tags": tags,
        "description": (
            "Guía completa paso a paso para empezar en criptomonedas con solo $100. "
            "Sin tecnicismos, sin jerga — todo explicado desde cero para principiantes. "
            "Dónde comprar, qué comprar, cómo evitar los errores más comunes. "
            "#cripto #bitcoin #principiantes #tutorial #2026"
        ),
        "status": "ready",
        "format": "long_16:9",
        "audience": "principiantes",
        "tags_seo": ["tutorial", "completo", "paso a paso", "guía", "2026", "principiantes"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════
# PRODUCCIÓN DE VIDEO
# ══════════════════════════════════════════════════════

def produce_video(script: dict) -> dict:
    """Produce el video usando producer.py — retorna resultado."""
    from producer import produce_single

    sid = script["id"]
    vtype = script["type"]
    out_dir = OUTPUT_TODAY / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  🎬 Produciendo {vtype}: {script['title'][:55]}...")
    try:
        result = produce_single(script, out_dir, use_ai_video=True)
        if result and result.get("status") == "done":
            print(f"  ✅ Video listo: {result.get('output_file', '?')}")
        else:
            status = result.get("status", "unknown") if result else "None"
            err    = result.get("error", "unknown error") if result else "produce_single returned None"
            print(f"  ⚠️ Producción terminó con status={status}: {err}")
        return result or {"status": "error", "error": "produce_single returned None"}
    except Exception as e:
        print(f"  ❌ Excepción en produce_single: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ══════════════════════════════════════════════════════
# UPLOAD QUEUE
# ══════════════════════════════════════════════════════

def add_to_queue(script: dict, prod_result: dict, scheduled_iso: str) -> dict:
    """Agrega una entrada a upload_queue.json y retorna la entrada."""
    queue = []
    if QUEUE_FILE.exists():
        try:
            queue = json.loads(QUEUE_FILE.read_text())
        except Exception:
            queue = []

    output_file = prod_result.get("output_file") if prod_result else None

    entry = {
        "script_id": script["id"],
        "title": script["title"],
        "type": script["type"],
        "status": "pending_upload" if (output_file and Path(output_file).exists()) else "pending_production",
        "output_file": output_file,
        "scheduled_date": scheduled_iso,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tags": script.get("tags", []),
        "description": script.get("description", ""),
        "audience": script.get("audience", "principiantes"),
        "production_status": prod_result.get("status", "unknown") if prod_result else "skipped",
        "production_error": prod_result.get("error") if prod_result else None,
        "gate_passed": False,
        "gate_issues": [],
    }

    # Avoid duplicates
    queue = [q for q in queue if q.get("script_id") != script["id"]]
    queue.append(entry)

    QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))
    print(f"  📅 Agendado: {scheduled_iso} → {script['title'][:50]}")
    return entry


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🎓 BITTRADER EDUCATIONAL PIVOT")
    print("   3 videos para principiantes hispanohablantes")
    print("=" * 60)

    # Horarios (ISO 8601 UTC)
    # Short 1: hoy 12:00 PM MT = 18:00 UTC
    # Short 2: hoy 6:00 PM MT  = 00:00 UTC del 25
    # Long:    mañana 12:00 PM MT = 18:00 UTC del 25
    SCHEDULES = {
        "short1": "2026-03-24T18:00:00Z",
        "short2": "2026-03-25T00:00:00Z",
        "long":   "2026-03-25T18:00:00Z",
    }

    results = []
    errors  = []

    # ── VIDEO 1: Short — El error del 90% ──────────────────────────────
    print("\n[1/3] SHORT — El error que cometen el 90% de nuevos traders")
    print("-" * 55)
    script1 = generate_short_1()
    if not script1:
        errors.append({"video": "short1", "step": "script", "error": "LLM falló al generar guión"})
        print("  ⚠️ Continuando con video 2...")
    else:
        print(f"  📄 Guión generado: {len(script1['script'])} chars")
        print(f"  📝 Preview: {script1['script'][:150]}...")
        prod1 = produce_video(script1)
        entry1 = add_to_queue(script1, prod1, SCHEDULES["short1"])
        results.append({
            "video": "Short 1",
            "title": script1["title"],
            "script_chars": len(script1["script"]),
            "production_status": prod1.get("status") if prod1 else "skipped",
            "output_file": prod1.get("output_file") if prod1 else None,
            "scheduled": SCHEDULES["short1"],
            "queue_status": entry1["status"],
        })
        if prod1 and prod1.get("status") != "done":
            errors.append({"video": "short1", "step": "production",
                           "error": prod1.get("error", "unknown")})

    time.sleep(2)

    # ── VIDEO 2: Short — Stop Loss ──────────────────────────────────────
    print("\n[2/3] SHORT — Stop Loss: la herramienta que te salva de perder todo")
    print("-" * 55)
    script2 = generate_short_2()
    if not script2:
        errors.append({"video": "short2", "step": "script", "error": "LLM falló al generar guión"})
        print("  ⚠️ Continuando con video 3...")
    else:
        print(f"  📄 Guión generado: {len(script2['script'])} chars")
        print(f"  📝 Preview: {script2['script'][:150]}...")
        prod2 = produce_video(script2)
        entry2 = add_to_queue(script2, prod2, SCHEDULES["short2"])
        results.append({
            "video": "Short 2",
            "title": script2["title"],
            "script_chars": len(script2["script"]),
            "production_status": prod2.get("status") if prod2 else "skipped",
            "output_file": prod2.get("output_file") if prod2 else None,
            "scheduled": SCHEDULES["short2"],
            "queue_status": entry2["status"],
        })
        if prod2 and prod2.get("status") != "done":
            errors.append({"video": "short2", "step": "production",
                           "error": prod2.get("error", "unknown")})

    time.sleep(2)

    # ── VIDEO 3: Long — Cómo empezar con $100 ──────────────────────────
    print("\n[3/3] LONG — Cómo empezar en cripto con $100")
    print("-" * 55)
    script3 = generate_long()
    if not script3:
        errors.append({"video": "long", "step": "script", "error": "LLM falló al generar guión"})
    else:
        print(f"  📄 Guión generado: {len(script3['script'])} chars")
        print(f"  📝 Preview: {script3['script'][:150]}...")
        prod3 = produce_video(script3)
        entry3 = add_to_queue(script3, prod3, SCHEDULES["long"])
        results.append({
            "video": "Long",
            "title": script3["title"],
            "script_chars": len(script3["script"]),
            "production_status": prod3.get("status") if prod3 else "skipped",
            "output_file": prod3.get("output_file") if prod3 else None,
            "scheduled": SCHEDULES["long"],
            "queue_status": entry3["status"],
        })
        if prod3 and prod3.get("status") != "done":
            errors.append({"video": "long", "step": "production",
                           "error": prod3.get("error", "unknown")})

    # ── REPORT ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 REPORTE FINAL")
    print("=" * 60)

    for r in results:
        status_icon = "✅" if r["production_status"] == "done" else "⚠️"
        print(f"\n{status_icon} {r['video']}: {r['title'][:50]}")
        print(f"   Guión: {r['script_chars']} chars")
        print(f"   Producción: {r['production_status']}")
        print(f"   Archivo: {r['output_file'] or 'N/A'}")
        print(f"   Agendado: {r['scheduled']}")
        print(f"   Queue status: {r['queue_status']}")

    if errors:
        print(f"\n⚠️ ERRORES ENCONTRADOS ({len(errors)}):")
        for e in errors:
            print(f"   [{e['video']}] {e['step']}: {e.get('error', '?')[:100]}")
    else:
        print("\n✅ Sin errores")

    # Guardar reporte
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "type": "educational_pivot",
        "results": results,
        "errors": errors,
        "queue_file": str(QUEUE_FILE),
        "output_dir": str(OUTPUT_TODAY),
    }
    report_path = DATA_DIR / f"edu_pivot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n📄 Reporte guardado: {report_path}")

    return report


if __name__ == "__main__":
    main()
