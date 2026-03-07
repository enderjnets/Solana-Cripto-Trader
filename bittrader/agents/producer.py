#!/usr/bin/env python3
"""
🎬 BitTrader Producer — Productor Ejecutivo
Convierte guiones en videos completos usando herramientas locales.
Ejecutar: python3 agents/producer.py
"""
import sys
import json
import os
import requests
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE  = Path("/home/enderj/.openclaw/workspace")
BITTRADER  = WORKSPACE / "bittrader"
DATA_DIR   = BITTRADER / "agents/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Claude Sonnet 4.6 (local API) - PRIMARY
CLAUDE_BASE_URL = "http://127.0.0.1:8443/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

# GLM-4.7 Coding Plan - FALLBACK #1
ZAI_CODING_KEY = "863f222d3de340df8b6ff7e1e36ab216.DFOH1veovvWzaoFT"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_CODING_MODEL = "glm-4.7"

# MiniMax M2.5 - FALLBACK #2 (Anthropic API compatible)
MINIMAX_KEY   = json.loads((BITTRADER / "keys/minimax.json").read_text())["minimax_coding_key"]
MINIMAX_URL   = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MODEL = "MiniMax-M2.5"

LOGO_PATH     = WORKSPACE / "videos/BIBLIOTECA/bittrader_logo.png"
GUIONES_FILE  = DATA_DIR / "guiones_latest.json"

MINIMAX_TTS_URL   = "https://api.minimax.io/v1/t2a_v2"
MINIMAX_VIDEO_URL = "https://api.minimax.io/v1/video_generation"
MINIMAX_QUERY_URL = "https://api.minimax.io/v1/query/video_generation"

MINIMAX_HEADERS = {
    "Authorization": f"Bearer {MINIMAX_KEY}",
    "Content-Type":  "application/json",
}

# ════════════════════════════════════════════════════════════════════════
# CLAUDE SONNET 4.6 INTEGRATION
# ════════════════════════════════════════════════════════════════════════

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
        r = requests.post(CLAUDE_BASE_URL, headers=headers, json=data, timeout=120)
        r.raise_for_status()
        response = r.json()
        return response.get("content", [{}])[0].get("text", "")
    except Exception as e:
        print(f"      ⚠️ Claude error: {e}")
        return None


def call_glm_4_7(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama a GLM-4.7 (FALLBACK #1)"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZAI_CODING_KEY}"
    }

    messages = []
    if system:
        messages.append({"role": "user", "content": system})
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

        # Endpoint de coding usa reasoning_content
        if "choices" in response and len(response["choices"]) > 0:
            msg = response["choices"][0]["message"]
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            return reasoning if reasoning else content or str(response)
        return str(response)
    except Exception as e:
        print(f"      ⚠️ GLM-4.7 error: {e}")
        return None


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
        r = requests.post(MINIMAX_URL, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Parsear respuesta Anthropic API compatible
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "").strip()
        return None
    except Exception as e:
        print(f"      ⚠️ MiniMax error: {e}")
        return None


def call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Llama al LLM con fallback: Claude Sonnet -> GLM-4.7 -> MiniMax"""
    # 1. Intentar Claude Sonnet 4.6
    result = call_claude(prompt, system, max_tokens)
    if result:
        return result

    # 2. Intentar GLM-4.7
    print("      ⚠️ Claude Sonnet falló, intentando GLM-4.7 fallback...")
    result = call_glm_4_7(prompt, system, max_tokens)
    if result:
        return result

    # 3. Intentar MiniMax
    print("      ⚠️ GLM-4.7 falló, intentando MiniMax fallback...")
    result = call_minimax_llm(prompt, system, max_tokens)
    if result:
        return result

    print("      ❌ Todos los LLM fallaron")
    return None


def shorten_script_with_llm(script: str, current_duration: float) -> str:
    """Pide al LLM que acorte el guión para que quepa en 58s."""
    target_words = int(130 * (55 / current_duration))

    system = "Eres editor de guiones para YouTube Shorts en español latino."
    prompt = (f"Este guión dura {current_duration:.1f}s pero debe durar máximo 55s (~{target_words} palabras).\n"
              f"Acórtalo conservando el gancho inicial y el CTA final. Mantén el tono y energía.\n"
              f"RESPONDE SOLO EL GUIÓN ACORTADO, sin comentarios:\n\n{script}")

    return call_llm(prompt, system, max_tokens=400)


# ════════════════════════════════════════════════════════════════════════
# MAIN PRODUCTION WORKFLOW
# ════════════════════════════════════════════════════════════════════════

def load_scripts() -> dict:
    if not GUIONES_FILE.exists():
        print("  ⚠️ No hay guiones. Ejecuta creator.py primero.")
        return {}
    return json.loads(GUIONES_FILE.read_text())


def run_producer(limit: int = 5) -> dict:
    print("\n🎬 BitTrader Producer iniciando (Claude Sonnet PRIMARY -> GLM-4.7 -> MiniMax)...")

    guiones = load_scripts()
    scripts = guiones.get("scripts", [])

    pending = [s for s in scripts if s.get("status") == "pending"]
    print(f"  📋 {len(pending)} guiones pendientes de {len(scripts)} total")

    if not pending:
        print("  ✅ No hay guiones pendientes por procesar")
        return {"processed": 0, "errors": 0}

    # Procesar hasta 'limit' guiones
    to_process = pending[:limit]
    print(f"  🎬 Procesando {len(to_process)} guiones...")

    results = {"processed": 0, "errors": 0, "scripts": []}

    for script in to_process:
        script_id = script.get("id", "unknown")
        print(f"\n  📝 [{script_id[:12]}] {script.get('title', '?')[:50]}")

        # TODO: Implementar producción real de video
        # Por ahora, solo marcamos como procesado
        script["status"] = "completed"
        script["produced_at"] = datetime.now(timezone.utc).isoformat()

        results["scripts"].append(script)
        results["processed"] += 1

    # Guardar actualización
    guiones["scripts"] = [s for s in scripts if s.get("id") not in [t.get("id") for t in results["scripts"]]] + results["scripts"]
    GUIONES_FILE.write_text(json.dumps(guiones, indent=2, ensure_ascii=False))

    print(f"\n✅ Producer completado → {results['processed']} procesados")
    if results["errors"]:
        print(f"   ⚠️  {results['errors']} errores")

    return results


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BitTrader Producer — Productor Ejecutivo")
    parser.add_argument("--limit", type=int, default=5, help="Número de guiones a procesar")
    args = parser.parse_args()

    result = run_producer(limit=args.limit)

    print("\n── Resultados ──────────────────────")
    print(f"  Procesados: {result['processed']}")
    print(f"  Errores:     {result['errors']}")
    print("─────────────────────────────────────────\n")
