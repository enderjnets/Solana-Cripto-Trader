#!/usr/bin/env python3
"""
Chat Agent — Responde mensajes del dashboard de trading en tiempo real.
Se ejecuta como proceso background, monitoreando agent_notes.json.
Cuando detecta un mensaje nuevo del usuario, responde inmediatamente.
"""

import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

# ── Paths ──────────────────────────────────────────────────────────
AGENTS_DIR = Path(__file__).parent
DATA_DIR = AGENTS_DIR / "data"
NOTES_FILE = DATA_DIR / "agent_notes.json"
STATE_FILE = DATA_DIR / "chat_agent_state.json"
LOG_FILE = Path("/home/enderj/.config/solana-jupiter-bot/chat_agent.log")

# ── Logging ──────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── State ────────────────────────────────────────────────────────
def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except:
        return {"last_user_ts": "", "processing": False}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ── Notes ────────────────────────────────────────────────────────
def load_notes():
    try:
        return json.loads(NOTES_FILE.read_text())
    except:
        return {"messages": [], "last_updated": None}

def save_notes(notes):
    notes["last_updated"] = datetime.now(timezone.utc).isoformat()
    NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2))

def add_agent_response(text):
    notes = load_notes()
    notes.setdefault("messages", []).append({
        "sender": "agent",
        "text": text,
        "ts": datetime.now(timezone.utc).isoformat()
    })
    save_notes(notes)
    log(f"Respuesta enviada: {text[:80]}...")

# ── Trading context ──────────────────────────────────────────────
def get_trading_context():
    """Obtiene contexto actual del bot para respuestas informadas."""
    try:
        with open(DATA_DIR / "portfolio.json") as f:
            port = json.load(f)
        with open(DATA_DIR / "trade_history.json") as f:
            th = json.load(f)
        trades = th.get("trades", th) if isinstance(th, dict) else th
        capital = port.get("capital_usd", 0)
        initial = port.get("initial_capital", 1000)
        ret_pct = ((capital - initial) / initial * 100) if initial > 0 else 0
        wins = len([t for t in trades if t.get("pnl_usd", 0) > 0])
        losses = len([t for t in trades if t.get("pnl_usd", 0) < 0])
        wr = wins / (wins + losses) * 100 if wins + losses > 0 else 0
        return {
            "capital": capital,
            "return_pct": ret_pct,
            "win_rate": wr,
            "positions": len([p for p in port.get("positions", []) if p.get("status") == "open"])
        }
    except Exception as e:
        log(f"Error context: {e}")
        return {}

# ── Chat response ────────────────────────────────────────────────
SYSTEM_PROMPT = dedent("""\
    Eres el Agente de Trading de Solana. Tu función es asistir al trader (Ender)
    con preguntas sobre el bot de trading, su rendimiento, posiciones, y cualquier
    duda técnica o de mercado.

    CONTEXTO ACTUAL: El bot está corriendo en paper trading mode.
    - Capital: ${capital}
    - Retorno: {return_pct:+.2f}%
    - Win Rate: {win_rate:.1f}%
    - Posiciones abiertas: {positions}

    REGLAS:
    - Responde en español, claro y directo
    - Máximo 2-3 oraciones para preguntas simples
    - Si pregunta por métricas, da números específicos
    - Si hay errores o problemas, sé proactivo en sugerir soluciones
    - NO inventes datos — si no lo sabes, di que lo verificas
    - El prefijo [AGENT] indica que eres el agente, no el usuario
""")

def build_prompt(user_msg, context):
    system = SYSTEM_PROMPT.format(**context)
    notes = load_notes()
    msgs = notes.get("messages", [])[-6:]  # últimos 6 mensajes para contexto
    history = "\n".join(f"[{'USER' if m.get('sender')=='user' else 'AGENT'}]: {m.get('text','')}"
                        for m in msgs if m.get('text'))
    return f"{system}\n\nHISTORIAL:\n{history}\n\nUSER: {user_msg}\n\nAGENT:"

# FIX: Use native MiniMax endpoint (same as trading LLM - faster, no timeout)
try:
    import json as _json
    _keys = _json.loads(open("/home/enderj/.openclaw/workspace/bittrader/keys/minimax.json").read_text())
    MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", _keys["minimax_api_key"])
except Exception:
    MINIMAX_KEY = "sk-cp-8tBIgoE2Vs8QE0AIoMjq4MTh8kiHtem3KWlOnNlAJZgKwAlYh_nt6oCq382Y0cmBi2buvch3nJJbMg7uqr_hIV6Z0ZqY3Q_qZ6AStHCUpKKT_IT-e0vEl4A"
MINIMAX_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"

def get_response(user_msg):
    """Obtiene respuesta usando Gemma 4 local (gratis) con fallback a MiniMax."""
    import requests as _req
    import re
    ctx = get_trading_context()
    prompt = build_prompt(user_msg, ctx)

    # Try Gemma 4 e2b local first (free, 0.3s, 100% GPU on ROG)
    try:
        from ollama_client import query_gemma4
        ctx = get_trading_context()
        system = f"Eres el agente de trading de Solana. Capital: ${ctx.get('capital',0):.0f}, Retorno: {ctx.get('return_pct',0):+.1f}%, WR: {ctx.get('win_rate',0):.0f}%, Pos: {ctx.get('positions',0)}. Responde en espanol, directo, max 3 oraciones."
        gemma_result = query_gemma4(prompt[:3000], system=system, max_tokens=200)
        if gemma_result:
            log(f"Respuesta via Gemma 4 local (gratis, 100% GPU)")
            return gemma_result[:500]
    except Exception as e:
        log(f"Gemma 4 fallback: {e}")

    # Fallback: MiniMax API
    try:
        headers = {
            "Authorization": f"Bearer {MINIMAX_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MINIMAX_MODEL,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": "Eres el agente de trading de Solana. Responde en espanol, claro y directo. Maximo 3-4 oraciones."},
                {"role": "user", "content": prompt[:4000]}
            ],
            "temperature": 0.7
        }
        resp = _req.post(MINIMAX_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            # Native API: choices[0].message.content
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "").strip()
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                if text:
                    return text[:500]
            # Fallback: Anthropic format
            content = data.get("content", [])
            if isinstance(content, list) and content:
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "").strip()[:500]
            return "Sin respuesta del modelo"
        else:
            return f"MiniMax {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        return f"Error: {str(e)[:100]}"


def set_thinking(thinking):
    """Escribe estado pensando para el dashboard."""
    state = load_state()
    state["thinking"] = thinking
    save_state(state)

# ── Main loop ────────────────────────────────────────────────────
def main():
    log("Chat Agent iniciado")
    set_thinking(False)

    while True:
        try:
            state = load_state()
            notes = load_notes()
            msgs = notes.get("messages", [])

            # Buscar último mensaje del usuario
            user_msgs = [m for m in msgs if m.get("sender") == "user"]
            if not user_msgs:
                time.sleep(1)
                continue

            last_user_msg = user_msgs[-1]
            last_ts = last_user_msg.get("ts", "")

            # ¿Es mensaje nuevo (no procesado)?
            if last_ts == state.get("last_user_ts"):
                time.sleep(1)
                continue

            # Nuevo mensaje detected
            user_text = last_user_msg.get("text", "").strip()
            if not user_text:
                time.sleep(1)
                continue

            log(f"NUEVO MENSAJE: {user_text[:80]}")

            # Marcar como procesado y pensando ATOMICAMENTE
            state["last_user_ts"] = last_ts
            state["processing"] = True
            state["thinking"] = True
            state["thinking_since"] = datetime.now(timezone.utc).isoformat()
            save_state(state)

            # Obtener respuesta (esto puede tardar 10-30s)
            response = get_response(user_text)

            # Guardar respuesta PRIMERO, luego limpiar thinking
            add_agent_response(response)

            # AHORA limpiar estado pensando (despues de que la respuesta esta guardada)
            state["processing"] = False
            state["thinking"] = False
            save_state(state)

        except Exception as e:
            log(f"ERROR: {e}")
            set_thinking(False)
            time.sleep(2)

if __name__ == "__main__":
    main()
