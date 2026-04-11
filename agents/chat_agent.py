#!/usr/bin/env python3
"""
Chat Agent v2 — Panel de Control Inteligente del Solana Cripto Trader.

Features:
- Onboarding: se presenta, pregunta nombre, capital, perfil de riesgo
- Conoce TODO sobre el bot: estrategia, parámetros, historial, market data
- Puede ajustar configuración según perfil de riesgo del usuario
- Responde preguntas sobre posiciones, trades, mercado, estrategia
- Proactivo: sugiere mejoras basadas en rendimiento
"""

import json
import os
import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

# ── Paths ──────────────────────────────────────────────────────────
AGENTS_DIR = Path(__file__).parent
DATA_DIR = AGENTS_DIR / "data"
NOTES_FILE = DATA_DIR / "agent_notes.json"
STATE_FILE = DATA_DIR / "chat_agent_state.json"
PROFILE_FILE = DATA_DIR / "user_profile.json"
LOG_FILE = Path("/home/enderj/.config/solana-jupiter-bot/chat_agent.log")

# ── Risk Profiles ────────────────────────────────────────────────
RISK_PROFILES = {
    "conservador": {
        "sl_pct": 0.03, "tp_pct": 0.06, "leverage_tier": 0,
        "risk_per_trade": 0.01, "max_positions": 3,
        "trailing_stop_pct": 3.0, "trailing_pct": 2.0,
        "description": "Bajo riesgo. SL 3%, TP 6%, leverage 3x, max 3 posiciones. Drawdown max 5%."
    },
    "moderado": {
        "sl_pct": 0.025, "tp_pct": 0.08, "leverage_tier": 1,
        "risk_per_trade": 0.015, "max_positions": 4,
        "trailing_stop_pct": 5.0, "trailing_pct": 3.0,
        "description": "Balance riesgo/retorno. SL 2.5%, TP 8%, leverage 5x, max 4 posiciones. Drawdown max 10%."
    },
    "agresivo": {
        "sl_pct": 0.02, "tp_pct": 0.10, "leverage_tier": 2,
        "risk_per_trade": 0.025, "max_positions": 6,
        "trailing_stop_pct": 8.0, "trailing_pct": 5.0,
        "description": "Alto riesgo/retorno. SL 2%, TP 10%, leverage 7x, max 6 posiciones. Drawdown max 15%."
    },
}

# ── Logging ──────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except:
        pass

# ── State/Notes ──────────────────────────────────────────────────
def load_state():
    try: return json.loads(STATE_FILE.read_text())
    except: return {"last_user_ts": "", "processing": False}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def load_notes():
    try: return json.loads(NOTES_FILE.read_text())
    except: return {"messages": [], "last_updated": None}

def load_archive():
    try: return json.loads((DATA_DIR / "agent_notes_archive.json").read_text())
    except: return []

def save_notes(notes):
    notes["last_updated"] = datetime.now(timezone.utc).isoformat()
    NOTES_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2))

def add_agent_response(text):
    notes = load_notes()
    notes.setdefault("messages", []).append({
        "sender": "agent", "text": text,
        "ts": datetime.now(timezone.utc).isoformat()
    })
    save_notes(notes)
    log(f"Respuesta enviada: {text[:80]}...")

# ── User Profile ─────────────────────────────────────────────────
def load_profile():
    try: return json.loads(PROFILE_FILE.read_text())
    except: return {}

def save_profile(profile):
    PROFILE_FILE.write_text(json.dumps(profile, indent=2, default=str))

# ── Trading Context (COMPLETO) ───────────────────────────────────
def get_full_context():
    """Lee TODOS los datos del sistema para respuestas informadas."""
    ctx = {}
    try:
        # Portfolio
        port = json.loads((DATA_DIR / "portfolio.json").read_text())
        positions = [p for p in port.get("positions", []) if p.get("status") == "open"]
        capital = port.get("capital_usd", 0)
        initial = port.get("initial_capital", 100)
        invested = sum(p.get("margin_usd", 0) for p in positions)
        unrealized = sum(p.get("pnl_usd", 0) for p in positions)
        equity = capital + invested + unrealized
        ctx["capital"] = round(capital, 2)
        ctx["initial"] = round(initial, 2)
        ctx["equity"] = round(equity, 2)
        ctx["return_pct"] = round((equity - initial) / initial * 100, 2) if initial > 0 else 0
        ctx["positions"] = []
        for p in positions:
            ctx["positions"].append({
                "symbol": p.get("symbol"), "direction": p.get("direction"),
                "entry": p.get("entry_price", 0), "current": p.get("current_price", 0),
                "pnl_usd": round(p.get("pnl_usd", 0), 4),
                "pnl_pct": round(p.get("pnl_pct", 0), 2),
                "margin": round(p.get("margin_usd", 0), 2),
                "leverage": p.get("leverage", 5),
                "strategy": p.get("strategy", "?"),
                "time_open": p.get("open_time", "")[:16],
            })

        # Trade History
        th = json.loads((DATA_DIR / "trade_history.json").read_text())
        trades = th.get("trades", th) if isinstance(th, dict) else th
        wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
        losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
        ctx["total_trades"] = len(trades)
        ctx["wins"] = len(wins)
        ctx["losses"] = len(losses)
        ctx["win_rate"] = round(len(wins) / len(trades) * 100, 1) if trades else 0
        ctx["realized_pnl"] = round(sum(t.get("pnl_usd", 0) for t in trades), 2)
        ctx["avg_win"] = round(sum(t["pnl_usd"] for t in wins) / len(wins), 2) if wins else 0
        ctx["avg_loss"] = round(sum(t["pnl_usd"] for t in losses) / len(losses), 2) if losses else 0

        # Auto Learner Params
        al = json.loads((DATA_DIR / "auto_learner_state.json").read_text())
        ctx["params"] = al.get("params", {})
        ctx["avoid"] = al.get("tokens_to_avoid", [])
        ctx["prefer"] = al.get("tokens_to_prefer", [])

        # Market Data
        mkt = json.loads((DATA_DIR / "market_latest.json").read_text())
        fg = mkt.get("fear_greed", {})
        ctx["fear_greed"] = fg.get("value", 50) if isinstance(fg, dict) else int(fg or 50)
        ctx["fg_label"] = fg.get("label", "Neutral") if isinstance(fg, dict) else "?"
        ctx["tokens_tracked"] = len(mkt.get("tokens", {}))

        # User Profile
        ctx["profile"] = load_profile()

    except Exception as e:
        log(f"Context error: {e}")

    return ctx


# ── Config Modification ─────────────────────────────────────────
def apply_risk_profile(profile_name):
    """Aplica un perfil de riesgo predefinido."""
    if profile_name not in RISK_PROFILES:
        return f"Perfil '{profile_name}' no existe. Opciones: conservador, moderado, agresivo."

    profile = RISK_PROFILES[profile_name]
    try:
        al_fp = DATA_DIR / "auto_learner_state.json"
        al = json.loads(al_fp.read_text())
        for key, value in profile.items():
            if key != "description" and key in al.get("params", {}):
                al["params"][key] = value
        al_fp.write_text(json.dumps(al, indent=4, default=str))

        # Update user profile
        up = load_profile()
        up["risk_profile"] = profile_name
        up["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_profile(up)

        return f"✅ Perfil '{profile_name}' aplicado:\n{profile['description']}\n\nParámetros actualizados. Se aplicarán en el próximo ciclo."
    except Exception as e:
        return f"Error aplicando perfil: {e}"


def modify_param(param, value):
    """Modifica un parámetro específico del auto_learner."""
    try:
        al_fp = DATA_DIR / "auto_learner_state.json"
        al = json.loads(al_fp.read_text())

        if param in al.get("params", {}):
            al["params"][param] = value
            al_fp.write_text(json.dumps(al, indent=4, default=str))
            return f"✅ {param} = {value}. Se aplica en el próximo ciclo."
        elif param == "avoid":
            avoid = al.get("tokens_to_avoid", [])
            if value.upper() not in avoid:
                avoid.append(value.upper())
                al["tokens_to_avoid"] = avoid
                al["params"]["tokens_to_avoid"] = avoid
                al_fp.write_text(json.dumps(al, indent=4, default=str))
                return f"✅ {value.upper()} agregado a lista de evitar."
            return f"{value.upper()} ya está en la lista de evitar."
        elif param == "prefer":
            prefer = al.get("tokens_to_prefer", [])
            if value.upper() not in prefer:
                prefer.append(value.upper())
                al["tokens_to_prefer"] = prefer
                al_fp.write_text(json.dumps(al, indent=4, default=str))
                return f"✅ {value.upper()} agregado a preferidos."
            return f"{value.upper()} ya está en preferidos."
        else:
            return f"Parámetro '{param}' no reconocido."
    except Exception as e:
        return f"Error: {e}"


# ── Command Processing ───────────────────────────────────────────
def process_command(text, ctx):
    """Procesa comandos del usuario. Retorna respuesta o None si no es comando."""
    t = text.lower().strip()

    # Onboarding check
    profile = ctx.get("profile", {})
    if not profile.get("onboarding_complete"):
        return handle_onboarding(text, profile)

    # Risk profile change
    for p in RISK_PROFILES:
        if p in t and ("perfil" in t or "profile" in t or "risk" in t or "cambia" in t):
            return apply_risk_profile(p)

    # Parameter changes
    sl_match = re.search(r'sl\s*(?:en\s*)?(\d+(?:\.\d+)?)\s*%', t)
    if sl_match:
        return modify_param("sl_pct", float(sl_match.group(1)) / 100)

    tp_match = re.search(r'tp\s*(?:en\s*)?(\d+(?:\.\d+)?)\s*%', t)
    if tp_match:
        return modify_param("tp_pct", float(tp_match.group(1)) / 100)

    leverage_match = re.search(r'leverage\s*(?:en\s*)?(\d+)', t)
    if leverage_match:
        lev = int(leverage_match.group(1))
        tier = 0 if lev <= 3 else 1 if lev <= 5 else 2
        return modify_param("leverage_tier", tier)

    maxpos_match = re.search(r'(?:max\s*pos|posiciones?\s*max)\s*(?:en\s*)?(\d+)', t)
    if maxpos_match:
        return modify_param("max_positions", int(maxpos_match.group(1)))

    if "evita" in t or "avoid" in t:
        tokens = re.findall(r'\b([A-Z]{2,10})\b', text.upper())
        if tokens:
            return modify_param("avoid", tokens[-1])

    if "prefer" in t or "favorit" in t:
        tokens = re.findall(r'\b([A-Z]{2,10})\b', text.upper())
        if tokens:
            return modify_param("prefer", tokens[-1])

    return None  # Not a command


def handle_onboarding(text, profile):
    """Maneja el flujo de onboarding paso a paso."""
    step = profile.get("onboarding_step", 0)

    if step == 0:
        # First message - ask name
        profile["onboarding_step"] = 1
        save_profile(profile)
        return (
            "🤖 ¡Hola! Soy tu **Agente de Trading de Solana**.\n\n"
            "Estoy aquí para ayudarte a configurar y monitorear tu bot de trading. "
            "Antes de empezar, me gustaría conocerte.\n\n"
            "📝 **¿Cuál es tu nombre?**"
        )

    elif step == 1:
        # Got name - explain risks and ask capital
        name = text.strip().split()[0].capitalize()
        profile["name"] = name
        profile["onboarding_step"] = 2
        save_profile(profile)
        return (
            f"¡Encantado, **{name}**! 🤝\n\n"
            "⚠️ **IMPORTANTE sobre riesgos:**\n"
            "• El trading de criptomonedas puede resultar en pérdida de capital\n"
            "• Este bot está en modo **paper trading** (simulación con dinero virtual)\n"
            "• Los resultados pasados no garantizan resultados futuros\n"
            "• Nunca inviertas más de lo que puedas permitirte perder\n\n"
            "💰 **¿Con cuánto capital quieres operar?** (ej: 100, 500, 1000)"
        )

    elif step == 2:
        # Got capital - ask risk profile
        cap_match = re.search(r'(\d+)', text)
        capital = int(cap_match.group(1)) if cap_match else 100
        profile["capital"] = capital
        profile["onboarding_step"] = 3
        save_profile(profile)
        return (
            f"Capital: **${capital}** ✅\n\n"
            "🎯 Última pregunta: **¿Cuál es tu tolerancia al riesgo?**\n\n"
            "1️⃣ **Conservador** — Proteger capital ante todo.\n"
            f"   • SL 3%, TP 6%, leverage 3x, max 3 posiciones\n"
            f"   • Con ${capital}: riesgo ~${capital*0.01:.0f}/trade, max pérdida ~${capital*0.05:.0f}\n\n"
            "2️⃣ **Moderado** — Balance riesgo/retorno.\n"
            f"   • SL 2.5%, TP 8%, leverage 5x, max 4 posiciones\n"
            f"   • Con ${capital}: riesgo ~${capital*0.015:.0f}/trade, max pérdida ~${capital*0.10:.0f}\n\n"
            "3️⃣ **Agresivo** — Máximo retorno, acepto volatilidad.\n"
            f"   • SL 2%, TP 10%, leverage 7x, max 6 posiciones\n"
            f"   • Con ${capital}: riesgo ~${capital*0.025:.0f}/trade, max pérdida ~${capital*0.15:.0f}\n\n"
            "Escribe **1**, **2** o **3**"
        )

    elif step == 3:
        # Got risk profile - apply and complete
        t = text.strip().lower()
        if "1" in t or "conserv" in t:
            profile_name = "conservador"
        elif "3" in t or "agresiv" in t:
            profile_name = "agresivo"
        else:
            profile_name = "moderado"

        profile["risk_profile"] = profile_name
        profile["onboarding_complete"] = True
        profile["created_at"] = datetime.now(timezone.utc).isoformat()
        save_profile(profile)

        result = apply_risk_profile(profile_name)
        name = profile.get("name", "Trader")

        return (
            f"🎉 ¡Perfecto, {name}! Tu perfil está configurado:\n\n"
            f"👤 Nombre: **{name}**\n"
            f"💰 Capital: **${profile.get('capital', 100)}**\n"
            f"🎯 Perfil: **{profile_name.upper()}**\n"
            f"📊 {RISK_PROFILES[profile_name]['description']}\n\n"
            f"{result}\n\n"
            "Ahora puedes preguntarme lo que quieras:\n"
            "• *¿Cómo va el trading?*\n"
            "• *¿Qué posiciones tengo?*\n"
            "• *Cambia mi perfil a agresivo*\n"
            "• *Explícame la estrategia*\n"
            "• *¿Cómo va el mercado?*"
        )

    return None


# ── Build Enhanced Prompt ────────────────────────────────────────
SYSTEM_PROMPT = dedent("""\
    Eres el Agente de Trading Inteligente del Solana Cripto Trader.
    Tu nombre es "Solana Trading Agent". Hablas en español.

    ## USUARIO
    Nombre: {user_name}
    Perfil de riesgo: {risk_profile}
    Capital: ${capital}

    ## ESTADO ACTUAL DEL BOT
    - Equity: ${equity} (retorno: {return_pct:+.2f}%)
    - Capital libre: ${capital_libre}
    - Posiciones abiertas: {n_positions}
    - Trades cerrados: {total_trades} (W:{wins} L:{losses}, WR:{win_rate:.1f}%)
    - P&L realizado: ${realized_pnl}
    - Fear & Greed: {fear_greed} ({fg_label})

    ## POSICIONES ABIERTAS
    {positions_text}

    ## PARÁMETROS ACTUALES
    - SL: {sl_pct:.1f}% | TP: {tp_pct:.1f}% | R:R: 1:{rr:.1f}
    - Leverage tier: {leverage_tier} (0=3x, 1=5x, 2=7x)
    - Max posiciones: {max_positions}
    - Risk/trade: {risk_per_trade:.1f}%
    - Tokens evitar: {avoid}
    - Tokens preferir: {prefer}

    ## ARQUITECTURA DEL BOT
    - Ejecuta ciclos cada 10-60 segundos
    - Usa Jupiter API para precios de Solana tokens
    - Simula Drift Protocol (perpetuos con leverage)
    - AI Strategy genera señales con GPT-5.4 (fallback MiniMax M2.7)
    - Risk Manager evalúa posiciones con LLM cada ciclo
    - Auto-Learner adapta parámetros cada 5 trades
    - Trailing stop progresivo (se tightea con profit)
    - Breakeven stop (SL a entry después de +1% por 30min)
    - Partial profit taking (50% al llegar a 50% del TP)

    ## TUS CAPACIDADES
    - Puedes cambiar perfil de riesgo (conservador/moderado/agresivo)
    - Puedes modificar parámetros: SL, TP, leverage, max posiciones
    - Puedes agregar/quitar tokens de avoid/prefer
    - Conoces todo sobre el mercado actual y el historial

    ## REGLAS
    - Responde en español, claro y directo
    - Para preguntas simples: máximo 2-3 oraciones
    - Para análisis: usa bullets y datos específicos
    - Si te piden cambiar config: confirma el cambio y explica el impacto
    - NUNCA inventes datos — usa solo los datos del contexto
    - Sé proactivo: si ves problemas, sugiere soluciones
""")


def build_prompt(user_msg, ctx):
    profile = ctx.get("profile", {})
    params = ctx.get("params", {})

    pos_text = "Sin posiciones abiertas."
    if ctx.get("positions"):
        lines = []
        for p in ctx["positions"]:
            sign = "+" if p["pnl_usd"] >= 0 else ""
            lines.append(f"  {p['symbol']} {p['direction'].upper()} | PnL: {sign}${p['pnl_usd']:.2f} ({p['pnl_pct']:.1f}%) | Margin: ${p['margin']:.2f} | {p['strategy']}")
        pos_text = "\n".join(lines)

    sl = params.get("sl_pct", 0.025) * 100
    tp = params.get("tp_pct", 0.08) * 100

    system = SYSTEM_PROMPT.format(
        user_name=profile.get("name", "Trader"),
        risk_profile=profile.get("risk_profile", "moderado"),
        capital=ctx.get("capital", 0),
        equity=ctx.get("equity", 0),
        return_pct=ctx.get("return_pct", 0),
        capital_libre=ctx.get("capital", 0),
        n_positions=len(ctx.get("positions", [])),
        total_trades=ctx.get("total_trades", 0),
        wins=ctx.get("wins", 0),
        losses=ctx.get("losses", 0),
        win_rate=ctx.get("win_rate", 0),
        realized_pnl=ctx.get("realized_pnl", 0),
        fear_greed=ctx.get("fear_greed", 50),
        fg_label=ctx.get("fg_label", "?"),
        positions_text=pos_text,
        sl_pct=sl, tp_pct=tp,
        rr=tp/sl if sl > 0 else 0,
        leverage_tier=params.get("leverage_tier", 1),
        max_positions=params.get("max_positions", 4),
        risk_per_trade=params.get("risk_per_trade", 0.015) * 100,
        avoid=ctx.get("avoid", []),
        prefer=ctx.get("prefer", []),
    )

    notes = load_notes()
    msgs_current = notes.get("messages", [])
    if len(msgs_current) < 4:
        archive = load_archive()
        if archive:
            prev = archive[-1].get("messages", [])[-4:]
            msgs = (prev + msgs_current)[-8:]
        else:
            msgs = msgs_current[-6:]
    else:
        msgs = msgs_current[-6:]
    history = "\n".join(f"[{'USER' if m.get('sender')=='user' else 'AGENT'}]: {m.get('text','')[:200]}"
                        for m in msgs if m.get('text'))

    return f"{system}\n\nHISTORIAL:\n{history}\n\nUSER: {user_msg}\n\nAGENT:"


# ── LLM Config ──────────────────────────────────────────────────
try:
    import json as _json
    _keys = _json.loads(open("/home/enderj/.openclaw/workspace/bittrader/keys/minimax.json").read())
    MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", _keys["minimax_api_key"])
except Exception:
    MINIMAX_KEY = ""
MINIMAX_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-M2.7"


def get_response(user_msg):
    """Obtiene respuesta usando Gemma 4 local con fallback a MiniMax."""
    ctx = get_full_context()

    # Check for commands first
    cmd_result = process_command(user_msg, ctx)
    if cmd_result:
        return cmd_result

    prompt = build_prompt(user_msg, ctx)

    # Try Gemma 4 local first
    try:
        from ollama_client import query_gemma4
        system = prompt.split("\n\nHISTORIAL:")[0]
        gemma_result = query_gemma4(prompt[:3000], system=system[:2000], max_tokens=300)
        if gemma_result:
            log("Respuesta via Gemma 4 local")
            return gemma_result[:500]
    except Exception as e:
        log(f"Gemma 4 fallback: {e}")

    # Fallback: MiniMax API
    try:
        import requests
        headers = {"Authorization": f"Bearer {MINIMAX_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": MINIMAX_MODEL, "max_tokens": 500,
            "messages": [
                {"role": "system", "content": "Responde en espanol. Max 3-4 oraciones."},
                {"role": "user", "content": prompt[:4000]}
            ],
            "temperature": 0.7
        }
        resp = requests.post(MINIMAX_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            choices = resp.json().get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "").strip()
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                if text:
                    return text[:500]
        return "Sin respuesta del modelo"
    except Exception as e:
        return f"Error: {str(e)[:100]}"


# ── Typing Indicator ─────────────────────────────────────────────
def set_thinking(thinking):
    state = load_state()
    state["thinking"] = thinking
    save_state(state)


# ── Main Loop ────────────────────────────────────────────────────
def main():
    log("Chat Agent v2 iniciado")
    set_thinking(False)

    while True:
        try:
            state = load_state()
            notes = load_notes()
            msgs = notes.get("messages", [])

            user_msgs = [m for m in msgs if m.get("sender") == "user"]
            if not user_msgs:
                # No messages yet - check if onboarding needed
                profile = load_profile()
                if not profile.get("onboarding_complete") and not profile.get("onboarding_step"):
                    # Auto-start onboarding
                    response = handle_onboarding("", {})
                    if response:
                        add_agent_response(response)
                time.sleep(1)
                continue

            last_user_msg = user_msgs[-1]
            last_ts = last_user_msg.get("ts", "")

            if last_ts == state.get("last_user_ts"):
                time.sleep(1)
                continue

            user_text = last_user_msg.get("text", "").strip()
            if not user_text:
                time.sleep(1)
                continue

            log(f"NUEVO MENSAJE: {user_text[:80]}")

            state["last_user_ts"] = last_ts
            state["processing"] = True
            state["thinking"] = True
            state["thinking_since"] = datetime.now(timezone.utc).isoformat()
            save_state(state)

            response = get_response(user_text)
            add_agent_response(response)

            state["processing"] = False
            state["thinking"] = False
            save_state(state)

        except Exception as e:
            log(f"ERROR: {e}")
            set_thinking(False)
            time.sleep(2)


if __name__ == "__main__":
    main()
