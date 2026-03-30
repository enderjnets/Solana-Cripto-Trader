#!/usr/bin/env python3
"""
Agent Inbox Monitor — checks agent_notes.json for new user messages
and triggers the CEO agent when Ender writes something.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents/data")
NOTES_FILE = DATA_DIR / "agent_notes.json"
STATE_FILE = DATA_DIR / "agent_inbox_state.json"

def load_notes():
    try:
        with open(NOTES_FILE) as f:
            return json.load(f)
    except:
        return {"messages": [], "last_updated": None}

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"last_seen_index": -1}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def main():
    notes = load_notes()
    state = load_state()
    msgs = notes.get("messages", [])
    
    last_index = state.get("last_seen_index", -1)
    
    # Find new user messages
    new_user_msgs = []
    for i, m in enumerate(msgs):
        if i > last_index and m.get("sender") == "user":
            new_user_msgs.append((i, m))
    
    if not new_user_msgs:
        print("No new messages. Silent.")
        return
    
    # Mark all as seen
    state["last_seen_index"] = len(msgs) - 1
    save_state(state)
    
    # Combine new messages
    combined = "\n".join([f"[{m['ts']}] Ender dice: {m['text']}" for _, m in new_user_msgs])
    print(f"NEW MESSAGE FROM ENDER:\n{combined}")
    print(f"\nTriggering CEO agent to respond...")
    
    # Trigger CEO agent via sessions_send to main session
    trigger_msg = f"""🚨 INBOX ALERT — Ender dejó un mensaje en el dashboard de BitTrader:

{combined}

ACCIÓN REQUERIDA:
1. Lee el mensaje completo de Ender desde /home/enderj/.openclaw/workspace/Solana-Cripto-Trader/agents/data/agent_notes.json
2. El mensaje es una DIRECTIVA para el equipo — reacciona accordingly
3. Responde SIEMPRE dejando una nota de respuesta en el archivo agent_notes.json
4. Agrega tu respuesta como mensaje con sender="agent" al archivo JSON
5. Si el mensaje requiere acción (producir videos, cambiar estrategia, etc.), ejecútala o inclúyela en tu plan

IMPORTANTE: Ender espera una respuesta directa del agente en el dashboard. Deja tu respuesta en agent_notes.json ANTES de terminar."""

    # Write trigger file for next CEO agent run
    trigger_file = DATA_DIR / "agent_inbox_trigger.json"
    with open(trigger_file, "w") as f:
        json.dump({
            "triggered_at": datetime.now().isoformat(),
            "messages": [m for _, m in new_user_msgs],
            "pending_response": True
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Trigger file written: {trigger_file}")
    sys.exit(0)

if __name__ == "__main__":
    main()
