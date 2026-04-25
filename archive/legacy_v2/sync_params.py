#!/usr/bin/env python3
"""
Bridge: Lee auto_learner_state.json y actualiza master_orchestrator con los parámetros recomendados.
Ejecutar antes de iniciar el bot o cada vez que el auto-learner adapte.
"""
import json
from pathlib import Path

AGENTS_DIR = Path(__file__).parent / "agents" / "data"
CONFIG_DIR = Path.home() / ".config" / "solana-jupiter-bot"
MASTER_STATE = CONFIG_DIR / "master_state.json"
LEARNER_STATE = AGENTS_DIR / "auto_learner_state.json"
MASTER_ORCH = Path(__file__).parent / "master_orchestrator.py"

def sync():
    # Leer parámetros del auto-learner
    try:
        learner = json.loads(LEARNER_STATE.read_text())
    except Exception as e:
        print(f"⚠️ No se pudo leer auto_learner_state: {e}")
        return False
    
    params = learner.get("params", {})
    tokens_avoid = learner.get("tokens_to_avoid", [])
    tokens_prefer = learner.get("tokens_to_prefer", [])
    
    sl_pct = params.get("sl_pct", 0.025)  # default 2.5%
    tp_pct = params.get("tp_pct", 0.05)   # default 5%
    trailing_pct = params.get("trailing_stop_pct", 3.0)
    leverage_tier = params.get("leverage_tier", 1)
    
    # Map tier to actual leverage
    leverage_map = {0: 2.0, 1: 3.0, 2: 5.0}
    leverage = leverage_map.get(leverage_tier, 3.0)
    
    print(f"📊 Parámetros del auto-learner:")
    print(f"   SL: {sl_pct*100:.2f}% | TP: {tp_pct*100:.2f}% | Trail: {trailing_pct}% | Leverage: {leverage}x")
    print(f"   Evitar: {tokens_avoid}")
    print(f"   Preferir: {tokens_prefer}")
    
    # Actualizar constantes en master_orchestrator.py
    content = MASTER_ORCH.read_text()
    
    replacements = [
        ("STOP_LOSS_PCT = -1.5", f"STOP_LOSS_PCT = {-sl_pct*100}  # from auto_learner"),
        ("TAKE_PROFIT_PCT = 4.0", f"TAKE_PROFIT_PCT = {tp_pct*100}  # from auto_learner"),
        ("TRAILING_STOP_PCT = 2.0", f"TRAILING_STOP_PCT = {trailing_pct}  # from auto_learner"),
        ("LEVERAGE = 5.0", f"LEVERAGE = {leverage}  # from auto_learner (tier {leverage_tier})"),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"   ✅ Actualizado: {old[:30]}...")
    
    MASTER_ORCH.write_text(content)
    print("✅ Parámetros sincronizados al master_orchestrator")
    return True

if __name__ == "__main__":
    sync()
