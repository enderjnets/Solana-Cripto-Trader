#!/usr/bin/env python3
"""
Script para arreglar el estado del paper trading
Elimina campos innecesarios que causan crash en init
"""

import json
from pathlib import Path

STATE_FILE = Path("/home/enderj/.openclaw/workspace/solana-jupiter-bot/data/paper_trading_state.json")

def fix_state():
    """Arreglar estado eliminando campos problemáticos"""
    print("🔧 Arreglando estado del paper trading...")

    with open(STATE_FILE) as f:
        state = json.load(f)

    print("Estado actual:")
    for key in state.keys():
        print(f"  {key}: {type(state[key])}")

    # Eliminar campos duplicados o problemáticos
    fields_to_remove = ["last_trade_time", "daily_stats", "best_streak", "worst_streak"]  # Campos que causan problemas

    for field in fields_to_remove:
        if field in state:
            print(f"  ⚠️  Eliminando campo: {field}")
            del state[field]

    # Guardar estado arreglado
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print("\n✅ Estado arreglado guardado")

if __name__ == "__main__":
    fix_state()
