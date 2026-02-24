#!/usr/bin/env python3
"""
Cierre Manual de Posiciones Excedentes
Uso: python3 close_excess_positions.py [--all] [--symbol SYMBOL]
"""

import json
from pathlib import Path

PROJECT_DIR = Path("/home/enderj/.openclaw/workspace/solana-jupiter-bot")
STATE_FILE = PROJECT_DIR / "data" / "paper_trading_state.json"

def load_state():
    """Cargar estado"""
    with open(STATE_FILE) as f:
        return json.load(f)

def save_state(state):
    """Guardar estado"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def close_excess_positions():
    """Cerrar posiciones excedentes manteniendo solo las 5 más recientes"""
    state = load_state()
    trades = state.get("trades", [])

    print(f"Posiciones actuales: {len(trades)}")

    if len(trades) <= 5:
        print("✅ No hay posiciones excedentes")
        return

    # Mantener solo las 5 más recientes (por timestamp)
    trades_sorted = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
    trades_to_keep = trades_sorted[:5]
    trades_to_close = trades_sorted[5:]

    print(f"\nPosiciones a cerrar: {len(trades_to_close)}")

    for i, trade in enumerate(trades_to_close, 1):
        symbol = trade.get("symbol", "UNKNOWN")
        entry = trade.get("entry_price", 0)
        size = trade.get("size", 0)
        pnl_pct = trade.get("pnl_pct", 0)

        print(f"{i}. {symbol} - ${size:.2f} - P&L: {pnl_pct:+.1f}%")

    # Confirmar
    confirm = input("\n¿Cerrar estas posiciones? (yes/no): ")
    if confirm.lower() != "yes":
        print("❌ Cancelado")
        return

    # Actualizar estado
    state["trades"] = trades_to_keep

    # Actualizar stats
    stats = state.get("stats", {})
    stats["total_trades"] = len(trades_to_keep) + stats.get("winning_trades", 0) + stats.get("losing_trades", 0)

    # Guardar
    save_state(state)

    print(f"\n✅ Posiciones cerradas")
    print(f"Posiciones restantes: {len(trades_to_keep)}")

if __name__ == "__main__":
    close_excess_positions()
