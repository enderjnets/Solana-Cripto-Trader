#!/usr/bin/env python3
"""
Script para resetear el paper trading a estado limpio
Mantiene el balance pero cierra todas las posiciones
"""

import json
from pathlib import Path
from datetime import datetime

STATE_FILE = Path("/home/enderj/.openclaw/workspace/solana-jupiter-bot/data/paper_trading_state.json")
BACKUP_FILE = STATE_FILE.parent / f"paper_trading_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def reset_paper_trading():
    """Resetear paper trading a estado limpio"""
    print("🔄 Reset Paper Trading...")
    print(f"Backup: {BACKUP_FILE}")

    # Cargar estado actual
    with open(STATE_FILE) as f:
        state = json.load(f)

    # Hacer backup
    import shutil
    shutil.copy(STATE_FILE, BACKUP_FILE)

    # Mantener balance pero cerrar posiciones
    current_balance = state.get("balance_usd", 700)
    current_stats = state.get("stats", {})

    # Crear estado limpio
    clean_state = {
        "balance_usd": current_balance,
        "trades": [],  # Cerrar todas las posiciones
        "stats": current_stats,  # Mantener estadísticas
        "last_trade_time": None,
        "daily_stats": {
            "total_pnl": 0,
            "trades_closed": 0,
            "winning_trades": 0,
            "losing_trades": 0
        },
        "best_streak": 0,
        "worst_streak": 0
    }

    # Guardar estado limpio
    with open(STATE_FILE, 'w') as f:
        json.dump(clean_state, f, indent=2)

    print(f"\n✅ Paper trading reseteado")
    print(f"   Balance: ${current_balance:,.2f}")
    print(f"   Posiciones: 0 (cerradas)")
    print(f"   Backup guardado en: {BACKUP_FILE}")

if __name__ == "__main__":
    reset_paper_trading()
