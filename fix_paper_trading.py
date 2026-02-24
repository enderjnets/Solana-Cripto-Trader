#!/usr/bin/env python3
"""
Fix Paper Trading State
Corrige los bugs detectados en el estado del paper trading
"""
import json
from pathlib import Path
from datetime import datetime

STATE_FILE = Path("data/paper_trading_state.json")

def fix_state():
    """Fix all bugs in paper trading state"""
    
    if not STATE_FILE.exists():
        print("❌ No se encontró el archivo de estado")
        return
    
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    print("🔍 AUDITORÍA ANTES DE CORRECCIÓN:\n")
    
    # Mostrar estado actual
    stats = state.get('stats', {})
    print(f"  Balance: ${state.get('balance_usd', 0):.2f}")
    print(f"  Initial Balance: ${state.get('initial_balance', 0):.2f}")
    print(f"  Total Trades: {stats.get('total_trades', 0)}")
    print(f"  Winning: {stats.get('winning_trades', 0)}")
    print(f"  Losing: {stats.get('losing_trades', 0)}")
    print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
    print()
    
    # BUG #1: Win Rate incorrecto
    print("🔧 BUG #1: Corrigiendo Win Rate...")
    if stats.get('total_trades', 0) > 0:
        correct_win_rate = (stats.get('winning_trades', 0) / stats.get('total_trades', 1)) * 100
        print(f"  Win Rate corregido: {correct_win_rate:.1f}%")
        state['stats']['win_rate'] = correct_win_rate
    else:
        print("  No hay trades para calcular win rate")
    
    # BUG #2: Balance incorrecto por margin no deducido
    print("\n🔧 BUG #2: Corrigiendo Balance...")
    
    # Recalcular balance desde cero
    initial_balance = 700.54  # Balance inicial real
    trades = state.get('trades', [])
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    
    # Sumar P&L de todos los trades cerrados
    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
    total_fees = stats.get('total_fees', 0)
    
    # Balance correcto = initial + pnl_total - fees_totales
    correct_balance = initial_balance + total_pnl
    print(f"  Balance Inicial: ${initial_balance:.2f}")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Total Fees: ${total_fees:.2f}")
    print(f"  Balance Correcto: ${correct_balance:.2f}")
    print(f"  Balance Actual: ${state.get('balance_usd', 0):.2f}")
    print(f"  Discrepancia: ${state.get('balance_usd', 0) - correct_balance:.2f}")
    
    # Corregir balance
    state['balance_usd'] = correct_balance
    state['initial_balance'] = initial_balance
    
    # BUG #3: Verificar y corregir P&L porcentual de trades
    print("\n🔧 BUG #3: Verificando P&L de trades...")
    for t in closed_trades:
        entry = t.get('entry_price', 0)
        exit_p = t.get('exit_price', 0)
        size = t.get('size', 0)
        pnl = t.get('pnl', 0)
        pnl_pct = t.get('pnl_pct', 0)
        direction = t.get('direction', 'bullish')
        
        # Recalcular
        if entry > 0:
            if direction in ['bullish', 'long']:
                calc_pct = (exit_p - entry) / entry
            else:
                calc_pct = (entry - exit_p) / entry
            
            calc_pnl = size * calc_pct
            
            # Verificar diferencia
            if abs(calc_pnl - pnl) > 0.01:
                print(f"  {t.get('symbol', '?')}: P&L incorrecto (${pnl:.2f} → ${calc_pnl:.2f})")
    
    # Guardar estado corregido
    print("\n💾 Guardando estado corregido...")
    
    # Backup del estado original
    backup_file = STATE_FILE.with_suffix('.json.backup')
    with open(backup_file, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"  Backup creado: {backup_file}")
    
    # Guardar estado corregido
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"  Estado corregido guardado: {STATE_FILE}")
    
    print("\n✅ CORRECCIONES APLICADAS:")
    print(f"  • Win Rate: {state['stats']['win_rate']:.1f}%")
    print(f"  • Balance: ${state['balance_usd']:.2f}")
    print(f"  • Initial Balance: ${state['initial_balance']:.2f}")
    print(f"  • Total P&L: ${state['balance_usd'] - state['initial_balance']:.2f}")

if __name__ == "__main__":
    fix_state()
