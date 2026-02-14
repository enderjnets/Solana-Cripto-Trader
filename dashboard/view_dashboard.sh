#!/bin/bash
echo "=========================================="
echo "     EKO TRADING SYSTEM - DASHBOARD"
echo "=========================================="
echo ""
python3 -c "
import json
from datetime import datetime
from pathlib import Path

state = json.loads(Path('unified_brain_state.json').read_text())
db = json.loads(Path('db/unified_trading.db').read_text())
trades = db.get('trades', [])
stats = state.get('stats', {})

print('STATUS: ONLINE')
print('-' * 40)
print(f'Version:      {state.get(\"version\", \"?\")}')
print(f'Total Trades:  {len(trades)}')
print(f'Daily P&L:    {stats.get(\"daily_pnl_pct\", 0):+.2f}%')
print(f'Cycles:       {stats.get(\"cycles\", 0)}')
print(f'Trades Today: {stats.get(\"trades_today\", 0)}')
print()
print('MODULOS:')
print('-' * 40)
for m, active in state.get('modules', {}).items():
    print(f'  [{(\"OK\" if active else \"OFF\").center(3)}] {m}')
print()
print('TRADES RECIENTES:')
print('-' * 40)
for t in reversed(trades[-10:]):
    s = t.get('symbol', '').ljust(6)
    d = t.get('direction', '')
    p = t.get('pnl_pct') or 0
    st = t.get('status', '')
    print(f'  {s} | {d:4} | {p:+7.2f}% | {st}')
print()
print(f'更新时间: {datetime.now().strftime(\"%H:%M:%S\")}')
"
echo ""
echo "=========================================="
echo "Web Dashboards:"
echo "  Simple:  http://127.0.0.1:8502"
echo "  Pro:     http://127.0.0.1:8501"
echo "=========================================="
