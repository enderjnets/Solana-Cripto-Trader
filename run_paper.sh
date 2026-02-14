#!/bin/bash
# Paper Trading Runner
# Wraps agent_brain.py to enable paper trading mode

echo "========================================"
echo "🚀 PAPER TRADING MODE"
echo "========================================"
echo ""
echo "Configuración:"
echo "   • Balance inicial: \$10,000 USD"
echo "   • Risk por trade: 10% del balance"
echo "   • Stop loss: -5%"
echo "   • Take profit: +10%"
echo "   • Target diario: +5%"
echo ""
echo "El sistema ejecutará PAPER TRADES automáticamente"
echo "sin enviar transacciones a la blockchain."
echo ""
echo "Presiona Ctrl+C para detener"
echo "========================================"
echo ""

# Start paper trading engine
python3 paper_trading_engine.py --start

# Run agent brain in fast mode (cycles every 120s)
python3 agent_brain.py --fast
