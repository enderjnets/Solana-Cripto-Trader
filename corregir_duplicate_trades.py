#!/usr/bin/env python3
"""
Corrección Adicional - Sistema de Trading
======================================
Correcciones para evitar múltiples trades en el mismo token.
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent

def corregir_duplicate_trades():
    """Agregar validación para evitar múltiples trades en el mismo token"""

    file_path = PROJECT_DIR / "unified_trading_system.py"
    content = file_path.read_text()

    # Agregar validación de token duplicado antes de execute_trade
    old_validate = '''    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute a trading signal (paper mode)"""
        
        # Check concurrent limit
        open_trades = self.paper_engine.get_open_trades()
        profile = self.get_trading_params()'''

    new_validate = '''    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute a trading signal (paper mode)"""
        
        # Check concurrent limit
        open_trades = self.paper_engine.get_open_trades()
        profile = self.get_trading_params()
        
        # Check for duplicate token (FIXED: Only 1 trade per token)
        symbol_trades = [t for t in open_trades if t["symbol"] == signal.symbol]
        if len(symbol_trades) > 0:
            logger.warning(f"⚠️ Trade rejected: Already have {len(symbol_trades)} open position(s) in {signal.symbol}")
            return False'''

    content = content.replace(old_validate, new_validate)

    # Guardar cambios
    file_path.write_text(content)
    print("✅ Validación de token duplicado agregada")

def resetear_sistema():
    """Resetear paper trading a $500"""
    print("\n🔄 Reseteando sistema...")

    state_file = PROJECT_DIR / "data" / "paper_trading_state.json"

    # Estado limpio
    clean_state = {
        "enabled": True,
        "start_time": datetime.now().isoformat(),
        "balance_usd": 500.0,
        "initial_balance": 500.0,
        "leverage": 1,
        "margin_used": 0.0,
        "trades": [],
        "stats": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_fees": 0.0,
            "liquidations": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "current_streak": 0,
            "best_streak": 0,
            "worst_streak": 0,
        },
        "signals": []
    }

    # Guardar estado limpio
    state_file.write_text(json.dumps(clean_state, indent=2))
    print("✅ Sistema reseteado a $500.00")

    # Resetear risk state
    risk_file = PROJECT_DIR / "data" / "risk_state.json"
    if risk_file.exists():
        risk_state = {
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "daily_start_balance": 500.0,
            "open_trades": [],
            "last_update": datetime.now().isoformat()
        }
        risk_file.write_text(json.dumps(risk_state, indent=2))
        print("✅ Risk state reseteado")

def actualizar_analisis():
    """Actualizar ANALISIS_FALLAS.md con la nueva falla"""
    print("\n📝 Actualizando análisis...")

    analisis_file = PROJECT_DIR / "ANALISIS_FALLAS.md"
    if analisis_file.exists():
        content = analisis_file.read_text()

        # Agregar nueva falla
        new_falla = """

### 8. **Falta de Validación de Token Duplicado** (NUEVO - 2026-02-25 12:30)
**Ubicación:** `unified_trading_system.py:1155` (función execute_trade)
**Problema:** El sistema abría múltiples trades en el mismo token consecutivamente
**Impacto:** 5 trades en JTO bullish abiertos en 30 minutos
**Corrección:** Agregar validación para máximo 1 trade por token
---

"""

        # Insertar antes del final
        content = content.replace("---\n\n**Generado por:** Eko 🦞", new_falla + "---\n\n**Generado por:** Eko 🦞")

        analisis_file.write_text(content)
        print("✅ ANALISIS_FALLAS.md actualizado")

def main():
    print("\n" + "="*60)
    print("🔧 CORRECIÓN ADICIONAL - Evitar Trades Duplicados")
    print("="*60)

    # 1. Corregir código
    print("\n📝 Agregando validación de token duplicado...")
    corregir_duplicate_trades()

    # 2. Resetear sistema
    resetear_sistema()

    # 3. Actualizar análisis
    actualizar_analisis()

    print("\n" + "="*60)
    print("✅ CORRECIÓN ADICIONAL COMPLETADA")
    print("="*60)
    print("\n📊 Resumen:")
    print("   • Validación: 1 trade máximo por token")
    print("   • Sistema: Reseteado a $500")
    print("\n🎯 Próximo paso: git commit -m 'fix: Prevent duplicate token trades'")
    print("="*60)

if __name__ == "__main__":
    main()
