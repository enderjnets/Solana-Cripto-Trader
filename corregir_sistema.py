#!/usr/bin/env python3
"""
Sistema de Corrección de Trading - Eko v2.0
==========================================
Aplica todas las correcciones identificadas en el análisis.
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent

def corregir_unified_trading_system():
    """Aplicar correcciones a unified_trading_system.py"""

    file_path = PROJECT_DIR / "unified_trading_system.py"
    content = file_path.read_text()

    # 1. Aumentar threshold de ensemble (0.05 → 0.20)
    print("✅ 1. Corrigiendo threshold de ensemble...")
    content = content.replace(
        'direction = "bullish" if ensemble > 0.05 else "bearish" if ensemble < -0.05 else "neutral"',
        'direction = "bullish" if ensemble > 0.20 else "bearish" if ensemble < -0.20 else "neutral"  # FIXED: Increased threshold from 0.05 to 0.20'
    )

    # 2. Aumentar filtro de confianza mínima (10% → 40%)
    print("✅ 2. Corrigiendo filtro de confianza mínima...")
    content = content.replace(
        'if signal["confidence"] < 10:  # Minimum 10% confidence',
        'if signal["confidence"] < 40:  # Minimum 40% confidence - FIXED: Increased from 10% for quality over quantity'
    )

    # 3. Corregir cálculo de confianza (no lineal)
    print("✅ 3. Corrigiendo cálculo de confianza...")
    old_confidence_calc = """        # Convert to confidence (0-95%)
        # ensemble is -1 to 1, confidence is absolute value scaled
        raw_confidence = abs(ensemble) * 95
        confidence = min(95, max(0, raw_confidence))"""

    new_confidence_calc = """        # Convert to confidence (0-95%) - FIXED: Non-linear to penalize weak signals
        # ensemble is -1 to 1, confidence uses exponential scaling
        # Weak signals (< 0.3) get very low confidence
        # Strong signals (> 0.6) get high confidence
        if abs(ensemble) < 0.3:
            # Weak signal - penalize heavily
            confidence = abs(ensemble) * 30  # Max 9% for weak signals
        elif abs(ensemble) < 0.6:
            # Moderate signal - moderate confidence
            confidence = 30 + (abs(ensemble) - 0.3) * 150  # 30-75%
        else:
            # Strong signal - high confidence
            confidence = 75 + (abs(ensemble) - 0.6) * 50  # 75-95%

        confidence = min(95, max(0, confidence))"""

    content = content.replace(old_confidence_calc, new_confidence_calc)

    # 4. No generar señales neutrales con confianza artificial
    print("✅ 4. Corrigiendo señales neutrales...")
    old_neutral = '''                else:
                    # Neutral zone
                    direction = "neutral"
                    confidence = 5
                    reason = f"RSI neutral: {rsi:.1f}"'''

    new_neutral = '''                else:
                    # Neutral zone - FIXED: Skip neutral signals instead of generating low confidence
                    continue  # Skip neutral RSI signals'''

    content = content.replace(old_neutral, new_neutral)

    # Guardar cambios
    file_path.write_text(content)
    print("✅ Archivo unified_trading_system.py corregido")

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

def actualizar_memoria():
    """Actualizar MEMORY.md con el análisis"""
    print("\n📝 Actualizando MEMORY.md...")

    memory_file = PROJECT_DIR.parent / "MEMORY.md"
    if memory_file.exists():
        content = memory_file.read_text()

        # Agregar entrada de hoy
        today_entry = f"""

## 2026-02-25 - Corrección de Sistema de Trading

### Análisis Granular Completado
- **Fallas identificadas:** 7 fallas críticas
- **Correcciones aplicadas:**
  1. Threshold de ensemble: 0.05 → 0.20
  2. Filtro de confianza mínima: 10% → 40%
  3. Cálculo de confianza no lineal
  4. Señales neutrales: eliminadas
  5. Max trades/día: ∞ → 15
  6. Position size: 15% → 8% (en config)
  7. Risk Agent: validación de confianza

### Estado Post-Corrección
- **Balance:** $500.00 (reseteado)
- **Sistema:** Detenido, listo para reiniciar
- **Archivo de análisis:** ANALISIS_FALLAS.md

### Próximos Pasos
- Reiniciar sistema con parámetros corregidos
- Monitorear calidad de señales
- Verificar win rate mejora

---
"""

        # Insertar después del encabezado principal
        if "## 2026-02-25" not in content:
            # Encontrar el final del encabezado
            lines = content.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('## Trading Systems'):
                    insert_pos = i + 1
                    break

            # Insertar entrada
            lines.insert(insert_pos, today_entry)
            memory_file.write_text('\n'.join(lines))
            print("✅ MEMORY.md actualizado")

def main():
    print("\n" + "="*60)
    print("🔧 SISTEMA DE CORRECCIÓN DE TRADING - EKO v2.0")
    print("="*60)

    # 1. Corregir código
    print("\n📝 Aplicando correcciones al código...")
    corregir_unified_trading_system()

    # 2. Resetear sistema
    resetear_sistema()

    # 3. Actualizar memoria
    actualizar_memoria()

    print("\n" + "="*60)
    print("✅ CORRECCIONES COMPLETADAS")
    print("="*60)
    print("\n📊 Resumen:")
    print("   • Threshold: 0.05 → 0.20")
    print("   • Min Confidence: 10% → 40%")
    print("   • Cálculo confianza: Lineal → No lineal")
    print("   • Señales neutrales: Eliminadas")
    print("   • Sistema: Reseteado a $500")
    print("\n🎯 Próximo paso: git commit -m 'fix: Critical trading strategy fixes'")
    print("="*60)

if __name__ == "__main__":
    main()
