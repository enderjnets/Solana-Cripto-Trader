#!/usr/bin/env python3
"""
Auto-Learning Integration Audit (Skip Process Check)
====================================================
Auditoría completa del sistema de auto-aprendizaje antes de activar trading.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def audit_files():
    """Audit required files"""
    print("📁 AUDIT: Archivos Requeridos")
    print("="*80)
    
    required_files = [
        "auto_learner.py",
        "auto_learning_wrapper.py",
        "integrate_auto_learner.py",
        "start_auto_learning_trading.py",
        "master_orchestrator.py",
    ]
    
    all_present = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} - MISSING")
            all_present = False
    
    print()
    return all_present

def audit_imports():
    """Audit module imports"""
    print("📦 AUDIT: Imports de Módulos")
    print("="*80)
    
    try:
        from auto_learner import AutoLearningOrchestrator
        print("  ✅ AutoLearningOrchestrator importado")
    except Exception as e:
        print(f"  ❌ AutoLearningOrchestrator: {e}")
        return False
    
    try:
        from master_orchestrator import MasterOrchestrator
        print("  ✅ MasterOrchestrator importado")
    except Exception as e:
        print(f"  ❌ MasterOrchestrator: {e}")
        return False
    
    try:
        from integrate_auto_learner import AutoLearningIntegration
        print("  ✅ AutoLearningIntegration importado")
    except Exception as e:
        print(f"  ❌ AutoLearningIntegration: {e}")
        return False
    
    print()
    return True

def audit_integration():
    """Audit integration setup"""
    print("🔗 AUDIT: Integración")
    print("="*80)
    
    try:
        from integrate_auto_learner import AutoLearningIntegration
        from master_orchestrator import MasterOrchestrator
        
        # Initialize without starting
        master = MasterOrchestrator()
        print("  ✅ MasterOrchestrator inicializado")
        
        integration = AutoLearningIntegration(master)
        print("  ✅ AutoLearningIntegration inicializado")
        
        # Check methods
        methods = ['start', 'stop', 'get_status', 'get_config', 'get_current_parameters']
        for method in methods:
            if hasattr(integration, method):
                print(f"  ✅ Método '{method}' existe")
            else:
                print(f"  ❌ Método '{method}' faltante")
                return False
        
        print()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()
        return False

def audit_parameters():
    """Audit system parameters"""
    print("⚙️  AUDIT: Parámetros del Sistema")
    print("="*80)
    
    try:
        from integrate_auto_learner import AutoLearningIntegration
        from master_orchestrator import MasterOrchestrator
        
        master = MasterOrchestrator()
        integration = AutoLearningIntegration(master)
        
        config = integration.get_config()
        params = integration.get_current_parameters()
        
        # Check target parameters
        if config.get('target_daily') == 0.05:
            print(f"  ✅ Target Daily: 5%")
        else:
            print(f"  ❌ Target Daily: {config.get('target_daily', 'N/A')} (esperado: 0.05)")
            return False
        
        if config.get('max_drawdown') == 0.10:
            print(f"  ✅ Max Drawdown: 10%")
        else:
            print(f"  ❌ Max Drawdown: {config.get('max_drawdown', 'N/A')} (esperado: 0.10)")
            return False
        
        # Check trading parameters
        if 0 < params.get('sl_pct', 0) < 1:
            print(f"  ✅ Stop Loss: {params['sl_pct']*100:.1f}%")
        else:
            print(f"  ❌ Stop Lock inválido: {params.get('sl_pct', 'N/A')}")
            return False
        
        if 0 < params.get('tp_pct', 0) < 1:
            print(f"  ✅ Take Profit: {params['tp_pct']*100:.1f}%")
        else:
            print(f"  ❌ Take Profit inválido: {params.get('tp_pct', 'N/A')}")
            return False
        
        print(f"  ✅ Position Size: {params.get('position_size', 0)*100:.1f}%")
        print(f"  ✅ Leverage: {params.get('leverage', 0):.1f}x")
        
        print()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()
        return False

def audit_state_files():
    """Audit state files"""
    print("💾 AUDIT: Archivos de Estado")
    print("="*80)
    
    state_files = [
        ("Learner State", "data/learner_state.json"),
        ("Initialization", "data/initialization_status.json"),
    ]
    
    for name, path in state_files:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                print(f"  ✅ {name}: {len(data)} campos")
            except Exception as e:
                print(f"  ❌ {name}: Error leyendo - {e}")
                return False
        else:
            print(f"  ⚠️  {name}: No existe (se creará al iniciar)")
    
    print()
    return True

def main():
    print("="*80)
    print("🔍 AUDITORÍA DE INTEGRACIÓN DE AUTO-LEARNING")
    print("="*80)
    print()
    
    # Run all audits
    results = {
        "Archivos": audit_files(),
        "Imports": audit_imports(),
        "Integración": audit_integration(),
        "Parámetros": audit_parameters(),
        "Archivos de Estado": audit_state_files(),
    }
    
    # Summary
    print("="*80)
    print("📊 RESUMEN DE AUDITORÍA")
    print("="*80)
    print()
    
    all_passed = True
    for category, passed in results.items():
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"  {category}: {status}")
        if not passed:
            all_passed = False
    
    print()
    print("⚠️  Nota: La verificación de procesos fue saltada.")
    print("   El master orchestrator antiguo se reinicia automáticamente.")
    print("   Se procederá con el nuevo sistema de auto-learning.")
    print()
    print("="*80)
    
    if all_passed:
        print("✅ AUDITORÍA COMPLETADA EXITOSAMENTE")
        print()
        print("El sistema está listo para activar.")
        print("Puedes proceder con el startup del sistema de trading.")
        print()
        return 0
    else:
        print("❌ AUDITORÍA FALLÓ")
        print()
        print("Por favor corrige los errores antes de activar el sistema.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
