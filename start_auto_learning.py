#!/usr/bin/env python3
"""
START AUTO-LEARNING SYSTEM
==========================
Script para iniciar el sistema de auto-aprendizaje y conectarlo
con el Master Orchestrator que ya está corriendo.

Uso:
    python3 start_auto_learning.py
"""

import sys
import time
import signal
from pathlib import Path
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("start_auto_learning")

# Import auto-learner
try:
    from auto_learner import AutoLearningOrchestrator
    from integrate_auto_learner import AutoLearningIntegration
    print("✅ Auto-learning modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("\nInstalling dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "pandas"])
    print("✅ Dependencies installed. Please restart.")
    sys.exit(1)


# ============================================================================
# AUTO-LEARNING STARTER
# ============================================================================
class AutoLearningStarter:
    """Starts and manages the auto-learning system"""
    
    def __init__(self):
        self.auto_learner = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n\n🛑 Shutdown signal received...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the auto-learning system"""
        print("\n" + "="*80)
        print("🚀 INICIANDO SISTEMA DE AUTO-APRENDIZAJE")
        print("="*80)
        
        # Initialize
        print("\n📊 Inicializando Auto-Learning Orchestrator...")
        self.auto_learner = AutoLearningOrchestrator()
        
        # Show current status
        status = self.auto_learner.get_status()
        print("\n📈 Estado Actual:")
        print(f"  Generación: {status['current_generation']}")
        print(f"  Trades totales: {status['total_trades']}")
        print(f"  PnL diario: {status['daily_pnl']:.2%}")
        print(f"  PnL total: ${status['total_pnl']:.2f}")
        print(f"  Tasa de exploración: {status['exploration_rate']:.3f}")
        
        # Show parameters
        print("\n⚙️  Parámetros Actuales:")
        params = status['current_parameters']
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Start the system
        print("\n🔥 Iniciando sistema...")
        self.auto_learner.start()
        self.running = True
        
        print("\n✅ Sistema de auto-aprendizaje ACTIVO")
        print("\n🎯 Objetivo: 5% diario")
        print("⚡ El sistema aprenderá de cada trade automáticamente")
        print("🧬 Evolución de estrategias cada hora (si hay suficientes trades)")
        print("🔄 Adaptación de parámetros cada 5 minutos")
        
        print("\n" + "="*80)
        print("Sistema corriendo. Presiona Ctrl+C para detener.")
        print("="*80)
        
        # Keep running and show periodic updates
        self._monitor_loop()
    
    def _monitor_loop(self):
        """Monitor and display learning progress"""
        update_count = 0
        
        while self.running:
            time.sleep(300)  # Update every 5 minutes
            update_count += 1
            
            status = self.auto_learner.get_status()
            
            print(f"\n{'='*80}")
            print(f"📊 UPDATE #{update_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            print(f"  Generación: {status['current_generation']}")
            print(f"  Trades: {status['total_trades']}")
            print(f"  PnL Diario: {status['daily_pnl']:.2%}")
            print(f"  Exploración: {status['exploration_rate']:.3f}")
            
            # Check if target reached
            if status['daily_pnl'] >= 0.05:
                print("\n🎉 ¡OBJETIVO DIARIO ALCANZADO!")
                print(f"   PnL: {status['daily_pnl']:.2%}")
            
            # Check drawdown
            if status['daily_pnl'] <= -0.10:
                print("\n⚠️ ¡MAX DRAWDOWN ALCANZADO!")
                print(f"   PnL: {status['daily_pnl']:.2%}")
                print("   Considerando pausar trading...")
            
            # Show best strategy if available
            if status['best_strategy']:
                print(f"\n🧬 Mejor Estrategia (Gen {status['best_strategy']['generation']}):")
                print(f"   Fitness: {status['best_strategy']['fitness']:.4f}")
                print(f"   Win Rate: {status['best_strategy']['win_rate']:.2%}")
            
            print(f"{'='*80}\n")
    
    def stop(self):
        """Stop the auto-learning system"""
        if self.auto_learner:
            print("\n🛑 Deteniendo sistema de auto-aprendizaje...")
            self.auto_learner.stop()
            
            # Show final status
            status = self.auto_learner.get_status()
            print("\n📊 Estado Final:")
            print(f"  Generaciones evolucionadas: {status['current_generation']}")
            print(f"  Total trades: {status['total_trades']}")
            print(f"  PnL total: ${status['total_pnl']:.2f}")
            
            print("\n✅ Sistema detenido correctamente")


# ============================================================================
# QUICK START FOR TESTING
# ============================================================================
def quick_test():
    """Quick test of the auto-learning system"""
    print("\n" + "="*80)
    print("🧪 MODO PRUEBA - Auto-Learning System")
    print("="*80)
    
    from auto_learner import AutoLearningOrchestrator, MarketState
    import numpy as np
    
    # Initialize
    learner = AutoLearningOrchestrator()
    
    # Simulate some trades
    print("\n📊 Simulando trades para prueba...")
    
    for i in range(20):
        # Create fake market state
        state = MarketState(
            rsi=np.random.uniform(30, 70),
            ema_diff=np.random.uniform(-0.05, 0.05),
            volatility=np.random.uniform(0.01, 0.05),
            volume_ratio=np.random.uniform(0.8, 1.5),
            price_momentum=np.random.uniform(-0.03, 0.03),
            bb_position=np.random.uniform(0.2, 0.8),
            trend_strength=np.random.uniform(0, 1),
            recent_pnl=learner.daily_pnl,
            trade_count=learner.total_trades,
            time_of_day=np.random.uniform(0, 1)
        )
        
        # Get decision
        should_trade, confidence, direction = learner.should_trade(state)
        
        # Simulate trade result
        if should_trade:
            win = np.random.random() > 0.4  # 60% win rate
            pnl_pct = np.random.uniform(0.01, 0.05) if win else np.random.uniform(-0.05, -0.02)
            
            trade_data = {
                "trade_id": f"test_{i}",
                "strategy_id": "test",
                "pnl": pnl_pct * 1000,  # Assuming $1000 position
                "pnl_pct": pnl_pct,
                "win": win,
                "holding_time": np.random.uniform(1, 10),
                "market_state": state.__dict__,
                "confidence": confidence,
                "direction": direction
            }
            
            learner.record_trade_result(trade_data)
            
            print(f"  Trade {i+1}: {'✅ WIN' if win else '❌ LOSS'} | "
                  f"PnL: {pnl_pct:+.2%} | "
                  f"Confidence: {confidence:.2f}")
    
    # Show status
    status = learner.get_status()
    print("\n📊 Estado después de 20 trades:")
    print(f"  Total PnL: {status['daily_pnl']:.2%}")
    print(f"  Exploration rate: {status['exploration_rate']:.3f}")
    
    print("\n✅ Prueba completada exitosamente")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Learning Trading System")
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--status', action='store_true', help='Show current status only')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    elif args.status:
        # Show status only
        from auto_learner import AutoLearningOrchestrator
        learner = AutoLearningOrchestrator()
        status = learner.get_status()
        
        print("\n📊 AUTO-LEARNING STATUS")
        print("="*60)
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    else:
        # Start full system
        starter = AutoLearningStarter()
        starter.start()
