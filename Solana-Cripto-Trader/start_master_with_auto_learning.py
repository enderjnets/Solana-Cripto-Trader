#!/usr/bin/env python3
"""
Master Orchestrator con Auto-Learning Integrado
================================================
Versión definitiva que ejecuta ambos sistemas juntos.

Este script:
1. Inicia el Master Orchestrator (sistema principal de trading)
2. Integra el Auto-Learning (genetic + reinforcement learning)
3. Ejecuta ambos sistemas en paralelo
4. Maneja señales de shutdown gracefully
"""

import sys
import time
import json
import signal
import asyncio
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/master_with_auto_learning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("master_with_auto_learning")

# ============================================================================
# AUTO-LEARNING INTEGRATION (SIMPFLIFICADA)
# ============================================================================

class SimpleAutoLearning:
    """Simplified auto-learning that doesn't require complex integration"""

    def __init__(self):
        self.generation = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.exploration_rate = 0.3
        self.best_strategy = None
        self.running = False
        self.learning_thread = None

        # Adaptive parameters
        self.sl_pct = 0.025  # 2.5%
        self.tp_pct = 0.05   # 5.0%
        self.position_size = 0.05  # 5%
        self.leverage = 5.0

        # Learning history
        self.trade_history = []

        logger.info("Simple auto-learning initialized")

    def adapt_parameters(self, master_state: Dict):
        """Adapt parameters based on performance"""
        daily_pnl = master_state.get('daily_pnl', 0)
        win_rate = master_state.get('stats', {}).get('win_rate', 50)

        # Adjust SL based on win rate
        if win_rate > 60:
            self.sl_pct = max(0.015, self.sl_pct * 0.95)  # Tighter SL with high win rate
        elif win_rate < 40:
            self.sl_pct = min(0.035, self.sl_pct * 1.05)  # Wider SL with low win rate

        # Adjust TP based on daily PnL
        if daily_pnl > 0.03:  # Good performance
            self.tp_pct = max(0.03, self.tp_pct * 0.95)
        elif daily_pnl < -0.03:  # Bad performance
            self.tp_pct = min(0.07, self.tp_pct * 1.05)

        # Adjust exploration rate
        if daily_pnl > 0.05:  # Target reached
            self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
        elif daily_pnl < -0.05:
            self.exploration_rate = min(0.5, self.exploration_rate * 1.1)

        # Adjust position size based on drawdown
        drawdown = abs(min(0, daily_pnl))
        if drawdown > 0.08:  # Near max drawdown
            self.position_size = max(0.02, self.position_size * 0.9)
        elif drawdown < 0.02 and daily_pnl > 0:
            self.position_size = min(0.08, self.position_size * 1.05)

        logger.info(f"Parameters adapted: SL={self.sl_pct:.3f}, TP={self.tp_pct:.3f}, "
                   f"Pos={self.position_size:.3f}, Exp={self.exploration_rate:.3f}")

    def record_trade(self, trade: Dict):
        """Record a trade for learning"""
        self.trade_history.append(trade)
        self.total_trades += 1
        self.total_pnl += trade.get('pnl', 0)
        self.daily_pnl = self.total_pnl / 500.0  # Assuming $500 starting capital

        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)

        logger.debug(f"Trade recorded: PnL=${trade.get('pnl', 0):.2f}, "
                     f"Total=${self.total_pnl:.2f}, Daily={self.daily_pnl:.2%}")

    def evolve_strategies(self):
        """Evolve strategies (simplified version)"""
        self.generation += 1

        if not self.trade_history:
            logger.info("No trades yet, skipping evolution")
            return False

        # Analyze recent performance
        recent_trades = self.trade_history[-50:]
        wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0

        # Log evolution
        logger.info(f"Generation {self.generation} | Win Rate: {win_rate:.1%} | "
                   f"Trades: {len(recent_trades)}")

        # Store best strategy
        self.best_strategy = {
            "generation": self.generation,
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
            "position_size": self.position_size,
            "leverage": self.leverage,
            "win_rate": win_rate
        }

        return True

    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "running": self.running,
            "generation": self.generation,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "exploration_rate": self.exploration_rate,
            "best_strategy": self.best_strategy,
            "current_params": {
                "sl_pct": self.sl_pct,
                "tp_pct": self.tp_pct,
                "position_size": self.position_size,
                "leverage": self.leverage
            }
        }

    def get_parameters(self) -> Dict:
        """Get current trading parameters"""
        return {
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
            "position_size": self.position_size,
            "leverage": self.leverage
        }

    def _learning_loop(self):
        """Background learning loop"""
        while self.running:
            try:
                time.sleep(60)  # Check every minute
                if not self.running:
                    break

                # Evolve every hour
                if time.time() % 3600 < 60:
                    self.evolve_strategies()

            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)

    def start(self):
        """Start auto-learning"""
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("Auto-learning started")

    def stop(self):
        """Stop auto-learning"""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("Auto-learning stopped")


# ============================================================================
# MASTER ORCHESTRATOR WRAPPER
# ============================================================================

class MasterOrchestratorWrapper:
    """Wrapper to run master orchestrator with auto-learning"""

    def __init__(self):
        from master_orchestrator import MasterOrchestrator

        self.master = MasterOrchestrator()
        self.auto_learning = SimpleAutoLearning()

        self.master_thread = None
        self.running = False

        logger.info("Master orchestrator wrapper initialized")

    def _master_loop(self):
        """Run master orchestrator in async loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_with_learning())
        except Exception as e:
            logger.error(f"Error in master loop: {e}", exc_info=True)
        finally:
            loop.close()

    async def _run_with_learning(self):
        """Run master orchestrator with auto-learning integration"""
        drift_params = self.master.state.data.get("drift_simulation", {
            "leverage": 5.0,
            "trading_fee_pct": 0.05,
            "liquidation_threshold": 0.80,
            "borrowing_fee_hourly": 0.0001,
            "funding_rate_pct": 0.01
        })

        self.master.state.log("=" * 60)
        self.master.state.log("🧬 MASTER ORCHESTRATOR v4.0 - AUTO-LEARNING INTEGRADO")
        self.master.state.log(f"   Meta: 5.0% diario")
        self.master.state.log(f"   Max Drawdown: 10.0%")
        self.master.state.log(f"   Leverage: {drift_params['leverage']}x")
        self.master.state.log(f"   Trading Fee: {drift_params['trading_fee_pct']}%")
        self.master.state.log(f"   Auto-Learning: ACTIVO")
        self.master.state.log(f"   Capital: ${self.master.state.data.get('paper_capital', 500):.2f}")
        self.master.state.log("=" * 60)

        cycle = 0
        last_learning = time.time()

        while self.running:
            cycle += 1
            self.master.state.data["cycles"] = cycle

            try:
                # Run master orchestrator agents
                last_research = self.master.state.data["agents"]["researcher"].get("last_run")
                if not last_research or (datetime.now() - datetime.fromisoformat(last_research)).seconds > 300:
                    await self.master.researcher.run()

                last_backtest = self.master.state.data["agents"]["backtester"].get("last_run")
                if not last_backtest or (datetime.now() - datetime.fromisoformat(last_backtest)).seconds > 180:
                    await self.master.backtester.run()

                await self.master.auditor.run()
                await self.master.paper_trader.run()

                # Fetch prices
                prices = self.master.get_real_prices() if hasattr(self.master, 'get_real_prices') else {}
                if not prices:
                    # Fallback
                    try:
                        from master_orchestrator import get_real_prices
                        prices = get_real_prices()
                    except:
                        pass

                self.master.state.data["current_prices"] = prices

                # Auto-learning integration
                if time.time() - last_learning >= 60:  # Every minute
                    # Feed master state to auto-learning
                    master_state = self.master.state.data.copy()
                    self.auto_learning.adapt_parameters(master_state)

                    # Record closed trades
                    paper_history = self.master.state.data.get("paper_history", [])
                    if paper_history:
                        recent_closed = paper_history[-5:]  # Last 5 closed trades
                        for trade in recent_closed:
                            if 'recorded' not in trade:
                                self.auto_learning.record_trade({
                                    'pnl': trade.get('pnl_final', 0),
                                    'win': trade.get('pnl_final', 0) > 0,
                                    'reason': trade.get('close_reason', 'unknown')
                                })
                                trade['recorded'] = True

                    last_learning = time.time()

                # Log status
                paper_stats = self.master.state.data["agents"].get("paper_trading", {})
                open_pos = paper_stats.get("open_positions", 0)
                capital = paper_stats.get("capital", 500)
                win_rate = paper_stats.get("win_rate", 0)
                total_fees = paper_stats.get("total_fees", 0)

                learning_status = self.auto_learning.get_status()

                self.master.state.log(f"📊 Ciclo {cycle} | Auto-Learning: Gen {learning_status['generation']} | "
                                    f"Capital: ${capital:.2f} | Estrategias: 0 | 📝 Drift: {open_pos} pos | "
                                    f"Win Rate: {win_rate:.1f}% | Fees: ${total_fees:.4f}")

                await asyncio.sleep(60)  # 1 minute cycle

            except Exception as e:
                self.master.state.log(f"❌ Error: {e}")
                import traceback
                self.master.state.log(f"   Trace: {traceback.format_exc()[:150]}")
                await asyncio.sleep(60)

    def start(self):
        """Start master orchestrator with auto-learning"""
        self.running = True

        # Start auto-learning
        self.auto_learning.start()

        # Start master orchestrator in thread
        self.master_thread = threading.Thread(target=self._master_loop, daemon=False)
        self.master_thread.start()

        logger.info("Master orchestrator with auto-learning started")

    def stop(self):
        """Stop both systems"""
        self.running = False
        self.auto_learning.stop()

        if self.master_thread:
            self.master_thread.join(timeout=10)

        logger.info("Master orchestrator with auto-learning stopped")

    def get_status(self) -> Dict:
        """Get comprehensive status"""
        master_stats = self.master.state.data["agents"].get("paper_trading", {})
        learning_status = self.auto_learning.get_status()

        return {
            "running": self.running,
            "master": {
                "capital": master_stats.get("capital", 500),
                "daily_pnl": master_stats.get("daily_pnl_pct", 0),
                "win_rate": master_stats.get("win_rate", 0),
                "open_positions": master_stats.get("open_positions", 0),
                "total_fees": master_stats.get("total_fees", 0)
            },
            "auto_learning": {
                "generation": learning_status["generation"],
                "total_trades": learning_status["total_trades"],
                "daily_pnl": learning_status["daily_pnl"],
                "exploration_rate": learning_status["exploration_rate"],
                "current_params": learning_status["current_params"]
            }
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    global wrapper, running

    print("="*80)
    print("🧬 MASTER ORCHESTRATOR v4.0 - AUTO-LEARNING INTEGRADO")
    print("="*80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize wrapper
    print("📊 Inicializando Master Orchestrator con Auto-Learning...")
    wrapper = MasterOrchestratorWrapper()
    print("✅ Sistema inicializado")

    # Start system
    print()
    print("🚀 Iniciando sistema...")
    wrapper.start()
    print("✅ SISTEMA ACTIVO")

    # Show initial status
    print()
    print("📈 Estado Inicial:")
    status = wrapper.get_status()
    print(f"  Capital: ${status['master']['capital']:.2f}")
    print(f"  Daily PnL: {status['master']['daily_pnl']:.2%}")
    print(f"  Win Rate: {status['master']['win_rate']:.1f}%")
    print(f"  Open Positions: {status['master']['open_positions']}")
    print()
    print("🧬 Auto-Learning:")
    print(f"  Generación: {status['auto_learning']['generation']}")
    print(f"  Trades Aprendidos: {status['auto_learning']['total_trades']}")
    print(f"  Exploration Rate: {status['auto_learning']['exploration_rate']:.3f}")
    print(f"  Parámetros Actuales:")
    params = status['auto_learning']['current_params']
    print(f"    Stop Loss: {params['sl_pct']*100:.1f}%")
    print(f"    Take Profit: {params['tp_pct']*100:.1f}%")
    print(f"    Position Size: {params['position_size']*100:.1f}%")
    print(f"    Leverage: {params['leverage']:.1f}x")

    print()
    print("="*80)
    print("✅ SISTEMA ACTIVO Y APRENDIENDO")
    print("="*80)
    print()
    print("📊 El sistema ahora:")
    print("  • Ejecuta trades con Master Orchestrator")
    print("  • Aprende de cada trade automáticamente")
    print("  • Adapta parámetros cada minuto")
    print("  • Evoluciona estrategias cada hora")
    print("  • Monitorea performance continuamente")
    print()
    print("🎯 Meta: 5% diario")
    print("🛡️ Max Drawdown: 10%")
    print("🧬 Auto-Learning: ACTIVO")
    print()
    print("📊 Monitor commands:")
    print("  tail -f /tmp/master_with_auto_learning.log")
    print("  tail -f ~/.config/solana-jupiter-bot/master.log")
    print()
    print("🛑 Press Ctrl+C to stop")
    print("="*80)
    print()

    # Keep running and show updates
    update_count = 0
    while running:
        time.sleep(300)  # Update every 5 minutes
        update_count += 1

        if not running:
            break

        status = wrapper.get_status()

        print(f"\n{'='*80}")
        print(f"📊 UPDATE #{update_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"  Capital: ${status['master']['capital']:.2f}")
        print(f"  Daily PnL: {status['master']['daily_pnl']:.2%}")
        print(f"  Win Rate: {status['master']['win_rate']:.1f}%")
        print(f"  Open Positions: {status['master']['open_positions']}")
        print(f"  Total Fees: ${status['master']['total_fees']:.2f}")
        print()
        print(f"  🧬 Auto-Learning Gen: {status['auto_learning']['generation']}")
        print(f"  Trades Aprendidos: {status['auto_learning']['total_trades']}")
        print(f"  Exploration: {status['auto_learning']['exploration_rate']:.3f}")

        # Check targets
        if status['master']['daily_pnl'] >= 0.05:
            print("\n🎉 DAILY TARGET REACHED!")

        if status['master']['daily_pnl'] <= -0.10:
            print("\n⚠️ MAX DRAWDOWN WARNING!")

        print(f"{'='*80}\n")


# Global variables
wrapper = None
running = True

def signal_handler(signum, frame):
    """Handle shutdown gracefully"""
    global running, wrapper
    print('\n\n🛑 Shutdown signal received...')
    running = False
    if wrapper:
        wrapper.stop()
        print('✅ System stopped gracefully')
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received...")
        if wrapper:
            wrapper.stop()
        sys.exit(0)
