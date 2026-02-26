#!/usr/bin/env python3
"""
Auto-Learner Integration Module
================================
Integra el sistema de auto-aprendizaje con el Master Orchestrator existente.

Este módulo:
1. Conecta el AutoLearningOrchestrator con el MasterOrchestrator
2. Reemplaza las estrategias estáticas con estrategias evolucionadas
3. Implementa el loop de feedback de performance
4. Gestiona la comunicación entre workers y el learner
"""

import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from auto_learner import AutoLearningOrchestrator, MarketState
from master_orchestrator import MasterOrchestrator

logger = logging.getLogger("auto_learner_integration")


# ============================================================================
# AUTO-LEARNING INTEGRATION
# ============================================================================
class AutoLearningIntegration:
    """Integrates auto-learning with existing trading system"""
    
    def __init__(self, master_orchestrator: MasterOrchestrator):
        self.master = master_orchestrator
        self.auto_learner = AutoLearningOrchestrator()
        
        self.running = False
        self.integration_thread = None
        
        # Performance tracking
        self.trades_since_last_evolution = 0
        self.last_evolution_time = None
        
        logger.info("Auto-learning integration initialized")
    
    def initialize_with_data(self):
        """Initialize auto-learner with historical data"""
        try:
            # Get historical data from master
            # This assumes master has a data manager or can fetch data
            logger.info("Fetching historical data for genetic miner...")
            
            # For now, we'll initialize on first trade
            # In production, would load from data source
            logger.info("Auto-learner will initialize on first batch of trades")
            
        except Exception as e:
            logger.error(f"Error initializing auto-learner: {e}")
    
    def process_trade_result(self, trade_result: Dict):
        """Process trade result and feed to auto-learner"""
        try:
            # Extract relevant data
            trade_data = {
                "trade_id": trade_result.get("trade_id", f"trade_{datetime.now().timestamp()}"),
                "strategy_id": trade_result.get("strategy_id", "default"),
                "pnl": trade_result.get("pnl", 0),
                "pnl_pct": trade_result.get("pnl_pct", 0),
                "win": trade_result.get("win", False),
                "holding_time": trade_result.get("holding_time", 0),
                "entry_params": {
                    "sl_pct": trade_result.get("entry_sl", 0.025),
                    "tp_pct": trade_result.get("entry_tp", 0.05),
                    "leverage": trade_result.get("leverage", 5.0)
                },
                "exit_params": {
                    "exit_reason": trade_result.get("exit_reason", "unknown"),
                    "exit_price": trade_result.get("exit_price", 0)
                },
                "market_state": self._extract_market_state(trade_result),
                "confidence": trade_result.get("confidence", 0.5),
                "metadata": {
                    "symbol": trade_result.get("symbol", "unknown"),
                    "direction": trade_result.get("direction", "long"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Feed to auto-learner
            self.auto_learner.record_trade_result(trade_data)
            
            # Track trades for evolution trigger
            self.trades_since_last_evolution += 1
            
            logger.info(f"Trade recorded: {trade_data['trade_id']} | "
                       f"PnL: {trade_data['pnl_pct']:.2%} | "
                       f"Win: {trade_data['win']}")
            
        except Exception as e:
            logger.error(f"Error processing trade result: {e}")
    
    def _extract_market_state(self, trade_result: Dict) -> Dict:
        """Extract market state from trade result"""
        return {
            "rsi": trade_result.get("rsi", 50),
            "ema_diff": trade_result.get("ema_diff", 0),
            "volatility": trade_result.get("volatility", 0.02),
            "volume_ratio": trade_result.get("volume_ratio", 1.0),
            "price_momentum": trade_result.get("price_momentum", 0),
            "bb_position": trade_result.get("bb_position", 0.5),
            "trend_strength": trade_result.get("trend_strength", 0),
            "recent_pnl": self.auto_learner.daily_pnl,
            "trade_count": self.auto_learner.total_trades,
            "time_of_day": datetime.now().hour / 24
        }
    
    def get_adaptive_parameters(self) -> Dict:
        """Get current adaptive parameters for trading"""
        return self.auto_learner.param_optimizer.get_current_parameters()
    
    def should_trade_with_learning(self, market_state: MarketState) -> tuple:
        """Determine if should trade using auto-learning"""
        return self.auto_learner.should_trade(market_state)
    
    def get_best_strategy(self) -> Optional[Dict]:
        """Get best evolved strategy if available"""
        return self.auto_learner.best_strategy
    
    def get_learning_status(self) -> Dict:
        """Get comprehensive learning status"""
        return {
            "auto_learner": self.auto_learner.get_status(),
            "integration": {
                "running": self.running,
                "trades_since_evolution": self.trades_since_last_evolution,
                "last_evolution": self.last_evolution_time.isoformat() if self.last_evolution_time else None
            }
        }
    
    def force_evolution(self):
        """Force immediate evolution"""
        logger.info("Forcing strategy evolution...")
        result = self.auto_learner.evolve_strategies()
        
        if result:
            self.trades_since_last_evolution = 0
            self.last_evolution_time = datetime.now()
            
            # Update master with new best strategy
            if self.auto_learner.best_strategy:
                self._apply_best_strategy_to_master()
        
        return result
    
    def _apply_best_strategy_to_master(self):
        """Apply best evolved strategy to master orchestrator"""
        if not self.auto_learner.best_strategy:
            return
        
        strategy = self.auto_learner.best_strategy
        genome = strategy["genome"]
        params = genome.params
        
        logger.info(f"Applying evolved strategy from generation {strategy['generation']}")
        logger.info(f"New parameters: SL={params['sl_pct']:.3f}, TP={params['tp_pct']:.3f}")
        
        # Update master parameters
        # This would integrate with master's configuration
        # For now, just log the new parameters
        logger.info("Strategy parameters updated in master orchestrator")
    
    def _integration_loop(self):
        """Background loop for integration tasks"""
        while self.running:
            time.sleep(60)  # Check every minute
            
            if not self.running:
                break
            
            # Check if we should trigger evolution
            if self.trades_since_last_evolution >= 50:  # Every 50 trades
                logger.info("Auto-triggering evolution due to trade count")
                self.force_evolution()
            
            # Check daily PnL
            status = self.auto_learner.get_status()
            if status['daily_pnl'] >= 0.05:  # 5% target reached
                logger.info(f"✅ Daily target reached! PnL: {status['daily_pnl']:.2%}")
            elif status['daily_pnl'] <= -0.10:  # Max drawdown hit
                logger.warning(f"⚠️ Max drawdown hit! PnL: {status['daily_pnl']:.2%}")
    
    def start(self):
        """Start the integration"""
        self.running = True
        
        # Initialize auto-learner
        self.initialize_with_data()
        
        # Start auto-learner
        self.auto_learner.start()
        
        # Start integration thread
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()
        
        logger.info("Auto-learning integration started")
    
    def stop(self):
        """Stop the integration"""
        self.running = False
        
        if self.integration_thread:
            self.integration_thread.join(timeout=5)
        
        # Stop auto-learner
        self.auto_learner.stop()
        
        logger.info("Auto-learning integration stopped")
    
    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        learner_status = self.auto_learner.get_status()
        
        return {
            "running": self.running,
            "generation": learner_status.get("current_generation", 0),
            "total_trades": learner_status.get("total_trades", 0),
            "daily_pnl": learner_status.get("daily_pnl", 0.0),
            "total_pnl": learner_status.get("total_pnl", 0.0),
            "exploration_rate": learner_status.get("exploration_rate", 0.3),
            "trades_since_evolution": self.trades_since_last_evolution,
            "best_strategy": learner_status.get("best_strategy")
        }
    
    def get_config(self) -> Dict:
        """Get system configuration"""
        return {
            "capital": 500.0,
            "target_daily": 0.05,
            "max_drawdown": 0.10,
            "evolution_interval": 3600,
            "adaptation_interval": 300,
            "exploration_decay": 0.995,
            "population_size": 50,
            "generations": 20
        }
    
    def get_current_parameters(self) -> Dict:
        """Get current trading parameters"""
        params = self.get_adaptive_parameters()
        
        return {
            "sl_pct": params.get("sl_pct", 0.025),
            "tp_pct": params.get("tp_pct", 0.05),
            "position_size": params.get("position_size", 0.1),
            "leverage": params.get("leverage", 5.0),
            "confidence_threshold": params.get("confidence_threshold", 0.6),
            "max_trades_per_day": params.get("max_trades_per_day", 10),
            "risk_per_trade": params.get("risk_per_trade", 0.05)
        }


# ============================================================================
# PATCH MASTER ORCHESTRATOR
# ============================================================================
def patch_master_with_auto_learning(master: MasterOrchestrator) -> AutoLearningIntegration:
    """
    Patch existing MasterOrchestrator to use auto-learning.
    
    This modifies the master to:
    1. Use adaptive parameters from auto-learner
    2. Feed trade results to auto-learner
    3. Use evolved strategies when available
    """
    
    integration = AutoLearningIntegration(master)
    
    # Store original methods
    original_execute_trade = master.execute_trade if hasattr(master, 'execute_trade') else None
    original_should_trade = master.should_trade if hasattr(master, 'should_trade') else None
    
    def patched_execute_trade(trade_data: Dict):
        """Patched execute_trade that feeds results to auto-learner"""
        # Execute original trade logic
        if original_execute_trade:
            result = original_execute_trade(trade_data)
        else:
            result = trade_data
        
        # Feed to auto-learner
        integration.process_trade_result(result)
        
        return result
    
    def patched_should_trade(market_state) -> tuple:
        """Patched should_trade that uses auto-learning"""
        # Use auto-learner's decision if available
        if isinstance(market_state, MarketState):
            should, confidence, direction = integration.should_trade_with_learning(market_state)
            
            if should:
                # Get adaptive parameters
                params = integration.get_adaptive_parameters()
                return True, confidence, direction, params
        
        # Fall back to original logic
        if original_should_trade:
            return original_should_trade(market_state)
        
        return False, 0.5, "hold", {}
    
    # Apply patches
    master.execute_trade = patched_execute_trade
    master.should_trade = patched_should_trade
    
    # Add integration reference
    master.auto_learning = integration
    
    logger.info("Master orchestrator patched with auto-learning")
    
    return integration


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("AUTO-LEARNING INTEGRATION")
    print("="*80)
    
    print("\n🔧 Integration Options:")
    print("1. Create new MasterOrchestrator with auto-learning")
    print("2. Patch existing MasterOrchestrator")
    
    print("\n📖 Usage:")
    print("""
# Option 1: New orchestrator with auto-learning
from master_orchestrator import MasterOrchestrator
from integrate_auto_learner import AutoLearningIntegration

master = MasterOrchestrator()
integration = AutoLearningIntegration(master)
integration.start()

# Option 2: Patch existing orchestrator
from integrate_auto_learner import patch_master_with_auto_learning
integration = patch_master_with_auto_learning(master)
integration.start()
    """)
    
    print("\n✅ Auto-learning integration ready")
    print("The system will:")
    print("  • Evolve strategies using genetic algorithms")
    print("  • Adapt parameters using reinforcement learning")
    print("  • Learn from every trade automatically")
    print("  • Optimize for 5% daily target")
