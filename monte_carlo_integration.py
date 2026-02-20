#!/usr/bin/env python3
"""
Monte Carlo Trading Integration
================================
Wrapper para integrar el motor de Monte Carlo existente en el sistema de trading.

El módulo monte_carlo_engine.py debe estar disponible en:
- /home/enderj/monte-carlo-trading/monte_carlo_engine.py
- O en la raíz del proyecto

Características:
- Análisis probabilístico de estrategias
- Risk of Ruin calculator
- Simulaciones bootstrap
- Integración con el sistema de trading
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger("monte_carlo_integration")


# ============================================================================
# PATH SETUP
# ============================================================================

# Buscar el módulo monte_carlo_engine
MONTE_CARLO_PATHS = [
    Path(__file__).parent / "monte_carlo_engine.py",
    Path(__file__).parent / "monte_carlo_trading" / "monte_carlo_engine.py",
    Path("/home/enderj/monte-carlo-trading/monte_carlo_engine.py"),
    Path("/home/enderj/monte-carlo-trading/trading_monte_carlo.py"),
]

# Intentar importar el módulo
MonteCarloEngine = None
FellerRiskCalculator = None
analyze_trading_system = None

for path in MONTE_CARLO_PATHS:
    if path.exists():
        try:
            # Agregar el directorio al path
            sys.path.insert(0, str(path.parent))
            
            # Importar el módulo
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("monte_carlo_engine", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Obtener las clases y funciones
                MonteCarloEngine = getattr(module, "MonteCarloEngine", None)
                FellerRiskCalculator = getattr(module, "FellerRiskCalculator", None)
                analyze_trading_system = getattr(module, "analyze_trading_system", None)
                
                if MonteCarloEngine:
                    logger.info(f"✅ Monte Carlo module loaded from: {path}")
                    break
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to load from {path}: {e}")
            continue

# Si no se encontró, crear implementaciones placeholder
if MonteCarloEngine is None:
    logger.warning("⚠️ Monte Carlo module not found - using placeholder")
    
    # Placeholder Monte Carlo Engine
    class MonteCarloEngine:
        """Placeholder - Monte Carlo Engine no disponible"""
        
        def __init__(self, initial_balance=10000, num_simulations=10000, random_seed=None):
            self.initial_balance = initial_balance
            self.num_simulations = num_simulations
            self.trade_history = []
            logger.warning("⚠️ Using placeholder MonteCarloEngine")
        
        def add_trade(self, pnl_percent, timestamp=None):
            self.trade_history.append({"pnl_percent": pnl_percent, "timestamp": timestamp})
        
        def add_trades_from_list(self, trades):
            self.trade_history.extend([{"pnl_percent": t} for t in trades])
        
        def run_bootstrap_simulation(self):
            if not self.trade_history:
                return {"error": "No trades in history"}
            
            import random
            import numpy as np
            
            results = []
            for _ in range(self.num_simulations):
                trades = random.choices(self.trade_history, k=len(self.trade_history))
                balance = self.initial_balance
                for t in trades:
                    balance *= (1 + t["pnl_percent"] / 100)
                results.append(balance)
            
            return {
                "num_simulations": self.num_simulations,
                "mean_balance": np.mean(results),
                "median_balance": np.median(results),
                "std_balance": np.std(results),
                "percentile_10": np.percentile(results, 10),
                "percentile_50": np.percentile(results, 50),
                "percentile_90": np.percentile(results, 90),
                "percentile_95": np.percentile(results, 95),
            }
    
    FellerRiskCalculator = None
    analyze_trading_system = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MonteCarloConfig:
    """Configuración del análisis Monte Carlo"""
    # Habilitar análisis
    enabled: bool = False
    
    # Parámetros de simulación
    initial_balance: float = 10000.0
    num_simulations: int = 10000
    random_seed: int = 42
    
    # Cuándo ejecutar análisis
    run_on_trade: bool = False  # Después de cada trade
    run_on_schedule: bool = True  # En schedule
    schedule_interval_minutes: int = 60
    
    # Threshold de riesgo
    max_risk_of_ruin: float = 0.05  # 5% máximo
    min_confidence_level: float = 0.95  # 95%
    
    # Guardar resultados
    save_results: bool = True
    results_dir: str = "data/monte_carlo"


# ============================================================================
# MONTE CARLO ANALYZER
# ============================================================================

class MonteCarloAnalyzer:
    """
    Analizador Monte Carlo para el sistema de trading.
    
    Uso:
        analyzer = MonteCarloAnalyzer(config)
        
        # Agregar resultados de trades
        analyzer.add_trade(2.5)  # 2.5% profit
        analyzer.add_trade(-1.2)  # 1.2% loss
        
        # Ejecutar análisis
        result = await analyzer.analyze()
        
        # Obtener recomendaciones
        recommendation = analyzer.get_recommendation(result)
    """
    
    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        
        # Motor Monte Carlo
        self.engine: Optional[MonteCarloEngine] = None
        
        # Historial de trades
        self.trade_history: List[float] = []
        
        # Último análisis
        self.last_analysis: Optional[Dict] = None
        
        # Resultados guardados
        self.analyses: List[Dict] = []
        
        if self.config.enabled:
            self._init_engine()
        
        logger.info(f"🎲 Monte Carlo Analyzer initialized (enabled: {self.config.enabled})")
    
    def _init_engine(self):
        """Inicializar el motor Monte Carlo"""
        try:
            self.engine = MonteCarloEngine(
                initial_balance=self.config.initial_balance,
                num_simulations=self.config.num_simulations,
                random_seed=self.config.random_seed
            )
        except Exception as e:
            logger.error(f"❌ Failed to init Monte Carlo engine: {e}")
    
    def add_trade(self, pnl_percent: float):
        """Agregar un trade al historial"""
        self.trade_history.append(pnl_percent)
        
        if self.config.enabled and self.engine:
            self.engine.add_trade(pnl_percent)
        
        logger.debug(f"📊 Trade added: {pnl_percent:.2f}%")
    
    def add_trades(self, trades: List[float]):
        """Agregar múltiples trades"""
        for trade in trades:
            self.add_trade(trade)
    
    def clear_history(self):
        """Limpiar historial de trades"""
        self.trade_history = []
        if self.engine:
            self.engine = MonteCarloEngine(
                initial_balance=self.config.initial_balance,
                num_simulations=self.config.num_simulations,
                random_seed=self.config.random_seed
            )
    
    async def analyze(self, method: str = "bootstrap") -> Dict:
        """
        Ejecutar análisis Monte Carlo.
        
        Args:
            method: Método de simulación ('bootstrap', 'random_walk', 'block_bootstrap')
            
        Returns:
            Diccionario con resultados del análisis
        """
        if not self.config.enabled:
            return {"error": "Monte Carlo analysis disabled"}
        
        if len(self.trade_history) < 10:
            return {"error": "Insufficient trades for analysis (minimum 10)"}
        
        try:
            # Ejecutar simulación
            if method == "bootstrap" and self.engine:
                result = self.engine.run_bootstrap_simulation()
                
                # Convertir a dict si es un objeto
                if hasattr(result, 'summary'):
                    analysis = result.summary()
                else:
                    analysis = result
            
            elif method == "random_walk" and self.engine:
                # Calcular stats de los trades
                import numpy as np
                mean_return = np.mean(self.trade_history)
                std_return = np.std(self.trade_history)
                
                result = self.engine.run_random_walk(
                    mean_return=mean_return,
                    std_return=std_return,
                    num_trades=len(self.trade_history)
                )
                
                analysis = result.summary() if hasattr(result, 'summary') else result
            
            else:
                return {"error": f"Unknown method: {method}"}
            
            # Risk of Ruin
            if FellerRiskCalculator and len(self.trade_history) > 0:
                import numpy as np
                mean_return = float(np.mean(self.trade_history))
                std_return = float(np.std(self.trade_history))
                
                ror = FellerRiskCalculator.calculate_risk_of_ruin(
                    mean_return=mean_return,
                    std_return=std_return,
                    initial_balance=self.config.initial_balance
                )
                analysis["risk_of_ruin"] = ror
            
            # Agregar metadata
            analysis["timestamp"] = datetime.now().isoformat()
            analysis["num_trades"] = len(self.trade_history)
            analysis["method"] = method
            
            # Guardar resultado
            self.last_analysis = analysis
            self.analyses.append(analysis)
            
            if self.config.save_results:
                self._save_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return {"error": str(e)}
    
    def _save_analysis(self, analysis: Dict):
        """Guardar análisis en archivo"""
        try:
            import os
            os.makedirs(self.config.results_dir, exist_ok=True)
            
            filename = f"{self.config.results_dir}/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"💾 Analysis saved: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Save error: {e}")
    
    def get_recommendation(self, analysis: Dict = None) -> Dict:
        """
        Obtener recomendación basada en el análisis.
        
        Returns:
            Diccionario con recommendation, action, y reason
        """
        analysis = analysis or self.last_analysis
        
        if not analysis:
            return {
                "recommendation": "NEUTRAL",
                "action": "NO_ACTION",
                "reason": "No analysis available"
            }
        
        # Verificar Risk of Ruin
        ror = analysis.get("risk_of_ruin", {})
        if ror:
            prob_ruin = ror.get("probability_ruin", 1.0)
            
            if prob_ruin > self.config.max_risk_of_ruin:
                return {
                    "recommendation": "HIGH_RISK",
                    "action": "REDUCE_POSITION",
                    "reason": f"Risk of Ruin {prob_ruin:.2%} exceeds {self.config.max_risk_of_ruin:.2%}",
                    "details": ror
                }
        
        # Verificar balance final
        mean_balance = analysis.get("mean_balance", 0)
        initial = self.config.initial_balance
        
        if mean_balance < initial:
            return {
                "recommendation": "LOSING",
                "action": "STOP_TRADING",
                "reason": f"Expected balance ${mean_balance:.2f} below initial ${initial:.2f}",
                "details": analysis
            }
        
        p95 = analysis.get("percentile_95", 0)
        if p95 > initial * 2:
            return {
                "recommendation": "EXCELLENT",
                "action": "INCREASE_POSITION",
                "reason": f"P95 balance ${p95:.2f} is 2x initial - consider increasing",
                "details": analysis
            }
        
        return {
            "recommendation": "ACCEPTABLE",
            "action": "CONTINUE",
            "reason": "System performs within expected parameters",
            "details": analysis
        }
    
    def get_summary(self) -> Dict:
        """Obtener resumen del estado actual"""
        return {
            "enabled": self.config.enabled,
            "total_trades": len(self.trade_history),
            "recent_pnl": sum(self.trade_history[-10:]) if len(self.trade_history) >= 10 else sum(self.trade_history),
            "avg_pnl": sum(self.trade_history) / len(self.trade_history) if self.trade_history else 0,
            "analyses_run": len(self.analyses),
            "last_analysis": self.last_analysis.get("timestamp") if self.last_analysis else None
        }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def create_monte_carlo_config(
    enabled: bool = False,
    initial_balance: float = 10000,
    num_simulations: int = 10000,
    **kwargs
) -> MonteCarloConfig:
    """Crear configuración Monte Carlo desde kwargs"""
    return MonteCarloConfig(
        enabled=enabled,
        initial_balance=initial_balance,
        num_simulations=num_simulations,
        run_on_trade=kwargs.get("run_on_trade", False),
        run_on_schedule=kwargs.get("run_on_schedule", True),
        save_results=kwargs.get("save_results", True)
    )


async def create_analyzer(
    enabled: bool = False,
    **kwargs
) -> MonteCarloAnalyzer:
    """Crear analizador con configuración"""
    config = create_monte_carlo_config(enabled, **kwargs)
    return MonteCarloAnalyzer(config)


# ============================================================================
# MAIN TEST
# ============================================================================

async def test_monte_carlo():
    """Test del analizador Monte Carlo"""
    print("=" * 60)
    print("🎲 Monte Carlo Analyzer Test")
    print("=" * 60)
    
    # Create config
    config = MonteCarloConfig(
        enabled=True,
        initial_balance=10000,
        num_simulations=1000  # Reduced for test
    )
    
    # Create analyzer
    analyzer = MonteCarloAnalyzer(config)
    
    # Add sample trades
    sample_trades = [
        2.5, -1.2, 3.1, -0.8, 1.5, 4.2, -2.0, 0.5,
        1.8, -0.3, 2.2, 3.5, -1.5, 0.9, 2.0, -0.7,
        1.2, 2.8, -1.1, 0.4, 3.0, -2.5, 1.0, 0.2
    ]
    
    print(f"\n📊 Adding {len(sample_trades)} sample trades...")
    analyzer.add_trades(sample_trades)
    
    # Run analysis
    print("\n🔄 Running Monte Carlo analysis...")
    result = await analyzer.analyze(method="bootstrap")
    
    # Show results
    if "error" not in result:
        print("\n📈 Analysis Results:")
        print(f"   Mean Balance: ${result.get('mean_balance', 0):,.2f}")
        print(f"   Median Balance: ${result.get('median_balance', 0):,.2f}")
        print(f"   P50: ${result.get('percentile_50', 0):,.2f}")
        print(f"   P90: ${result.get('percentile_90', 0):,.2f}")
        print(f"   P95: ${result.get('percentile_95', 0):,.2f}")
        
        if "risk_of_ruin" in result:
            ror = result["risk_of_ruin"]
            print(f"\n⚠️  Risk of Ruin: {ror.get('probability_ruin', 0):.4%}")
            print(f"   Recommendation: {ror.get('recommendation', 'N/A')}")
        
        # Get trading recommendation
        print("\n🎯 Trading Recommendation:")
        rec = analyzer.get_recommendation(result)
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"   Action: {rec['action']}")
        print(f"   Reason: {rec['reason']}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    # Show summary
    print("\n📋 Summary:")
    summary = analyzer.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_monte_carlo())
