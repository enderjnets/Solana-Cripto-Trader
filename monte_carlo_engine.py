#!/usr/bin/env python3
"""
Monte Carlo Trading Engine
==========================
Implementación de análisis probabilístico para sistemas de trading.
Basado en: "Arquitectura e Implementación del Análisis de Montecarlo en Sistemas de Trading Algorítmico"

Características:
- Simulaciones de Monte Carlo con reshuffling de trades
- Cálculo de Risk of Ruin (RoR)
- Distribución de resultados (percentiles)
- Confidence intervals
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime


@dataclass
class TradeResult:
    """Resultado de un trade individual."""
    pnl_percent: float  # Porcentaje de gain/loss
    timestamp: Optional[str] = None
    
    @property
    def is_win(self) -> bool:
        return self.pnl_percent > 0


@dataclass
class MonteCarloResult:
    """Resultado de una simulación Monte Carlo."""
    num_simulations: int
    initial_balance: float
    final_balances: List[float]
    max_drawdowns: List[float]
    trade_sequences: List[List[TradeResult]]
    
    @property
    def mean_final_balance(self) -> float:
        return np.mean(self.final_balances)
    
    @property
    def median_final_balance(self) -> float:
        return np.median(self.final_balances)
    
    @property
    def std_final_balance(self) -> float:
        return np.std(self.final_balances)
    
    def percentile(self, p: float) -> float:
        """Percentil de balance final (0-100)."""
        return np.percentile(self.final_balances, p)
    
    @property
    def mean_max_drawdown(self) -> float:
        return np.mean(self.max_drawdowns)
    
    def percentile_drawdown(self, p: float) -> float:
        return np.percentile(self.max_drawdowns, p)
    
    @property
    def risk_of_ruin(self) -> float:
        """Probabilidad de perder >90% del capital."""
        ruin_count = sum(1 for b in self.final_balances if b < self.initial_balance * 0.1)
        return ruin_count / len(self.final_balances)
    
    def summary(self) -> Dict:
        """Resumen ejecutivo de resultados."""
        return {
            "simulations": self.num_simulations,
            "initial_balance": self.initial_balance,
            "mean_balance": self.mean_final_balance,
            "median_balance": self.median_final_balance,
            "std_balance": self.std_final_balance,
            "p10_balance": self.percentile(10),
            "p50_balance": self.percentile(50),
            "p90_balance": self.percentile(90),
            "p95_balance": self.percentile(95),
            "p99_balance": self.percentile(99),
            "mean_drawdown": self.mean_max_drawdown,
            "p95_drawdown": self.percentile_drawdown(95),
            "p99_drawdown": self.percentile_drawdown(99),
            "risk_of_ruin": self.risk_of_ruin,
            "profit_factor_expected": self.mean_final_balance / self.initial_balance
        }


class MonteCarloEngine:
    """
    Motor de simulaciones Monte Carlo para trading.
    
    Utiliza múltiples métodos de simulación:
    - Bootstrapping: Reshuffle de trades históricos
    - Random Walk: Simulación con distribución normal
    - Block Bootstrap: Reshuffle en bloques
    """
    
    def __init__(
        self,
        initial_balance: float = 10000,
        num_simulations: int = 10000,
        random_seed: Optional[int] = None
    ):
        self.initial_balance = initial_balance
        self.num_simulations = num_simulations
        self.trade_history: List[TradeResult] = []
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def add_trade(self, pnl_percent: float, timestamp: Optional[str] = None):
        """Agregar un trade al historial."""
        self.trade_history.append(TradeResult(pnl_percent=pnl_percent, timestamp=timestamp))
    
    def add_trades_from_list(self, trades: List[float]):
        """Agregar múltiples trades desde lista de PnL percentages."""
        for pnl in trades:
            self.add_trade(pnl)
    
    def _calculate_equity_curve(self, trades: List[TradeResult]) -> List[float]:
        """Calcular curva de equity para una secuencia de trades."""
        balance = self.initial_balance
        curve = [balance]
        
        for trade in trades:
            balance *= (1 + trade.pnl_percent / 100)
            curve.append(balance)
        
        return curve
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calcular drawdown máximo de una curva de equity."""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _reshuffle_trades(self, trades: List[TradeResult], length: int) -> List[TradeResult]:
        """Hacer reshuffle aleatorio de trades."""
        return random.choices(trades, k=length)
    
    def run_bootstrap_simulation(self) -> MonteCarloResult:
        """
        Ejecutar simulación Monte Carlo con método Bootstrap.
        
        Reshuffle aleatorio de trades históricos para generar
        miles de "historias alternativas".
        """
        if len(self.trade_history) == 0:
            raise ValueError("No hay trades en el historial. Agregar trades primero.")
        
        final_balances = []
        max_drawdowns = []
        
        trade_length = len(self.trade_history)
        
        for _ in range(self.num_simulations):
            # Reshuffle de trades
            shuffled = self._reshuffle_trades(self.trade_history, trade_length)
            
            # Calcular equity curve
            equity = self._calculate_equity_curve(shuffled)
            
            # Guardar resultados
            final_balances.append(equity[-1])
            max_drawdowns.append(self._calculate_max_drawdown(equity))
        
        return MonteCarloResult(
            num_simulations=self.num_simulations,
            initial_balance=self.initial_balance,
            final_balances=final_balances,
            max_drawdowns=max_drawdowns,
            trade_sequences=[]
        )
    
    def run_block_bootstrap(self, block_size: int = 10) -> MonteCarloResult:
        """
        Block Bootstrap - mantiene coherencia temporal de trades.
        """
        if len(self.trade_history) < block_size:
            raise ValueError(f"Necesitas al menos {block_size} trades para block bootstrap.")
        
        trades = self.trade_history
        n = len(trades)
        
        final_balances = []
        max_drawdowns = []
        
        for _ in range(self.num_simulations):
            # Crear secuencia con bloques
            sequence = []
            while len(sequence) < n:
                start = random.randint(0, n - block_size)
                block = trades[start:start + block_size]
                sequence.extend(block)
            
            # Recortar al tamaño original
            sequence = sequence[:n]
            
            equity = self._calculate_equity_curve(sequence)
            final_balances.append(equity[-1])
            max_drawdowns.append(self._calculate_max_drawdown(equity))
        
        return MonteCarloResult(
            num_simulations=self.num_simulations,
            initial_balance=self.initial_balance,
            final_balances=final_balances,
            max_drawdowns=max_drawdowns,
            trade_sequences=[]
        )
    
    def run_random_walk(
        self,
        mean_return: float,
        std_return: float,
        num_trades: int
    ) -> MonteCarloResult:
        """
        Random Walk - simula trades con distribución normal.
        
        Útil cuando no hay historial suficiente o para stress testing.
        """
        final_balances = []
        max_drawdowns = []
        
        for _ in range(self.num_simulations):
            # Generar returns aleatorios
            returns = np.random.normal(mean_return, std_return, num_trades)
            trades = [TradeResult(pnl_percent=r) for r in returns]
            
            equity = self._calculate_equity_curve(trades)
            final_balances.append(equity[-1])
            max_drawdowns.append(self._calculate_max_drawdown(equity))
        
        return MonteCarloResult(
            num_simulations=self.num_simulations,
            initial_balance=self.initial_balance,
            final_balances=final_balances,
            max_drawdowns=max_drawdowns,
            trade_sequences=[]
        )


class FellerRiskCalculator:
    """
    Calculadora de Risk of Ruin usando la fórmula de Feller.
    
    Para sistemas con tamaño de posición fraccional fijo:
    P(ruin) ≈ exp(-2 * z * μ / σ²)
    
    Donde:
    - z = fracción de capital que se pierde (e.g., 0.9 para 90%)
    - μ = retorno promedio por trade
    - σ = desviación estándar de retornos
    """
    
    @staticmethod
    def calculate_risk_of_ruin(
        mean_return: float,
        std_return: float,
        initial_balance: float,
        ruin_threshold: float = 0.1,  # 10% del capital
        fractional_loss: float = 0.9   # 90% de pérdida = ruin
    ) -> Dict:
        """
        Calcular probabilidad de ruina usando aproximación de Feller.
        
        Args:
            mean_return: Retorno promedio por trade (%)
            std_return: Desviación estándar de retornos (%)
            initial_balance: Balance inicial
            ruin_threshold: Balance mínimo como fracción (0.1 = 10%)
            fractional_loss: Fracción de pérdida que define ruina
            
        Returns:
            Diccionario con métricas de riesgo
        """
        if std_return == 0:
            # No variance - use deterministic calculation
            if mean_return > 0:
                return {
                    "probability_ruin": 0.0,
                    "probability_survival": 1.0,
                    "expected_trades_to_ruin": float('inf'),
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "risk_per_trade_percent": mean_return,
                    "recommendation": "🟢 Sin varianza - Sistema determinista"
                }
            else:
                return {
                    "probability_ruin": 1.0,
                    "probability_survival": 0.0,
                    "expected_trades_to_ruin": 0,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "risk_per_trade_percent": abs(mean_return),
                    "recommendation": "⛔ Sistema perdedor sin varianza"
                }
        
        # Parámetros
        z = fractional_loss
        mu = mean_return
        sigma = std_return
        
        # Aproximación de Feller (para ruina en frontera inferior)
        if mu > 0:
            # Sistema rentable - probabilidad de ruina muy baja
            exponent_capital = -2 * (initial_balance * (1 - ruin_threshold) * mu) / (sigma ** 2)
            prob_ruin = min(1.0, np.exp(exponent_capital / 100))
        else:
            # Sistema no rentable
            exponent = -2 * z * abs(mu) / (sigma ** 2)
            prob_ruin = min(1.0, np.exp(exponent))
        
        # Expected number of trades until ruin
        if mu > 0:
            expected_trades = np.log(ruin_threshold) / np.log(1 + mu / 100)
        else:
            expected_trades = float('inf')
        
        return {
            "probability_ruin": prob_ruin,
            "probability_survival": 1 - prob_ruin,
            "expected_trades_to_ruin": expected_trades if expected_trades > 0 else float('inf'),
            "mean_return": mu,
            "std_return": sigma,
            "risk_per_trade_percent": abs(mu) + 2 * sigma,
            "recommendation": FellerRiskCalculator._get_recommendation(prob_ruin)
        }
    
    @staticmethod
    def _get_recommendation(prob_ruin: float) -> str:
        """Obtener recomendación basado en probabilidad de ruina."""
        if prob_ruin < 0.01:
            return "🟢 EXCELENTE: Riesgo de ruina muy bajo (<1%)"
        elif prob_ruin < 0.05:
            return "🟡 BUENO: Riesgo bajo (1-5%)"
        elif prob_ruin < 0.10:
            return "🟠 MODERADO: Riesgo moderado (5-10%)"
        elif prob_ruin < 0.25:
            return "🔴 ELEVADO: Considerar reducir riesgo (10-25%)"
        else:
            return "⛔ CRÍTICO: Alto riesgo de ruina (>25%)"


def analyze_trading_system(
    trades: List[float],
    initial_balance: float = 10000,
    num_simulations: int = 10000
) -> Dict:
    """
    Análisis completo de un sistema de trading usando Monte Carlo.
    """
    mc = MonteCarloEngine(
        initial_balance=initial_balance,
        num_simulations=num_simulations,
        random_seed=42
    )
    mc.add_trades_from_list(trades)
    
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if losses and avg_loss > 0 else float('inf')
    
    mean_return = np.mean(trades)
    std_return = np.std(trades)
    
    result = mc.run_bootstrap_simulation()
    summary = result.summary()
    
    feller = FellerRiskCalculator.calculate_risk_of_ruin(
        mean_return=mean_return,
        std_return=std_return,
        initial_balance=initial_balance
    )
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "num_trades": len(trades),
            "initial_balance": initial_balance,
            "num_simulations": num_simulations
        },
        "basic_stats": {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "mean_return": mean_return,
            "std_return": std_return
        },
        "monte_carlo": summary,
        "risk_of_ruin": feller,
        "verdict": _generate_verdict(summary, feller)
    }
    
    return analysis


def _generate_verdict(mc_summary: Dict, feller: Dict) -> str:
    """Generar veredicto final del sistema."""
    ror = feller.get("probability_ruin", 1.0)
    p95_balance = mc_summary.get("p95_balance", 0)
    p95_dd = mc_summary.get("p95_drawdown", 100)
    
    score = 0
    
    if ror < 0.01:
        score += 30
    elif ror < 0.05:
        score += 20
    elif ror < 0.10:
        score += 10
    
    if p95_balance > mc_summary["initial_balance"] * 2:
        score += 40
    elif p95_balance > mc_summary["initial_balance"] * 1.5:
        score += 30
    elif p95_balance > mc_summary["initial_balance"]:
        score += 20
    
    if p95_dd < 10:
        score += 30
    elif p95_dd < 20:
        score += 20
    elif p95_dd < 30:
        score += 10
    
    if score >= 80:
        return f"🟢 EXCELENTE (Score: {score}/100) - Sistema robusto"
    elif score >= 60:
        return f"🟡 BUENO (Score: {score}/100) - Aceptable"
    elif score >= 40:
        return f"🟠 MODERADO (Score: {score}/100) - Requiere optimización"
    else:
        return f"🔴 RIESGO ALTO (Score: {score}/100) - No recomendado"


if __name__ == "__main__":
    sample_trades = [
        2.5, -1.2, 3.1, -0.8, 1.5, 4.2, -2.0, 0.5, 
        1.8, -0.3, 2.2, 3.5, -1.5, 0.9, 2.0, -0.7,
        1.2, 2.8, -1.1, 0.4, 3.0, -2.5, 1.0, 0.2
    ]
    
    print("=" * 60)
    print("MONTE CARLO TRADING ANALYSIS")
    print("=" * 60)
    
    analysis = analyze_trading_system(
        trades=sample_trades,
        initial_balance=10000,
        num_simulations=10000
    )
    
    print(f"\n📊 ESTADÍSTICAS BÁSICAS")
    print(f"   Trades: {analysis['basic_stats']['total_trades']}")
    print(f"   Win Rate: {analysis['basic_stats']['win_rate']:.1%}")
    print(f"   Profit Factor: {analysis['basic_stats']['profit_factor']:.2f}")
    
    mc = analysis['monte_carlo']
    print(f"\n🎲 RESULTADOS MONTE CARLO")
    print(f"   Balance Medio: ${mc['mean_balance']:,.2f}")
    print(f"   Balance P50: ${mc['p50_balance']:,.2f}")
    print(f"   Balance P95: ${mc['p95_balance']:,.2f}")
    print(f"   Drawdown P95: {mc['p95_drawdown']:.2f}%")
    
    ror = analysis['risk_of_ruin']
    print(f"\n⚠️  RISK OF RUIN")
    print(f"   Prob Ruina: {ror['probability_ruin']:.4%}")
    print(f"   {ror['recommendation']}")
    
    print(f"\n🏆 VEREDICTO: {analysis['verdict']}")
