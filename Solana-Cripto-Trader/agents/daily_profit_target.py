#!/usr/bin/env python3
"""
🎯 DAILY PROFIT TARGET MANAGER
Sistema automatizado para alcanzar 5% de profit diario

LÓGICA:
1. Sistema abre posiciones automáticamente cuando haya capital y señales válidas
2. Cuando TODAS las posiciones estén >1% profit, cerrar o risk manager decide
3. Risk Manager decide si dejar correr hasta 2%, 3%, 5% o más
4. Risk Manager decide si salir de posiciones que no ayudan
5. Objetivo: acumular profit hasta llegar al 5% diario

ARCHIVO: agents/daily_profit_target.py
"""

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATA_DIR = Path.home() / ".config" / "solana-jupiter-bot"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
STATE_FILE = DATA_DIR / "daily_profit_state.json"

# Targets escalonados
TARGETS = {
    "TIER_1": 0.01,  # 1% - asegurar ganancias mínimas
    "TIER_2": 0.02,  # 2% - dejar correr
    "TIER_3": 0.03,  # 3% - dejar correr
    "TIER_4": 0.05,  # 5% - objetivo diario
    "MAX": 0.10      # 10% - máximo permitido (salir rápido)
}

# ============================================================================
# ESTADO DEL DÍA
# ============================================================================
class DailyProfitState:
    """Gestiona el estado del profit diario"""

    def __init__(self):
        self.state_file = STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        """Carga el estado del archivo"""
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            self.daily_start_balance = data.get("daily_start_balance", 500.0)
            self.daily_profit_pct = data.get("daily_profit_pct", 0.0)
            self.daily_profit_usd = data.get("daily_profit_usd", 0.0)
            self.current_tier = data.get("current_tier", "TIER_1")
            self.date = data.get("date", "2026-01-01")
            self.trades_today = data.get("trades_today", 0)
            self.positions_closed_today = data.get("positions_closed_today", 0)
            self.last_decision = data.get("last_decision", {})
        else:
            self.reset_daily()

    def save(self):
        """Guarda el estado del archivo"""
        data = {
            "date": self.date,
            "daily_start_balance": self.daily_start_balance,
            "daily_profit_pct": self.daily_profit_pct,
            "daily_profit_usd": self.daily_profit_usd,
            "current_tier": self.current_tier,
            "trades_today": self.trades_today,
            "positions_closed_today": self.positions_closed_today,
            "last_decision": self.last_decision,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def reset_daily(self):
        """Resetea el estado para un nuevo día"""
        self.date = datetime.now(timezone.utc).date().isoformat()
        self.daily_start_balance = 500.0  # Capital inicial
        self.daily_profit_pct = 0.0
        self.daily_profit_usd = 0.0
        self.current_tier = "TIER_1"
        self.trades_today = 0
        self.positions_closed_today = 0
        self.last_decision = {}
        self.save()

    def check_new_day(self):
        """Verifica si es un nuevo día y reseta si es necesario"""
        today = datetime.now(timezone.utc).date().isoformat()
        if self.date != today:
            self.reset_daily()

    def update_profit(self, profit_usd: float, total_equity: float):
        """Actualiza el profit diario"""
        self.daily_profit_usd = profit_usd
        self.daily_profit_pct = profit_usd / self.daily_start_balance if self.daily_start_balance > 0 else 0.0

        # Actualizar tier actual
        self.current_tier = self.get_current_tier()
        self.save()

    def get_current_tier(self) -> str:
        """Determina el tier actual basado en el profit"""
        for tier, threshold in sorted(TARGETS.items(), key=lambda x: x[1], reverse=True):
            if self.daily_profit_pct >= threshold:
                return tier
        return "TIER_1"

    def add_trade(self):
        """Incrementa el contador de trades del día"""
        self.trades_today += 1
        self.save()

    def add_position_closed(self):
        """Incrementa el contador de posiciones cerradas del día"""
        self.positions_closed_today += 1
        self.save()

    def record_decision(self, decision: str, reason: str):
        """Registra una decisión del Risk Manager"""
        self.last_decision = {
            "decision": decision,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profit_pct": self.daily_profit_pct,
            "tier": self.current_tier
        }
        self.save()

# ============================================================================
# EVALUADOR DE PORTFOLIO
# ============================================================================
class PortfolioEvaluator:
    """Evalúa el portfolio para tomar decisiones de profit take"""

    def __init__(self):
        self.state = DailyProfitState()

    def load_portfolio(self) -> dict:
        """Carga el portfolio desde archivo"""
        if PORTFOLIO_FILE.exists():
            return json.loads(PORTFOLIO_FILE.read_text())
        return {}

    def calculate_total_pnl(self, portfolio: dict) -> Tuple[float, float]:
        """
        Calcula el P&L total del portfolio

        Returns:
            (pnl_usd, pnl_pct) - P&L en USD y porcentaje
        """
        positions = portfolio.get("positions", [])
        capital_initial = portfolio.get("initial_capital", 500.0)

        total_pnl = 0.0
        for pos in positions:
            if pos.get("status") == "open":
                total_pnl += pos.get("pnl_usd", 0.0)

        # Equity real = capital_libre + invertido + P&L abierto
        capital_free = portfolio.get("capital_usd", 0.0)
        total_invested = sum(p.get("margin_usd", 0.0) for p in positions if p.get("status") == "open")
        total_equity = capital_free + total_invested + total_pnl

        pnl_pct = (total_pnl / capital_initial) if capital_initial > 0 else 0.0

        return total_pnl, pnl_pct

    def evaluate_all_positive_threshold(self, portfolio: dict, threshold_pct: float = 0.01) -> bool:
        """
        Evalúa si TODAS las posiciones están en positivo por encima del threshold

        Args:
            portfolio: Portfolio actual
            threshold_pct: Threshold mínimo de profit por posición (default 1%)

        Returns:
            True si TODAS las posiciones > threshold_pct
        """
        positions = portfolio.get("positions", [])
        if not positions:
            return False

        for pos in positions:
            if pos.get("status") == "open":
                pnl_pct = pos.get("pnl_pct", 0.0)
                if pnl_pct < threshold_pct:
                    return False

        return True

    def count_positive_positions(self, portfolio: dict) -> int:
        """Cuenta cuántas posiciones están en positivo"""
        positions = portfolio.get("positions", [])
        count = 0
        for pos in positions:
            if pos.get("status") == "open" and pos.get("pnl_pct", 0.0) > 0:
                count += 1
        return count

    def count_negative_positions(self, portfolio: dict) -> int:
        """Cuenta cuántas posiciones están en negativo"""
        positions = portfolio.get("positions", [])
        count = 0
        for pos in positions:
            if pos.get("status") == "open" and pos.get("pnl_pct", 0.0) < 0:
                count += 1
        return count

    def get_worst_position(self, portfolio: dict) -> Optional[dict]:
        """Retorna la posición con el peor P&L"""
        positions = portfolio.get("positions", [])
        if not positions:
            return None

        worst = None
        worst_pnl = float('inf')

        for pos in positions:
            if pos.get("status") == "open":
                pnl = pos.get("pnl_pct", 0.0)
                if pnl < worst_pnl:
                    worst_pnl = pnl
                    worst = pos

        return worst

    def get_best_position(self, portfolio: dict) -> Optional[dict]:
        """Retorna la posición con el mejor P&L"""
        positions = portfolio.get("positions", [])
        if not positions:
            return None

        best = None
        best_pnl = -float('inf')

        for pos in positions:
            if pos.get("status") == "open":
                pnl = pos.get("pnl_pct", 0.0)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best = pos

        return best

# ============================================================================
# RISK MANAGER - DECISIONES DE PROFIT TAKE
# ============================================================================
class ProfitRiskManager:
    """Risk Manager especializado en decisiones de profit take"""

    def __init__(self):
        self.state = DailyProfitState()
        self.evaluator = PortfolioEvaluator()

    def evaluate_and_decide(self, portfolio: dict) -> dict:
        """
        Evalúa el portfolio y toma una decisión de profit take

        Returns:
            dict: {
                "action": "CLOSE_ALL" | "CLOSE_PARTIAL" | "CONTINUE" | "CLOSE_WORST",
                "reason": "explicación",
                "target_positions": [...],  # posiciones a cerrar
                "new_tier": "TIER_X",
                "confidence": float
            }
        """
        self.state.check_new_day()

        # Calcular profit actual
        total_pnl_usd, total_pnl_pct = self.evaluator.calculate_total_pnl(portfolio)
        self.state.update_profit(total_pnl_usd, 0)  # Actualizar con equity real

        # Contar posiciones
        positive_count = self.evaluator.count_positive_positions(portfolio)
        negative_count = self.evaluator.count_negative_positions(portfolio)
        total_positions = len(portfolio.get("positions", []))

        # Evaluar condiciones
        result = {
            "action": "CONTINUE",
            "reason": "",
            "target_positions": [],
            "new_tier": self.state.current_tier,
            "confidence": 0.5,
            "metrics": {
                "total_pnl_usd": total_pnl_usd,
                "total_pnl_pct": total_pnl_pct * 100,
                "positive_positions": positive_count,
                "negative_positions": negative_count,
                "total_positions": total_positions
            }
        }

        # CONDICIÓN 1: TODAS las posiciones >1% → Cerrar o risk manager decide
        all_positive = self.evaluator.evaluate_all_positive_threshold(portfolio, 0.01)

        if all_positive:
            # Evaluar tier actual
            current_tier_value = TARGETS.get(self.state.current_tier, 0.01)

            if total_pnl_pct >= 0.05:  # 5% alcanzado
                result["action"] = "CLOSE_ALL"
                result["reason"] = f"Objetivo diario alcanzado ({total_pnl_pct*100:.2f}% >= 5%)"
                result["confidence"] = 0.95
                result["target_positions"] = [p["id"] for p in portfolio.get("positions", []) if p.get("status") == "open"]

            elif total_pnl_pct >= 0.03:  # 3% - dejar correr o cerrar
                # Risk decision: si ya tuvimos un buen run, cerrar
                if self.state.trades_today >= 5 and total_pnl_pct > 0.04:
                    result["action"] = "CLOSE_ALL"
                    result["reason"] = f"3% alcanzado con buen run (+{total_pnl_pct*100:.2f}%, {self.state.trades_today} trades)"
                    result["confidence"] = 0.7
                    result["target_positions"] = [p["id"] for p in portfolio.get("positions", []) if p.get("status") == "open"]
                else:
                    result["action"] = "CONTINUE"
                    result["reason"] = f"3% alcanzado, dejar correr hacia 5% (actual: {total_pnl_pct*100:.2f}%)"
                    result["confidence"] = 0.6

            elif total_pnl_pct >= 0.02:  # 2% - dejar correr
                result["action"] = "CONTINUE"
                result["reason"] = f"2% asegurado, dejar correr hacia 3-5% (actual: {total_pnl_pct*100:.2f}%)"
                result["confidence"] = 0.7

            else:  # 1% - asegurarlo
                # Risk decision: cerrar para asegurar o dejar correr
                if positive_count == total_positions and total_pnl_pct < 0.015:
                    result["action"] = "CLOSE_ALL"
                    result["reason"] = f"1% asegurado para evitar riesgos (actual: {total_pnl_pct*100:.2f}%)"
                    result["confidence"] = 0.75
                    result["target_positions"] = [p["id"] for p in portfolio.get("positions", []) if p.get("status") == "open"]
                else:
                    result["action"] = "CONTINUE"
                    result["reason"] = f"1% alcanzado, dejando correr hacia 2%+ (actual: {total_pnl_pct*100:.2f}%)"
                    result["confidence"] = 0.65

        # CONDICIÓN 2: Posiciones negativas → considerar cerrar las peores
        elif negative_count > 0 and total_pnl_pct < 0:
            worst = self.evaluator.get_worst_position(portfolio)
            if worst and worst.get("pnl_pct", 0) < -0.02:  # Perdiendo más de 2%
                result["action"] = "CLOSE_WORST"
                result["reason"] = f"Cerrar posición perdedora ({worst['symbol']}: {worst.get('pnl_pct', 0)*100:.2f}%)"
                result["confidence"] = 0.8
                result["target_positions"] = [worst["id"]]
            else:
                result["action"] = "CONTINUE"
                result["reason"] = f"Todas las posiciones no son positivas, esperando ({negative_count} negativas)"
                result["confidence"] = 0.5

        # CONDICIÓN 3: Posiciones mixtas (algunas positivas, otras negativas)
        else:
            if positive_count > 0 and negative_count > 0:
                worst = self.evaluator.get_worst_position(portfolio)
                if worst and worst.get("pnl_pct", 0) < -0.015:
                    result["action"] = "CLOSE_WORST"
                    result["reason"] = f"Mixed: cerrar posición perdedora que no ayuda ({worst['symbol']})"
                    result["confidence"] = 0.7
                    result["target_positions"] = [worst["id"]]
                else:
                    result["action"] = "CONTINUE"
                    result["reason"] = f"Mixed: esperando que todas sean positivas ({positive_count}/{total_positions})"
                    result["confidence"] = 0.55

        # Actualizar tier
        result["new_tier"] = self.state.current_tier

        # Registrar decisión
        self.state.record_decision(result["action"], result["reason"])

        return result

# ============================================================================
# FUNCIÓN PRINCIPAL - Para integración con Orchestrator
# ============================================================================
def evaluate_profit_take_decision() -> dict:
    """
    Función principal para llamar desde Orchestrator

    Returns:
        dict: Decisión de profit take
    """
    risk_manager = ProfitRiskManager()
    portfolio = risk_manager.evaluator.load_portfolio()

    if not portfolio:
        return {
            "action": "CONTINUE",
            "reason": "No portfolio disponible",
            "target_positions": [],
            "new_tier": "TIER_1",
            "confidence": 0.0,
            "metrics": {}
        }

    decision = risk_manager.evaluate_and_decide(portfolio)
    return decision

# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🎯 DAILY PROFIT TARGET MANAGER - TEST")
    print("=" * 70)

    # Test 1: Portfolio vacío
    print("\n📋 Test 1: Portfolio vacío")
    decision = evaluate_profit_take_decision()
    print(f"  Action: {decision['action']}")
    print(f"  Reason: {decision['reason']}")

    # Test 2: Simular portfolio
    print("\n📋 Test 2: Portfolio simulado")
    test_portfolio = {
        "capital_usd": 125.0,
        "initial_capital": 500.0,
        "positions": [
            {
                "id": "TEST_1",
                "symbol": "SOL",
                "direction": "long",
                "pnl_usd": 10.0,
                "pnl_pct": 0.02,
                "status": "open"
            },
            {
                "id": "TEST_2",
                "symbol": "GOAT",
                "direction": "long",
                "pnl_usd": 5.0,
                "pnl_pct": 0.015,
                "status": "open"
            }
        ]
    }

    # Crear archivo temporal para test
    TEMP_FILE = Path.home() / ".config" / "solana-jupiter-bot" / "portfolio_test.json"
    TEMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEMP_FILE.write_text(json.dumps(test_portfolio, indent=2))

    risk_manager = ProfitRiskManager()
    decision = risk_manager.evaluate_and_decide(test_portfolio)

    print(f"  Action: {decision['action']}")
    print(f"  Reason: {decision['reason']}")
    print(f"  Metrics: {decision['metrics']}")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)
