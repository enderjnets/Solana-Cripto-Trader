#!/usr/bin/env python3
"""
🧠 Agente: Auto-Learner - Sistema de Autoaprendizaje Simplificado
===================================================================
Versión simplificada para integración con sistema modular.

Características:
- Adaptive Parameters - Ajuste dinámico de SL/TP/leverage
- Performance Feedback - Aprendizaje de resultados
- Confidence Tuning - Ajuste de confianza basado en historial
- Risk Adaptation - Gestión de riesgo adaptativa

Objetivo: Optimizar parámetros en tiempo real
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import sqlite3

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
HISTORY_FILE = DATA_DIR / "trade_history.json"
LEARNER_STATE = DATA_DIR / "auto_learner_state.json"
LEARNER_DB = DATA_DIR / "auto_learner.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("auto_learner")

# ─── Parámetros de Aprendizaje ────────────────────────────────────────────────

ADAPTATION_RATE = 0.2    # Qué tan rápido adaptar (0-1) — aumentado de 0.1
MIN_TRADES_TO_LEARN = 5  # Mínimo trades antes de ajustar — reducido de 10
PERFORMANCE_WINDOW = 30  # Últimos 30 trades para análisis — aumentado de 20
MIN_CONFIDENCE = 0.5    # Confianza mínima
MAX_CONFIDENCE = 0.95   # Confianza máxima

# Parámetros adaptables
DEFAULT_PARAMS = {
    "sl_pct": 0.025,        # 2.5% SL
    "tp_pct": 0.05,         # 5% TP
    "leverage_tier": 2,     # 0=conservative, 1=moderate, 2=aggressive
    "risk_per_trade": 0.02, # 2% por trade
    "max_positions": 5      # Máximo posiciones
}

# Tiers de leverage
LEVERAGE_TIERS = {
    0: {"name": "CONSERVATIVE", "min_lev": 1.0, "max_lev": 2.0},
    1: {"name": "MODERATE", "min_lev": 2.0, "max_lev": 5.0},
    2: {"name": "AGGRESSIVE", "min_lev": 5.0, "max_lev": 10.0}
}

# ─── Base de Datos ────────────────────────────────────────────────────────────

class LearnerDB:
    """Base de datos para historial de aprendizaje."""

    def __init__(self, db_path: Path = LEARNER_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    trade_id TEXT UNIQUE,
                    symbol TEXT,
                    direction TEXT,
                    sl_pct REAL,
                    tp_pct REAL,
                    leverage REAL,
                    pnl_usd REAL,
                    pnl_pct REAL,
                    win BOOLEAN,
                    confidence REAL,
                    holding_time REAL
                )
            """)
            # Limpiar duplicados existentes (mantener solo la primera inserción por trade_id)
            conn.execute("""
                DELETE FROM trade_results
                WHERE id NOT IN (
                    SELECT MIN(id) FROM trade_results
                    WHERE trade_id IS NOT NULL
                    GROUP BY trade_id
                )
                AND trade_id IS NOT NULL
            """)
            # Crear índice único si no existe (por si la tabla ya existía sin él)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_id_unique
                ON trade_results(trade_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    sl_pct REAL,
                    tp_pct REAL,
                    leverage_tier INTEGER,
                    risk_per_trade REAL,
                    win_rate REAL,
                    avg_pnl REAL
                )
            """)

    def record_trade(self, trade_data: dict):
        """Registra resultado de trade. INSERT OR IGNORE evita duplicados por trade_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO trade_results (
                    trade_id, symbol, direction, sl_pct, tp_pct, leverage,
                    pnl_usd, pnl_pct, win, confidence, holding_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get("trade_id"),
                trade_data.get("symbol"),
                trade_data.get("direction"),
                trade_data.get("sl_pct"),
                trade_data.get("tp_pct"),
                trade_data.get("leverage"),
                trade_data.get("pnl_usd"),
                trade_data.get("pnl_pct"),
                trade_data.get("win"),
                trade_data.get("confidence"),
                trade_data.get("holding_time")
            ))
            conn.commit()

    def get_recent_performance(self, limit: int = PERFORMANCE_WINDOW) -> dict:
        """Obtiene performance reciente."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM trade_results
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

        trades = [dict(row) for row in rows]

        if not trades:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "avg_pnl": 0}

        wins = sum(1 for t in trades if t["win"])
        losses = len(trades) - wins
        win_rate = wins / len(trades) if trades else 0
        avg_pnl = sum(t["pnl_usd"] for t in trades) / len(trades)

        return {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "data": trades
        }

    def record_parameters(self, params: dict, performance: dict):
        """Registra parámetros y performance."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO parameter_history (
                    sl_pct, tp_pct, leverage_tier, risk_per_trade, win_rate, avg_pnl
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                params.get("sl_pct"),
                params.get("tp_pct"),
                params.get("leverage_tier"),
                params.get("risk_per_trade"),
                performance.get("win_rate", 0),
                performance.get("avg_pnl", 0)
            ))
            conn.commit()


# ─── Carga / Guardado de Estado ───────────────────────────────────────────

def load_portfolio() -> dict:
    """Carga portfolio."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"capital_usd": 500.0, "positions": []}


def load_history() -> list:
    """Carga historial de trades."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def load_learner_state() -> dict:
    """Carga estado del auto-learner."""
    if LEARNER_STATE.exists():
        with open(LEARNER_STATE) as f:
            state = json.load(f)
            # Actualizar params si faltan
            if "params" not in state:
                state["params"] = DEFAULT_PARAMS.copy()
            return state
    return {
        "params": DEFAULT_PARAMS.copy(),
        "total_trades_learned": 0,
        "last_adaptation": None,
        "adaptation_count": 0
    }


def save_learner_state(state: dict):
    """Guarda estado del auto-learner."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(LEARNER_STATE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Lógica de Aprendizaje ───────────────────────────────────────────────────

def calculate_adaptive_params(state: dict, performance: dict) -> dict:
    """
    Calcula nuevos parámetros adaptados basados en performance.

    Reglas:
    - Win rate alto (>60%) → Más agresivo (aumentar leverage, riesgo)
    - Win rate bajo (<40%) → Más conservador (reducir leverage, riesgo)
    - Avg PnL positivo → Mantener dirección
    - Avg PnL negativo → Revertir dirección
    """
    params = state["params"].copy()

    win_rate = performance.get("win_rate", 0.5)
    avg_pnl = performance.get("avg_pnl", 0)
    trades = performance.get("trades", 0)

    # No adaptar si hay pocos trades
    if trades < MIN_TRADES_TO_LEARN:
        return params

    log.info(f"🧠 Adaptando parámetros... WR: {win_rate*100:.1f}% | Avg PnL: ${avg_pnl:.2f}")

    # Ajustar SL/TP basado en win rate
    if win_rate > 0.6:
        # Buen performance → Dar más espacio a ganancias
        params["sl_pct"] = max(0.015, params["sl_pct"] * (1 - ADAPTATION_RATE * 0.5))
        params["tp_pct"] = min(0.08, params["tp_pct"] * (1 + ADAPTATION_RATE * 0.3))
        log.info(f"  ✅ Buen WR → Ajustando SL: {params['sl_pct']*100:.2f}%, TP: {params['tp_pct']*100:.2f}%")
    elif win_rate < 0.4:
        # Mal performance → Proteger capital (SL más estricto)
        params["sl_pct"] = min(0.04, params["sl_pct"] * (1 + ADAPTATION_RATE * 0.5))
        params["tp_pct"] = max(0.03, params["tp_pct"] * (1 - ADAPTATION_RATE * 0.3))
        log.info(f"  ⚠️ Mal WR → Ajustando SL: {params['sl_pct']*100:.2f}%, TP: {params['tp_pct']*100:.2f}%")

    # Ajustar leverage tier basado en performance
    if win_rate > 0.65 and avg_pnl > 0:
        # Excelente performance → Aumentar tier
        params["leverage_tier"] = min(2, params["leverage_tier"] + 1)
        log.info(f"  🚀 Excelente performance → Tier: {LEVERAGE_TIERS[params['leverage_tier']]['name']}")
    elif win_rate < 0.35 or (win_rate < 0.45 and avg_pnl < 0):
        # Mal performance → Reducir tier
        params["leverage_tier"] = max(0, params["leverage_tier"] - 1)
        log.info(f"  🛡️ Mal performance → Tier: {LEVERAGE_TIERS[params['leverage_tier']]['name']}")

    # Ajustar riesgo por trade
    if win_rate > 0.6 and avg_pnl > 0:
        # Confianza alta → Aumentar riesgo ligeramente
        params["risk_per_trade"] = min(0.03, params["risk_per_trade"] * (1 + ADAPTATION_RATE * 0.2))
    elif win_rate < 0.4 or avg_pnl < -5:
        # Pérdidas → Reducir riesgo
        params["risk_per_trade"] = max(0.01, params["risk_per_trade"] * (1 - ADAPTATION_RATE * 0.5))

    # Asegurar límites
    params["sl_pct"] = max(0.01, min(0.05, params["sl_pct"]))
    params["tp_pct"] = max(0.03, min(0.10, params["tp_pct"]))
    params["risk_per_trade"] = max(0.005, min(0.04, params["risk_per_trade"]))

    return params


def calculate_confidence(history: list, recent_performance: dict) -> float:
    """
    Calcula confianza dinámica basada en historial.

    Factores:
    - Win rate reciente
    - Tendencia de PnL
    - Volatilidad de resultados
    """
    if not history or len(history) < MIN_TRADES_TO_LEARN:
        return MIN_CONFIDENCE

    win_rate = recent_performance.get("win_rate", 0.5)
    avg_pnl = recent_performance.get("avg_pnl", 0)

    # Base de confianza = win rate
    confidence = win_rate

    # Boost si PnL positivo
    if avg_pnl > 0:
        confidence += 0.1 * min(1, avg_pnl / 10)
    elif avg_pnl < 0:
        confidence -= 0.1 * min(1, abs(avg_pnl) / 10)

    # Limitar rango
    confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

    return round(confidence, 2)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(debug: bool = False) -> dict:
    log.info("=" * 50)
    log.info("🧠 AUTO-LEARNER — Sistema de Autoaprendizaje")
    log.info("=" * 50)

    # Cargar datos
    portfolio = load_portfolio()
    history = load_history()
    state = load_learner_state()
    db = LearnerDB()

    # Sincronizar historial con DB
    recent_trades = history[-PERFORMANCE_WINDOW:]
    for trade in recent_trades:
        if trade.get("status") == "closed":
            try:
                db.record_trade({
                    "trade_id": trade.get("id"),
                    "symbol": trade.get("symbol"),
                    "direction": trade.get("direction"),
                    "sl_pct": state["params"].get("sl_pct"),
                    "tp_pct": state["params"].get("tp_pct"),
                    "leverage": 1.0,  # Ajustar si hay leverage
                    "pnl_usd": trade.get("pnl_usd", 0),
                    "pnl_pct": trade.get("pnl_pct", 0),
                    "win": trade.get("pnl_usd", 0) > 0,
                    "confidence": trade.get("confidence", 0.5),
                    "holding_time": 0  # Calcular si hay timestamps
                })
            except Exception as e:
                pass

    # Obtener performance reciente
    performance = db.get_recent_performance()
    win_rate = performance["win_rate"]
    avg_pnl = performance["avg_pnl"]
    total_trades = performance["trades"]

    log.info(f"📊 Performance reciente:")
    log.info(f"   Trades: {total_trades}")
    log.info(f"   Win Rate: {win_rate*100:.1f}%")
    log.info(f"   Avg PnL: ${avg_pnl:+.2f}")

    # Calcular nueva confianza
    confidence = calculate_confidence(recent_trades, performance)
    log.info(f"🎯 Confianza dinámica: {confidence*100:.0f}%")

    # Adaptar parámetros si hay suficientes trades
    if total_trades >= MIN_TRADES_TO_LEARN:
        new_params = calculate_adaptive_params(state, performance)

        # Guardar historial de parámetros
        db.record_parameters(new_params, performance)

        # Actualizar estado
        state["params"] = new_params
        state["total_trades_learned"] += total_trades
        state["last_adaptation"] = datetime.now(timezone.utc).isoformat()
        state["adaptation_count"] += 1

        log.info(f"✅ Parámetros adaptados (adaptación #{state['adaptation_count']})")
    else:
        log.info(f"⏸️ Esperando más trades (necesita {MIN_TRADES_TO_LEARN})")

    # Guardar estado
    save_learner_state(state)

    # Reporte
    params = state["params"]
    tier = LEVERAGE_TIERS[params["leverage_tier"]]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_trades_learned": state["total_trades_learned"],
        "adaptation_count": state["adaptation_count"],
        "last_adaptation": state.get("last_adaptation"),
        "performance": {
            "win_rate": round(win_rate * 100, 1),
            "avg_pnl": round(avg_pnl, 2),
            "recent_trades": total_trades
        },
        "confidence": round(confidence, 2),
        "adaptive_params": {
            "sl_pct": round(params["sl_pct"] * 100, 2),
            "tp_pct": round(params["tp_pct"] * 100, 2),
            "leverage_tier": params["leverage_tier"],
            "leverage_tier_name": tier["name"],
            "leverage_range": f"{tier['min_lev']:.0f}x-{tier['max_lev']:.0f}x",
            "risk_per_trade": round(params["risk_per_trade"] * 100, 1),
            "max_positions": params["max_positions"]
        }
    }

    # Guardar reporte
    output_file = DATA_DIR / "auto_learner_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"💾 Reporte guardado en {output_file}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-Learner - Sistema de Autoaprendizaje")
    parser.add_argument("--debug", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("auto_learner").setLevel(logging.DEBUG)

    run(debug=args.debug)
