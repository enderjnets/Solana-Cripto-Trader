#!/usr/bin/env python3
"""
AAA-M Self-Evolution Harness — Phase 2
AlphaEvolver-style parameter optimization for MiniMax M2.7.

Aplica automaticamente cambios de parametros sugeridos por el LLM,
con safety guards y rollback automatico.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger("aaa_m_evolution")

# -- Paths ------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "aaa_m_config.json"

# -- Safety Limits ----------------------------------------------------------

SAFETY_LIMITS = {
    "max_leverage": 10.0,
    "min_sl_pct": 0.01,
    "max_sl_pct": 0.10,
    "min_tp_pct": 0.02,
    "max_tp_pct": 0.20,
    "min_margin_pct": 0.005,
    "max_margin_pct": 0.10,
    "max_risk_per_trade_pct": 0.05,
    "min_liquidity_usd": 100000,
    "min_momentum_pct": 0.5,
    "max_momentum_pct": 10.0,
    "min_confidence_open": 0.50,
    "max_confidence_open": 0.95,
    "min_hold_minutes": 5,
    "max_hold_minutes": 120,
    "max_kelly_multiplier": 2.0,
}

# -- Config Loader ----------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Carga configuracion dinamica. Si no existe, crea defaults."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            log.warning(f"Error cargando config: {e}, usando defaults")
    config = build_default_config()
    save_config(config)
    return config


def build_default_config() -> Dict[str, Any]:
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "cycle_interval_sec": 30,
            "max_positions": 20,
            "min_confidence_open": 0.60,
            "min_momentum_pct": 2.0,
            "min_liquidity_usd": 200000,
            "default_sl_pct": 0.025,
            "default_tp_pct": 0.06,
            "default_leverage": 3,
            "default_margin_pct": 0.02,
            "max_hold_minutes": 30,
            "trailing_trigger_pct": 0.03,
            "trailing_stop_pct": 0.015,
            "kelly_multiplier": 1.0,
            "max_risk_per_trade_pct": 0.03,
        },
        "evolution_history": [],
        "last_applied": None,
    }


def save_config(config: Dict[str, Any]) -> None:
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


# -- Safety Guard -----------------------------------------------------------

class SafetyGuard:
    """Valida que los cambios de parametros sean seguros."""

    @staticmethod
    def validate_param_changes(changes: Dict[str, float]) -> tuple[bool, List[str]]:
        violations = []
        checks = [
            ("default_leverage", "max_leverage", ">", "Leverage excede maximo", None),
            ("default_sl_pct", "max_sl_pct", ">", "SL excede maximo", None),
            ("default_sl_pct", "min_sl_pct", "<", "SL por debajo del minimo", None),
            ("default_tp_pct", "max_tp_pct", ">", "TP excede maximo", None),
            ("default_tp_pct", "min_tp_pct", "<", "TP por debajo del minimo", None),
            ("default_margin_pct", "max_margin_pct", ">", "Margin excede maximo", None),
            ("default_margin_pct", "min_margin_pct", "<", "Margin por debajo del minimo", None),
            ("max_risk_per_trade_pct", "max_risk_per_trade_pct", ">", "Riesgo excede maximo", None),
            ("min_liquidity_usd", "min_liquidity_usd", "<", "Liquidez minima insuficiente", None),
            ("min_momentum_pct", "max_momentum_pct", ">", "Momentum excede maximo", None),
            ("min_momentum_pct", "min_momentum_pct", "<", "Momentum por debajo del minimo", None),
            ("min_confidence_open", "max_confidence_open", ">", "Confianza excede maximo", None),
            ("min_confidence_open", "min_confidence_open", "<", "Confianza por debajo del minimo", None),
            ("max_hold_minutes", "max_hold_minutes", ">", "Hold time excede maximo", None),
            ("max_hold_minutes", "min_hold_minutes", "<", "Hold time por debajo del minimo", None),
            ("kelly_multiplier", "max_kelly_multiplier", ">", "Kelly excede maximo", None),
        ]
        for param, limit_key, op, msg, _ in checks:
            if param not in changes:
                continue
            value = changes[param]
            limit = SAFETY_LIMITS.get(limit_key)
            if limit is None:
                continue
            if op == ">" and value > limit:
                violations.append(f"{msg}: {value} > {limit}")
            elif op == "<" and value < limit:
                violations.append(f"{msg}: {value} < {limit}")
        return len(violations) == 0, violations

    @staticmethod
    def should_rollback(current_sharpe: float, baseline_sharpe: float) -> bool:
        if baseline_sharpe <= 0:
            return False
        delta = (current_sharpe - baseline_sharpe) / abs(baseline_sharpe)
        return delta < -0.30


# -- Parameter Applier ------------------------------------------------------

class ParameterApplier:
    """Aplica cambios de parametros validados al config."""

    @staticmethod
    def apply_changes(config: Dict[str, Any], changes: Dict[str, float],
                      confidence: float, analysis: str) -> Dict[str, Any]:
        is_valid, violations = SafetyGuard.validate_param_changes(changes)
        if not is_valid:
            log.warning(f"Cambios RECHAZADOS por violaciones: {violations}")
            config["evolution_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "REJECTED",
                "changes": changes,
                "confidence": confidence,
                "violations": violations,
                "analysis": analysis[:200],
            })
            save_config(config)
            return config

        if confidence < 0.75:
            log.info(f"Cambios DEFERRED: confianza {confidence:.0%} < 75%")
            config["evolution_history"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "DEFERRED",
                "changes": changes,
                "confidence": confidence,
                "analysis": analysis[:200],
            })
            save_config(config)
            return config

        old_params = dict(config["parameters"])
        applied = {}
        for key, value in changes.items():
            if key in config["parameters"]:
                old_val = config["parameters"][key]
                config["parameters"][key] = value
                applied[key] = {"old": old_val, "new": value}
            else:
                log.warning(f"Parametro desconocido: {key}")

        config["last_applied"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_params": old_params,
            "applied_changes": applied,
            "confidence": confidence,
            "analysis": analysis[:200],
        }
        config["evolution_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "APPLIED",
            "changes": changes,
            "confidence": confidence,
            "analysis": analysis[:200],
        })
        save_config(config)
        log.info(f"Parametros evolutivos APLICADOS: {list(applied.keys())}")
        return config

    @staticmethod
    def rollback(config: Dict[str, Any], reason: str) -> Dict[str, Any]:
        last = config.get("last_applied")
        if not last or not last.get("baseline_params"):
            log.warning("No hay baseline para rollback")
            return config
        baseline = last["baseline_params"]
        config["parameters"].update(baseline)
        config["last_applied"] = None
        config["evolution_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "ROLLBACK",
            "reason": reason,
            "restored_params": baseline,
        })
        save_config(config)
        log.info(f"Rollback completado. Restaurados: {list(baseline.keys())}")
        return config


# -- Integration Helpers ----------------------------------------------------

def get_effective_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    return config.get("parameters", build_default_config()["parameters"])


def check_and_rollback_if_needed(current_sharpe: float,
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    last = config.get("last_applied")
    if not last:
        return config
    baseline_sharpe = last.get("baseline_sharpe", 0.0)
    if baseline_sharpe == 0.0:
        return config
    if SafetyGuard.should_rollback(current_sharpe, baseline_sharpe):
        config = ParameterApplier.rollback(config,
            f"Sharpe degradado: {current_sharpe:.2f} vs baseline {baseline_sharpe:.2f}")
    return config


def record_baseline_sharpe(sharpe: float,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    last = config.get("last_applied")
    if last and "baseline_sharpe" not in last:
        last["baseline_sharpe"] = sharpe
        save_config(config)
    return config
