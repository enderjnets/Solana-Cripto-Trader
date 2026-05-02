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



# =============================================================================
# PHASE 3: Auto-Strategy Evolution via A/B Testing
# =============================================================================

class VariantRegistry:
    """Manages strategy variants in the config file."""

    @staticmethod
    def get_active_variant(config: Dict[str, Any]) -> Dict[str, Any]:
        vid = config.get("active_variant_id", "v1")
        return config.get("variants", {}).get(vid, {})

    @staticmethod
    def get_variant(config: Dict[str, Any], vid: str) -> Optional[Dict[str, Any]]:
        return config.get("variants", {}).get(vid)

    @staticmethod
    def create_variant(config: Dict[str, Any], variant_id: str, name: str,
                       system_prompt_addon: str, filter_rules: Dict[str, Any],
                       entry_logic: str = "momentum_breakout",
                       exit_logic: str = "fixed_sl_tp") -> Dict[str, Any]:
        variants = config.setdefault("variants", {})
        variants[variant_id] = {
            "name": name,
            "system_prompt_addon": system_prompt_addon,
            "filter_rules": filter_rules,
            "entry_logic": entry_logic,
            "exit_logic": exit_logic,
            "sharpe_baseline": None,
            "trades_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Purge old variants if > 5
        if len(variants) > 5:
            sorted_v = sorted(variants.items(),
                              key=lambda x: x[1].get("sharpe_baseline") or -999)
            to_remove = [k for k, _ in sorted_v[:-5]]
            for k in to_remove:
                if k != config.get("active_variant_id") and k != config.get("ab_test", {}).get("control_variant"):
                    del variants[k]
        save_config(config)
        return variants[variant_id]

    @staticmethod
    def promote_variant(config: Dict[str, Any], variant_id: str) -> None:
        if variant_id in config.get("variants", {}):
            config["active_variant_id"] = variant_id
            save_config(config)
            log.info(f"Variant {variant_id} PROMOTED to active")


class ABTestManager:
    """Runs A/B tests between control and test variants."""

    TEST_CYCLES = 100  # 100 cycles ~ 50 min
    MIN_TRADES_FOR_EVAL = 3
    IMPROVEMENT_THRESHOLD = 1.10  # 10% improvement required
    MAX_TEST_DURATION_HOURS = 48

    @staticmethod
    def start_test(config: Dict[str, Any], control_id: str, test_id: str) -> None:
        ab = config.setdefault("ab_test", {})
        ab["status"] = "running"
        ab["control_variant"] = control_id
        ab["test_variant"] = test_id
        ab["cycles_remaining"] = ABTestManager.TEST_CYCLES
        ab["sharpe_control"] = 0.0
        ab["sharpe_test"] = 0.0
        ab["trades_control"] = 0
        ab["trades_test"] = 0
        ab["start_time"] = datetime.now(timezone.utc).isoformat()
        save_config(config)
        log.info(f"A/B TEST STARTED: {control_id} vs {test_id}")

    @staticmethod
    def record_cycle(config: Dict[str, Any], current_variant_id: str,
                     sharpe: float, trade_count: int) -> None:
        ab = config.get("ab_test", {})
        if ab.get("status") != "running":
            return
        control_id = ab.get("control_variant")
        test_id = ab.get("test_variant")
        if current_variant_id == control_id:
            ab["sharpe_control"] = sharpe
            ab["trades_control"] = trade_count
        elif current_variant_id == test_id:
            ab["sharpe_test"] = sharpe
            ab["trades_test"] = trade_count
        ab["cycles_remaining"] = max(0, ab.get("cycles_remaining", 0) - 1)
        save_config(config)

    @staticmethod
    def should_switch_variant(config: Dict[str, Any]) -> bool:
        ab = config.get("ab_test", {})
        if ab.get("status") != "running":
            return False
        cycles = ab.get("cycles_remaining", 0)
        total = ABTestManager.TEST_CYCLES
        # First half = control, second half = test
        return cycles > total // 2

    @staticmethod
    def get_current_variant_id(config: Dict[str, Any]) -> str:
        ab = config.get("ab_test", {})
        if ab.get("status") != "running":
            return config.get("active_variant_id", "v1")
        # First half of test cycles = control, second half = test
        cycles = ab.get("cycles_remaining", 0)
        total = ABTestManager.TEST_CYCLES
        if cycles > total // 2:
            return ab.get("control_variant", "v1")
        return ab.get("test_variant", "v1")

    @staticmethod
    def evaluate_and_finalize(config: Dict[str, Any]) -> Optional[str]:
        ab = config.get("ab_test", {})
        if ab.get("status") != "running":
            return None

        # Check timeout
        start = ab.get("start_time")
        if start:
            started = datetime.fromisoformat(start)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 3600
            if elapsed > ABTestManager.MAX_TEST_DURATION_HOURS:
                ab["status"] = "timeout"
                save_config(config)
                log.warning("A/B TEST TIMEOUT after 48h")
                return "timeout"

        cycles = ab.get("cycles_remaining", 0)
        if cycles > 0:
            return None  # Still running

        # Evaluation
        sharpe_c = ab.get("sharpe_control", 0.0)
        sharpe_t = ab.get("sharpe_test", 0.0)
        trades_c = ab.get("trades_control", 0)
        trades_t = ab.get("trades_test", 0)

        log.info(f"A/B TEST EVALUATION: Control={sharpe_c:.2f}({trades_c}T) vs Test={sharpe_t:.2f}({trades_t}T)")

        if trades_c < ABTestManager.MIN_TRADES_FOR_EVAL and trades_t < ABTestManager.MIN_TRADES_FOR_EVAL:
            ab["status"] = "insufficient_data"
            save_config(config)
            log.info("A/B TEST: Insufficient trades, descartado")
            return "insufficient_data"

        # If test is clearly worse
        if sharpe_t < 0 and sharpe_c >= 0:
            ab["status"] = "test_failed"
            save_config(config)
            log.info("A/B TEST: Test failed (negative sharpe vs positive control)")
            return "test_failed"

        # Check improvement threshold
        if sharpe_c <= 0:
            baseline = 0.001
        else:
            baseline = sharpe_c
        improvement = sharpe_t / baseline if baseline > 0 else 0

        if improvement >= ABTestManager.IMPROVEMENT_THRESHOLD:
            test_id = ab.get("test_variant")
            VariantRegistry.promote_variant(config, test_id)
            ab["status"] = "promoted"
            save_config(config)
            log.info(f"A/B TEST: Test PROMOTED (improvement={improvement:.2f}x)")
            return "promoted"

        ab["status"] = "test_rejected"
        save_config(config)
        log.info(f"A/B TEST: Test rejected (improvement={improvement:.2f}x < {ABTestManager.IMPROVEMENT_THRESHOLD})")
        return "test_rejected"


class VariantGenerator:
    """Uses MiniMax M2.7 to generate new strategy variants."""

    @staticmethod
    def generate_variant(analysis: Dict[str, Any], current_variant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ask M2.7 to generate a new strategy variant based on analysis."""
        try:
            from llm_config import call_minimax_m2_7
        except ImportError:
            log.warning("call_minimax_m2_7 no disponible para VariantGenerator")
            return None

        recommendations = analysis.get("recommendations", [])
        param_changes = analysis.get("param_changes", {})
        current_prompt_addon = current_variant.get("system_prompt_addon", "")
        current_filters = current_variant.get("filter_rules", {})

        system = "Eres un quant strategist. Diseña variantes de estrategia de trading. Responde SOLO en JSON valido."
        prompt = f"""Basado en este analisis de trading, genera una NUEVA variante de estrategia:

ANALISIS:
- Recomendaciones: {recommendations}
- Cambios de parametros sugeridos: {param_changes}
- Confianza: {analysis.get('confidence', 0):.0%}

VARIANTE ACTUAL:
- Prompt addon: {current_prompt_addon or '(ninguno)'}
- Filtros: {current_filters}
- Entry logic: {current_variant.get('entry_logic', 'momentum_breakout')}

Genera una variante que TESTEE una hipotesis diferente. No solo ajustes numericos — cambia el ENFOQUE estrategico.

Responde en JSON:
{{
  "variant_name": "Nombre descriptivo",
  "system_prompt_addon": "Instruccion adicional de max 300 chars para el system prompt",
  "filter_changes": {{
    "min_momentum_pct": valor,
    "min_liquidity_usd": valor,
    "min_volume_24h": valor
  }},
  "entry_logic": "descripcion corta",
  "exit_logic": "descripcion corta",
  "confidence": 0.0-1.0
}}"""

        response = call_minimax_m2_7(prompt, system=system, max_tokens=1500)
        if not response:
            return None

        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            result = json.loads(text)
            if result.get("confidence", 0) < 0.75:
                log.info(f"VariantGenerator: confianza {result.get('confidence', 0):.0%} < 75%, descartada")
                return None
            return result
        except Exception as e:
            log.warning(f"VariantGenerator: error parseando respuesta: {e}")
            return None


# -- Phase 3 Integration Helpers --------------------------------------------

def get_active_variant(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    # Check if A/B test is running and we should use control or test
    if config.get("ab_test", {}).get("status") == "running":
        return VariantRegistry.get_variant(config, ABTestManager.get_current_variant_id(config)) or {}
    return VariantRegistry.get_active_variant(config)


def maybe_start_ab_test(config: Dict[str, Any], analysis: Dict[str, Any]) -> bool:
    """If analysis suggests strategic changes, generate variant and start A/B test."""
    ab = config.get("ab_test", {})
    if ab.get("status") == "running":
        return False  # Already running

    # Only start if there are non-numeric recommendations
    recommendations = analysis.get("recommendations", [])
    has_strategic = any(r for r in recommendations if any(word in r.lower() for word in [
        "filtro", "filter", "momentum", "volumen", "volume", "breakout", "tendencia", "trend",
        "prompt", "estrategia", "strategy", "enfoque", "focus"
    ]))
    if not has_strategic:
        return False

    current = VariantRegistry.get_active_variant(config)
    variant_data = VariantGenerator.generate_variant(analysis, current)
    if not variant_data:
        return False

    vid = f"v{len(config.get('variants', {})) + 1}"
    VariantRegistry.create_variant(
        config, vid,
        name=variant_data.get("variant_name", f"Variant {vid}"),
        system_prompt_addon=variant_data.get("system_prompt_addon", ""),
        filter_rules={**current.get("filter_rules", {}), **variant_data.get("filter_changes", {})},
        entry_logic=variant_data.get("entry_logic", current.get("entry_logic", "momentum_breakout")),
        exit_logic=variant_data.get("exit_logic", current.get("exit_logic", "fixed_sl_tp")),
    )

    control_id = config.get("active_variant_id", "v1")
    ABTestManager.start_test(config, control_id, vid)
    return True
