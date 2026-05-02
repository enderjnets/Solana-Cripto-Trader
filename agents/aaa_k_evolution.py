#!/usr/bin/env python3
"""
AAA-K Self-Evolution Harness — Phase 2+3
AlphaEvolver-style parameter + strategy optimization for Kimi 2.6.

Adaptado del harness de AAA-M con limites conservadores:
- SL: 2-8% (vs 1-10% de M)
- Leverage max: 5x (vs 10x de M)
- A/B test mas lento: 50 ciclos = ~100min
- LLM: Kimi 2.6 (mas analitico, menos impulsivo)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from aaa_alerts import alert_ab_test_result

log = logging.getLogger("aaa_k_evolution")

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "aaa_k_config.json"

# -- Conservative Safety Limits ---------------------------------------------

SAFETY_LIMITS = {
    "max_leverage": 5.0,
    "min_sl_pct": 0.02,
    "max_sl_pct": 0.08,
    "min_tp_pct": 0.04,
    "max_tp_pct": 0.15,
    "min_margin_pct": 0.01,
    "max_margin_pct": 0.08,
    "max_risk_per_trade_pct": 0.04,
    "min_liquidity_usd": 200000,
    "min_momentum_pct": 0.5,
    "max_momentum_pct": 8.0,
    "min_confidence_open": 0.55,
    "max_confidence_open": 0.95,
    "min_hold_hours": 0.5,
    "max_hold_hours": 12,
    "max_kelly_multiplier": 1.0,
}

# -- Config Loader ----------------------------------------------------------

def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            log.warning(f"Error cargando config K: {e}, usando defaults")
    config = build_default_config()
    save_config(config)
    return config


def build_default_config() -> Dict[str, Any]:
    return {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "cycle_interval_sec": 120,
            "max_positions": 10,
            "min_confidence_open": 0.60,
            "min_momentum_pct": 1.0,
            "min_liquidity_usd": 500000,
            "default_sl_pct": 0.04,
            "default_tp_pct": 0.08,
            "default_leverage": 2,
            "default_margin_pct": 0.03,
            "max_hold_hours": 8,
            "trailing_trigger_pct": 0.05,
            "trailing_stop_pct": 0.025,
            "kelly_multiplier": 0.5,
            "max_risk_per_trade_pct": 0.03,
        },
        "evolution_history": [],
        "last_applied": None,
        "active_variant_id": "v1",
        "variants": {
            "v1": {
                "name": "Trend Breakout v1",
                "system_prompt_addon": "",
                "filter_rules": {
                    "min_momentum_pct": 1.0,
                    "min_liquidity_usd": 500000,
                    "min_volume_24h": 2000000,
                },
                "entry_logic": "trend_breakout",
                "exit_logic": "fixed_sl_tp_trailing",
                "sharpe_baseline": 0.0,
                "trades_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        },
        "ab_test": {
            "status": "idle",
            "control_variant": "v1",
            "test_variant": None,
            "cycles_remaining": 0,
            "sharpe_control": 0.0,
            "sharpe_test": 0.0,
            "trades_control": 0,
            "trades_test": 0,
            "start_time": None,
        },
    }


def save_config(config: Dict[str, Any]) -> None:
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


# -- Safety Guard -----------------------------------------------------------

class SafetyGuard:
    @staticmethod
    def validate_param_changes(changes: Dict[str, float]) -> tuple[bool, List[str]]:
        violations = []
        checks = [
            ("default_leverage", "max_leverage", ">", "Leverage excede maximo"),
            ("default_sl_pct", "max_sl_pct", ">", "SL excede maximo"),
            ("default_sl_pct", "min_sl_pct", "<", "SL por debajo del minimo"),
            ("default_tp_pct", "max_tp_pct", ">", "TP excede maximo"),
            ("default_tp_pct", "min_tp_pct", "<", "TP por debajo del minimo"),
            ("default_margin_pct", "max_margin_pct", ">", "Margin excede maximo"),
            ("default_margin_pct", "min_margin_pct", "<", "Margin por debajo del minimo"),
            ("max_risk_per_trade_pct", "max_risk_per_trade_pct", ">", "Riesgo excede maximo"),
            ("min_liquidity_usd", "min_liquidity_usd", "<", "Liquidez minima insuficiente"),
            ("min_momentum_pct", "max_momentum_pct", ">", "Momentum excede maximo"),
            ("min_momentum_pct", "min_momentum_pct", "<", "Momentum por debajo del minimo"),
            ("min_confidence_open", "max_confidence_open", ">", "Confianza excede maximo"),
            ("min_confidence_open", "min_confidence_open", "<", "Confianza por debajo del minimo"),
            ("max_hold_hours", "max_hold_hours", ">", "Hold time excede maximo"),
            ("max_hold_hours", "min_hold_hours", "<", "Hold time por debajo del minimo"),
            ("kelly_multiplier", "max_kelly_multiplier", ">", "Kelly excede maximo"),
        ]
        for param, limit_key, op, msg in checks:
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
    @staticmethod
    def apply_changes(config: Dict[str, Any], changes: Dict[str, float],
                      confidence: float, analysis: str) -> Dict[str, Any]:
        is_valid, violations = SafetyGuard.validate_param_changes(changes)
        if not is_valid:
            log.warning(f"[K] Cambios RECHAZADOS: {violations}")
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
            log.info(f"[K] Cambios DEFERRED: confianza {confidence:.0%} < 75%")
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
                log.warning(f"[K] Parametro desconocido: {key}")

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
        log.info(f"[K] Parametros APLICADOS: {list(applied.keys())}")
        return config

    @staticmethod
    def rollback(config: Dict[str, Any], reason: str) -> Dict[str, Any]:
        last = config.get("last_applied")
        if not last or not last.get("baseline_params"):
            log.warning("[K] No hay baseline para rollback")
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
        log.info(f"[K] Rollback completado")
        return config


# -- Variant Registry -------------------------------------------------------

class VariantRegistry:
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
                       entry_logic: str = "trend_breakout",
                       exit_logic: str = "fixed_sl_tp_trailing") -> Dict[str, Any]:
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
        if len(variants) > 5:
            sorted_v = sorted(variants.items(), key=lambda x: x[1].get("sharpe_baseline") or -999)
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
            log.info(f"[K] Variant {variant_id} PROMOTED")


# -- AB Test Manager --------------------------------------------------------

class ABTestManager:
    TEST_CYCLES = 50  # 50 * 120s = ~100 min
    MIN_TRADES_FOR_EVAL = 3
    IMPROVEMENT_THRESHOLD = 1.10
    MAX_TEST_DURATION_HOURS = 72

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
        log.info(f"[K] A/B TEST: {control_id} vs {test_id}")

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
        return cycles > total // 2

    @staticmethod
    def get_current_variant_id(config: Dict[str, Any]) -> str:
        ab = config.get("ab_test", {})
        if ab.get("status") != "running":
            return config.get("active_variant_id", "v1")
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
        start = ab.get("start_time")
        if start:
            started = datetime.fromisoformat(start)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 3600
            if elapsed > ABTestManager.MAX_TEST_DURATION_HOURS:
                ab["status"] = "timeout"
                save_config(config)
                log.warning("[K] A/B TEST TIMEOUT")
                return "timeout"
        cycles = ab.get("cycles_remaining", 0)
        if cycles > 0:
            return None
        sharpe_c = ab.get("sharpe_control", 0.0)
        sharpe_t = ab.get("sharpe_test", 0.0)
        trades_c = ab.get("trades_control", 0)
        trades_t = ab.get("trades_test", 0)
        log.info(f"[K] A/B EVAL: C={sharpe_c:.2f}({trades_c}T) vs T={sharpe_t:.2f}({trades_t}T)")
        if trades_c < ABTestManager.MIN_TRADES_FOR_EVAL and trades_t < ABTestManager.MIN_TRADES_FOR_EVAL:
            ab["status"] = "insufficient_data"
            save_config(config)
            return "insufficient_data"
        if sharpe_t < 0 and sharpe_c >= 0:
            ab["status"] = "test_failed"
            save_config(config)
            return "test_failed"
        baseline = sharpe_c if sharpe_c > 0 else 0.001
        improvement = sharpe_t / baseline if baseline > 0 else 0
        if improvement >= ABTestManager.IMPROVEMENT_THRESHOLD:
            test_id = ab.get("test_variant")
            VariantRegistry.promote_variant(config, test_id)
            ab["status"] = "promoted"
            save_config(config)
            log.info(f"[K] A/B PROMOTED (improvement={improvement:.2f}x)")
            alert_ab_test_result("AAA-K", test_id, "promoted", improvement)
            return "promoted"
        ab["status"] = "test_rejected"
        save_config(config)
        log.info(f"[K] A/B REJECTED (improvement={improvement:.2f}x)")
        return "test_rejected"


# -- Variant Generator ------------------------------------------------------

class VariantGenerator:
    @staticmethod
    def generate_variant(analysis: Dict[str, Any], current_variant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            from llm_config import call_kimi
        except ImportError:
            log.warning("[K] call_kimi no disponible")
            return None
        recommendations = analysis.get("recommendations", [])
        param_changes = analysis.get("param_changes", {})
        current_prompt_addon = current_variant.get("system_prompt_addon", "")
        current_filters = current_variant.get("filter_rules", {})
        system = "Eres un quant strategist conservador. Diseña variantes de estrategia de trading. Responde SOLO en JSON valido."
        prompt = f"""Basado en este analisis, genera una NUEVA variante de estrategia CONSERVADORA:

ANALISIS:
- Recomendaciones: {recommendations}
- Cambios sugeridos: {param_changes}
- Confianza: {analysis.get('confidence', 0):.0%}

VARIANTE ACTUAL:
- Prompt addon: {current_prompt_addon or '(ninguno)'}
- Filtros: {current_filters}
- Entry: {current_variant.get('entry_logic', 'trend_breakout')}

Genera una variante que TESTEE una hipotesis diferente pero conservadora.

JSON:
{{
  "variant_name": "Nombre",
  "system_prompt_addon": "max 300 chars",
  "filter_changes": {{
    "min_momentum_pct": valor,
    "min_liquidity_usd": valor,
    "min_volume_24h": valor
  }},
  "entry_logic": "...",
  "exit_logic": "...",
  "confidence": 0.0-1.0
}}"""
        response = call_kimi(prompt, system=system, max_tokens=1500)
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
                log.info(f"[K] Variant conf {result.get('confidence', 0):.0%} < 75%, descartada")
                return None
            return result
        except Exception as e:
            log.warning(f"[K] VariantGenerator parse error: {e}")
            return None


# -- Integration Helpers ----------------------------------------------------

def get_active_variant(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    if config.get("ab_test", {}).get("status") == "running":
        return VariantRegistry.get_variant(config, ABTestManager.get_current_variant_id(config)) or {}
    return VariantRegistry.get_active_variant(config)


def maybe_start_ab_test(config: Dict[str, Any], analysis: Dict[str, Any]) -> bool:
    ab = config.get("ab_test", {})
    if ab.get("status") == "running":
        return False
    recommendations = analysis.get("recommendations", [])
    has_strategic = any(r for r in recommendations if any(word in r.lower() for word in [
        "filtro", "filter", "momentum", "volumen", "volume", "breakout", "tendencia", "trend",
        "prompt", "estrategia", "strategy", "enfoque", "focus", "conservador", "agresivo"
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
        entry_logic=variant_data.get("entry_logic", current.get("entry_logic", "trend_breakout")),
        exit_logic=variant_data.get("exit_logic", current.get("exit_logic", "fixed_sl_tp_trailing")),
    )
    control_id = config.get("active_variant_id", "v1")
    ABTestManager.start_test(config, control_id, vid)
    return True


def check_and_rollback_if_needed(current_sharpe: float, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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


def record_baseline_sharpe(sharpe: float, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    last = config.get("last_applied")
    if last and "baseline_sharpe" not in last:
        last["baseline_sharpe"] = sharpe
        save_config(config)
    return config


def get_effective_params(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_config()
    return config.get("parameters", build_default_config()["parameters"])
