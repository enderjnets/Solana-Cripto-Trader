#!/usr/bin/env python3
"""
AAA Meta-Arbitro — Juez entre AAA-K y AAA-M

Funciones:
  1. Scoreboard continuo (Sharpe, PF, WR, DD, PnL)
  2. Peso de voto dinamico
  3. Gate de transferencia progresiva (6 fases)
  4. Deteccion de drift/anomalia
  5. Estado compartido escrito a meta_arbitro_state.json

Ciclo: Cada 30 minutos
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from aaa_shared import load_portfolio, calculate_metrics, load_trade_history, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("aaa_meta_arbitro")

STATE_FILE = DATA_DIR / "meta_arbitro_state.json"
TRANSFER_LOG = DATA_DIR / "transfer_log.json"

# ─── Gate Configuration ─────────────────────────────────────────────────────

GATE_PHASES = [
    {"phase": 0, "name": "Observacion", "condition": "Inicio", "action": "Nada"},
    {"phase": 1, "name": "Whitelist", "condition": "Ambos Sharpe > 1.0 por 3 dias", "action": "Ampliar token whitelist"},
    {"phase": 2, "name": "Params", "condition": "Uno Sharpe > 1.5 por 5 dias", "action": "Adoptar SL/TP del lider"},
    {"phase": 3, "name": "Sizing", "condition": "Ambos Sharpe > 1.5 por 7 dias", "action": "Adoptar sizing/risk del mejor"},
    {"phase": 4, "name": "Veto", "condition": "Peso de uno > 70% por 10 dias", "action": "Ese agente puede proponer 1 trade real/ciclo"},
    {"phase": 5, "name": "Full Merge", "condition": "Ambos Sharpe > 2.0 por 14 dias", "action": "Reemplazar motor de decisiones"},
]


def load_agent_metrics(agent: str) -> dict:
    """Carga metricas de un agente."""
    trades = load_trade_history(agent)
    metrics = calculate_metrics(trades, 50000.0)
    portfolio = load_portfolio(agent)
    open_pos = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    total_equity = portfolio.get("capital_usd", 0) + sum(
        p.get("margin_usd", 0) + p.get("pnl_usd", 0) for p in open_pos
    )
    metrics["capital_usd"] = portfolio.get("capital_usd", 0)
    metrics["total_equity"] = round(total_equity, 2)
    metrics["open_positions"] = len(open_pos)
    metrics["cycle_count"] = portfolio.get("cycle_count", 0)
    metrics["agent"] = agent
    return metrics


def compute_weights(metrics_k: dict, metrics_m: dict) -> dict:
    """Calcula pesos dinamicos basados en Sharpe relativo."""
    sharpe_k = max(0.1, metrics_k.get("sharpe_ratio", 0.1))
    sharpe_m = max(0.1, metrics_m.get("sharpe_ratio", 0.1))

    total = sharpe_k + sharpe_m
    weight_k = sharpe_k / total
    weight_m = sharpe_m / total

    # Suavizar: no permitir que un agente tenga < 10% de peso
    weight_k = max(0.1, min(0.9, weight_k))
    weight_m = max(0.1, min(0.9, weight_m))

    # Normalizar de nuevo
    total2 = weight_k + weight_m
    weight_k = round(weight_k / total2, 2)
    weight_m = round(weight_m / total2, 2)

    return {"K": weight_k, "M": weight_m}


def evaluate_gate(metrics_k: dict, metrics_m: dict, history: list) -> int:
    """Evalua la fase actual del gate de transferencia."""
    if len(history) < 3 * 24:  # Necesitamos al menos 3 dias de datos
        return 0

    # Calcular Sharpe promedio de los ultimos 3, 5, 7, 10, 14 dias
    def avg_sharpe(agent: str, hours: int) -> float:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        relevant = [h for h in history if h["timestamp"] > cutoff.isoformat() and h["agent"] == agent]
        if not relevant:
            return 0.0
        return sum(h.get("sharpe_ratio", 0) for h in relevant) / len(relevant)

    sharpe_k_3d = avg_sharpe("K", 72)
    sharpe_m_3d = avg_sharpe("M", 72)
    sharpe_k_5d = avg_sharpe("K", 120)
    sharpe_m_5d = avg_sharpe("M", 120)
    sharpe_k_7d = avg_sharpe("K", 168)
    sharpe_m_7d = avg_sharpe("M", 168)
    sharpe_k_10d = avg_sharpe("K", 240)
    sharpe_m_10d = avg_sharpe("M", 240)
    sharpe_k_14d = avg_sharpe("K", 336)
    sharpe_m_14d = avg_sharpe("M", 336)

    # Fase 1: Ambos Sharpe > 1.0 por 3 dias
    if sharpe_k_3d > 1.0 and sharpe_m_3d > 1.0:
        # Fase 2: Uno Sharpe > 1.5 por 5 dias
        if sharpe_k_5d > 1.5 or sharpe_m_5d > 1.5:
            # Fase 3: Ambos Sharpe > 1.5 por 7 dias
            if sharpe_k_7d > 1.5 and sharpe_m_7d > 1.5:
                # Fase 4: Peso de uno > 70% por 10 dias
                weights = compute_weights(
                    {"sharpe_ratio": sharpe_k_10d},
                    {"sharpe_ratio": sharpe_m_10d}
                )
                if weights["K"] > 0.7 or weights["M"] > 0.7:
                    # Fase 5: Ambos Sharpe > 2.0 por 14 dias
                    if sharpe_k_14d > 2.0 and sharpe_m_14d > 2.0:
                        return 5
                    return 4
                return 3
            return 2
        return 1
    return 0


def detect_drift(metrics_k: dict, metrics_m: dict, history: list) -> list:
    """Detecta si algun agente ha empeorado drasticamente."""
    alerts = []

    # Sharpe cayo > 30% en 48h
    def recent_sharpe(agent: str, hours: int) -> float:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        relevant = [h for h in history if h["timestamp"] > cutoff.isoformat() and h["agent"] == agent]
        if not relevant:
            return 0.0
        return sum(h.get("sharpe_ratio", 0) for h in relevant) / len(relevant)

    for agent, metrics in [("K", metrics_k), ("M", metrics_m)]:
        sharpe_now = metrics.get("sharpe_ratio", 0)
        sharpe_48h = recent_sharpe(agent, 48)
        if sharpe_48h > 0.5 and sharpe_now < sharpe_48h * 0.7:
            alerts.append(f"DRIFT: {agent} Sharpe cayo de {sharpe_48h:.2f} a {sharpe_now:.2f} en 48h")

    return alerts


def run_cycle():
    """Ejecuta un ciclo del Meta-Arbitro."""
    log.info("=" * 60)
    log.info(f"⚖️ Meta-Arbitro — {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    log.info("=" * 60)

    metrics_k = load_agent_metrics("AAA-K")
    metrics_m = load_agent_metrics("AAA-M")

    log.info(f"📊 K: Equity=${metrics_k['total_equity']:.2f} WR={metrics_k['win_rate']:.1f}% PF={metrics_k['profit_factor']:.2f} Sharpe={metrics_k['sharpe_ratio']:.2f} DD={metrics_k['max_drawdown_pct']:.1f}%")
    log.info(f"📊 M: Equity=${metrics_m['total_equity']:.2f} WR={metrics_m['win_rate']:.1f}% PF={metrics_m['profit_factor']:.2f} Sharpe={metrics_m['sharpe_ratio']:.2f} DD={metrics_m['max_drawdown_pct']:.1f}%")

    weights = compute_weights(metrics_k, metrics_m)
    log.info(f"⚖️ Pesos: K={weights['K']:.0%} | M={weights['M']:.0%}")

    # Cargar historial de metricas
    history = []
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            history = state.get("history", [])
        except Exception:
            pass

    # Agregar metricas actuales al historial
    now = datetime.now(timezone.utc).isoformat()
    history.append({"timestamp": now, "agent": "K", **metrics_k})
    history.append({"timestamp": now, "agent": "M", **metrics_m})
    # Limitar historial a 30 dias
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    history = [h for h in history if h["timestamp"] > cutoff]

    # Evaluar gate
    gate_phase = evaluate_gate(metrics_k, metrics_m, history)
    gate_info = GATE_PHASES[gate_phase]
    log.info(f"🚪 Gate Fase {gate_phase}: {gate_info['name']} — {gate_info['condition']}")

    # Detectar drift
    alerts = detect_drift(metrics_k, metrics_m, history)
    if alerts:
        for alert in alerts:
            log.warning(f"   🚨 {alert}")

    # Leader
    leader = "K" if metrics_k["sharpe_ratio"] > metrics_m["sharpe_ratio"] else "M"
    if metrics_k["sharpe_ratio"] == metrics_m["sharpe_ratio"]:
        leader = "TIE"

    # Guardar estado
    state = {
        "timestamp": now,
        "scoreboard": {
            "K": metrics_k,
            "M": metrics_m,
        },
        "weights": weights,
        "leader": leader,
        "gate_phase": gate_phase,
        "gate_name": gate_info["name"],
        "gate_action": gate_info["action"],
        "alerts": alerts,
        "history": history,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    log.info(f"💾 Estado guardado en {STATE_FILE}")

    # Log de transferencia si cambio de fase
    prev_phase = 0
    if TRANSFER_LOG.exists():
        try:
            logs = json.loads(TRANSFER_LOG.read_text())
            if logs:
                prev_phase = logs[-1].get("phase", 0)
        except Exception:
            pass

    if gate_phase != prev_phase:
        log_entry = {
            "timestamp": now,
            "phase": gate_phase,
            "name": gate_info["name"],
            "action": gate_info["action"],
            "weights": weights,
            "leader": leader,
        }
        logs = []
        if TRANSFER_LOG.exists():
            try:
                logs = json.loads(TRANSFER_LOG.read_text())
            except Exception:
                pass
        logs.append(log_entry)
        TRANSFER_LOG.write_text(json.dumps(logs, indent=2))
        log.info(f"📝 Transfer log actualizado: Fase {gate_phase}")

    return state


def run_continuous(interval_minutes: int = 30):
    log.info(f"🚀 Meta-Arbitro iniciado (intervalo: {interval_minutes}min)")
    while True:
        try:
            run_cycle()
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            log.info("🛑 Detenido")
            break
        except Exception as e:
            log.error(f"❌ Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    if args.once:
        result = run_cycle()
        print(json.dumps(result, indent=2))
    else:
        run_continuous(interval_minutes=args.interval)
