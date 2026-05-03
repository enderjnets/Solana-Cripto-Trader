import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

log = logging.getLogger("aaa_advisory")


def load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def generate_advisories(data_dir: Path) -> Dict:
    """Genera advisories basados en estados de K, M y Meta-Arbitro."""
    import sys
    sys.path.insert(0, str(data_dir.parent))
    from aaa_shared import load_portfolio, calculate_metrics, load_trade_history

    advisories = []
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%H:%M:%S")

    # Load K metrics
    try:
        k_port = load_portfolio('AAA-K')
        k_trades = load_trade_history('AAA-K')
        k_metrics = calculate_metrics(k_trades, k_port.get('initial_capital', 1000.0))
    except Exception as e:
        log.warning(f"Error loading K metrics: {e}")
        k_port, k_metrics = {}, {}

    # Load M metrics
    try:
        m_port = load_portfolio('AAA-M')
        m_trades = load_trade_history('AAA-M')
        m_metrics = calculate_metrics(m_trades, m_port.get('initial_capital', 1000.0))
    except Exception as e:
        log.warning(f"Error loading M metrics: {e}")
        m_port, m_metrics = {}, {}

    k_state = load_json(data_dir / "aaa_k_state.json")
    m_state = load_json(data_dir / "aaa_m_state.json")
    meta = load_json(data_dir / "meta_arbitro_state.json")

    # System baseline
    advisories.append({
        "time": now_str,
        "sev": "info",
        "title": "SYSTEM ONLINE",
        "body": "AAA Command Center telemetry active. K cycles every 120s, M every 30s.",
        "agent": "system",
    })

    # Meta-Arbitro gate
    if meta:
        gp = meta.get("gate_phase", 0)
        phases = ["Observation", "Whitelist", "Parameters", "Sizing", "Single Trade", "Full Ensemble"]
        next_p = phases[gp + 1] if gp < 5 else "MAX CLEARANCE"
        advisories.append({
            "time": now_str,
            "sev": "info" if gp >= 5 else "warning",
            "title": f"GATE PHASE {gp}: {phases[gp]}",
            "body": f"Meta-Arbitro leader: {meta.get('leader', '--')}. Next: {next_p}. Action: {meta.get('gate_action', 'None')}.",
            "agent": "meta",
        })

    # K analysis
    if k_metrics:
        ret = k_metrics.get("return_pct", 0)
        dd = k_metrics.get("max_drawdown_pct", 0)
        trades = k_metrics.get("total_trades", 0)
        wr = k_metrics.get("win_rate", 0)
        if dd > 5:
            advisories.append({"time": now_str, "sev": "critical", "title": "AAA-K DRAWDOWN ALERT", "body": f"Max drawdown {dd:.2f}% exceeds 5% threshold.", "agent": "K"})
        elif ret < -5:
            advisories.append({"time": now_str, "sev": "critical", "title": "AAA-K CAPITAL AT RISK", "body": f"Return {ret:.2f}% below -5%. Consider halting.", "agent": "K"})
        elif ret < -3:
            advisories.append({"time": now_str, "sev": "warning", "title": "AAA-K NEGATIVE RETURN", "body": f"Return {ret:.2f}%. Adverse market conditions.", "agent": "K"})
        elif trades > 5 and wr < 0.3:
            advisories.append({"time": now_str, "sev": "warning", "title": "AAA-K LOW WIN RATE", "body": f"Win rate {wr*100:.0f}% after {trades} trades.", "agent": "K"})
        else:
            advisories.append({"time": now_str, "sev": "info", "title": "AAA-K NOMINAL", "body": f"Return {ret:.2f}%, DD {dd:.2f}%. Operating normally.", "agent": "K"})

        if k_state and k_state.get("cooldown_active"):
            syms = ", ".join(k_state["cooldown_active"])
            advisories.append({"time": now_str, "sev": "info", "title": "AAA-K SHIELD ACTIVE", "body": f"Cooldown guard protecting: {syms}", "agent": "K"})

    # M analysis
    if m_metrics:
        ret = m_metrics.get("return_pct", 0)
        dd = m_metrics.get("max_drawdown_pct", 0)
        trades = m_metrics.get("total_trades", 0)
        wr = m_metrics.get("win_rate", 0)
        if dd > 5:
            advisories.append({"time": now_str, "sev": "critical", "title": "AAA-M DRAWDOWN ALERT", "body": f"Max drawdown {dd:.2f}% exceeds 5% threshold.", "agent": "M"})
        elif ret < -5:
            advisories.append({"time": now_str, "sev": "critical", "title": "AAA-M CAPITAL AT RISK", "body": f"Return {ret:.2f}% below -5%. Consider halting.", "agent": "M"})
        elif ret < -3:
            advisories.append({"time": now_str, "sev": "warning", "title": "AAA-M NEGATIVE RETURN", "body": f"Return {ret:.2f}%. Adverse market conditions.", "agent": "M"})
        elif trades > 10 and wr < 0.3:
            advisories.append({"time": now_str, "sev": "warning", "title": "AAA-M LOW WIN RATE", "body": f"Win rate {wr*100:.0f}% after {trades} trades.", "agent": "M"})
        else:
            advisories.append({"time": now_str, "sev": "info", "title": "AAA-M NOMINAL", "body": f"Return {ret:.2f}%, DD {dd:.2f}%. Scalping active.", "agent": "M"})

        if m_state and m_state.get("cooldown_active"):
            syms = ", ".join(m_state["cooldown_active"])
            advisories.append({"time": now_str, "sev": "info", "title": "AAA-M SHIELD ACTIVE", "body": f"Cooldown guard protecting: {syms}", "agent": "M"})

    return {
        "last_updated": now.isoformat(),
        "advisories": advisories,
    }


if __name__ == "__main__":
    import sys
    d = Path(__file__).parent / "aaa_data"
    result = generate_advisories(d)
    print(json.dumps(result, indent=2, ensure_ascii=False))
