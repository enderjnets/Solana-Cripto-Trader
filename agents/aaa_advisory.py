#!/usr/bin/env python3
"""
💡 AAA Live Advisory Mode — SOLAAA-63
======================================
Lee el estado de los agentes AAA (K, M, Meta-Arbitro) y genera un reporte
advisory que el live bot consume en modo "susurro": loguea pero NO ejecuta.

Compara recomendaciones AAA con las señales del live bot y detecta divergencias.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AAA_DATA_DIR = BASE_DIR / "aaa_data"
ADVISORY_FILE = DATA_DIR / "aaa_advisory.json"
ALERT_COOLDOWN_FILE = DATA_DIR / "aaa_advisory_cooldowns.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("aaa_advisory")


# ─── Config ───────────────────────────────────────────────────────────────────

DIVERGENCE_THRESHOLD = 0.70   # Alertar si confianza AAA >= 70% y diverge
ALERT_COOLDOWN_SEC = 1800     # 30 min entre alerts de la misma divergencia


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default if default is not None else {}


def _save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ─── AAA State Loaders ────────────────────────────────────────────────────────

def load_aaa_portfolio(agent: str) -> dict:
    """Carga portfolio paper de AAA-K o AAA-M."""
    path = AAA_DATA_DIR / f"aaa_{agent.lower()}_portfolio.json"
    return _load_json(path, {"positions": [], "equity": 50000.0, "sharpe": 0.0})


def load_aaa_knowledge(agent: str) -> dict:
    """Carga knowledge de AAA-K o AAA-M."""
    path = AAA_DATA_DIR / f"knowledge_{agent.lower()}.json"
    return _load_json(path, {})


def load_meta_arbitro_state() -> dict:
    """Carga estado del Meta-Arbitro."""
    path = AAA_DATA_DIR / "meta_arbitro_state.json"
    return _load_json(path, {
        "leader": "M",
        "gate_phase": 0,
        "gate_name": "Observation",
        "weights": {"K": 0.0, "M": 1.0},
        "scoreboard": {}
    })


def load_aaa_config(agent: str) -> dict:
    """Carga config dinámica de AAA-K o AAA-M (si existe)."""
    path = AAA_DATA_DIR / f"aaa_{agent.lower()}_config.json"
    return _load_json(path, {})


# ─── Recommendation Extractors ────────────────────────────────────────────────

def extract_recommendations_from_portfolio(portfolio: dict, agent: str) -> List[dict]:
    """Extrae recomendaciones de posiciones abiertas en un portfolio AAA."""
    recs = []
    positions = portfolio.get("positions", [])
    for pos in positions:
        if pos.get("status") == "open":
            recs.append({
                "symbol": pos.get("symbol", "?"),
                "direction": pos.get("direction", "?").upper(),
                "confidence": _infer_confidence(pos, portfolio),
                "reason": pos.get("strategy", "unknown"),
                "entry_price": pos.get("entry_price", 0),
                "current_price": pos.get("current_price", 0),
                "pnl_usd": pos.get("pnl_usd", 0),
                "pnl_pct": pos.get("pnl_pct", 0),
                "leverage": pos.get("leverage", 1),
                "margin_usd": pos.get("margin_usd", 0),
                "agent": agent,
            })
    return recs


def _infer_confidence(position: dict, portfolio: dict) -> float:
    """Infiere confianza de una posición AAA basado en datos disponibles."""
    # Si el portfolio tiene un campo confidence, usarlo
    conf = position.get("confidence", 0)
    if conf and isinstance(conf, (int, float)) and conf > 0:
        return min(1.0, float(conf))
    # Fallback: basado en P&L positivo (posiciones ganando = más confianza)
    pnl_pct = position.get("pnl_pct", 0)
    if pnl_pct > 2:
        return 0.75
    elif pnl_pct > 0:
        return 0.60
    elif pnl_pct > -2:
        return 0.45
    else:
        return 0.30


def extract_aaa_recommendations() -> Dict[str, List[dict]]:
    """Extrae todas las recomendaciones activas de AAA."""
    k_portfolio = load_aaa_portfolio("k")
    m_portfolio = load_aaa_portfolio("m")

    k_recs = extract_recommendations_from_portfolio(k_portfolio, "K")
    m_recs = extract_recommendations_from_portfolio(m_portfolio, "M")

    return {
        "K": k_recs,
        "M": m_recs,
        "k_sharpe": k_portfolio.get("sharpe", 0.0),
        "k_equity": k_portfolio.get("equity", k_portfolio.get("capital_usd", 50000.0)),
        "m_sharpe": m_portfolio.get("sharpe", 0.0),
        "m_equity": m_portfolio.get("equity", m_portfolio.get("capital_usd", 50000.0)),
    }


# ─── Live Bot Loaders ─────────────────────────────────────────────────────────

def load_live_signals() -> List[dict]:
    """Carga señales del live bot desde signals_latest.json y strategy_llm.json."""
    signals = []
    # Señales técnicas
    sig_path = DATA_DIR / "signals_latest.json"
    sig_data = _load_json(sig_path, {})
    for s in sig_data.get("signals", []):
        signals.append({
            "symbol": s.get("symbol", s.get("token", "?")),
            "direction": s.get("direction", "?").upper(),
            "source": "technical",
            "confidence": s.get("confidence", 0.5),
        })
    # Señales LLM
    llm_path = DATA_DIR / "strategy_llm.json"
    llm_data = _load_json(llm_path, {})
    for s in llm_data.get("signals", []):
        signals.append({
            "symbol": s.get("symbol", s.get("token", "?")),
            "direction": s.get("direction", "?").upper(),
            "source": "llm",
            "confidence": s.get("confidence", 0.5),
        })
    return signals


def load_live_positions() -> List[dict]:
    """Carga posiciones abiertas del live bot."""
    port_path = DATA_DIR / "portfolio.json"
    portfolio = _load_json(port_path, {"positions": []})
    live_positions = []
    for p in portfolio.get("positions", []):
        if p.get("status") == "open":
            live_positions.append({
                "symbol": p.get("symbol", "?"),
                "direction": p.get("direction", "?").upper(),
                "source": "live_position",
                "confidence": 1.0,  # Posición abierta = 100% "confianza" de que está activa
            })
    return live_positions


# ─── Divergence Engine ────────────────────────────────────────────────────────

def compare_aaa_vs_live(aaa_recs_k: List[dict], aaa_recs_m: List[dict],
                        live_signals: List[dict], live_positions: List[dict]) -> dict:
    """Compara recomendaciones AAA con señales/posiciones del live bot."""

    # Merge all AAA recs (K + M) — deduplicate by symbol+direction
    aaa_by_symbol = {}
    for rec in aaa_recs_k + aaa_recs_m:
        key = rec["symbol"].upper()
        # Keep highest confidence if duplicate
        if key not in aaa_by_symbol or rec["confidence"] > aaa_by_symbol[key]["confidence"]:
            aaa_by_symbol[key] = rec

    # Merge live signals + positions
    live_by_symbol = {}
    for sig in live_signals + live_positions:
        key = sig["symbol"].upper()
        if key not in live_by_symbol:
            live_by_symbol[key] = sig

    convergences = []
    divergences = []
    only_aaa = []
    only_live = []

    all_symbols = set(aaa_by_symbol.keys()) | set(live_by_symbol.keys())

    for sym in all_symbols:
        aaa = aaa_by_symbol.get(sym)
        live = live_by_symbol.get(sym)

        if aaa and live:
            if aaa["direction"] == live["direction"]:
                convergences.append({
                    "symbol": sym,
                    "direction": aaa["direction"],
                    "aaa_confidence": aaa["confidence"],
                    "live_source": live.get("source", "unknown"),
                    "aaa_agent": aaa.get("agent", "?"),
                })
            else:
                divergences.append({
                    "symbol": sym,
                    "aaa_direction": aaa["direction"],
                    "live_direction": live["direction"],
                    "aaa_confidence": aaa["confidence"],
                    "live_source": live.get("source", "unknown"),
                    "aaa_agent": aaa.get("agent", "?"),
                    "aaa_reason": aaa.get("reason", ""),
                })
        elif aaa and not live:
            only_aaa.append({
                "symbol": sym,
                "direction": aaa["direction"],
                "aaa_confidence": aaa["confidence"],
                "aaa_agent": aaa.get("agent", "?"),
                "aaa_reason": aaa.get("reason", ""),
            })
        elif live and not aaa:
            only_live.append({
                "symbol": sym,
                "direction": live["direction"],
                "live_source": live.get("source", "unknown"),
            })

    return {
        "convergences": convergences,
        "divergences": divergences,
        "only_aaa": only_aaa,
        "only_live": only_live,
        "summary": {
            "total_symbols": len(all_symbols),
            "n_convergences": len(convergences),
            "n_divergences": len(divergences),
            "n_only_aaa": len(only_aaa),
            "n_only_live": len(only_live),
        }
    }


# ─── Alert Cooldown Manager ───────────────────────────────────────────────────

def _load_cooldowns() -> dict:
    return _load_json(ALERT_COOLDOWN_FILE, {})


def _save_cooldowns(cd: dict):
    _save_json(ALERT_COOLDOWN_FILE, cd)


def _cooldown_key(divergence: dict) -> str:
    return f"{divergence['symbol']}:{divergence['aaa_direction']}:{divergence['live_direction']}"


def should_alert_divergence(divergence: dict, threshold: float = DIVERGENCE_THRESHOLD,
                            cooldown_sec: int = ALERT_COOLDOWN_SEC) -> bool:
    """Determina si una divergencia debe generar alerta (confianza + cooldown)."""
    if divergence.get("aaa_confidence", 0) < threshold:
        return False

    cd = _load_cooldowns()
    key = _cooldown_key(divergence)
    last = cd.get(key, 0)
    now = time.time()
    if now - last < cooldown_sec:
        return False

    cd[key] = now
    _save_cooldowns(cd)
    return True


# ─── Telegram Alert ───────────────────────────────────────────────────────────

def send_divergence_alert(divergence: dict) -> bool:
    """Envia alerta Telegram de divergencia usando reporter.send_telegram si existe."""
    try:
        from reporter import send_telegram
        if not send_telegram:
            return False

        sym = divergence["symbol"]
        aaa_dir = divergence["aaa_direction"]
        live_dir = divergence["live_direction"]
        conf = divergence.get("aaa_confidence", 0)
        agent = divergence.get("aaa_agent", "?")
        reason = divergence.get("aaa_reason", "N/A")

        msg = (
            f"⚠️ *AAA Divergencia Detectada*\n\n"
            f"*Token:* {sym}\n"
            f"*AAA-{agent}:* {aaa_dir} (conf {conf:.0%})\n"
            f"*Live Bot:* {live_dir}\n"
            f"*Razón AAA:* {reason[:100]}\n\n"
            f"💡 Live Advisory Mode — sin ejecución automática"
        )
        send_telegram(msg)
        return True
    except Exception as e:
        log.warning(f"   ⚠️ No se pudo enviar alerta Telegram: {e}")
        return False


# ─── Master Entry Point ───────────────────────────────────────────────────────

def generate_advisory_report() -> dict:
    """Genera el reporte advisory completo."""

    # Check if AAA data exists at all
    if not AAA_DATA_DIR.exists():
        log.info("   ℹ️  AAA data dir no existe — skip advisory")
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "aaa_available": False}

    # Load AAA state
    aaa_recs = extract_aaa_recommendations()
    meta = load_meta_arbitro_state()

    # Load live bot state
    live_signals = load_live_signals()
    live_positions = load_live_positions()

    # Compare
    divergence = compare_aaa_vs_live(
        aaa_recs.get("K", []),
        aaa_recs.get("M", []),
        live_signals,
        live_positions,
    )

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aaa_available": True,
        "aaa_k": {
            "recommendations": aaa_recs.get("K", []),
            "sharpe": aaa_recs.get("k_sharpe", 0.0),
            "equity": aaa_recs.get("k_equity", 50000.0),
        },
        "aaa_m": {
            "recommendations": aaa_recs.get("M", []),
            "sharpe": aaa_recs.get("m_sharpe", 0.0),
            "equity": aaa_recs.get("m_equity", 50000.0),
        },
        "meta_arbitro": {
            "leader": meta.get("leader", "M"),
            "gate_phase": meta.get("gate_phase", 0),
            "gate_name": meta.get("gate_name", "Observation"),
            "weights": meta.get("weights", {"K": 0.0, "M": 1.0}),
        },
        "live_bot": {
            "signals": live_signals,
            "positions": live_positions,
        },
        "divergence": divergence,
    }

    # Save report
    _save_json(ADVISORY_FILE, report)

    # Process alerts for high-confidence divergences
    for div in divergence.get("divergences", []):
        if should_alert_divergence(div):
            send_divergence_alert(div)

    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    report = generate_advisory_report()
    print(json.dumps(report, indent=2, default=str))
