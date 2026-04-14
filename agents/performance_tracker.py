"""
Performance Tracker — monitorea la salud de cada estrategia en producción.
Lee trade_history.json, computa métricas por estrategia, detecta degradación.
"""
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

log = logging.getLogger("performance_tracker")

DATA_DIR = Path(__file__).parent / "data"
PERF_FILE = DATA_DIR / "strategy_performance.json"

# Thresholds de salud
HEALTH_WR_DROP_THRESHOLD  = 0.15   # Alerta si WR cae >15pp vs baseline
HEALTH_MIN_SHARPE          = 0.50   # Sharpe < 0.5 → degradada
HEALTH_MIN_TRADES          = 10     # Mínimo para evaluar
HEALTH_DEGRADED_SCORE      = 0.40   # Score < 0.40 → reducir tamaño
HEALTH_PAUSE_SCORE         = 0.20   # Score < 0.20 → pausar estrategia

# Baselines del backtest (simulación 357,984 runs)
BACKTEST_BASELINES = {
    "stoch_rsi_scalp":  {"wr": 0.645, "pf": 38.53},
    "rsi_bb_scalp":     {"wr": 0.759, "pf": 17.25},
    "golden_cross":     {"wr": 0.730, "pf":  5.42},
    "death_cross":      {"wr": 0.620, "pf":  3.20},  # estimado
    "macd_cross":       {"wr": 0.580, "pf":  2.50},
    "breakout":         {"wr": 0.560, "pf":  2.10},
    "oversold_bounce":  {"wr": 0.610, "pf":  3.00},
    "trend_momentum":   {"wr": 0.590, "pf":  2.80},
    "scalping":         {"wr": 0.550, "pf":  2.00},
}


def _load_trade_history() -> list:
    f = DATA_DIR / "trade_history.json"
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text())
        if isinstance(data, list):
            return data
        return data.get("trades", [])
    except Exception:
        return []


def _sharpe(returns: list) -> float:
    """Sharpe ratio anualizado (asume ~252 trading days)."""
    if len(returns) < 3:
        return 0.0
    n = len(returns)
    mean = sum(returns) / n
    var  = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std  = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(252)


def _profit_factor(returns: list) -> float:
    wins  = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    return wins / losses if losses > 0 else (float("inf") if wins > 0 else 1.0)


def get_strategy_metrics(strategy_name: str, window_days: int = 30) -> dict:
    """
    Computa métricas de rendimiento para una estrategia en los últimos N días.
    Returns dict con: trades, wr, pf, sharpe, avg_pnl, total_pnl
    """
    trades = _load_trade_history()
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    relevant = []
    for t in trades:
        if t.get("strategy", "").lower() not in (strategy_name.lower(), strategy_name.replace("_", "")):
            # Fuzzy match: accept if strategy_name is substring
            strat = t.get("strategy", "")
            if strategy_name not in strat and strat not in strategy_name:
                continue
        ct = t.get("close_time") or t.get("open_time", "")
        try:
            ts = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                continue
        except Exception:
            pass
        if "pnl_usd" in t:  # BUG FIX: trade_history.json never sets "status" field
            relevant.append(t)

    if not relevant:
        return {"trades": 0, "wr": 0.0, "pf": 1.0, "sharpe": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0}

    returns = [t.get("pnl_pct", 0.0) for t in relevant]
    wins    = sum(1 for r in returns if r > 0)

    return {
        "trades":    len(relevant),
        "wr":        wins / len(relevant),
        "pf":        _profit_factor(returns),
        "sharpe":    _sharpe(returns),
        "avg_pnl":   sum(returns) / len(returns),
        "total_pnl": sum(returns),
    }


def strategy_health(strategy_name: str, window_days: int = 30) -> float:
    """
    Retorna score de salud 0.0-1.0:
      > 0.70 → estrategia sana (tamaño normal)
      0.40-0.70 → degradación leve (tamaño ×0.75)
      0.20-0.40 → degradación seria (tamaño ×0.50)
      < 0.20 → pausar estrategia
    """
    m = get_strategy_metrics(strategy_name, window_days)

    if m["trades"] < HEALTH_MIN_TRADES:
        return 0.65  # sin datos suficientes — neutral-positivo (no penalizar al inicio)

    baseline = BACKTEST_BASELINES.get(strategy_name, {"wr": 0.55, "pf": 2.0})
    base_wr  = baseline["wr"]
    base_pf  = baseline["pf"]

    # Componente 1: Win Rate relativo al baseline (40% del score)
    wr_ratio = min(m["wr"] / base_wr, 1.2) / 1.2  # cap at 1.0
    if m["wr"] < base_wr - HEALTH_WR_DROP_THRESHOLD:
        wr_ratio *= 0.5  # penalización extra por caída grave

    # Componente 2: Sharpe (35% del score)
    sharpe_score = min(max(m["sharpe"] / 2.0, 0.0), 1.0)  # normalizar a [0,1]

    # Componente 3: Profit Factor relativo (25% del score)
    pf_ratio = min(m["pf"] / max(base_pf, 1.0), 1.2) / 1.2

    score = wr_ratio * 0.40 + sharpe_score * 0.35 + pf_ratio * 0.25
    return round(max(0.0, min(1.0, score)), 3)


def get_position_multiplier(strategy_name: str) -> float:
    """
    Retorna multiplicador de tamaño de posición basado en la salud.
    0.5 = degradada, 0.75 = leve degradación, 1.0 = sana
    """
    health = strategy_health(strategy_name)
    if health < HEALTH_PAUSE_SCORE:
        log.warning(f"⚠️ {strategy_name}: PAUSADA (health={health:.2f}) — no abrir nuevas posiciones")
        return 0.0
    if health < HEALTH_DEGRADED_SCORE:
        log.warning(f"⚠️ {strategy_name}: degradada (health={health:.2f}) — tamaño ×0.50")
        return 0.50
    if health < 0.65:
        log.info(f"ℹ️ {strategy_name}: leve degradación (health={health:.2f}) — tamaño ×0.75")
        return 0.75
    return 1.0


def save_performance_snapshot():
    """
    Guarda snapshot de métricas de todas las estrategias conocidas.
    Llamar al final de cada ciclo del orchestrator.
    """
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": {}
    }
    for name in BACKTEST_BASELINES:
        m = get_strategy_metrics(name)
        h = strategy_health(name)
        m["health"] = h
        m["multiplier"] = get_position_multiplier(name)
        report["strategies"][name] = m

    try:
        PERF_FILE.write_text(json.dumps(report, indent=2))
    except Exception as e:
        log.warning(f"⚠️ No se pudo guardar performance snapshot: {e}")


def print_dashboard():
    """Imprime tabla de rendimiento por estrategia en los logs."""
    log.info("📊 ── Performance Dashboard (últimos 30 días) ─────────────────")
    for name in BACKTEST_BASELINES:
        m = get_strategy_metrics(name)
        h = strategy_health(name)
        mx = get_position_multiplier(name)
        if m["trades"] == 0:
            log.info(f"   {name:<22} — sin trades todavía")
        else:
            status = "✅" if h >= 0.65 else ("⚠️" if h >= 0.40 else "🔴")
            log.info(
                f"   {status} {name:<22} "
                f"n={m['trades']:>3} WR={m['wr']*100:.0f}% "
                f"PF={m['pf']:.1f}x Sharpe={m['sharpe']:.2f} "
                f"health={h:.2f} mult={mx:.2f}"
            )
    log.info("─" * 60)


# ── Alertas de degradación ────────────────────────────────────────────────────
_last_alert: dict = {}   # {strategy_name: timestamp} para no spam

def check_and_alert(cooldown_hours: float = 6.0):
    """
    Revisa la salud de todas las estrategias y envía alerta Telegram si alguna
    está degradada. Cooldown de 6h por estrategia para evitar spam.
    """
    import time
    now = time.time()
    alerts_sent = 0

    for name, baseline in BACKTEST_BASELINES.items():
        h = strategy_health(name)
        m = get_strategy_metrics(name)

        if m["trades"] < HEALTH_MIN_TRADES:
            continue  # sin datos suficientes

        # Calcular cuando fue la ultima alerta
        last = _last_alert.get(name, 0)
        if (now - last) < cooldown_hours * 3600:
            continue

        if h < HEALTH_PAUSE_SCORE:
            emoji = "🔴"
            level = "PAUSADA"
        elif h < HEALTH_DEGRADED_SCORE:
            emoji = "🟠"
            level = "DEGRADADA"
        elif h < 0.65:
            emoji = "🟡"
            level = "leve degradacion"
        else:
            continue  # sana — no alertar

        msg = (
            f"{emoji} *Solana Bot — Estrategia {level}*\n"
            f"`{name}`\n"
            f"Health: {h:.2f} | Trades: {m['trades']} | WR: {m['wr']*100:.0f}% "
            f"(base {baseline['wr']*100:.0f}%)\n"
            f"PF: {m['pf']:.1f}x | Sharpe: {m['sharpe']:.2f}\n"
            f"Multiplier: {get_position_multiplier(name):.2f}x"
        )
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            import reporter
            reporter.send_telegram(msg)
            _last_alert[name] = now
            alerts_sent += 1
            log.warning(f"⚠️ Alerta enviada: {name} health={h:.2f}")
        except Exception as e:
            log.warning(f"No se pudo enviar alerta Telegram: {e}")

    return alerts_sent


def integrate_multiplier(strategy_name: str, base_size_usd: float) -> float:
    """
    Ajusta el tamaño de posición por salud de la estrategia.
    Llamar desde executor.py antes de abrir posición.
    """
    mult = get_position_multiplier(strategy_name)
    if mult < 1.0:
        log.info(f"📉 {strategy_name}: size ajustado {base_size_usd:.2f} → {base_size_usd*mult:.2f} (health mult={mult:.2f})")
    return base_size_usd * mult
