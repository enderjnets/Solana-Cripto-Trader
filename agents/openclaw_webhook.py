"""
OpenClaw Webhook Integration — Solana Cripto Trader
Non-blocking (timeout 3s, try/except) — bot no se detiene si OpenClaw falla.
"""
import requests
import logging

log = logging.getLogger("openclaw_wh")

_URL = "http://127.0.0.1:18789/plugins/webhooks/solana-alerts"
_SECRET = "fb57daf6f5549f7fd4b16bb6f6f17931f1a2d44b6afca2f3a84b8a0af4116194"
_HEADERS = {"Authorization": f"Bearer {_SECRET}", "Content-Type": "application/json"}
_TIMEOUT = 3


def _fire(goal: str, notify: str = "done_only") -> bool:
    """Dispara un TaskFlow en EkoRog. Silencioso si falla."""
    try:
        r = requests.post(
            _URL,
            json={"action": "create_flow", "goal": goal, "status": "queued", "notifyPolicy": notify},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if r.ok:
            log.debug(f"OpenClaw wh OK: {goal[:60]}")
            return True
        log.debug(f"OpenClaw wh {r.status_code}")
    except Exception as e:
        log.debug(f"OpenClaw wh unavailable: {e}")
    return False


def on_trade_opened(pos: dict):
    sym = pos.get("symbol", "?")
    d = pos.get("direction", "?").upper()
    lev = pos.get("leverage", 1)
    entry = pos.get("entry_price", 0)
    margin = pos.get("margin_usd", 0)
    strategy = pos.get("strategy", "?")
    _fire(
        f"Trade ABIERTO: {sym} {d} {lev}x @ ${entry:.4f} | margen ${margin:.2f} | "        f"estrategia {strategy} | Notifica al usuario via Telegram de forma concisa."
    )


def on_trade_closed(pos: dict):
    sym = pos.get("symbol", "?")
    d = pos.get("direction", "?").upper()
    pnl = pos.get("pnl_usd", 0)
    pct = pos.get("pnl_pct", 0)
    reason = pos.get("close_reason", "?")
    sign = "+" if pnl >= 0 else ""
    result = "WIN ✅" if pnl >= 0 else "LOSS ❌"
    _fire(
        f"Trade CERRADO {result}: {sym} {d} {sign}${pnl:.2f} ({sign}{pct:.1f}%) "        f"razon:{reason} | Notifica al usuario via Telegram con el resultado."
    )


def on_wild_chain_closed(sym: str, pnl: float, n_levels: int, margin: float, reason: str = "AI_CLOSE"):
    sign = "+" if pnl >= 0 else ""
    result = "WIN ✅" if pnl >= 0 else "LOSS ❌"
    _fire(
        f"WILD cadena CERRADA {result}: {sym} {sign}${pnl:.2f} | {n_levels} niveles | "        f"margen ${margin:.2f} | {reason} | Notifica via Telegram con razonamiento del Motor Martingala."
    )


def on_wild_level_opened(sym: str, level: int, direction: str, margin: float, multiplier: float, chain_pnl: float):
    kind = "Martingala" if direction == "same" else "Cobertura"
    sign = "+" if chain_pnl >= 0 else ""
    _fire(
        f"WILD nivel {level} ({kind}): {sym} margen ${margin:.2f} x{multiplier:.1f} | "        f"PnL cadena: {sign}${chain_pnl:.2f} | Notifica brevemente via Telegram."
    )


def on_critical_error(msg: str, context: str = ""):
    _fire(
        f"ERROR CRITICO bot Solana: {msg[:120]} | {context[:80]} | "        f"Investiga y notifica al usuario con urgencia.",
        notify="state_changes",
    )
