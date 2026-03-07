#!/usr/bin/env python3
"""
⚡ Agente 4: Executor
Abre y cierra posiciones según señales del Strategy Agent.
Persiste estado real en portfolio.json y trade_history.json.

MODOS:
  - safe=True  → Paper trading (precio real Jupiter, sin wallet)
  - safe=False → Trades reales via Jupiter Swap API (requiere keypair)

Uso:
    python3 executor.py          # Paper trading (default)
    python3 executor.py --live   # Modo real (requiere configuración de wallet)
    python3 executor.py --debug  # Con output detallado
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SIGNALS_FILE   = DATA_DIR / "signals_latest.json"
MARKET_FILE    = DATA_DIR / "market_latest.json"
RISK_FILE      = DATA_DIR / "risk_report.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
HISTORY_FILE   = DATA_DIR / "trade_history.json"

# .env del proyecto para Telegram/wallet
ENV_FILE = Path(__file__).parent.parent / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("executor")

# ─── Parámetros ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 500.0   # Capital paper inicial
PAPER_MODE      = True    # Cambia a False para trades reales
MAKER_FEE       = 0.0025  # 0.25% fee Jupiter

# ─── Carga / Guardado de Estado ──────────────────────────────────────────────

def load_portfolio() -> dict:
    """Carga portfolio desde disco. Crea uno nuevo si no existe."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "capital_usd": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "positions": [],
        "status": "ACTIVE",
        "mode": "paper",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
    }


def save_portfolio(portfolio: dict):
    portfolio["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def load_history() -> list:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_signals() -> dict:
    if not SIGNALS_FILE.exists():
        return {"signals": []}
    with open(SIGNALS_FILE) as f:
        return json.load(f)


def load_market() -> dict:
    if not MARKET_FILE.exists():
        return {"tokens": {}}
    with open(MARKET_FILE) as f:
        return json.load(f)


def get_current_price(symbol: str, market: dict) -> float:
    """Obtiene precio actual del snapshot de mercado."""
    token_data = market.get("tokens", {}).get(symbol, {})
    return float(token_data.get("price", 0))


def load_risk_report() -> dict:
    """Carga el reporte del Risk Manager."""
    if RISK_FILE.exists():
        with open(RISK_FILE) as f:
            return json.load(f)
    return {}


def close_positions_emergency(portfolio: dict, symbols: list, market: dict, history: list) -> list:
    """
    Cierra posiciones por emergencia (sin importar SL/TP).
    Usado cuando Risk Manager detecta condiciones extremas.
    """
    closed = []
    now = datetime.now(timezone.utc).isoformat()

    for pos in portfolio["positions"]:
        if pos.get("status") != "open":
            continue

        if pos["symbol"] in symbols:
            # Actualizar precio actual
            pos["current_price"] = get_current_price(pos["symbol"], market)

            pos["status"] = "closed"
            pos["close_time"] = now
            pos["close_reason"] = "EMERGENCY_CLOSE"
            pos["close_price"] = pos["current_price"]

            # Actualizar P&L
            pnl_pct = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"]
            if pos["direction"] == "short":
                pnl_pct = -pnl_pct
            pnl_usd = pos["size_usd"] * pnl_pct

            pos["pnl_pct"] = round(pnl_pct, 4)
            pos["pnl_usd"] = round(pnl_usd, 4)

            # Actualizar stats del portfolio
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            if pnl_usd > 0:
                portfolio["wins"] = portfolio.get("wins", 0) + 1
            else:
                portfolio["losses"] = portfolio.get("losses", 0) + 1

            # Devolver capital al portfolio
            portfolio["capital_usd"] += pos["size_usd"] + pnl_usd

            # Agregar al historial
            history.append({
                "id": pos["id"],
                "symbol": pos["symbol"],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": pos["close_price"],
                "size_usd": pos["size_usd"],
                "pnl_usd": pos["pnl_usd"],
                "pnl_pct": pos["pnl_pct"],
                "open_time": pos["open_time"],
                "close_time": pos["close_time"],
                "close_reason": "EMERGENCY_CLOSE",
                "strategy": pos.get("strategy", "unknown"),
            })

            closed.append(pos)
            log.error(f"🚨 EMERGENCY CLOSE: {pos['symbol']} | P&L: ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%)")

    return closed


# ─── Paper Trading ────────────────────────────────────────────────────────────

def paper_open_position(signal: dict, portfolio: dict, market: dict) -> Optional[dict]:
    """Abre una posición en paper trading usando precio real de Jupiter."""
    symbol = signal["symbol"]
    price = get_current_price(symbol, market)

    if price <= 0:
        log.warning(f"⚠️  Sin precio para {symbol}, no se puede abrir posición")
        return None

    size_usd = signal.get("suggested_size_usd", 0)
    if size_usd <= 0:
        size_usd = portfolio["capital_usd"] * 0.02  # 2% fallback

    # Verificar capital suficiente
    if portfolio["capital_usd"] < size_usd:
        log.warning(f"⚠️  Capital insuficiente: ${portfolio['capital_usd']:.2f} < ${size_usd:.2f}")
        return None

    fee = size_usd * MAKER_FEE
    tokens_bought = (size_usd - fee) / price

    sl_pct = 0.025
    tp_pct = 0.05
    sl_price = price * (1 - sl_pct) if signal["direction"] == "long" else price * (1 + sl_pct)
    tp_price = price * (1 + tp_pct) if signal["direction"] == "long" else price * (1 - tp_pct)

    # Si el signal tiene SL/TP del risk manager, usarlos
    if signal.get("sl_price", 0) > 0:
        sl_price = signal["sl_price"]
    if signal.get("tp_price", 0) > 0:
        tp_price = signal["tp_price"]

    position = {
        "id": f"{symbol}_{int(time.time())}",
        "symbol": symbol,
        "direction": signal["direction"],
        "strategy": signal["strategy"],
        "entry_price": round(price, 8),
        "current_price": round(price, 8),
        "size_usd": round(size_usd, 2),
        "tokens": round(tokens_bought, 6),
        "sl_price": round(sl_price, 8),
        "tp_price": round(tp_price, 8),
        "fee_entry": round(fee, 4),
        "pnl_usd": 0.0,
        "pnl_pct": 0.0,
        "status": "open",
        "open_time": datetime.now(timezone.utc).isoformat(),
        "close_time": None,
        "mode": "paper",
        "confidence": signal.get("confidence", 0),
    }

    # Descontar del capital (la reserva mientras la posición está abierta)
    portfolio["capital_usd"] = round(portfolio["capital_usd"] - size_usd, 2)
    portfolio["positions"].append(position)

    return position


def paper_update_positions(portfolio: dict, market: dict, history: list) -> list:
    """
    Actualiza P&L de todas las posiciones abiertas.
    Cierra automáticamente si toca SL o TP.
    """
    closed = []
    remaining = []

    for pos in portfolio["positions"]:
        if pos["status"] != "open":
            remaining.append(pos)
            continue

        symbol = pos["symbol"]
        current_price = get_current_price(symbol, market)

        if current_price <= 0:
            remaining.append(pos)
            continue

        # Actualizar precio actual
        pos["current_price"] = round(current_price, 8)

        # Calcular P&L
        if pos["direction"] == "long":
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

        pnl_usd = pos["tokens"] * abs(current_price - pos["entry_price"])
        if pnl_pct < 0:
            pnl_usd = -pnl_usd

        pos["pnl_pct"] = round(pnl_pct * 100, 4)
        pos["pnl_usd"] = round(pnl_usd, 4)

        # Verificar SL/TP
        hit_sl = False
        hit_tp = False

        if pos["direction"] == "long":
            hit_sl = current_price <= pos["sl_price"]
            hit_tp = current_price >= pos["tp_price"]
        else:
            hit_sl = current_price >= pos["sl_price"]
            hit_tp = current_price <= pos["tp_price"]

        if hit_sl or hit_tp:
            close_reason = "TP" if hit_tp else "SL"
            pos["status"] = "closed"
            pos["close_time"] = datetime.now(timezone.utc).isoformat()
            pos["close_reason"] = close_reason
            pos["close_price"] = current_price

            # Devolver capital + P&L
            fee_exit = pos["size_usd"] * MAKER_FEE
            returned = pos["size_usd"] + pnl_usd - fee_exit
            pos["fee_exit"] = round(fee_exit, 4)
            portfolio["capital_usd"] = round(portfolio["capital_usd"] + returned, 2)

            # Estadísticas
            is_win = pnl_usd > 0
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            if is_win:
                portfolio["wins"] = portfolio.get("wins", 0) + 1
            else:
                portfolio["losses"] = portfolio.get("losses", 0) + 1

            result_emoji = "✅" if is_win else "❌"
            log.info(f"  {result_emoji} [{close_reason}] {symbol} cerrada | P&L: ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%)")

            history.append({**pos})
            closed.append(pos)
        else:
            remaining.append(pos)

    portfolio["positions"] = remaining
    return closed


# ─── Ejecución Real (Jupiter Swap API) ───────────────────────────────────────

def real_open_position(signal: dict, portfolio: dict) -> Optional[dict]:
    """
    Ejecuta trade real via Jupiter Swap API.
    REQUIERE: keypair configurado en .env (HOT_WALLET_PRIVATE_KEY)
    """
    log.warning("⚠️  Modo LIVE no implementado aún — activar manualmente")
    log.warning("   Para activar trades reales, configura HOT_WALLET_PRIVATE_KEY en .env")
    log.warning("   y revisa MAINNET_GUIDE.md para el proceso seguro")
    return None


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(safe: bool = True, debug: bool = False) -> dict:
    log.info("=" * 50)
    mode_label = "📄 PAPER" if safe else "🔴 LIVE"
    log.info(f"⚡ EXECUTOR — {mode_label} MODE")
    log.info("=" * 50)

    portfolio = load_portfolio()
    history = load_history()
    signals_data = load_signals()
    market = load_market()
    risk_report = load_risk_report()

    signals = signals_data.get("signals", [])
    portfolio_status = portfolio.get("status", "ACTIVE")

    # Chequear emergency close del Risk Manager
    emergency_close = risk_report.get("emergency_close", {})
    if emergency_close.get("triggered", False):
        symbols_to_close = emergency_close.get("symbols", [])
        reason = emergency_close.get("reason", "")
        log.warning(f"⚠️ RISK MANAGER solicitó emergency close: {reason}")
        emergency_closed = close_positions_emergency(portfolio, symbols_to_close, market, history)
        if emergency_closed:
            log.error(f"🚨 {len(emergency_closed)} posiciones cerradas por emergencia")

    # Actualizar precios y cerrar posiciones que tocaron SL/TP
    open_before = len([p for p in portfolio["positions"] if p.get("status") == "open"])
    closed_this_cycle = paper_update_positions(portfolio, market, history)
    open_after = len([p for p in portfolio["positions"] if p.get("status") == "open"])

    if closed_this_cycle:
        log.info(f"🔒 {len(closed_this_cycle)} posición(es) cerrada(s) este ciclo")

    # Estado del sistema
    if portfolio_status in ("PAUSED", "STOPPED"):
        log.warning(f"⛔ Sistema {portfolio_status} — no se abren nuevas posiciones")
        save_portfolio(portfolio)
        save_history(history)
        return {
            "status": portfolio_status,
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # Abrir nuevas posiciones según señales
    opened = []
    open_count = len([p for p in portfolio["positions"] if p.get("status") == "open"])

    for signal in signals:
        if open_count >= 5:
            log.info("📊 Máximo de posiciones alcanzado (5)")
            break

        symbol = signal["symbol"]
        # Verificar que no hay posición abierta para este token
        existing = [p for p in portfolio["positions"]
                    if p["symbol"] == symbol and p["status"] == "open"]
        if existing:
            if debug:
                log.info(f"  ⏭️  {symbol}: posición ya abierta, skip")
            continue

        if safe:
            pos = paper_open_position(signal, portfolio, market)
        else:
            pos = real_open_position(signal, portfolio)

        if pos:
            opened.append(pos)
            open_count += 1
            arrow = "🟢" if pos["direction"] == "long" else "🔴"
            log.info(f"  {arrow} ABIERTA {symbol} [{signal['strategy']}] "
                     f"${pos['size_usd']:.2f} @ ${pos['entry_price']:.6f}")

    # Calcular métricas actuales
    total_trades = portfolio.get("total_trades", 0)
    wins = portfolio.get("wins", 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    open_positions = [p for p in portfolio["positions"] if p.get("status") == "open"]
    unrealized_pnl = sum(p.get("pnl_usd", 0) for p in open_positions)

    # Guardar
    save_portfolio(portfolio)
    save_history(history)

    log.info(f"💰 Capital: ${portfolio['capital_usd']:.2f}")
    log.info(f"📊 Posiciones abiertas: {len(open_positions)}")
    log.info(f"📈 P&L no realizado: ${unrealized_pnl:+.2f}")
    log.info(f"🏆 Win rate: {win_rate:.1f}% ({wins}W/{total_trades-wins}L de {total_trades} trades)")
    log.info(f"💾 Guardado en {PORTFOLIO_FILE}")

    return {
        "status": "OK",
        "mode": "paper" if safe else "live",
        "capital": portfolio["capital_usd"],
        "open_positions": len(open_positions),
        "opened_this_cycle": len(opened),
        "closed_this_cycle": len(closed_this_cycle),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor Agent")
    parser.add_argument("--live", action="store_true",
                        help="Activar modo real (cuidado: trades reales en Solana)")
    parser.add_argument("--debug", action="store_true", help="Output detallado")
    args = parser.parse_args()

    safe_mode = not args.live
    if not safe_mode:
        log.warning("🚨 MODO LIVE ACTIVADO — se ejecutarán trades reales en Solana")
        confirm = input("Confirmar modo LIVE (escribe 'SI' para continuar): ")
        if confirm.strip() != "SI":
            log.info("Cancelado.")
            sys.exit(0)

    run(safe=safe_mode, debug=args.debug)
    sys.exit(0)
