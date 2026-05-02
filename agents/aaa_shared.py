#!/usr/bin/env python3
"""
AAA Shared Utilities — Agente de Autoaprendizaje Acelerado
Funciones compartidas entre AAA-K y AAA-M:
  - Paper trading engine simplificado
  - Métricas de performance (Sharpe, PF, WR, etc.)
  - Persistencia de estado
  - Scanner wrapper
"""

import json
import math
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "aaa_data"
DATA_DIR.mkdir(exist_ok=True)

log = logging.getLogger("aaa_shared")

# ─── Paper Trading Constants ────────────────────────────────────────────────

TAKER_FEE = 0.001          # 0.1%
FUNDING_HOURLY = 0.0001    # 0.01%/hour
MAX_LEVERAGE = 10

# Slippage por tier de liquidez
def get_slippage(symbol: str, liquidity: float = 0) -> float:
    """Retorna slippage estimado basado en liquidez."""
    if liquidity >= 10_000_000:
        return 0.001    # 0.1% — bluechips
    if liquidity >= 1_000_000:
        return 0.002    # 0.2% — mid-cap
    if liquidity >= 500_000:
        return 0.003    # 0.3% — low-cap
    return 0.005        # 0.5% — memes / thin liquidity


def get_current_price(symbol: str, market: dict) -> float:
    """Obtiene precio actual de market data."""
    tokens = market.get("tokens", {})
    token = tokens.get(symbol, {})
    price = token.get("price", 0)
    if price <= 0:
        # Fallback: buscar por mint
        for t in tokens.values():
            if t.get("symbol") == symbol:
                price = t.get("price", 0)
                break
    return float(price)


def get_token_liquidity(symbol: str, market: dict) -> float:
    """Obtiene liquidez del token."""
    tokens = market.get("tokens", {})
    token = tokens.get(symbol, {})
    return float(token.get("liquidity", 0))


# ─── Paper Trading Core ─────────────────────────────────────────────────────

def paper_open_position(
    symbol: str,
    direction: str,
    entry_price: float,
    margin_usd: float,
    leverage: float,
    sl_pct: float,
    tp_pct: float,
    portfolio: dict,
    market: dict,
    strategy: str = "default",
    agent: str = "unknown"
) -> Optional[dict]:
    """
    Abre una posición en paper trading.
    Similar a executor.py pero simplificado para AAA.
    """
    capital = portfolio.get("capital_usd", 0)
    if capital < margin_usd:
        log.warning(f"[{agent}] Capital insuficiente: ${capital:.2f} < ${margin_usd:.2f}")
        return None

    notional = margin_usd * leverage
    liquidity = get_token_liquidity(symbol, market)
    slippage = get_slippage(symbol, liquidity)
    fee_entry = notional * (TAKER_FEE + slippage)

    if direction == "long":
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
        liq_price = entry_price * (1 - (margin_usd - fee_entry) / notional)
    else:
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)
        liq_price = entry_price * (1 + (margin_usd - fee_entry) / notional)

    position = {
        "id": f"{symbol}_{agent}_{int(time.time() * 1000)}",
        "symbol": symbol,
        "direction": direction,
        "entry_price": round(entry_price, 8),
        "current_price": round(entry_price, 8),
        "margin_usd": round(margin_usd, 2),
        "notional_value": round(notional, 2),
        "leverage": leverage,
        "sl_price": round(sl_price, 8),
        "tp_price": round(tp_price, 8),
        "liquidation_price": round(liq_price, 8),
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "fee_entry": round(fee_entry, 4),
        "funding_accumulated": 0.0,
        "funding_hours": 0.0,
        "pnl_usd": 0.0,
        "pnl_pct": 0.0,
        "status": "open",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "agent": agent,
        "mode": "paper",
        "peak_price": entry_price,
        "trailing_activated": False,
        "trailing_sl": None,
    }

    portfolio["capital_usd"] = round(capital - margin_usd, 2)
    portfolio.setdefault("positions", []).append(position)
    log.info(f"[{agent}] OPEN {direction.upper()} {symbol} @ ${entry_price:.4f} | Margin=${margin_usd:.2f} Lev={leverage}x | SL={sl_pct*100:.1f}% TP={tp_pct*100:.1f}%")
    return position


def apply_funding_rate(pos: dict) -> float:
    """Aplica funding rate simulado a una posición."""
    hours = 1.0 / 60.0  # Asumimos llamada cada minuto
    funding = pos.get("notional_value", 0) * FUNDING_HOURLY * hours
    pos["funding_accumulated"] = round(pos.get("funding_accumulated", 0) + funding, 6)
    pos["funding_hours"] = round(pos.get("funding_hours", 0) + hours, 2)
    return funding


def paper_update_positions(portfolio: dict, market: dict, agent: str = "unknown") -> List[dict]:
    """
    Actualiza P&L de posiciones abiertas y cierra las que tocan SL/TP/liq.
    Retorna lista de posiciones cerradas en este ciclo.
    """
    closed = []
    remaining = []

    for pos in portfolio.get("positions", []):
        if pos.get("status") != "open":
            remaining.append(pos)
            continue

        symbol = pos["symbol"]
        current_price = get_current_price(symbol, market)
        if current_price <= 0:
            remaining.append(pos)
            continue

        pos["current_price"] = round(current_price, 8)
        apply_funding_rate(pos)

        notional = pos.get("notional_value", 0)
        margin = pos.get("margin_usd", 0)
        direction = pos["direction"]
        entry = pos["entry_price"]
        funding = pos.get("funding_accumulated", 0)

        if direction == "long":
            price_pnl_pct = (current_price - entry) / entry
            pos["peak_price"] = max(pos.get("peak_price", entry), current_price)
        else:
            price_pnl_pct = (entry - current_price) / entry
            pos["peak_price"] = min(pos.get("peak_price", entry), current_price)

        pnl_from_price = notional * price_pnl_pct
        gross_pnl = pnl_from_price + funding
        fee_exit_est = notional * (TAKER_FEE + get_slippage(symbol, get_token_liquidity(symbol, market)))
        net_pnl = gross_pnl - pos.get("fee_entry", 0) - fee_exit_est

        pos["pnl_usd"] = round(net_pnl, 4)
        pos["pnl_pct"] = round((net_pnl / margin * 100) if margin > 0 else 0, 4)
        pos["fee_exit_est"] = round(fee_exit_est, 4)

        # Check triggers
        sl_price = pos.get("sl_price", 0)
        tp_price = pos.get("tp_price", 0)
        liq_price = pos.get("liquidation_price", 0)

        hit_sl = (direction == "long" and current_price <= sl_price) or (direction == "short" and current_price >= sl_price)
        hit_tp = (direction == "long" and current_price >= tp_price) or (direction == "short" and current_price <= tp_price)
        hit_liq = (direction == "long" and current_price <= liq_price) or (direction == "short" and current_price >= liq_price)

        # Trailing stop check
        trailing_activated = pos.get("trailing_activated", False)
        trailing_sl = pos.get("trailing_sl")
        if trailing_activated and trailing_sl:
            if direction == "long" and current_price <= trailing_sl:
                hit_sl = True
            elif direction == "short" and current_price >= trailing_sl:
                hit_sl = True

        if hit_liq:
            close_reason = "LIQUIDATED"
        elif hit_sl:
            close_reason = "STOP_LOSS"
        elif hit_tp:
            close_reason = "TAKE_PROFIT"
        else:
            # Time exit: max 4h for AAA (vs 24h for live)
            opened = datetime.fromisoformat(pos["opened_at"].replace("Z", "+00:00"))
            hours_open = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
            max_hold = 4.0 if agent == "AAA-M" else 8.0
            if hours_open >= max_hold:
                close_reason = "TIME_EXIT"
            else:
                remaining.append(pos)
                continue

        # Close position
        pos["status"] = "closed"
        pos["close_time"] = datetime.now(timezone.utc).isoformat()
        pos["close_reason"] = close_reason
        pos["close_price"] = round(current_price, 8)
        pos["hours_open"] = round(hours_open, 2)

        # Return margin + PnL
        portfolio["capital_usd"] = round(portfolio.get("capital_usd", 0) + margin + net_pnl, 2)

        log.info(f"[{agent}] CLOSE {symbol} {close_reason} @ ${current_price:.4f} | PnL=${net_pnl:+.2f} ({pos['pnl_pct']:+.2f}%) | Hold={pos.get('hours_open', 0):.1f}h")
        closed.append(pos)

    portfolio["positions"] = remaining
    return closed


# ─── Performance Metrics ────────────────────────────────────────────────────

def calculate_metrics(trades: List[dict], capital_start: float = 50000.0) -> dict:
    """Calcula métricas de performance a partir de una lista de trades cerrados."""
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "total_pnl": 0.0,
            "return_pct": 0.0, "avg_hold_hours": 0.0,
        }

    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]

    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    gross_profit = sum(t.get("pnl_usd", 0) for t in wins)
    gross_loss = sum(abs(t.get("pnl_usd", 0)) for t in losses)

    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0

    # Sharpe ratio (simplificado): usando pnl_pct por trade como returns
    returns = [t.get("pnl_pct", 0) for t in trades]
    sharpe = 0.0
    if len(returns) >= 5:
        avg_ret = sum(returns) / len(returns)
        variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0
        # Minimum std to avoid division by near-zero
        std_ret = max(std_ret, 0.001)
        sharpe = (avg_ret / std_ret * math.sqrt(365))
    elif len(returns) >= 2:
        # With 2-4 trades, use a simplified estimate
        avg_ret = sum(returns) / len(returns)
        sharpe = avg_ret * 2  # Rough estimate without full std calc

    # Max drawdown
    equity = capital_start
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.get("pnl_usd", 0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    avg_hold = sum(t.get("hours_open", 0) for t in trades) / len(trades) if trades else 0.0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round((total_pnl / capital_start) * 100, 2),
        "avg_hold_hours": round(avg_hold, 2),
    }


# ─── Persistence ────────────────────────────────────────────────────────────

def load_portfolio(agent: str) -> dict:
    """Carga portfolio de un agente."""
    path = DATA_DIR / f"portfolio_aaa_{agent.lower()}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            log.warning(f"Error cargando portfolio {agent}: {e}")
    return {
        "capital_usd": 50000.0,
        "initial_capital": 50000.0,
        "positions": [],
        "status": "ACTIVE",
        "agent": agent,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_portfolio(portfolio: dict, agent: str):
    """Guarda portfolio de un agente."""
    path = DATA_DIR / f"portfolio_aaa_{agent.lower()}.json"
    path.write_text(json.dumps(portfolio, indent=2))


def load_trade_history(agent: str) -> List[dict]:
    """Carga historial de trades de un agente."""
    path = DATA_DIR / f"trade_history_aaa_{agent.lower()}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
            return data.get("trades", [])
        except Exception as e:
            log.warning(f"Error cargando trade history {agent}: {e}")
    return []


def save_trade_history(trades: List[dict], agent: str):
    """Guarda historial de trades de un agente."""
    path = DATA_DIR / f"trade_history_aaa_{agent.lower()}.json"
    path.write_text(json.dumps(trades, indent=2))


def append_trades(new_trades: List[dict], agent: str):
    """Añade trades al historial existente."""
    history = load_trade_history(agent)
    history.extend(new_trades)
    save_trade_history(history, agent)
    return history


def load_market_data() -> dict:
    """Carga market_latest.json del live bot."""
    path = BASE_DIR / "data" / "market_latest.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"tokens": {}, "fear_greed": {"value": 50}}


def load_signals() -> dict:
    """Carga signals_latest.json del live bot."""
    path = BASE_DIR / "data" / "signals_latest.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


# ─── Equity History ─────────────────────────────────────────────────────────

def update_equity_history(equity: float, agent: str):
    """Actualiza historial de equity para gráficos."""
    path = DATA_DIR / f"equity_history_aaa_{agent.lower()}.json"
    data = {"equity": [], "dates": []}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            pass
    data.setdefault("equity", [])
    data.setdefault("dates", [])
    data["equity"].append(round(equity, 2))
    data["dates"].append(datetime.now(timezone.utc).isoformat())
    # Limitar a 5000 puntos
    if len(data["equity"]) > 5000:
        data["equity"] = data["equity"][-5000:]
        data["dates"] = data["dates"][-5000:]
    path.write_text(json.dumps(data, indent=2))


def get_equity_history(agent: str) -> dict:
    """Retorna historial de equity."""
    path = DATA_DIR / f"equity_history_aaa_{agent.lower()}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"equity": [], "dates": []}
