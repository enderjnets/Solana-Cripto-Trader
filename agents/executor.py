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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from safe_io import atomic_write_json, safe_read_json
try:
    from agents.paperclip_client import on_trade_opened, on_trade_closed, on_daily_report
    _PAPERCLIP = True
except ImportError:
    _PAPERCLIP = False
try:
    import agents.openclaw_webhook as _ocwh
    _OPENCLAW_WH = True
except ImportError:
    _OPENCLAW_WH = False
from typing import Optional

try:
    from compound_engine import update_compound_capital
    _COMPOUND_ENABLED = True
except ImportError:
    _COMPOUND_ENABLED = False

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SIGNALS_FILE       = DATA_DIR / "signals_latest.json"
SIGNALS_LLM_FILE   = DATA_DIR / "strategy_llm.json"
MARKET_FILE        = DATA_DIR / "market_latest.json"
RISK_FILE          = DATA_DIR / "risk_report.json"
PORTFOLIO_FILE     = DATA_DIR / "portfolio.json"
HISTORY_FILE       = DATA_DIR / "trade_history.json"

# Máxima antigüedad de señales LLM para considerarlas válidas (en segundos)
LLM_SIGNALS_MAX_AGE_SEC = 600  # 10 minutos (5 ciclos de 2min)

# ─── FIX 2: Cooldown después de emergency close (02-abr-2026) ─────────────────
# Si un símbolo tuvo emergency close, esperar 30 min antes de reabrir
EMERGENCY_COOLDOWN_FILE = DATA_DIR / "symbol_cooldown.json"
EMERGENCY_COOLDOWN_SEC = 15 * 60  # E6: 15min cooldown (was 30min)  # 30 minutos

def _load_symbol_cooldown() -> dict:
    """Retorna dict de {symbol: timestamp_utc} de cooldown activo."""
    if not EMERGENCY_COOLDOWN_FILE.exists():
        return {}
    try:
        return json.loads(EMERGENCY_COOLDOWN_FILE.read_text())
    except Exception:
        return {}

def _is_in_emergency_cooldown(symbol: str) -> bool:
    """True si el símbolo está en cooldown por emergency close reciente."""
    cooldowns = _load_symbol_cooldown()
    if symbol.upper() not in cooldowns:
        return False
    last_close = cooldowns[symbol.upper()]
    elapsed = time.time() - last_close
    if elapsed > EMERGENCY_COOLDOWN_SEC:
        # Cooldown expirado, limpiar
        del cooldowns[symbol.upper()]
        EMERGENCY_COOLDOWN_FILE.write_text(json.dumps(cooldowns))
        return False
    remaining = int(EMERGENCY_COOLDOWN_SEC - elapsed)
    log.info(f"⏱️  {symbol} en emergency cooldown — {remaining}s restantes")
    return True

def record_emergency_cooldown(symbol: str):
    """Registra que este símbolo tuvo emergency close — inicia cooldown de 30 min."""
    cooldowns = _load_symbol_cooldown()
    cooldowns[symbol.upper()] = time.time()
    EMERGENCY_COOLDOWN_FILE.write_text(json.dumps(cooldowns))
    log.info(f"⏱️  Emergency cooldown registrado para {symbol.upper()} — 30 min")

def clear_all_cooldowns():
    """Limpia todos los cooldowns de emergencia."""
    EMERGENCY_COOLDOWN_FILE.write_text("{}")
    log.info("⏱️  Todos los cooldowns de emergencia limpiados")

# ─── FIX 1: Anti-rebound — evitar shorts cuando FG<15 y el precio ya rebotó ──
# Si FG está en pánico (<15) pero el precio ya subió >2% en 1h,
# el rebote está en progreso y shorts son arriesgados.
def _is_rebounding_from_fear(symbol: str, market: dict, price_history: dict) -> bool:
    """
    Detecta si el precio ya rebotó desde mínimos de Fear extremo.
    Retorna True si: FG < 15 Y precio subió >2% en la última hora.

    price_history se ordena oldest→newest (hist[0] = más antiguo, hist[-1] = más nuevo).
    """
    # fear_greed puede ser dict {'value': 12} o int
    fg_raw = market.get("fear_greed", 50)
    if isinstance(fg_raw, dict):
        fg = fg_raw.get("value", 50)
    else:
        fg = int(fg_raw) if fg_raw else 50
    if fg >= 15:
        return False  # No hay miedo extremo

    # Obtener precio actual del market
    tokens = market.get("tokens", {})
    token_info = tokens.get(symbol.upper()) or tokens.get(symbol.lower())
    if not token_info:
        return False
    current_price = token_info.get("price")
    if not current_price:
        return False

    # Obtener price history — oldest FIRST, newest LAST
    hist = price_history.get(symbol.upper(), [])
    if not hist or len(hist) < 2:
        return False

    # Parsear timestamps para encontrar el precio de ~1 hora atrás
    from datetime import datetime, timezone, timedelta
    now_ts = datetime.now(timezone.utc).timestamp()

    price_1h_ago = None
    for entry in reversed(hist):  # iterate newest→oldest para encontrar el más cercano a 1h
        ts_str = entry.get("ts", "")
        try:
            # Parse ISO timestamp
            if "+" in ts_str or "Z" in ts_str:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(ts_str)
            entry_ts = dt.timestamp()
            age_seconds = now_ts - entry_ts
            if 45 * 60 <= age_seconds <= 75 * 60:  # entre 45 min y 75 min = ~1h
                price_1h_ago = entry.get("price")
                break
        except Exception:
            continue

    if not price_1h_ago or price_1h_ago <= 0:
        return False

    change_1h = (current_price - price_1h_ago) / price_1h_ago * 100
    if change_1h > 2.0:
        log.info(f"🔴 ANTI-REBOUND: {symbol} FG={fg} +{change_1h:.1f}% 1h — SHORT bloqueado (rebote en progreso)")
        return True
    return False


def _get_token_trend(symbol: str, signal: dict, market: dict) -> str:
    """
    Returns the trend for a token: 'bullish', 'bearish', or 'neutral'.
    Priority:
    1. signal['trend'] if present and not None (from strategy/AI signals)
    2. EMA7 > EMA21 → 'bullish', EMA7 < EMA21 → 'bearish' (from strategy indicators)
    3. market-level 'price_1h_trend' from market_data (global fallback)
    4. 'neutral' if no trend data available
    """
    # 1. Signal-level trend (from strategy/AI)
    sig_trend = signal.get("trend")
    if sig_trend in ("up", "bullish"):
        return "bullish"
    if sig_trend in ("down", "bearish"):
        return "bearish"

    # 2. EMA-based trend from strategy indicators (ema7, ema21)
    ema7 = signal.get("ema7")
    ema21 = signal.get("ema21")
    if ema7 is not None and ema21 is not None:
        return "bullish" if ema7 > ema21 else "bearish"

    # 3. Market-level 1h trend from market_data
    tokens = market.get("tokens", {})
    token_info = tokens.get(symbol.upper(), {})
    price_trend = token_info.get("price_1h_trend")
    if price_trend in ("up", "down", "sideways"):
        if price_trend == "up":
            return "bullish"
        elif price_trend == "down":
            return "bearish"
        else:
            return "neutral"

    return "neutral"


def _should_block_short_rebound(signal: dict, market: dict) -> tuple[bool, str]:
    """
    Returns (should_block, reason) for blocking a SHORT due to rebound risk.

    Blocks SHORT when ALL:
      - RSI < 30 (oversold / rebound territory)
      - Trend is bullish (token trend up, EMA bullish, or global research bullish)

    In dry-run mode: logs the potential block but returns should_block=False.
    Controlled by env vars (read each call to avoid stale module-level values):
      - SHORT_REBOUND_FILTER_ENABLED (default true)
      - SHORT_REBOUND_FILTER_DRY_RUN (default false)
    """
    enabled = os.environ.get("SHORT_REBOUND_FILTER_ENABLED", "true").lower() == "true"
    if not enabled:
        return False, ""

    dry_run = os.environ.get("SHORT_REBOUND_FILTER_DRY_RUN", "false").lower() == "true"

    # E1: Don't block shorts in extreme fear - rebounds are shorting opportunities
    fg = get_fear_greed_index()
    if fg < 15:
        return False, "extreme fear - rebounds are shorting opportunities"

    direction = signal.get("direction", "").lower()
    if direction != "short":
        return False, ""

    rsi = signal.get("rsi", 50)
    if rsi >= 30:
        return False, ""

    symbol = signal.get("symbol", "UNKNOWN")
    trend = _get_token_trend(symbol, signal, market)
    if trend != "bullish":
        return False, ""

    fg = get_fear_greed_index()
    reason = f"SHORT {symbol} bloqueado: RSI={rsi:.1f} < 30 + tendencia {trend} (FG={fg})"

    if dry_run:
        log.info(f"[DRY-RUN] {reason} — dry-run activo, no se bloquea")
        return False, reason

    log.info(f"🔴 ANTI-REBOUND: {reason}")
    return True, reason

# .env del proyecto para Telegram/wallet
ENV_FILE = Path(__file__).parent.parent / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("executor")

# ─── Parámetros ──────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 1000.0   # Fallback - actual initial read from portfolio.json at runtime
PAPER_MODE      = True    # Cambia a False para trades reales

# ─── Anti-Rebound SHORT Filter ───────────────────────────────────────────────
SHORT_REBOUND_FILTER_ENABLED = os.environ.get("SHORT_REBOUND_FILTER_ENABLED", "true").lower() == "true"
SHORT_REBOUND_FILTER_DRY_RUN = os.environ.get("SHORT_REBOUND_FILTER_DRY_RUN", "false").lower() == "true"

# ─── Risk Management (ajustado 31-Mar-2026 — orden de Ender) ─────────────────
MIN_CONFIDENCE      = 0.55     # E1: Lowered from 0.70 for more trades     # Bajado de 0.85 para aprovechar más señales en extremos (2026-03-31)

# ── v2.10.0: Safety nets críticos (post-desastre 2026-04-18 -$57.27) ──
TRADE_WHITELIST_STRICT: frozenset = frozenset({"SOL", "BTC", "ETH", "XRP", "JUP"})
# Meme coins EXCLUIDOS: rug pulls frecuentes, liquidez ilusoria, fees 10-12% del margen
# Hardcoded (NO usar auto-learner.tokens_to_avoid — puede vaciarse al re-aprender)

MIN_SL_DISTANCE_PCT = 2.0      # Rechazar trades con SL a <2% del entry (con 5x lev = -10% max margin)
MAX_NOTIONAL_PCT_EQUITY = 0.50 # Cap notional <=50% del equity actual por trade (evita concentración)
BLOCK_LONGS_FG      = 10       # E1: Lowered from 20 — allow longs during F&G recovery bounces
MAX_TRADES_PER_DAY  = 0        # 0 = sin límite

# ─── Regla de Ender (SOL-8): Dual sizing — Wild vs Pure Mode ────────────────
# Wild Mode: small initial sizes so martingale has room (1.5%/10x at conf=0.85)
# Pure Mode: moderate swing sizes, lower leverage, no martingale (2%/5x)
MAX_RISK_PCT        = 0.020   # 2.0% base (reduced from 2.5%)
PORTFOLIO_RISK_PCT  = 0.10    # 10% of capital total risk (unchanged)
MIN_PROFIT_PCT      = 0.002   # 0.2% of capital min profit per pos (unchanged)

def _is_wild_mode_active() -> bool:
    """Lee wild_mode_state.json para detectar si Wild Mode está ON."""
    try:
        state = safe_read_json(DATA_DIR / "wild_mode_state.json", default={})
        return bool(state.get("active", False))
    except Exception:
        return False

# Wild Mode: smaller positions → martingale 2 chains × $37.50 × 2.3x = $172 (34% cap) ✅
_WILD_TIERS = [
    (0.85, 0.015, 10),   # conf ≥ 0.85: 1.5% capital, 10x lev → ~$37.50 margin at $500
    (0.75, 0.012,  7),   # conf ≥ 0.75: 1.2% capital, 7x lev
    (0.65, 0.010,  5),   # conf ≥ 0.65: 1.0% capital, 5x lev
    (0.00, 0.008,  3),   # else:         0.8% capital, 3x lev
]
# Pure Mode: swing sizes, no martingale → 2 positions × $100 = $200 (40% cap) ✅
_PURE_TIERS = [
    (0.85, 0.020,  5),   # conf ≥ 0.85: 2.0% capital, 5x lev → ~$100 margin at $500
    (0.75, 0.015,  4),   # conf ≥ 0.75: 1.5% capital, 4x lev
    (0.65, 0.010,  3),   # conf ≥ 0.65: 1.0% capital, 3x lev
    (0.00, 0.008,  2),   # else:         0.8% capital, 2x lev
]

def _get_risk_for_capital(capital: float, confidence: float = 0.7,
                          wild_mode: "Optional[bool]" = None) -> tuple:
    """Returns (max_risk_usd, leverage) based on capital, confidence and mode.
    Wild Mode: smaller initial sizes to leave room for martingale.
    Pure Mode: moderate swing sizes, lower leverage.
    """
    if wild_mode is None:
        wild_mode = _is_wild_mode_active()
    tiers = _WILD_TIERS if wild_mode else _PURE_TIERS
    for min_conf, risk_pct, lev in tiers:
        if confidence >= min_conf:
            risk = max(0.50, round(capital * risk_pct, 2))
            return risk, lev
    return max(0.50, round(capital * 0.008, 2)), 2

# Legacy compatibility
MAX_RISK_USD        = 5.00     # Overridden by _get_risk_for_capital() at runtime     # Máximo $3 de riesgo por POSICIÓN individual (ajustado 2026-03-31 para mercado lento)

# ─── Portfolio Take Profit (31-Mar-2026) — orden de Ender ─────────────────
# FIX 1.4+1.5: Portfolio TP y MIN dinamicos basados en capital
_PORTFOLIO_TP_USD     = 2.00     # Fallback estatico
_PORTFOLIO_MIN_USD    = 1.00     # Fallback estatico

def get_portfolio_tp(portfolio: dict = None) -> float:
    """Dynamic portfolio TP: 1% del capital, min $5, max $50."""
    if portfolio is None:
        return _PORTFOLIO_TP_USD
    capital = portfolio.get("capital_usd", 1000)
    return max(1.0, min(capital * 0.01, 50.0))  # Min $1 for small capital

def get_portfolio_min(portfolio: dict = None) -> float:
    """Dynamic portfolio MIN: 0.5% del capital, min $3, max $25."""
    if portfolio is None:
        return _PORTFOLIO_MIN_USD
    capital = portfolio.get("capital_usd", 1000)
    return max(0.50, min(capital * 0.005, 25.0))  # Min $0.50 for small capital

# Backward-compatible aliases
PORTFOLIO_TP_USD = _PORTFOLIO_TP_USD
PORTFOLIO_MIN_USD = _PORTFOLIO_MIN_USD

# ─── Coordinated Portfolio Sizing (31-Mar-2026 v2) — orden de Ender ───────
# El sizing se calcula a nivel de PORTAFOLIO, no por posición aislada.
# La IA decide cuántas posiciones abrir (N), y el sistema distribuye el
# target de ganancias entre todas ellas, ajustando por volatilidad.
# Cuando el P&L combinado llega al PORTFOLIO_TP → cerrar TODO.
PORTFOLIO_MAX_RISK_USD = 20.00  # Legacy - overridden by PORTFOLIO_RISK_PCT at runtime  # Riesgo total máximo del portafolio (todas las posiciones)
MIN_PROFIT_PER_POS_USD = 0.50   # Legacy - overridden by MIN_PROFIT_PCT at runtime   # Profit mínimo por posición ($0.50, flexible según N)

# ─── E4: Scalping Mode ───────────────────────────────────────────────
SCALP_TOKENS = {"SOL", "BTC", "ETH"}
SCALP_SL = 0.01       # 1% SL
SCALP_TP = 0.015      # 1.5% TP (R:R 1:1.5)
SCALP_MAX_HOLD = 1800  # 30 min max hold
SCALP_MAX_SIMULTANEOUS = 1

# ─── Drift Protocol Simulation ───────────────────────────────────────────────
TAKER_FEE           = 0.001    # 0.1% taker fee (Drift Protocol)
MAKER_FEE           = 0.001    # 0.1% maker fee (Drift Protocol)
DEFAULT_LEVERAGE    = 5        # 5x leverage por defecto (subido de 3x — orden Ender 2026-03-31)
MAX_LEVERAGE        = 10       # Máximo 10x
MAINTENANCE_MARGIN  = 0.05     # 5% margen de mantenimiento
FUNDING_RATE        = 0.0001   # 0.01% por hora (funding rate simulado)

# FIX 2.4: Modelo de slippage por tier de liquidez
SLIPPAGE_TIERS = {
    "BTC": 0.001, "ETH": 0.001, "SOL": 0.001,   # 0.1% high liquidity
    "JUP": 0.002, "RAY": 0.002,                   # 0.2% medium liquidity
}
SLIPPAGE_MEME = 0.005   # 0.5% for meme tokens
SLIPPAGE_DEFAULT = 0.003  # 0.3% default
MEME_TOKENS = {"BONK", "FARTCOIN", "MOODENG", "GOAT", "WIF", "POPCAT", "PENGU", "TRUMP", "MELANIA"}

def get_slippage(symbol: str) -> float:
    """Returns estimated slippage for a token."""
    if symbol in MEME_TOKENS:
        return SLIPPAGE_MEME
    return SLIPPAGE_TIERS.get(symbol, SLIPPAGE_DEFAULT)

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
        "paperclip_issue_id": None,  # Set by Paperclip integration
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
    }


def save_portfolio(portfolio: dict):
    portfolio["last_updated"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(PORTFOLIO_FILE, portfolio)


def load_history() -> list:
    """Carga trade_history.json — maneja formato lista O dict {"trades": [...]}."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):           # formato del orchestrator
                return data.get("trades", [])
        except Exception:
            pass
    return []


def save_history(history: list):
    atomic_write_json(HISTORY_FILE, history)


def load_signals() -> dict:
    """
    Carga señales priorizando AI Strategy (strategy_llm.json) sobre técnico (signals_latest.json).
    
    Lógica:
    1. Si strategy_llm.json existe y tiene señales recientes (< 10 min) → usar señales IA
    2. Fallback → signals_latest.json (señales técnicas de strategy.py)
    """
    # Intentar cargar señales del AI Strategy primero
    if SIGNALS_LLM_FILE.exists():
        try:
            with open(SIGNALS_LLM_FILE) as f:
                llm_data = json.load(f)
            
            # Verificar antigüedad del archivo
            generated_at = llm_data.get("generated_at", "")
            if generated_at:
                try:
                    ts = datetime.fromisoformat(generated_at)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
                    
                    llm_signals = llm_data.get("signals", [])
                    # Solo usar señales LLM si son recientes y hay señales válidas
                    if age_sec <= LLM_SIGNALS_MAX_AGE_SEC and llm_signals:
                        # Normalizar formato: ai_strategy usa sl_price/tp_price directamente
                        # pero necesita direction != "none"
                        valid_llm_signals = [
                            s for s in llm_signals
                            if s.get("direction", "none") not in ("none", "")
                            and "|" not in s.get("direction", "")  # reject template placeholder
                            and s.get("symbol", "") not in ("", "TOKEN")  # reject template placeholder
                            and s.get("confidence", 0) >= 0.55  # reject low-confidence/template signals
                        ]
                        if valid_llm_signals:
                            # FIX 1.6b: Filtrar tokens_to_avoid TAMBIEN en senales LLM
                            try:
                                _al_path = DATA_DIR / "auto_learner_state.json"
                                if _al_path.exists():
                                    _al = json.loads(_al_path.read_text())
                                    _avoid = set(
                                        _al.get("params", {}).get("tokens_to_avoid", []) or
                                        _al.get("tokens_to_avoid", [])
                                    )
                                    if _avoid:
                                        _before = len(valid_llm_signals)
                                        valid_llm_signals = [s for s in valid_llm_signals if s.get("symbol") not in _avoid]
                                        _skipped = _before - len(valid_llm_signals)
                                        if _skipped > 0:
                                            log.info(f"   🧠 LLM filter: excluyendo {_skipped} tokens (avoid: {_avoid})")
                            except Exception:
                                pass
                            if valid_llm_signals:
                                log.info(f"🤖 Usando señales AI Strategy ({len(valid_llm_signals)} señales, {age_sec:.0f}s de antigüedad)")
                                return {"signals": valid_llm_signals, "source": "ai_strategy"}
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            log.warning(f"⚠️ Error leyendo strategy_llm.json: {e}")
    
    # Fallback: señales técnicas de strategy.py
    if not SIGNALS_FILE.exists():
        return {"signals": [], "source": "none"}
    
    log.info("📊 Usando señales técnicas (signals_latest.json) — fallback")
    with open(SIGNALS_FILE) as f:
        data = json.load(f)
    data["source"] = "technical"
    return data


def load_market() -> dict:
    if not MARKET_FILE.exists():
        return {"tokens": {}}
    with open(MARKET_FILE) as f:
        data = json.load(f)
    # FIX 3.3: Staleness detection
    ts = data.get("timestamp", "")
    if ts:
        try:
            from datetime import datetime, timezone
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds()
            if age > 600:
                log.warning(f"\u26a0\ufe0f Market data is {age:.0f}s old - REFUSING stale data")
                data["_stale"] = True
            elif age > 300:
                log.warning(f"\u26a0\ufe0f Market data is {age:.0f}s old")
        except Exception:
            pass
    return data


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


def close_positions_emergency(portfolio: dict, symbols: list, market: dict, history: list, reason: str = "EMERGENCY_CLOSE", ai_reasoning: str = "") -> list:
    """
    Cierra posiciones por emergencia (sin importar SL/TP).
    Usado cuando Risk Manager detecta condiciones extremas, Portfolio TP, Smart Rotation, etc.
    ai_reasoning: texto opcional del LLM explicando por qué se cerró.
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
            pos["close_reason"] = reason
            pos["close_price"] = pos["current_price"]

            # Actualizar P&L con leverage
            notional = pos.get("notional_value", pos.get("size_usd", 0))
            margin = pos.get("margin_usd", pos.get("size_usd", 0))

            pnl_pct = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"]
            if pos["direction"] == "short":
                pnl_pct = -pnl_pct
            # FIX 1.2: Incluir exit fee en emergency close
            # FIX D (2026-04-18): slippage symmetry with normal close (executor.py:1300)
            fee_exit = notional * (TAKER_FEE + get_slippage(pos["symbol"]))
            pnl_usd = notional * pnl_pct + pos.get("funding_accumulated", 0) - fee_exit - pos.get("fee_entry", 0)
            pos["fee_exit"] = round(fee_exit, 4)

            pos["pnl_pct"] = round(pnl_pct * 100, 4)
            pos["pnl_usd"] = round(pnl_usd, 4)

            # Actualizar stats del portfolio
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            if pnl_usd > 0:
                portfolio["wins"] = portfolio.get("wins", 0) + 1
            else:
                portfolio["losses"] = portfolio.get("losses", 0) + 1

            # Devolver margen + P&L al portfolio
            returned = max(0, margin + pnl_usd)
            portfolio["capital_usd"] += returned
            # FIX A (2026-04-18): when margin floor clamps capital to 0, align
            # the recorded pnl_usd with the actual capital delta (-margin).
            # Previously the record kept the unclamped (more negative) value,
            # causing recorded pnl_usd < true capital loss and an accounting gap.
            if margin + pnl_usd < 0:
                pos["pnl_usd"] = round(-margin, 4)
                pos["pnl_pct"] = round(-100.0, 4)

            # Agregar al historial
            record = {
                "id": pos["id"],
                "symbol": pos["symbol"],
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": pos["close_price"],
                "current_price": pos["close_price"],
                "size_usd": pos["size_usd"],
                "notional_value": pos.get("notional_value", pos.get("size_usd", 0)),
                "margin_usd": pos.get("margin_usd", 0),
                "leverage": pos.get("leverage", 1),
                "fee_entry": pos.get("fee_entry", 0),
                "fee_exit": pos.get("fee_exit", 0),
                "funding_accumulated": pos.get("funding_accumulated", 0),
                "pnl_usd": pos["pnl_usd"],
                "pnl_pct": pos["pnl_pct"],
                "open_time": pos["open_time"],
                "close_time": pos["close_time"],
                "close_reason": reason,  # Usar la razón correcta, no hardcodear
                "strategy": pos.get("strategy", "unknown"),
                "mode": pos.get("mode", "paper"),
            }
            if ai_reasoning:
                record["ai_reasoning"] = ai_reasoning
            history.append(record)

            closed.append(pos)
            log.error(f"🚨 EMERGENCY CLOSE: {pos['symbol']} | P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")

    # Remove closed positions from portfolio
    portfolio["positions"] = [p for p in portfolio["positions"] if p.get("status") != "closed"]

    return closed


# ─── Paper Trading ────────────────────────────────────────────────────────────

def get_fear_greed_index() -> int:
    """Obtiene el Fear & Greed index actual (0-100)."""
    try:
        fg_file = DATA_DIR / "market_latest.json"
        if fg_file.exists():
            with open(fg_file) as f:
                data = json.load(f)
                return int(data.get("fear_greed", {}).get("value", 50))
    except Exception:
        pass
    return 50  # Neutral por defecto


def paper_open_position(signal: dict, portfolio: dict, market: dict) -> Optional[dict]:
    """
    Abre una posición en paper trading simulando Drift Protocol.
    - Leverage configurable (default 3x, max 10x)
    - Solo se descuenta el margen del capital
    - Calcula precio de liquidación
    """
    symbol = signal["symbol"]
    direction = signal.get("direction", "")
    confidence = signal.get("confidence", 0)
    
    # ─── Risk Filters (27-Mar-2026) ───────────────────────────────────────────
    # 1. Filtrar por confidence mínima
    # Liquidity check for non-core tokens
    CORE_TOKENS_SET = {"SOL", "BTC", "ETH", "JUP", "RAY", "WIF", "PENGU", "FARTCOIN", "GOAT", "POPCAT", "MOODENG", "BONK"}
    if symbol not in CORE_TOKENS_SET:
        _token_data = market.get("tokens", {}).get(symbol, {})
        _liq = _token_data.get("liquidity", 0)
        if _liq < 500000:
            log.warning(f"\u26a0\ufe0f Trade rejected: {symbol} liquidity ${_liq:.0f} < $500K min")
            return None

    # M7: Reject trades on stale market data
    if isinstance(market, dict) and market.get("_stale"):
        log.warning(f"\u26a0\ufe0f Trade rejected: market data is stale")
        return None

    # E1: Confidence boost for shorts in extreme fear
    fear_greed = get_fear_greed_index()
    if direction == "short" and fear_greed < 15:
        confidence = min(0.99, confidence + 0.15)
        log.info(f"   \U0001f525 EXTREME FEAR boost: {symbol} conf +15% -> {confidence:.2f}")

    if confidence < MIN_CONFIDENCE:
        log.info(f"⏭️  Señal {symbol} ignorada: confidence {confidence:.2f} < {MIN_CONFIDENCE}")
        return None
    
    # 2. Bloquear LONGs en mercado bajista (Fear & Greed < threshold)
    # EXCEPTION: permitir LONGs si RSI < 40 (sobreextendido a la baja = oportunidad de rebote)
    if direction == "long":
        fear_greed = get_fear_greed_index()
        rsi = signal.get("rsi", 50)
        # E5: Bypass FG block for mean reversion longs
        if signal.get("strategy") == "mean_reversion" and signal.get("bypass_fg_block"):
            log.info(f"   \U0001f504 Mean reversion LONG {symbol} bypasses FG block (RSI={rsi:.0f})")
        elif fear_greed < BLOCK_LONGS_FG and rsi >= 40:
            log.info(f"⏭️  LONG {symbol} bloqueado: Fear & Greed {fear_greed} < {BLOCK_LONGS_FG} y RSI={rsi:.1f} no sobrevendido")
            return None
        elif fear_greed < BLOCK_LONGS_FG and rsi < 40:
            log.info(f"   ✅ LONG {symbol} PERMITIDO: FG={fear_greed} bajo pero RSI={rsi:.1f} sobrevendido — esperando rebote")
    
    # ─── ANTI-REBOUND SHORT FILTER (SOLAA-23) — safety net en paper_open_position ─
    # Bloquear SHORT cuando RSI<30 + tendencia bullish (ver helper _should_block_short_rebound)
    if direction == "short":
        block, reason = _should_block_short_rebound(signal, market)
        if block:
            log.info(f"⏭️  SHORT {symbol} bloqueado: {reason}")
            return None

    # ─── FIX 2: Cooldown después de emergency close ────────────────────────────
    if _is_in_emergency_cooldown(symbol):
        log.info(f"⏭️  {symbol} en cooldown — skip")
        return None

    price = get_current_price(symbol, market)

    if price <= 0:
        log.warning(f"⚠️  Sin precio para {symbol}, no se puede abrir posición")
        return None

    # Determinar leverage (signal puede sugerirlo, si no, default)
    # Capital-proportional confidence-based sizing
    _capital = portfolio.get("capital_usd", 500) + sum(
        p.get("margin_usd", 0) for p in portfolio.get("positions", []) if p.get("status") == "open"
    )
    _wm = _is_wild_mode_active()
    _dyn_risk, leverage = _get_risk_for_capital(_capital, confidence, wild_mode=_wm)
    signal["_coordinated_risk"] = _dyn_risk
    log.info(f"   \U0001f4b0 Sizing: capital=${_capital:.0f} conf={confidence:.2f} -> risk=${_dyn_risk:.2f} lev={leverage}x")
    leverage = max(1, min(leverage, MAX_LEVERAGE))
    # WILD MODE: signal may force leverage (inherited from chain base position)
    if signal.get("_force_leverage"):
        leverage = max(1, min(int(signal["_force_leverage"]), MAX_LEVERAGE))
        log.info(f"   \U0001f525 Wild mode: leverage forced to {leverage}x")

    # ─── Position Sizing basado en Regla de Ender ─────────────────────────
    # Regla: ganar mín $4 neto después de comisiones, arriesgar máx $2
    #
    # Cálculo:
    #   max_risk = notional * sl_pct + notional * fee * 2  ≤  MAX_RISK_USD
    #   min_profit = notional * tp_pct - notional * fee * 2  ≥  MIN_NET_PROFIT_USD
    #
    # De max_risk: notional ≤ MAX_RISK_USD / (sl_pct + fee*2)
    # De min_profit: notional ≥ MIN_NET_PROFIT_USD / (tp_pct - fee*2)

    # SL/TP: usar los del signal si existen, sino defaults
    sl_pct = 0.02   # 2% SL (ajustado para regla de Ender)
    tp_pct = 0.05   # 5% TP (2.5:1 R:R → $1.09 profit, $0.50 risk)

    if signal.get("sl_price", 0) > 0 and signal.get("tp_price", 0) > 0:
        if signal["direction"] == "long":
            sl_pct = abs(price - signal["sl_price"]) / price
            tp_pct = abs(signal["tp_price"] - price) / price
        else:
            sl_pct = abs(signal["sl_price"] - price) / price
            tp_pct = abs(price - signal["tp_price"]) / price

    # MEJORA C: Adaptive TP - en mercado lateral (FG extremo), TP mas corto
    try:
        _mkt_file = DATA_DIR / "market_latest.json"
        if _mkt_file.exists():
            _mkt = json.loads(_mkt_file.read_text())
            _fg_raw = _mkt.get("fear_greed", 50)
            _fg = _fg_raw.get("value", 50) if isinstance(_fg_raw, dict) else int(_fg_raw or 50)
            if _fg <= 20 or _fg >= 80:
                max_tp = 0.05  # E3: 5% max TP en extremos (was 3%)
                if tp_pct > max_tp:
                    log.info(f"🌡 Adaptive TP: F&G={_fg} extreme -> TP {tp_pct*100:.1f}% -> {max_tp*100:.1f}%")
                    tp_pct = max_tp
    except Exception:
        pass

    # Asegurar R:R minimo de 1.2:1
    if tp_pct < sl_pct * 1.2:
        tp_pct = sl_pct * 1.5
        log.info(f"📐 TP ajustado a {tp_pct*100:.1f}% para mantener R:R >= 1.5:1")

    fee_round_trip = TAKER_FEE * 2  # Entry + exit fees

    # m1: Division by zero protection
    if tp_pct <= fee_round_trip:
        log.warning(f"\u26a0\ufe0f Trade rejected: tp_pct {tp_pct:.4f} <= fees {fee_round_trip:.4f}")
        return None

    # ─── Volatility-Adaptive + Coordinated Sizing (2026-03-31 v2) ────────
    # El sistema coordina el sizing de TODAS las posiciones del portafolio:
    # 1. La IA decide cuántas posiciones abrir (N)
    # 2. El profit target se distribuye: target_por_pos = PORTFOLIO_TP / N
    # 3. El notional se calcula para alcanzar ese target, ajustado por ATR
    # 4. Portfolio TP cierra TODO cuando se alcanza el target combinado
    REFERENCE_VOL = 0.025    # 2.5% = volatilidad "normal" de referencia
    VOL_MULT_MIN  = 0.5      # Alta vol → posición más chica
    VOL_MULT_MAX  = 3.0      # Baja vol → posición hasta 3x más grande
    VOL_ATR_FLOOR = 0.003    # Piso para evitar extremos (0.3%)

    # Obtener ATR% del token desde signals_latest.json
    token_atr_pct = None
    try:
        signals_file = DATA_DIR / "signals_latest.json"
        if signals_file.exists():
            sig_data = json.loads(signals_file.read_text())
            ind_summary = sig_data.get("indicator_summary", {})
            token_atr_pct = ind_summary.get(symbol, {}).get("atr_pct")
    except Exception:
        pass

    vol_factor = 1.0
    if token_atr_pct and token_atr_pct > 0:
        effective_atr = max(token_atr_pct / 100.0, VOL_ATR_FLOOR)
        vol_factor = REFERENCE_VOL / effective_atr
        vol_factor = max(VOL_MULT_MIN, min(vol_factor, VOL_MULT_MAX))

    # Usar sizing coordinado si viene del orchestrator
    coordinated_risk = signal.get("_coordinated_risk", MAX_RISK_USD)
    coordinated_profit = signal.get("_coordinated_profit_target", PORTFOLIO_TP_USD)
    n_positions = signal.get("_coordinated_n", 1)

    # Aplicar factor de volatilidad al riesgo coordinado
    adjusted_risk = coordinated_risk * vol_factor
    # Hard cap: adjusted_risk × N no puede exceder PORTFOLIO_MAX_RISK_USD
    _portfolio_max = max(_capital * PORTFOLIO_RISK_PCT, PORTFOLIO_MAX_RISK_USD) if '_capital' in dir() else PORTFOLIO_MAX_RISK_USD
    max_adj_per_pos = _portfolio_max / max(n_positions, 1)
    adjusted_risk = min(adjusted_risk, max_adj_per_pos)

    # Cap risk so margin stays under 20% of equity
    _cap_eq = portfolio.get("capital_usd", 100) + sum(
        p.get("margin_usd", 0) for p in portfolio.get("positions", []) if p.get("status") == "open"
    )
    _cap_max_margin = _cap_eq * 0.20
    _cap_max_notional = _cap_max_margin * leverage
    _cap_max_risk = _cap_max_notional * (sl_pct + fee_round_trip)
    if adjusted_risk > _cap_max_risk:
        log.info(f"   \U0001f4cf Risk capped for 20% margin: ${adjusted_risk:.2f} -> ${_cap_max_risk:.2f} (eq=${_cap_eq:.0f})")
        adjusted_risk = _cap_max_risk

    log.info(f"📊 Coordinated sizing: {n_positions} pos | risk=${coordinated_risk:.2f}×vol{vol_factor:.1f}=${adjusted_risk:.2f} | target=${coordinated_profit:.2f}/pos"
             + (f" | ATR={token_atr_pct:.2f}%" if token_atr_pct else ""))

    # Calcular notional óptimo
    max_notional_by_risk = adjusted_risk / (sl_pct + fee_round_trip)
    min_notional_by_profit = coordinated_profit / (tp_pct - fee_round_trip)

    if min_notional_by_profit > max_notional_by_risk:
        # Imposible cumplir ambas reglas → ajustar TP para que funcione
        needed_tp = (coordinated_profit / max_notional_by_risk) + fee_round_trip
        log.info(f"📐 Ajustando TP de {tp_pct*100:.1f}% a {needed_tp*100:.1f}% para cumplir regla coordinada")
        tp_pct = needed_tp

    notional_value = max_notional_by_risk  # Usar max permitido por riesgo
    margin_usd = notional_value / leverage

    # WILD MODE: signal may force exact margin (bypass risk-based sizing)
    if signal.get("_force_margin"):
        margin_usd = float(signal["_force_margin"])
        notional_value = margin_usd * leverage
        log.info(f"   \U0001f525 Wild mode: margin forced to ${margin_usd:.2f} (notional ${notional_value:.2f})")

    # Hard cap: max 25% of equity per position (safety for small accounts)
    _eq = portfolio.get("capital_usd", 100) + sum(
        p.get("margin_usd", 0) for p in portfolio.get("positions", []) if p.get("status") == "open"
    )
    _max_margin = _eq * 0.20  # 20% max per position (tighter for small accounts)
    if margin_usd > _max_margin:
        log.info(f"   \U0001f4cf Margin capped: ${margin_usd:.2f} -> ${_max_margin:.2f} (25% of ${_eq:.0f})")
        margin_usd = round(_max_margin, 2)
        notional_value = round(margin_usd * leverage, 2)

    # Verificar capital suficiente
    if portfolio["capital_usd"] < margin_usd:
        log.warning(f"⚠️  Capital insuficiente: ${portfolio['capital_usd']:.2f} < ${margin_usd:.2f}")
        return None

    # Verificar que el trade cumple la regla antes de ejecutar
    expected_profit = notional_value * tp_pct - notional_value * fee_round_trip
    expected_risk = notional_value * sl_pct + notional_value * fee_round_trip
    log.info(f"💰 Sizing: notional=${notional_value:.1f} margin=${margin_usd:.1f} SL={sl_pct*100:.1f}% TP={tp_pct*100:.1f}%")
    log.info(f"   Expected profit: ${expected_profit:.2f} | Max risk: ${expected_risk:.2f} | R:R 1:{expected_profit/expected_risk:.1f}")

    if expected_profit < coordinated_profit - 0.01:  # epsilon para rounding
        log.warning(f"⚠️  Trade rechazado: profit esperado ${expected_profit:.2f} < ${coordinated_profit:.2f}")
        return None

    # Fees sobre el notional
    fee_entry = notional_value * (TAKER_FEE + get_slippage(symbol))  # FIX 2.4: includes slippage
    tokens = (notional_value - fee_entry) / price

    # Margen de mantenimiento
    margin_maintenance = notional_value * MAINTENANCE_MARGIN

    # Precio de liquidación
    if signal["direction"] == "long":
        liq_price = price * (1 - (margin_usd - fee_entry) / notional_value)
    else:
        liq_price = price * (1 + (margin_usd - fee_entry) / notional_value)

    # Calcular SL/TP prices
    sl_price = price * (1 - sl_pct) if signal["direction"] == "long" else price * (1 + sl_pct)
    tp_price = price * (1 + tp_pct) if signal["direction"] == "long" else price * (1 - tp_pct)

    # v2.10.0 Fix B: MIN_SL_DISTANCE_PCT - rechazar trades con SL muy ajustado
    sl_distance_pct = abs(price - sl_price) / price * 100 if price > 0 else 0
    if sl_distance_pct < MIN_SL_DISTANCE_PCT:
        log.warning(f"\u26a0\ufe0f {symbol}: SL a {sl_distance_pct:.2f}% del entry < min {MIN_SL_DISTANCE_PCT}%, skip "
                    f"(evita MOODENG-pattern: SL 0.38% + 10x = -33% en 35min)")
        return None

    # v2.10.0 Fix C: MAX_NOTIONAL_PCT_EQUITY - cap notional a 50% del equity
    current_equity = float(portfolio.get("capital_usd", 0))
    if current_equity > 0:
        max_notional_allowed = current_equity * MAX_NOTIONAL_PCT_EQUITY
        if notional_value > max_notional_allowed:
            log.warning(f"\u26a0\ufe0f {symbol}: notional ${notional_value:.2f} > {MAX_NOTIONAL_PCT_EQUITY*100:.0f}% equity "
                        f"(${max_notional_allowed:.2f}), skip "
                        f"(MOODENG-pattern: notional 160% equity)")
            return None

    # FIX: Validate sizing - reject if margin or notional is zero/negative
    if notional_value <= 0 or margin_usd <= 0:
        log.warning(f"\u26a0\ufe0f Position rejected: {symbol} has zero sizing (notional=${notional_value:.2f}, margin=${margin_usd:.2f})")
        return None

    # FIX: Validate TP price is sane (positive, and on correct side of entry)
    if tp_price <= 0:
        log.warning(f"\u26a0\ufe0f Position rejected: {symbol} has invalid tp_price={tp_price:.8f}")
        return None
    if signal["direction"] == "long" and tp_price <= price:
        log.warning(f"\u26a0\ufe0f Position rejected: LONG {symbol} tp_price={tp_price:.8f} <= entry={price:.8f}")
        return None
    if signal["direction"] == "short" and tp_price >= price:
        log.warning(f"\u26a0\ufe0f Position rejected: SHORT {symbol} tp_price={tp_price:.8f} >= entry={price:.8f}")
        return None

    position = {
        "id": f"{symbol}_{int(time.time())}",
        "symbol": symbol,
        "direction": signal["direction"],
        "strategy": signal["strategy"],
        "entry_price": round(price, 8),
        "current_price": round(price, 8),
        "margin_usd": round(margin_usd, 2),          # Lo que puso el trader
        "notional_value": round(notional_value, 2),   # Tamaño real de la posición
        "leverage": leverage,
        "size_usd": round(notional_value, 2),         # Compat: = notional
        "tokens": round(tokens, 6),
        "sl_price": round(sl_price, 8),
        "tp_price": round(tp_price, 8),
        "liquidation_price": round(liq_price, 8),
        "margin_maintenance": round(margin_maintenance, 4),
        "fee_entry": round(fee_entry, 4),
        "funding_accumulated": 0.0,
        "pnl_usd": 0.0,
        "pnl_pct": 0.0,
        "status": "open",
        "open_time": datetime.now(timezone.utc).isoformat(),
        "close_time": None,
        "mode": "paper",
        "confidence": signal.get("confidence", 0),
        "last_funding_time": datetime.now(timezone.utc).isoformat(),
        # Trailing stop support (2026-03-31: default = trailing para preservar ganancias)
        "exit_mode": signal.get("exit_mode", "trailing"),  # "fixed" or "trailing"
        "trailing_pct": signal.get("trailing_pct", 0.015),  # 1.5% pullback desde peak
        "peak_price": round(price, 8),  # highest price seen (for long) / lowest (for short)
        "trailing_sl": 0.0,  # dynamic SL that follows the peak
    }

    # Descontar solo el MARGEN del capital (no el notional completo)
    _new_cap = round(portfolio["capital_usd"] - margin_usd, 2)
    if _new_cap < 0:
        log.error(f"\U0001f6d1 NEGATIVE CAPITAL prevented: ${portfolio['capital_usd']:.2f} - ${margin_usd:.2f} = ${_new_cap:.2f} - REJECTING TRADE")
        return None
    portfolio["capital_usd"] = _new_cap
    # Paperclip: track trade open
    if _PAPERCLIP:
        try:
            _pc_id = on_trade_opened(position)
            if _pc_id:
                position['paperclip_issue_id'] = _pc_id
        except Exception:
            pass
    if _OPENCLAW_WH:
        try: _ocwh.on_trade_opened(position)
        except Exception: pass

    portfolio["positions"].append(position)

    log.info(f"    📐 Leverage: {leverage}x | Margen: ${margin_usd:.2f} | Notional: ${notional_value:.2f} | Liq: ${liq_price:.6f}")

    return position


def apply_funding_rate(pos: dict) -> float:
    """
    Simula el funding rate de Drift Protocol.
    Se aplica cada hora sobre el notional value.
    Longs pagan cuando funding > 0, shorts pagan cuando funding < 0.
    Retorna el monto de funding aplicado (negativo = pagado, positivo = recibido).
    """
    now = datetime.now(timezone.utc)
    last_funding_str = pos.get("last_funding_time", pos.get("open_time"))
    try:
        last_funding = datetime.fromisoformat(last_funding_str)
        if last_funding.tzinfo is None:
            last_funding = last_funding.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        last_funding = now

    # Calcular horas transcurridas desde último funding
    hours_elapsed = (now - last_funding).total_seconds() / 3600.0
    if hours_elapsed < 1.0:
        return 0.0

    # Número de períodos de funding a aplicar
    periods = int(hours_elapsed)
    notional = pos.get("notional_value", pos.get("size_usd", 0))

    # Funding rate simulado: longs pagan, shorts reciben (mercado normal)
    # En mercado bajista sería al revés, pero simplificamos con rate fijo
    funding_per_period = notional * FUNDING_RATE
    total_funding = funding_per_period * periods

    # Longs pagan funding (negativo), shorts reciben (positivo)
    if pos["direction"] == "long":
        funding_impact = -total_funding
    else:
        funding_impact = total_funding

    pos["funding_accumulated"] = round(pos.get("funding_accumulated", 0) + funding_impact, 6)
    pos["last_funding_time"] = now.isoformat()

    if abs(funding_impact) > 0.001:
        direction_text = "pagó" if funding_impact < 0 else "recibió"
        log.info(f"    💸 {pos['symbol']} {direction_text} ${abs(funding_impact):.4f} funding ({periods}h)")

    return funding_impact


def paper_update_positions(portfolio: dict, market: dict, history: list) -> list:
    """
    Actualiza P&L de todas las posiciones abiertas.
    Simula Drift Protocol: leverage, liquidación, funding rate.
    Cierra automáticamente si toca SL, TP o precio de liquidación.
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

        # Aplicar funding rate
        funding_impact = apply_funding_rate(pos)

        # Calcular P&L basado en NOTIONAL (leverage amplifica ganancias Y pérdidas)
        notional = pos.get("notional_value", pos.get("size_usd", 0))
        margin = pos.get("margin_usd", pos.get("size_usd", 0))
        leverage = pos.get("leverage", 1)

        if pos["direction"] == "long":
            price_pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        else:
            price_pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

        # P&L = movimiento de precio * notional + funding acumulado
        pnl_from_price = notional * price_pnl_pct
        funding_total = pos.get("funding_accumulated", 0)
        pnl_usd = pnl_from_price + funding_total

        # P&L % relativo al MARGEN (no al notional) — refleja retorno real del trader
        pnl_pct_on_margin = (pnl_usd / margin * 100) if margin > 0 else 0

        pos["pnl_usd"] = round(pnl_usd, 4)
        pos["pnl_pct"] = round(pnl_pct_on_margin, 4)

        # ─── Verificar LIQUIDACIÓN (Drift Protocol) ─────────────────────
        liq_price = pos.get("liquidation_price", 0)
        hit_liquidation = False

        if liq_price > 0:
            if pos["direction"] == "long":
                hit_liquidation = current_price <= liq_price
            else:
                hit_liquidation = current_price >= liq_price

        if hit_liquidation:
            pos["status"] = "closed"
            pos["close_time"] = datetime.now(timezone.utc).isoformat()
            pos["close_reason"] = "LIQUIDATED"
            pos["close_price"] = current_price

            # En liquidación se pierde todo el margen
            pos["pnl_usd"] = round(-margin, 4)
            pos["pnl_pct"] = -100.0
            pos["fee_exit"] = 0.0

            # No se devuelve nada — margen perdido completamente
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            portfolio["losses"] = portfolio.get("losses", 0) + 1

            log.error(f"  💀 [LIQUIDATED] {symbol} {leverage}x {pos['direction']} | "
                      f"Perdido: ${margin:.2f} margen completo | Liq: ${liq_price:.6f}")

            history.append({**pos})
            closed.append(pos)
            continue

        # ─── Trailing Stop Logic ─────────────────────────────────────────
        exit_mode = pos.get("exit_mode", "fixed")
        trailing_pct = pos.get("trailing_pct", 0.0)

        # FIX 2.3: ATR-based dynamic trailing distance
        if exit_mode == "trailing" and trailing_pct > 0:
            try:
                import json as _json
                sig_path = DATA_DIR / "signals_latest.json"
                if sig_path.exists():
                    _sig = _json.loads(sig_path.read_text())
                    _atr = _sig.get("indicator_summary", {}).get(symbol, {}).get("atr_pct")
                    if _atr and _atr > 0:
                        dynamic_trail = (_atr / 100.0) * 1.5  # 1.5x ATR
                        dynamic_trail = max(0.01, min(dynamic_trail, 0.08))  # Clamp 1%-8%
                        pos["trailing_pct"] = dynamic_trail
                        trailing_pct = dynamic_trail
            except Exception:
                pass
            # PROGRESSIVE TRAILING: tighten based on profit level
            if pnl_pct_on_margin > 0 and trailing_pct > 0:
                if pnl_pct_on_margin >= 15:    # +15% margin = very profitable
                    trailing_pct = min(trailing_pct, 0.004)  # 0.4% tight trail
                    pos["trailing_pct"] = trailing_pct
                elif pnl_pct_on_margin >= 10:  # +10% margin
                    trailing_pct = min(trailing_pct, 0.006)  # 0.6% trail
                    pos["trailing_pct"] = trailing_pct
                elif pnl_pct_on_margin >= 5:   # +5% margin
                    trailing_pct = min(trailing_pct, 0.008)  # 0.8% trail
                    pos["trailing_pct"] = trailing_pct

            if pos["direction"] == "long":
                # Update peak if price made new high
                if current_price > pos.get("peak_price", pos["entry_price"]):
                    pos["peak_price"] = round(current_price, 8)
                # Trailing SL follows peak downward
                pos["trailing_sl"] = round(pos["peak_price"] * (1 - trailing_pct), 8)
            else:
                # Short: update peak (lowest) if price made new low
                if current_price < pos.get("peak_price", pos["entry_price"]):
                    pos["peak_price"] = round(current_price, 8)
                pos["trailing_sl"] = round(pos["peak_price"] * (1 + trailing_pct), 8)

        # ─── FIX 2.2: Partial Profit Taking (50% at halfway to TP) ────────
        partial_taken = pos.get("partial_taken", False)
        if not partial_taken and exit_mode == "trailing" and notional > 0 and margin > 0:
            tp_price = pos.get("tp_price", 0)
            entry = pos["entry_price"]
            if tp_price > 0 and entry > 0:
                if pos["direction"] == "long":
                    halfway = entry + (tp_price - entry) * 0.5
                    reached_half = current_price >= halfway
                else:
                    halfway = entry - (entry - tp_price) * 0.5
                    reached_half = current_price <= halfway
                if reached_half:
                    reduce_frac = 0.5
                    reduced_notional = notional * reduce_frac
                    reduced_margin = margin * reduce_frac
                    # FIX C (2026-04-18): include slippage in partial exit fee (parity with Fix D)
                    fee_partial = reduced_notional * (TAKER_FEE + get_slippage(pos["symbol"]))
                    partial_pnl = pnl_usd * reduce_frac - fee_partial
                    returned = max(0, reduced_margin + partial_pnl)
                    portfolio["capital_usd"] = round(portfolio["capital_usd"] + returned, 2)
                    pos["notional_value"] = round(notional - reduced_notional, 2)
                    pos["margin_usd"] = round(margin - reduced_margin, 2)
                    pos["size_usd"] = pos["notional_value"]
                    # FIX C (2026-04-18): apportion fee_entry on remaining pos so final
                    # close does not re-subtract the full original entry fee.
                    pos["fee_entry"] = round(pos.get("fee_entry", 0) * (1 - reduce_frac), 4)
                    if pos.get("tokens", 0) > 0:
                        pos["tokens"] = round(pos["tokens"] * (1 - reduce_frac), 8)
                    pos["sl_price"] = pos["entry_price"]  # Move SL to breakeven
                    pos["partial_taken"] = True
                    # FIX 3: Tighten trailing after partial take
                    _old_trail = pos.get("trailing_pct", 0.015)
                    pos["trailing_pct"] = max(0.004, _old_trail * 0.5)  # 50% tighter
                    if pos["direction"] == "long":
                        pos["trailing_sl"] = round(pos.get("peak_price", current_price) * (1 - pos["trailing_pct"]), 8)
                    else:
                        pos["trailing_sl"] = round(pos.get("peak_price", current_price) * (1 + pos["trailing_pct"]), 8)
                    log.info(f"  \u2702\ufe0f Post-partial trailing tightened: {_old_trail*100:.1f}% -> {pos['trailing_pct']*100:.1f}%")
                    # Update local vars for remaining checks
                    notional = pos["notional_value"]
                    margin = pos["margin_usd"]
                    log.info(f"  \u2702\ufe0f PARTIAL TAKE: {symbol} 50% at halfway to TP, SL->breakeven, returned ${returned:.2f}")

                    # Record partial take as a trade in history (fix accounting gap)
                    # FIX B (2026-04-18): removed local datetime import that shadowed module-level binding
                    history.append({
                        "id": f"{pos['id']}_partial",
                        "symbol": symbol,
                        "direction": pos["direction"],
                        "strategy": pos.get("strategy", "unknown"),
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "close_price": current_price,
                        "margin_usd": round(reduced_margin, 2),
                        "notional_value": round(reduced_notional, 2),
                        "size_usd": round(reduced_notional, 2),
                        "leverage": pos.get("leverage", 5),
                        "pnl_usd": round(partial_pnl, 4),
                        "pnl_pct": round((partial_pnl / reduced_margin * 100) if reduced_margin > 0 else 0, 4),
                        "open_time": pos.get("open_time", ""),
                        "close_time": datetime.now(timezone.utc).isoformat(),
                        "close_reason": "PARTIAL_TAKE",
                        "fee_entry": round(pos.get("fee_entry", 0) * reduce_frac, 4),
                        "fee_exit": round(fee_partial, 4),
                        "status": "closed",
                        "partial_taken": True,
                    })
                    # Update portfolio stats
                    portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
                    if partial_pnl > 0:
                        portfolio["wins"] = portfolio.get("wins", 0) + 1
                    else:
                        portfolio["losses"] = portfolio.get("losses", 0) + 1

        # ─── MEJORA A: Breakeven Stop ────────────────────────────────────
        # Si la posicion estuvo en profit > 1% de margen por 30+ min, mover SL a entry
        if not pos.get("breakeven_activated", False) and margin > 0:
            # Track max profit seen
            if pnl_pct_on_margin > pos.get("max_pnl_pct", 0):
                pos["max_pnl_pct"] = round(pnl_pct_on_margin, 4)
                pos["max_pnl_time"] = datetime.now(timezone.utc).isoformat()

            if pos.get("max_pnl_pct", 0) >= 1.0:  # Fue >= +1% en margen
                max_time_str = pos.get("max_pnl_time", "")
                if max_time_str:
                    try:
                        max_time = datetime.fromisoformat(max_time_str)
                        if max_time.tzinfo is None:
                            max_time = max_time.replace(tzinfo=timezone.utc)
                        mins_since_peak = (datetime.now(timezone.utc) - max_time).total_seconds() / 60
                        if mins_since_peak >= 30:
                            pos["sl_price"] = pos["entry_price"]
                            pos["breakeven_activated"] = True
                            log.info(f"  \U0001f6e1 BREAKEVEN: {symbol} SL moved to entry (was +{pos['max_pnl_pct']:.1f}% margin, 30min+ ago)")
                    except Exception:
                        pass

        # ─── MEJORA B: Aggressive trailing activation ────────────────────
        # En modo fixed, activar trailing automatico si profit > 0.3% en precio
        if exit_mode == "fixed" and not pos.get("auto_trailing_activated", False):
            if price_pnl_pct > 0.005:  # E3: 0.5% activation (was 0.3%)  # +0.3% en precio (= +1.5% en margen con 5x)
                pos["exit_mode"] = "trailing"
                pos["trailing_pct"] = 0.008  # E3: 0.8% trail (was 0.5%)  # 0.5% trailing (tight, for lateral markets)
                pos["peak_price"] = current_price
                pos["auto_trailing_activated"] = True
                exit_mode = "trailing"
                trailing_pct = 0.005
                # Update trailing_sl immediately
                if pos["direction"] == "long":
                    pos["trailing_sl"] = round(current_price * (1 - trailing_pct), 8)
                else:
                    pos["trailing_sl"] = round(current_price * (1 + trailing_pct), 8)
                log.info(f"  \U0001f3af AUTO-TRAIL: {symbol} activated at +{price_pnl_pct*100:.2f}% price, trail=0.5%")

        # ─── MEJORA D: Time-based exit if profit reversed ────────────────
        # Si posicion tiene 4h+, fue profitable, y ahora pierde, cerrar
        hit_time_exit = False
        try:
            open_time = datetime.fromisoformat(pos.get("open_time", ""))
            if open_time.tzinfo is None:
                open_time = open_time.replace(tzinfo=timezone.utc)
            hours_open = (datetime.now(timezone.utc) - open_time).total_seconds() / 3600
            was_profitable = pos.get("max_pnl_pct", 0) >= 0.5
            now_losing = pnl_pct_on_margin < -0.5
            if hours_open >= 4 and was_profitable and now_losing:
                hit_time_exit = True

                # FIX 1: Respect breakeven SL - don't TIME_EXIT if near entry
                if pos.get("breakeven_activated", False):
                    dist = abs(current_price - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0
                    if dist < 0.035:  # <3.5% from entry - let breakeven SL handle
                        hit_time_exit = False
                        log.info(f"  \u23f0 TIME_EXIT deferred: breakeven active, {dist*100:.1f}% from entry")

                # FIX 2: Don't TIME_EXIT if SL gives better exit
                if hit_time_exit and not pos.get("breakeven_activated"):
                    sl_price = pos.get("sl_price", 0)
                    if sl_price > 0:
                        if pos["direction"] == "short":
                            cur_loss = (current_price - pos["entry_price"]) / pos["entry_price"]
                            sl_loss = (sl_price - pos["entry_price"]) / pos["entry_price"]
                        else:
                            cur_loss = (pos["entry_price"] - current_price) / pos["entry_price"]
                            sl_loss = (pos["entry_price"] - sl_price) / pos["entry_price"]
                        if cur_loss > sl_loss and sl_loss > 0:
                            hit_time_exit = False
                            log.info(f"  \u23f0 TIME_EXIT deferred: loss {cur_loss*100:.1f}% > SL {sl_loss*100:.1f}%")
                log.info(f"  \u23f0 TIME EXIT: {symbol} {hours_open:.1f}h open, was +{pos.get('max_pnl_pct',0):.1f}% now {pnl_pct_on_margin:.1f}%")
        except Exception:
            pass

        # ─── Verificar SL/TP ─────────────────────────────────────────────
        hit_sl = False
        hit_tp = False
        hit_trailing = False

        if pos["direction"] == "long":
            hit_sl = current_price <= pos["sl_price"]
            # FIX 1.1: En modo trailing, NO evaluar hard TP — dejar que trailing capture mas profit
            hit_tp = (current_price >= pos["tp_price"]) if exit_mode != "trailing" else False
            # Trailing SL only activates after price has moved past entry (in profit)
            if exit_mode == "trailing" and pos.get("trailing_sl", 0) > pos["entry_price"]:
                hit_trailing = current_price <= pos["trailing_sl"]
        else:
            hit_sl = current_price >= pos["sl_price"]
            hit_tp = (current_price <= pos["tp_price"]) if exit_mode != "trailing" else False
            if exit_mode == "trailing" and pos.get("trailing_sl", 0) < pos["entry_price"]:
                hit_trailing = current_price >= pos["trailing_sl"]

        if hit_sl or hit_tp or hit_trailing or hit_time_exit:
            if hit_time_exit:
                close_reason = "TIME_EXIT"
            elif hit_trailing:
                close_reason = "TRAILING_SL"
            elif hit_tp:
                close_reason = "TP"
            else:
                close_reason = "SL"
            pos["status"] = "closed"
            pos["close_time"] = datetime.now(timezone.utc).isoformat()
            pos["close_reason"] = close_reason
            pos["close_price"] = current_price

            # Fee de salida sobre el notional
            fee_exit = notional * (TAKER_FEE + get_slippage(symbol))  # FIX 2.4: includes slippage
            pos["fee_exit"] = round(fee_exit, 4)

            # FIX B (2026-04-18): normal close now records net lifetime pnl (gross - fees)
            # matching emergency close semantics. Capital delta == recorded pnl_usd.
            fee_entry = pos.get("fee_entry", 0)
            net_pnl = pnl_usd - fee_exit - fee_entry
            pos["pnl_usd"] = round(net_pnl, 4)
            pos["pnl_pct"] = round((net_pnl / margin * 100) if margin > 0 else 0, 4)

            # Devolver margen + net_pnl al portfolio
            returned = margin + net_pnl
            returned = max(0, returned)  # No puede ser negativo (ya se descontó margen)
            portfolio["capital_usd"] = round(portfolio["capital_usd"] + returned, 2)
            # FIX B (2026-04-18): if margin floor clamps capital, align record with
            # actual capital delta (= -margin). Mirrors Fix A in emergency close.
            if margin + net_pnl < 0:
                pos["pnl_usd"] = round(-margin, 4)
                pos["pnl_pct"] = round(-100.0, 4)

            # Estadísticas
            is_win = net_pnl > 0
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            if is_win:
                portfolio["wins"] = portfolio.get("wins", 0) + 1
            else:
                portfolio["losses"] = portfolio.get("losses", 0) + 1

            result_emoji = "✅" if is_win else "❌"
            log.info(f"  {result_emoji} [{close_reason}] {symbol} {leverage}x {pos['direction']} | "
                     f"P&L: ${net_pnl:+.2f} ({pnl_pct_on_margin:+.1f}% on margin) | "
                     f"Funding: ${funding_total:+.4f}")

            # Attach last known LLM reasoning for this symbol (from position_decisions.json)
            try:
                _pd_path = DATA_DIR / "position_decisions.json"
                if _pd_path.exists():
                    import json as _j2
                    _pd = _j2.loads(_pd_path.read_text())
                    _decisions = _pd.get("decisions", []) if isinstance(_pd, dict) else []
                    _sym_dec = next((x for x in _decisions if x.get("symbol") == symbol), None)
                    if _sym_dec and _sym_dec.get("llm_reasoning"):
                        _r = _sym_dec["llm_reasoning"].strip()
                        # Only store if it looks like real reasoning (>20 chars, not a header block)
                        if len(_r) > 20 and "workdir:" not in _r:
                            pos["ai_reasoning"] = _r
            except Exception:
                pass

            history.append({**pos})
            closed.append(pos)

            # Paperclip: track trade close
            if _PAPERCLIP:
                try:
                    on_trade_closed(pos)
                except Exception:
                    pass
            if _OPENCLAW_WH:
                try: _ocwh.on_trade_closed(pos)
                except Exception: pass

            # 📈 Compound Engine: actualizar capital base tras cada cierre
            if _COMPOUND_ENABLED:
                try:
                    state = update_compound_capital(net_pnl)
                    log.info(f"   📈 Compound: capital_base=${state['capital_base']:.2f} | "
                             f"return={state['total_return_pct']:.2f}% | "
                             f"cycles={state['compound_cycles']}")
                except Exception as e:
                    log.warning(f"   ⚠️ Compound update error: {e}")
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
            for pos in emergency_closed:
                sym = pos.get("symbol", "")
                if sym:
                    record_emergency_cooldown(sym)

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

    # ── Safety Nets ──────────────────────────────────────────────────────
    # FIX 3: Auto-clear DAILY_TARGET_HIT si es de un día anterior (reset diario)
    TARGET_HIT_FILE = DATA_DIR / "DAILY_TARGET_HIT"
    if TARGET_HIT_FILE.exists():
        try:
            content = TARGET_HIT_FILE.read_text()
            # Parsear timestamp del archivo (formato: "Daily target reached: 2026-04-02T...")
            import re
            m = re.search(r"(\d{4}-\d{2}-\d{2})", content)
            if m:
                file_date = m.group(1)
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if file_date != today:
                    TARGET_HIT_FILE.unlink()
                    log.info(f"🛡️ DAILY_TARGET_HIT limpiado (era del {file_date}, hoy es {today})")
            else:
                # Sin timestamp — limpiar por seguridad
                TARGET_HIT_FILE.unlink()
                log.info("🛡️ DAILY_TARGET_HIT limpiado (sin fecha — limpio por seguridad)")
        except Exception:
            pass

    # Kill switch: if STOP_TRADING file exists, don't open new positions
    STOP_FILE = DATA_DIR / "STOP_TRADING"
    if STOP_FILE.exists():
        log.warning("🛑 KILL SWITCH ACTIVE — STOP_TRADING file detected. No new positions.")
        save_history(history)  # BUG FIX: persist any trades closed this cycle before early return
        save_portfolio(portfolio)
        return {
            "status": "kill_switch_active",
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # FIX 3: Daily Target Hit con RE-EVALUACIÓN inteligente
    # Solo lockea si HAY posiciones abiertas. Si no hay, desbloquea automáticamente.
    # También resetea si cambió el día (medianoche Denver = 6AM UTC)
    TARGET_HIT_FILE = DATA_DIR / "DAILY_TARGET_HIT"
    DT_STATE_FILE = DATA_DIR / "daily_target_state.json"
    open_now = [p for p in portfolio["positions"] if p.get("status") == "open"]

    # Check si es un NUEVO DÍA (medianoche en Denver = 06:00 UTC)
    today_denver = datetime.now(timezone(timedelta(hours=-6))).strftime("%Y-%m-%d")
    if TARGET_HIT_FILE.exists():
        # Verificar si cambió el día
        state_date = None
        if DT_STATE_FILE.exists():
            try:
                dt_state = json.loads(DT_STATE_FILE.read_text())
                state_date = dt_state.get("date")
            except:
                pass
        if state_date != today_denver:
            # Nuevo día — borrar lock y dejar operar
            TARGET_HIT_FILE.unlink()
            log.info(f"   🟢 DAILY_TARGET: nuevo día ({today_denver}) — lock reseteado")
        elif len(open_now) == 0:
            # No hay posiciones abiertas — desbloquear
            TARGET_HIT_FILE.unlink()
            log.info("   🟢 DAILY_TARGET: sin posiciones abiertas — desbloqueado")
        else:
            # Hay posiciones y mismo día — reevaluar condiciones
            try:
                signals_file = DATA_DIR / "signals_latest.json"
                market_file = DATA_DIR / "market_latest.json"
                rsi = 50
                fear_greed = 50
                if signals_file.exists() and signals_file.stat().st_size > 0:
                    try:
                        sig_data = json.loads(signals_file.read_text())
                        rsi_vals = [s.get("rsi", 50) for s in sig_data.get("signals", []) if "rsi" in s]
                        rsi = sum(rsi_vals) / len(rsi_vals) if rsi_vals else 50
                    except:
                        pass
                if market_file.exists() and market_file.stat().st_size > 0:
                    try:
                        mkt_data = json.loads(market_file.read_text())
                        fg_raw = mkt_data.get("fear_greed", 50)
                        if isinstance(fg_raw, dict):
                            fear_greed = fg_raw.get("value", 50)
                        elif isinstance(fg_raw, (int, float)):
                            fear_greed = int(fg_raw)
                    except:
                        pass

                # Razones para DESBLOQUEAR:
                # 1. Fear & Greed < 20 (Extreme Fear / sobrevendido) → oportunidad de compra
                # 2. RSI < 35 (oversold) → rebote esperado
                should_unlock = fear_greed < 20 or rsi < 35

                if should_unlock:
                    reason = "Extreme Fear" if fear_greed < 20 else "RSI oversold"
                    log.info(f"   🟢 DAILY_TARGET overridden: {reason} (FG={fear_greed}, RSI={rsi:.1f}) — permitiendo posiciones")
                    TARGET_HIT_FILE.unlink()
                else:
                    log.warning(f"   🛡️ DAILY_TARGET_HIT activo — FG={fear_greed}, RSI={rsi:.1f}, manteniendo lock")
                    save_history(history)  # BUG FIX: persist any trades closed this cycle before early return
                    save_portfolio(portfolio)
                    return {
                        "status": "daily_target_hit",
                        "capital": portfolio["capital_usd"],
                        "opened": 0,
                        "closed": len(closed_this_cycle),
                        "fear_greed": fear_greed,
                        "rsi": rsi,
                    }
            except Exception as e:
                log.warning(f"   🛡️ DAILY_TARGET_HIT — error en re-eval: {e}, manteniendo lock")
                save_history(history)  # BUG FIX: persist any trades closed this cycle before early return
                save_portfolio(portfolio)
                return {
                    "status": "daily_target_hit",
                    "capital": portfolio["capital_usd"],
                    "opened": 0,
                    "closed": len(closed_this_cycle),
                }

    # Max daily loss: if realized P&L today < -$25, stop trading
    MAX_DAILY_LOSS = 25.0
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_pnl = sum(
        t.get("pnl_usd", 0) for t in closed_this_cycle
    )
    # Also count previously closed trades today from history
    if HISTORY_FILE.exists():
        try:
            all_history = json.loads(HISTORY_FILE.read_text())
            today_pnl += sum(
                t.get("pnl_usd", 0) for t in all_history
                if isinstance(t, dict) and t.get("close_time", "").startswith(today_str)
            )
        except Exception:
            pass

    if today_pnl < -MAX_DAILY_LOSS:
        log.warning(f"🛑 MAX DAILY LOSS hit: ${today_pnl:.2f} < -${MAX_DAILY_LOSS}. Stopping new trades.")
        save_history(history)  # BUG FIX: persist any trades closed this cycle before early return
        save_portfolio(portfolio)
        return {
            "status": "max_daily_loss",
            "daily_pnl": round(today_pnl, 2),
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # ── Max trades per day: 0 = sin límite (orden Ender 2026-03-29) ──
    MAX_TRADES_PER_DAY = 0
    today_trade_count = portfolio.get("total_trades", 0)
    if MAX_TRADES_PER_DAY > 0 and today_trade_count >= MAX_TRADES_PER_DAY:
        log.warning(f"🛑 MAX TRADES/DAY hit: {today_trade_count} >= {MAX_TRADES_PER_DAY}. No new trades today.")
        save_portfolio(portfolio)
        save_history(history)
        return {
            "status": "max_daily_trades",
            "trades_today": today_trade_count,
            "capital": portfolio["capital_usd"],
            "opened": 0,
            "closed": len(closed_this_cycle),
        }

    # ─── Coordinated Portfolio Opening (v2) ─────────────────────────────
    opened = []
    open_count = len([p for p in portfolio["positions"] if p.get("status") == "open"])

    # Leer MAX_POSITIONS desde auto_learner si disponible
    try:
        import json as _json
        from pathlib import Path
        _af = Path(__file__).parent / "data" / "auto_learner_state.json"
        if _af.exists():
            _adata = _json.loads(_af.read_text())
            _params = _adata.get("params", {})
            if _params and "max_positions" in _params:
                MAX_POSITIONS = _params["max_positions"]
                log.info(f"   🧠 Auto-Learner: MAX_POSITIONS={MAX_POSITIONS}")
            else:
                MAX_POSITIONS = 2
        else:
            MAX_POSITIONS = 2
    except Exception:
        MAX_POSITIONS = 2

    slots_available = MAX_POSITIONS - open_count

    # v2.9.6 B2: no abrir posiciones nuevas si ya estamos cerca del daily target
    # (>=80% del TARGET_MAX_PCT). Esas posiciones no tienen tiempo de desarrollar
    # antes del force-close por DAILY_TARGET_MAX_REACHED -> cierran en pérdida.
    _near_target = False
    try:
        _dt_state_file = DATA_DIR / "daily_target_state.json"
        if _dt_state_file.exists():
            _dts = json.loads(_dt_state_file.read_text())
            _curr_pct = float(_dts.get("current_pnl_pct", 0))
            _target_pct = float(_dts.get("target_pct", 0.05))
            # NOTE: current_pnl_pct se guarda como fraccion (0.04 = 4%) en algunos sitios,
            # pero como porcentaje (4.0) en otros. Normalizamos.
            if _curr_pct > 1:
                _curr_pct = _curr_pct / 100.0
            if _target_pct > 1:
                _target_pct = _target_pct / 100.0
            if _target_pct > 0 and _curr_pct >= _target_pct * 0.8:
                _near_target = True
                log.info(f"   🎯 DAILY_TARGET near-limit: daily_pnl {_curr_pct*100:.2f}% >= 80%% del target ({_target_pct*100:.1f}%%) — no abrir nuevas posiciones")
    except Exception as _e_dt_guard:
        log.debug(f"daily_target near-check error (non-fatal): {_e_dt_guard}")

    if slots_available <= 0:
        log.info(f"📊 Máximo de posiciones alcanzado ({MAX_POSITIONS})")
    elif _near_target:
        # Skip opening new positions near daily target (v2.9.6 B2)
        pass
    else:
        # Paso 1: Filtrar señales válidas (no duplicadas, con confidence, whitelist)
        valid_signals = []
        open_symbols = {p["symbol"] for p in portfolio["positions"] if p.get("status") == "open"}
        skipped_whitelist = []
        for signal in signals:
            sym = signal["symbol"]
            # v2.10.0: Whitelist estricta - solo majors (fix post-desastre 2026-04-18)
            if sym not in TRADE_WHITELIST_STRICT:
                skipped_whitelist.append(sym)
                if debug:
                    log.info(f"  skip {sym}: fuera de TRADE_WHITELIST_STRICT")
                continue
            if sym in open_symbols:
                if debug:
                    log.info(f"  skip {sym}: posicion ya abierta")
                continue
            if signal.get("confidence", 0) < MIN_CONFIDENCE:
                continue
            valid_signals.append(signal)
            if len(valid_signals) >= slots_available:
                break
        if skipped_whitelist:
            log.info(f"   [WL v2.10] excluidos {len(skipped_whitelist)} tokens fuera de majors: {skipped_whitelist[:6]}")

        # Paso 1b: FILTRO AI STRATEGY — excluir tokens que el Auto-Learner dice evitar
        avoid_file = DATA_DIR / "auto_learner_state.json"
        if avoid_file.exists():
            try:
                al_data = json.loads(avoid_file.read_text())
                # FIX 1.6: Leer tokens_to_avoid de ambos niveles del JSON
                avoid_tokens = set(
                    al_data.get("params", {}).get("tokens_to_avoid", []) or
                    al_data.get("tokens_to_avoid", [])
                )
                prefer_tokens = al_data.get("params", {}).get("tokens_to_prefer", [])
                if avoid_tokens:
                    before = len(valid_signals)
                    valid_signals = [s for s in valid_signals if s["symbol"] not in avoid_tokens]
                    skipped = before - len(valid_signals)
                    if skipped > 0:
                        log.info(f"   🧠 AI Strategy filter: excluyendo {skipped} tokens (avoid: {avoid_tokens})")
                        if prefer_tokens:
                            log.info(f"   🧠 AI Strategy prefers: {prefer_tokens}")
            except Exception as e:
                log.warning(f"   ⚠️ AI Strategy filter error: {e}")

        # Paso 1c: FILTRO ANTI-REBOUND — bloquear SHORTs cuando RSI<30 + tendencia bullish
        # (SOLAA-23: reemplaza el filtro antiguo de Extreme Fear RSI<15)
        market_file = DATA_DIR / "market_latest.json"
        market = {}
        if market_file.exists():
            try:
                market = json.loads(market_file.read_text())
            except:
                pass
        before = len(valid_signals)
        valid_signals = [s for s in valid_signals
            if s.get("direction", "").upper() != "SHORT" or not _should_block_short_rebound(s, market)[0]]
        skipped = before - len(valid_signals)
        if skipped > 0:
            rsi_kept = sum(1 for s in valid_signals if s.get("direction", "").upper() == "SHORT")
            log.warning(f"   ⚠️ ANTI-REBOUND SHORT FILTER: removidos {skipped} SHORTs (RSI<30+bullish), mantenidos {rsi_kept}")

        # Paso 2: Calcular sizing coordinado para N posiciones
        n_planned = len(valid_signals)
        if n_planned > 0:
            # Calcular riesgo ACTUAL de posiciones ya abiertas
            existing_risk = 0.0
            for epos in portfolio.get("positions", []):
                if epos.get("status") != "open":
                    continue
                e_entry = epos.get("entry_price", 0)
                e_notional = epos.get("notional_value", 0)
                e_sl = epos.get("sl_price", 0)
                if e_entry > 0 and e_notional > 0:
                    sl_dist = abs(e_sl - e_entry) / e_entry
                    existing_risk += e_notional * sl_dist + e_notional * TAKER_FEE * 2
            
            # Presupuesto de riesgo restante para nuevas posiciones
            remaining_risk_budget = max(0, PORTFOLIO_MAX_RISK_USD - existing_risk)
            
            if remaining_risk_budget <= 0:
                log.warning(f"⚠️ Sin presupuesto de riesgo: existente ${existing_risk:.2f} ≥ cap ${PORTFOLIO_MAX_RISK_USD:.2f}")
                n_planned = 0
                valid_signals = []
            else:
                # El profit target por posición se distribuye entre todas (existentes + nuevas)
                total_positions = open_count + n_planned
                _min_profit = max(_capital * MIN_PROFIT_PCT, MIN_PROFIT_PER_POS_USD) if '_capital' in dir() else MIN_PROFIT_PER_POS_USD
                profit_per_pos = max(PORTFOLIO_TP_USD / total_positions, _min_profit)
                # El riesgo por posición = presupuesto RESTANTE / N nuevas posiciones
                risk_per_pos = remaining_risk_budget / n_planned
                # También respetar el límite individual
                risk_per_pos = min(risk_per_pos, MAX_RISK_USD)

                log.info(f"📊 Coordinated sizing: {n_planned} nuevas + {open_count} existentes")
                log.info(f"   Risk existente: ${existing_risk:.2f} | Restante: ${remaining_risk_budget:.2f}")
                log.info(f"   Target: ${profit_per_pos:.2f}/pos | Risk: ${risk_per_pos:.2f}/pos")

            # Inyectar sizing coordinado en cada signal
            for sig in valid_signals:
                sig["_coordinated_n"] = n_planned
                sig["_coordinated_risk"] = risk_per_pos
                sig["_coordinated_profit_target"] = profit_per_pos

        # Paso 3: Abrir posiciones con sizing coordinado
        for signal in valid_signals:
            try:
                if safe:
                    pos = paper_open_position(signal, portfolio, market)
                else:
                    pos = real_open_position(signal, portfolio)
            except Exception as e:
                import traceback
                log.error(f"   💥 Error abriendo posición {signal.get('symbol','?')}: {e}")
                log.error(f"   Signal keys: {list(signal.keys())}")
                log.error(f"   Signal _coordinated_risk: {signal.get('_coordinated_risk')} ({type(signal.get('_coordinated_risk')).__name__})")
                log.error(f"   Signal _coordinated_profit_target: {signal.get('_coordinated_profit_target')} ({type(signal.get('_coordinated_profit_target')).__name__})")
                log.error(traceback.format_exc()[:500])
                continue

            if pos:
                opened.append(pos)
                open_count += 1
                open_symbols.add(signal["symbol"])
                arrow = "🟢" if pos["direction"] == "long" else "🔴"
                lev = pos.get("leverage", 1)
                margin = pos.get("margin_usd", pos.get("size_usd", 0))
                notional = pos.get("notional_value", pos.get("size_usd", 0))
                log.info(f"  {arrow} ABIERTA {signal['symbol']} [{signal['strategy']}] "
                         f"{lev}x | Margen: ${margin:.2f} | Notional: ${notional:.2f} @ ${pos['entry_price']:.6f}"
                         f" | Coord: {n_planned} pos, target ${profit_per_pos:.2f}/pos")

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
