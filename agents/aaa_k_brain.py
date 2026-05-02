#!/usr/bin/env python3
"""
AAA-K Brain — Wrapper Kimi 2.6 (Claude Sonnet 4.6 via OpenClaw)
"El Estratega" — Análisis macro, asignación de capital, diseño de estrategias.

Frecuencia: Cada 2 minutos
Input: Snapshot del mercado (top 50 tokens), contexto de portafolio
Output: Decisiones de trading + análisis de riesgo
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

log = logging.getLogger("aaa_k_brain")

# ─── LLM Config ─────────────────────────────────────────────────────────────

try:
    from llm_config import call_kimi
    _HAS_KIMI = True
except ImportError:
    _HAS_KIMI = False
    log.warning("llm_config.call_kimi no disponible")


def _call_llm(prompt: str, system: str = "", max_tokens: int = 4000) -> Optional[str]:
    """Llama a Kimi 2.6 via OpenClaw. Fallback a call_llm genérico."""
    if _HAS_KIMI:
        try:
            return call_kimi(prompt, system=system, max_tokens=max_tokens, temperature=0.3)
        except Exception as e:
            log.warning(f"Kimi error: {e}")
    # Fallback genérico
    try:
        from llm_config import call_llm
        return call_llm(prompt, system=system, max_tokens=max_tokens)
    except Exception as e:
        log.warning(f"LLM fallback error: {e}")
    return None


# ─── Prompt Builders ────────────────────────────────────────────────────────

def build_market_snapshot(market: dict, top_n: int = 30) -> str:
    """Construye snapshot de mercado para el prompt."""
    tokens = market.get("tokens", {})
    fg = market.get("fear_greed", {}).get("value", 50)

    # Ordenar por momentum/volumen
    token_list = []
    for sym, data in tokens.items():
        if data.get("price", 0) <= 0:
            continue
        momentum_score = abs(data.get("price_5min_change_pct", 0)) + abs(data.get("price_24h_change_pct", 0)) * 0.3
        token_list.append((sym, data, momentum_score))

    token_list.sort(key=lambda x: x[2], reverse=True)
    top_tokens = token_list[:top_n]

    lines = [f"FEAR & GREED: {fg}/100"]
    lines.append(f"{'Symbol':<8} {'Price':>12} {'5m%':>8} {'24h%':>8} {'Liq$M':>8} {'Vol$M':>8} {'Trend':>8}")
    lines.append("-" * 70)

    for sym, data, _ in top_tokens:
        price = data.get("price", 0)
        chg_5m = data.get("price_5min_change_pct", 0)
        chg_24h = data.get("price_24h_change_pct", 0)
        liq = data.get("liquidity", 0) / 1e6
        vol = data.get("volume_24h", 0) / 1e6
        trend = data.get("price_1h_trend", "?")
        lines.append(f"{sym:<8} ${price:>10.4f} {chg_5m:>+7.2f}% {chg_24h:>+7.2f}% ${liq:>7.2f} ${vol:>7.2f} {trend:>8}")

    return "\n".join(lines)


def build_portfolio_context(portfolio: dict) -> str:
    """Construye contexto de portafolio para el prompt."""
    capital = portfolio.get("capital_usd", 0)
    initial = portfolio.get("initial_capital", 50000)
    positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    pnl_total = sum(p.get("pnl_usd", 0) for p in positions)

    lines = [
        f"Capital disponible: ${capital:.2f} (inicial: ${initial:.2f})",
        f"Posiciones abiertas: {len(positions)}",
        f"PnL no realizado: ${pnl_total:+.2f}",
        f"Return total: {((capital + sum(p.get('margin_usd', 0) for p in positions) - initial) / initial * 100):+.2f}%",
    ]

    if positions:
        lines.append("\nPosiciones activas:")
        for p in positions:
            lines.append(f"  {p['symbol']} {p['direction'].upper()} | Entry: ${p['entry_price']:.4f} | Current: ${p['current_price']:.4f} | PnL: ${p.get('pnl_usd', 0):+.2f} ({p.get('pnl_pct', 0):+.2f}%)")

    return "\n".join(lines)


def build_trade_history_context(trades: List[dict], max_trades: int = 10) -> str:
    """Construye contexto de historial de trades recientes."""
    recent = sorted(trades, key=lambda x: x.get("close_time", ""), reverse=True)[:max_trades]
    if not recent:
        return "Sin trades cerrados aún."

    lines = [f"Últimos {len(recent)} trades cerrados:"]
    for t in recent:
        lines.append(f"  {t['symbol']} {t['direction'].upper()} | {t.get('close_reason', '?')} | PnL: ${t.get('pnl_usd', 0):+.2f} | Hold: {t.get('hours_open', 0):.1f}h")

    # Stats rápidas
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    wr = (len(wins) / len(trades) * 100) if trades else 0
    lines.append(f"\nStats: {len(wins)}W / {len(losses)}L | Win Rate: {wr:.1f}% | Total trades: {len(trades)}")

    return "\n".join(lines)


# ─── Main Decision Function ─────────────────────────────────────────────────



def apply_variant_filters_k(market: dict, filter_rules: dict) -> dict:
    """Apply variant-specific filter rules to market data for K."""
    if not market or "tokens" not in market:
        return market
    filtered = {}
    min_mom = filter_rules.get("min_momentum_pct", 0)
    min_liq = filter_rules.get("min_liquidity_usd", 0)
    min_vol = filter_rules.get("min_volume_24h", 0)
    for sym, data in market["tokens"].items():
        liq = data.get("liquidity", 0)
        vol = data.get("volume_24h", 0)
        mom_5m = abs(data.get("price_5min_change_pct", 0))
        if liq < min_liq:
            continue
        if vol < min_vol:
            continue
        if min_mom > 0 and mom_5m < min_mom:
            continue
        filtered[sym] = data
    market_copy = dict(market)
    market_copy["tokens"] = filtered
    return market_copy


def make_trading_decision(
    market: dict,
    portfolio: dict,
    trade_history: List[dict],
    max_positions: int = 10,
    variant: Optional[Dict] = None,
) -> Dict:
    """
    Pide a Kimi 2.6 una decisión de trading basada en el contexto actual.
    Retorna dict con: action, symbol, direction, confidence, reasoning, sizing_params
    """
    # Apply variant filter rules if present
    if variant and variant.get("filter_rules"):
        market = apply_variant_filters_k(market, variant["filter_rules"])
    market_snapshot = build_market_snapshot(market, top_n=30)
    portfolio_ctx = build_portfolio_context(portfolio)
    history_ctx = build_trade_history_context(trade_history, max_trades=10)

    open_symbols = {p["symbol"] for p in portfolio.get("positions", []) if p.get("status") == "open"}
    open_count = len(open_symbols)

    # Build variant-aware system prompt
    system_base = """Eres un gestor de riesgo experto en crypto trading. Tu estilo es CONSERVADOR y MACRO.
Analizas el mercado completo y decides si abrir, cerrar, o mantener posiciones.
Responde SOLO en JSON válido. No uses markdown fences."""
    variant_addon = variant.get("system_prompt_addon", "") if variant else ""
    if variant_addon:
        system = system_base + "\n\nVARIANTE ACTIVA: " + variant_addon
    else:
        system = system_base

    prompt = f"""Eres AAA-K ("El Estratega"), un agente de trading conservador con capital de $50,000.

REGLAS CRÍTICAS:
1. Máximo {max_positions} posiciones abiertas simultáneas. Actualmente tienes {open_count} abiertas.
2. NO abras más de 1 posición por token.
3. Si una posición abierta está perdiendo >-5% Y el mercado se deterioró claramente, recomienda CLOSE. NO cierres por micro-pérdidas.
4. Si una posición lleva >8h abierta sin mejorar, considera CLOSE (eficiencia de capital). Posiciones <1h NUNCA se cierran por "sin mejorar".
5. Fear & Greed < 20 o > 80 → reduce exposición, no abras nuevas posiciones.
6. Prioriza tokens con: liquidez >$1M, volumen alto, y tendencia clara a favor.
7. SL: 4-6%, TP: 6-10%, Leverage: 2-3x para bluechips, 1-2x para memes.
8. Kelly criterion conservador: 0.5× Kelly. Riesgo máximo por trade: 2% del capital.
9. IMPORTANTE: Una posición recién abierta tendrá P&L negativo inicial por comisiones (~0.8%). Esto es NORMAL. No es señal de cierre. Déjala correr hasta SL o TP.

SNAPSHOT DE MERCADO:
{market_snapshot}

CONTEXTO DE PORTAFOLIO:
{portfolio_ctx}

HISTORIAL RECIENTE:
{history_ctx}

Tokens ya en posiciones abiertas: {list(open_symbols) if open_symbols else 'ninguno'}

DECIDE:
- Si hay oportunidad clara: abre UNA posición (la mejor).
- Si hay posiciones que deben cerrarse: indica cuál y por qué.
- Si no hay nada claro: HOLD.

Responde en JSON exacto:
{{
  "action": "OPEN|CLOSE|HOLD",
  "symbol": "TOKEN",
  "direction": "long|short",
  "confidence": 0.0-1.0,
  "reasoning": "máx 2 oraciones",
  "sl_pct": 0.04,
  "tp_pct": 0.08,
  "leverage": 2,
  "margin_pct": 0.02,
  "close_target": "SYMBOL"  // solo si action=CLOSE
}}"""

    response = _call_llm(prompt, system=system, max_tokens=4000)
    if not response:
        log.warning("Kimi no respondió, fallback a HOLD")
        return {"action": "HOLD", "confidence": 0.0, "reasoning": "LLM no disponible"}

    # Parse JSON
    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        result = json.loads(text)

        # Validar campos
        result.setdefault("action", "HOLD")
        result.setdefault("confidence", 0.0)
        result.setdefault("reasoning", "")
        result.setdefault("symbol", "")
        result.setdefault("direction", "long")
        result.setdefault("sl_pct", 0.04)
        result.setdefault("tp_pct", 0.08)
        result.setdefault("leverage", 2)
        result.setdefault("margin_pct", 0.02)

        return result
    except Exception as e:
        log.warning(f"Error parseando respuesta de Kimi: {e}")
        # Intentar extraer acción del texto
        action = "HOLD"
        if "OPEN" in response.upper():
            action = "OPEN"
        elif "CLOSE" in response.upper():
            action = "CLOSE"
        return {"action": action, "confidence": 0.5, "reasoning": response[:200]}


# ─── Portfolio Rebalancing / Analysis ───────────────────────────────────────

def analyze_portfolio_health(portfolio: dict, market: dict, trade_history: List[dict]) -> Dict:
    """
    Pide a Kimi un análisis de salud del portafolio y recomendaciones de rebalanceo.
    Se ejecuta cada 30 minutos.
    """
    portfolio_ctx = build_portfolio_context(portfolio)
    history_ctx = build_trade_history_context(trade_history, max_trades=20)

    # Calcular métricas actuales
    from aaa_shared import calculate_metrics
    metrics = calculate_metrics(trade_history)

    system = "Eres un analista de portafolio experto. Responde SOLO en JSON válido."

    prompt = f"""Analiza la salud del siguiente portafolio de trading:

MÉTRICAS ACTUALES:
- Win Rate: {metrics.get('win_rate', 0)}%
- Profit Factor: {metrics.get('profit_factor', 0)}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0)}
- Max Drawdown: {metrics.get('max_drawdown_pct', 0)}%
- Total PnL: ${metrics.get('total_pnl', 0):+.2f}
- Return: {metrics.get('return_pct', 0):+.2f}%

{portfolio_ctx}

{history_ctx}

Recomienda:
1. ¿Hay alguna posición que deba cerrarse AHORA? (riesgo, tiempo, o deterioro)
2. ¿Hay algún sector/token sobre-exposado?
3. ¿Deberíamos ajustar el tamaño de las posiciones?
4. ¿Alguna lección aprendida del historial reciente?

Responde en JSON:
{{
  "health_score": 0-100,
  "recommendations": ["acción 1", "acción 2"],
  "positions_to_close": ["SYMBOL1", "SYMBOL2"],
  "lessons": ["lección 1", "lección 2"],
  "confidence": 0.0-1.0
}}"""

    response = _call_llm(prompt, system=system, max_tokens=3000)
    if not response:
        return {"health_score": 50, "recommendations": [], "positions_to_close": [], "lessons": [], "confidence": 0.0}

    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        log.warning(f"Error parseando análisis de portafolio: {e}")
        return {"health_score": 50, "recommendations": [], "positions_to_close": [], "lessons": [response[:200]], "confidence": 0.5}


# ─── Self-Analysis for Evolution (Phase 2) ─────────────────────────────────

def analyze_recent_trades_k(trades: List[dict], max_trades: int = 20) -> Dict:
    """
    Pide a Kimi que analice trades recientes y sugiera mejoras.
    Se ejecuta cada 30 ciclos (~1h) para el ciclo de auto-evolucion.
    """
    recent = sorted(trades, key=lambda x: x.get("close_time", ""), reverse=True)[:max_trades]
    if not recent:
        return {"analysis": "Sin trades suficientes", "recommendations": [], "param_changes": [], "confidence": 0.0}

    wins = [t for t in recent if t.get("pnl_usd", 0) > 0]
    losses = [t for t in recent if t.get("pnl_usd", 0) <= 0]

    lines = [f"Ultimos {len(recent)} trades:"]
    for t in recent:
        lines.append(f"  {t['symbol']} {t['direction'].upper()} | {t.get('close_reason', '?')} | PnL: ${t.get('pnl_usd', 0):+.2f} | Hold: {t.get('hours_open', 0):.1f}h")
    lines.append(f"\nResumen: {len(wins)}W / {len(losses)}L")

    system = "Eres un quant developer experto. Analiza trades y sugiere mejoras de codigo. Responde SOLO en JSON."
    prompt = f"""Analiza estos trades recientes y sugiere mejoras para el algoritmo conservador:

{chr(10).join(lines)}

Identifica:
1. Patrones de perdida recurrentes
2. Patrones de ganancia recurrentes
3. Parametros que deberian ajustarse (SL, TP, leverage, hold time)
4. Filtros que deberian agregarse o quitarse

Responde en JSON:
{{
  "analysis": "resumen del problema",
  "recommendations": ["ajuste 1", "ajuste 2"],
  "param_changes": {{"sl_pct": 0.03, "tp_pct": 0.06, "leverage": 3}},
  "confidence": 0.0-1.0
}}"""

    response = _call_llm(prompt, system=system, max_tokens=2000)
    if not response:
        return {"analysis": "LLM no disponible", "recommendations": [], "param_changes": {}, "confidence": 0.0}

    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        log.warning(f"Error parseando analisis K: {e}")
        return {"analysis": response[:300], "recommendations": [], "param_changes": {}, "confidence": 0.5}
