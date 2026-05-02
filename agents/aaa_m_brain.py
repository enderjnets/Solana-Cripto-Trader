#!/usr/bin/env python3
"""
AAA-M Brain — Wrapper MiniMax M2.7
"El Cazador Evolutivo" — Scalping, momentum, breakouts.

Frecuencia: Cada 30 segundos
Input: Snapshot del mercado (top 50 tokens), contexto de portafolio
Output: Decisiones de trading rápidas
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

log = logging.getLogger("aaa_m_brain")

# ─── LLM Config ─────────────────────────────────────────────────────────────

try:
    from llm_config import call_minimax_m2_7
    _HAS_MINIMAX = True
except ImportError:
    _HAS_MINIMAX = False
    log.warning("llm_config.call_minimax_m2_7 no disponible")


def _call_llm(prompt: str, system: str = "", max_tokens: int = 2000) -> Optional[str]:
    """Llama a MiniMax M2.7."""
    if _HAS_MINIMAX:
        try:
            return call_minimax_m2_7(prompt, system=system, max_tokens=max_tokens)
        except Exception as e:
            log.warning(f"MiniMax error: {e}")
    return None


# ─── Prompt Builders ────────────────────────────────────────────────────────

def build_momentum_snapshot(market: dict, top_n: int = 40) -> str:
    """Construye snapshot enfocado en momentum para el prompt."""
    tokens = market.get("tokens", {})
    fg = market.get("fear_greed", {}).get("value", 50)

    token_list = []
    for sym, data in tokens.items():
        if data.get("price", 0) <= 0:
            continue
        # Score de momentum: combina cambio 5min, 24h, y volumen
        mom_5m = abs(data.get("price_5min_change_pct", 0))
        mom_24h = abs(data.get("price_24h_change_pct", 0))
        vol = data.get("volume_24h", 0) / 1e6
        liq = data.get("liquidity", 0) / 1e6
        momentum_score = mom_5m * 2 + mom_24h * 0.5 + min(vol * 0.1, 5) + min(liq * 0.01, 3)
        token_list.append((sym, data, momentum_score))

    token_list.sort(key=lambda x: x[2], reverse=True)
    top_tokens = token_list[:top_n]

    lines = [f"FEAR & GREED: {fg}/100 | Foco: MOMENTUM Y BREAKOUTS"]
    lines.append(f"{'Symbol':<8} {'Price':>12} {'5m%':>8} {'24h%':>8} {'Vol$M':>8} {'Liq$M':>8} {'1hTrend':>8}")
    lines.append("-" * 70)

    for sym, data, _ in top_tokens:
        price = data.get("price", 0)
        chg_5m = data.get("price_5min_change_pct", 0)
        chg_24h = data.get("price_24h_change_pct", 0)
        vol = data.get("volume_24h", 0) / 1e6
        liq = data.get("liquidity", 0) / 1e6
        trend = data.get("price_1h_trend", "?")
        lines.append(f"{sym:<8} ${price:>10.4f} {chg_5m:>+7.2f}% {chg_24h:>+7.2f}% ${vol:>7.2f} ${liq:>7.2f} {trend:>8}")

    return "\n".join(lines)


def build_portfolio_context(portfolio: dict) -> str:
    """Contexto de portafolio para M."""
    capital = portfolio.get("capital_usd", 0)
    initial = portfolio.get("initial_capital", 50000)
    positions = [p for p in portfolio.get("positions", []) if p.get("status") == "open"]
    pnl_total = sum(p.get("pnl_usd", 0) for p in positions)

    lines = [
        f"Capital disponible: ${capital:.2f} (inicial: ${initial:.2f})",
        f"Posiciones abiertas: {len(positions)}",
        f"PnL no realizado: ${pnl_total:+.2f}",
    ]

    if positions:
        lines.append("\nPosiciones activas:")
        for p in positions:
            hours = 0
            try:
                opened = datetime.fromisoformat(p["opened_at"].replace("Z", "+00:00"))
                hours = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
            except:
                pass
            lines.append(f"  {p['symbol']} {p['direction'].upper()} | PnL: ${p.get('pnl_usd', 0):+.2f} | Hold: {hours:.1f}h")

    return "\n".join(lines)


# ─── Main Decision Function ─────────────────────────────────────────────────



def apply_variant_filters(market: dict, filter_rules: dict) -> dict:
    """Apply variant-specific filter rules to market data."""
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
    max_positions: int = 20,
    dynamic_params: Optional[Dict] = None,
    variant: Optional[Dict] = None,
) -> Dict:
    """
    Pide a MiniMax M2.7 una decisión de trading basada en momentum.
    """
    # Apply variant filter rules if present
    if variant and variant.get("filter_rules"):
        market = apply_variant_filters(market, variant["filter_rules"])
    market_snapshot = build_momentum_snapshot(market, top_n=40)
    portfolio_ctx = build_portfolio_context(portfolio)

    open_symbols = {p["symbol"] for p in portfolio.get("positions", []) if p.get("status") == "open"}
    open_count = len(open_symbols)

    # Build variant-aware system prompt
    system_base = "Eres un scalper experto en crypto. Tu estilo es AGRESIVO y RAPIDO. Detectas momentum, breakouts y volumen spikes. Responde SOLO en JSON valido. No uses markdown fences."
    variant_addon = variant.get("system_prompt_addon", "") if variant else ""
    if variant_addon:
        system = system_base + "\n\nVARIANTE ACTIVA: " + variant_addon
    else:
        system = system_base

    prompt = f"""Eres AAA-M ("El Cazador"), un agente de trading agresivo con capital de $50,000.

REGLAS CRITICAS:
1. Maximo {max_positions} posiciones abiertas. Actualmente tienes {open_count} abiertas.
2. NO abras mas de 1 posicion por token.
3. Si una posicion abierta lleva >30 min sin mejorar, cierrala (eficiencia de capital).
4. Si una posicion tiene PnL > +3%, activa trailing stop al 1.5%.
5. Foco en tokens con: volumen alto, momentum 5min > 2%, y liquidez >$500K.
6. SL: 2-3%, TP: 4-8% o trailing, Leverage: 3-5x.
7. Kelly agresivo: 1.0x Kelly. Riesgo maximo por trade: 3% del capital.
8. Solo opera LONG en este mercado (shorts solo si Fear & Greed > 75 y token claramente debil).
9. Si no hay setup claro con momentum, HOLD. No forces trades.

SNAPSHOT DE MERCADO (ordenado por momentum):
{market_snapshot}

CONTEXTO DE PORTAFOLIO:
{portfolio_ctx}

Tokens ya en posiciones abiertas: {list(open_symbols) if open_symbols else 'ninguno'}

DECIDE:
- Si hay breakout/momentum claro: abre UNA posicion (la mejor).
- Si hay posiciones que deben cerrarse por tiempo o perdida: indica cuál.
- Si no hay nada claro: HOLD.

Responde en JSON exacto:
{{
  "action": "OPEN|CLOSE|HOLD",
  "symbol": "TOKEN",
  "direction": "long|short",
  "confidence": 0.0-1.0,
  "reasoning": "max 2 oraciones",
  "sl_pct": 0.025,
  "tp_pct": 0.06,
  "leverage": 3,
  "margin_pct": 0.02,
  "close_target": "SYMBOL"
}}"""

    response = _call_llm(prompt, system=system, max_tokens=2000)
    if not response:
        log.warning("MiniMax no respondio, fallback a HOLD")
        return {"action": "HOLD", "confidence": 0.0, "reasoning": "LLM no disponible"}

    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        result = json.loads(text)

        result.setdefault("action", "HOLD")
        result.setdefault("confidence", 0.0)
        result.setdefault("reasoning", "")
        result.setdefault("symbol", "")
        result.setdefault("direction", "long")
        result.setdefault("sl_pct", 0.025)
        result.setdefault("tp_pct", 0.06)
        result.setdefault("leverage", 3)
        result.setdefault("margin_pct", 0.02)

        return result
    except Exception as e:
        log.warning(f"Error parseando respuesta de MiniMax: {e}")
        action = "HOLD"
        if "OPEN" in response.upper():
            action = "OPEN"
        elif "CLOSE" in response.upper():
            action = "CLOSE"
        return {"action": action, "confidence": 0.5, "reasoning": response[:200]}


# ─── Self-Analysis for Evolution ───────────────────────────────────────────

def analyze_recent_trades(trades: List[dict], max_trades: int = 20) -> Dict:
    """
    Pide a MiniMax que analice sus trades recientes y sugiera mejoras.
    Se ejecuta cada 6h para el ciclo de auto-evolucion.
    """
    recent = sorted(trades, key=lambda x: x.get("close_time", ""), reverse=True)[:max_trades]
    if not recent:
        return {"analysis": "Sin trades suficientes", "recommendations": [], "code_changes": []}

    wins = [t for t in recent if t.get("pnl_usd", 0) > 0]
    losses = [t for t in recent if t.get("pnl_usd", 0) <= 0]

    lines = [f"Ultimos {len(recent)} trades:"]
    for t in recent:
        lines.append(f"  {t['symbol']} {t['direction'].upper()} | {t.get('close_reason', '?')} | PnL: ${t.get('pnl_usd', 0):+.2f} | Hold: {t.get('hours_open', 0):.1f}h | Strategy: {t.get('strategy', '?')}")

    lines.append(f"\nResumen: {len(wins)}W / {len(losses)}L")

    system = "Eres un quant developer experto. Analiza trades y sugiere mejoras de codigo. Responde SOLO en JSON."

    prompt = f"""Analiza estos trades recientes y sugiere mejoras para el algoritmo de trading:

{chr(10).join(lines)}

Identifica:
1. Patrones de perdida recurrentes (que hace que pierdas?)
2. Patrones de ganancia recurrentes (que hace que ganes?)
3. Parametros que deberian ajustarse (SL, TP, leverage, hold time)
4. Filtros que deberian agregarse o quitarse

Responde en JSON:
{{
  "analysis": "resumen del problema",
  "recommendations": ["ajuste 1", "ajuste 2"],
  "param_changes": {{"sl_pct": 0.03, "tp_pct": 0.05, "leverage": 4}},
  "confidence": 0.0-1.0
}}"""

    response = _call_llm(prompt, system=system, max_tokens=2000)
    if not response:
        return {"analysis": "LLM no disponible", "recommendations": [], "code_changes": []}

    try:
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        log.warning(f"Error parseando analisis: {e}")
        return {"analysis": response[:300], "recommendations": [], "code_changes": []}
