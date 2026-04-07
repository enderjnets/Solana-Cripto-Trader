#!/usr/bin/env python3
"""
🎯 AI Strategy Agent - Solana Trading Bot (Enhanced con LLM)
Genera señales de trading combinando análisis del Researcher + indicadores técnicos

Función:
- Generar señales LONG/SHORT
- Calcular SL/TP óptimos
- Determinar tamaño de posición (Kelly Criterion)
- Justificación de cada señal con LLM

Input:
- Research del AI Researcher (research_latest.json)
- Indicadores técnicos (del market_data.py)
- Estado del portafolio (portfolio.json)

Output:
- strategy_llm.json (señales con justificación)
"""

import os
import sys
import json
import logging

# Gemma 4 local second opinion (non-blocking, optional)
try:
    from ollama_client import validate_trading_signal as _gemma4_validate
    _HAS_GEMMA4 = True
except ImportError:
    _HAS_GEMMA4 = False
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import math

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MARKET_FILE = DATA_DIR / "market_latest.json"
RESEARCH_FILE = DATA_DIR / "research_latest.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
STRATEGY_FILE = DATA_DIR / "strategy_llm.json"
SIGNALS_FILE = DATA_DIR / "signals_latest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ai_strategy")

# Importar configuración LLM
try:
    from llm_config import call_llm
except ImportError:
    def call_llm(prompt, system=""):
        return {"error": "llm_config.py no disponible"}

# ─── Parámetros de Trading ────────────────────────────────────────────────

INITIAL_CAPITAL = 1000.0
MAX_POSITIONS = 5
RISK_PER_TRADE = 0.02  # 2% del capital
MIN_VOLATILITY = 0.005  # 0.5% volatilidad mínima

# SL/TP Configuración
BASE_SL_PCT = 0.025  # 2.5%
BASE_TP_PCT = 0.05   # 5.0% (2x SL)

# ─── Carga de Datos ───────────────────────────────────────────────────────────

def load_market() -> dict:
    if MARKET_FILE.exists():
        with open(MARKET_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_research() -> dict:
    if RESEARCH_FILE.exists():
        with open(RESEARCH_FILE, 'r') as f:
            return json.load(f)
    return {"trend": "NEUTRAL", "confidence": 0.5}

def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {
        "capital_usd": INITIAL_CAPITAL,
        "initial_capital": INITIAL_CAPITAL,
        "positions": []
    }

def load_rsi_map() -> dict:
    """Carga mapa symbol→rsi desde signals_latest.json si existe."""
    rsi_map = {}
    try:
        if SIGNALS_FILE.exists():
            data = json.loads(SIGNALS_FILE.read_text())
            for sig in data.get("signals", []):
                sym = sig.get("symbol")
                rsi_val = sig.get("rsi")
                if sym and rsi_val is not None:
                    rsi_map[sym] = rsi_val
            # También intentar desde indicator_summary
            for sym, ind in data.get("indicator_summary", {}).items():
                if sym not in rsi_map and ind.get("rsi") is not None:
                    rsi_map[sym] = ind["rsi"]
    except Exception:
        pass
    return rsi_map

def save_strategy(strategy: dict):
    strategy["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(STRATEGY_FILE, 'w') as f:
        json.dump(strategy, f, indent=2)

# ─── Cálculo de Indicadores Técnicos ──────────────────────────────────────

def calculate_rsi(prices: list, period: int = 14) -> Optional[float]:
    """Calcula RSI usando el método Wilder's Smoothing."""
    if len(prices) < period + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(change if change > 0 else 0)
        losses.append(abs(change) if change < 0 else 0)
    
    # Promedio inicial
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Wilder's smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)

def calculate_volatility(prices: list) -> Optional[float]:
    """Calcula volatilidad como desviación estándar de cambios."""
    if len(prices) < 2:
        return None
    
    changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    mean = sum(changes) / len(changes)
    variance = sum((c - mean) ** 2 for c in changes) / len(changes)
    
    return math.sqrt(variance)

# ─── Generación de Señales con LLM ────────────────────────────────────────

def generate_signals_with_llm(market: dict, research: dict, portfolio: dict) -> list:
    """Usa LLM para generar señales de trading."""
    
    # Obtener tokens disponibles
    tokens_data = market.get("tokens", {})
    
    # Filtrar tokens con datos suficientes
    valid_tokens = []
    for symbol, data in tokens_data.items():
        if data.get("price", 0) > 0:
            valid_tokens.append((symbol, data))
    
    # Ordenar por market cap (mayor primero)
    valid_tokens.sort(key=lambda x: x[1].get("market_cap", 0), reverse=True)
    
    # Tomar los top 10
    top_tokens = valid_tokens[:10]
    
    # Obtener portafolio actual
    positions = portfolio.get("positions", [])
    open_symbols = [p["symbol"] for p in positions if p.get("status") == "open"]
    
    # Cargar RSI desde signals_latest.json (más preciso que el hardcoded 50)
    rsi_map = load_rsi_map()

    # Preparar datos para el LLM
    tokens_summary = []
    for symbol, data in top_tokens:
        price = data.get("price", 0)
        change_24h = data.get("price_24h_change_pct", 0)
        # Preferir RSI calculado por strategy.py; fallback al del market o 50
        rsi = rsi_map.get(symbol, data.get("rsi", 50))
        mc = data.get("market_cap", 0)
        
        # Calcular volatilidad aproximada
        volatility = abs(change_24h) / 100 if change_24h else 0.001
        
        # Determinar estado del token en portafolio
        position_status = "NONE"
        if symbol in open_symbols:
            for pos in positions:
                if pos["symbol"] == symbol:
                    position_status = f"{pos['direction'].upper()}"
        
        tokens_summary.append({
            "symbol": symbol,
            "price": price,
            "change_24h": change_24h,
            "rsi": rsi,
            "volatility": volatility,
            "market_cap": mc,
            "position": position_status
        })
    
    # Prompt del sistema
    system_prompt = """Eres un trader experto de criptomonedas especializado en Solana y tokens relacionados. Tu tarea es generar señales de trading basadas en análisis técnico y fundamental.

IMPORTANTE - FORMATO DE RESPUESTA:
Responde ÚNICAMENTE en formato JSON válido con esta estructura:
{
  "signals": [
    {
      "symbol": "TOKEN",
      "direction": "long|short|none",
      "entry_price": 0.0,
      "sl_price": 0.0,
      "tp_price": 0.0,
      "exit_mode": "fixed|trailing",
      "trailing_pct": 0.0,
      "size_usd": 0.0,
      "confidence": 0.0,
      "strategy": "momentum|breakout|oversold",
      "reasoning": "Breve explicación",
      "factors": ["factor1", "factor2"]
    }
  ],
  "summary": {
    "total_signals": 0,
    "long_signals": 0,
    "short_signals": 0,
    "avg_confidence": 0.0
  }
}

REGLAS DE TRADING:
0. MODO SPOT/PAPER: Solo operamos LONGs. NO generar SHORTs — es imposible shortear en Jupiter/Raydium spot. Ignorar cualquier señal short.
1. RIESGO POR TRADE: 2% del capital ($10 USD)
2. STOP LOSS: 2.5-3% del entry
3. TAKE PROFIT: 5-6% del entry (2x SL) — o trailing stop si las condiciones lo ameritan
4. MÁXIMO 5 señales por ciclo — APROVECHA condiciones extremas del mercado
5. NO generar señal si el token YA tiene posición abierta

REGLA DE SENTIMIENTO EXTREMO (OBLIGATORIA):
- Fear & Greed ≤ 20 (Extreme Fear): mercado en PÁNICO → SOLO genera LONGs en tokens con RSI < 30 (oversold bounce). NO generes SHORTs (operamos en spot, no se puede shortear). Busca oportunidades de rebote.
- Fear & Greed ≤ 35 (Fear): mercado con miedo → genera LONGs selectivos en tokens sobrevendidos (RSI < 35). NO generes SHORTs (operamos en spot). Sé conservador con el sizing.
- Fear & Greed ≥ 75 (Greed): mercado eufórico → genera la MAYOR cantidad de LONGs posibles (3-5). La euforia tiende a continuar. SHORTs solo en tokens con RSI > 80.
- Fear & Greed ≥ 85 (Extreme Greed): → genera 4-5 LONGs agresivos. La tendencia es tu amiga.
- Fear & Greed 35-75 (Neutral): genera señales normales basadas en técnico (2-3).

OBJETIVO: Aprovechar al MÁXIMO las condiciones del mercado abriendo tantas posiciones como la situación lo permita, siempre respetando el riesgo.

NOTA SOBRE CONFIDENCE: En condiciones extremas (F&G ≤ 20 o ≥ 80), usa confidence de 0.80-0.95 para shorts/longs respectivamente. La probabilidad de éxito es alta cuando el sentimiento es extremo. No pongas confidence < 0.75 en señales que generes durante condiciones extremas.

EXIT MODE — Elige el modo de salida más apropiado para CADA señal:
- "fixed": TP fijo. Ideal para mercados laterales, baja volatilidad, o reversión a la media (oversold_bounce).
- "trailing": Trailing stop. Ideal para mercados con momentum fuerte, breakouts confirmados, o tendencias claras.
  - Si eliges "trailing", establece "trailing_pct" (porcentaje de retroceso desde el máximo/mínimo).
  - Ejemplo: trailing_pct=0.02 = cierra si el precio retrocede 2% desde su máximo alcanzado.
  - El tp_price sigue siendo el TP máximo de seguridad (el trade cierra ahí si no hay trailing mejor).
  - Rango recomendado trailing_pct: 0.015 (tight, scalping) a 0.04 (wide, swing).

CRITERIOS para elegir trailing vs fixed:
- Volatilidad > 3% y RSI 50-65 con tendencia alcista → TRAILING (puede capturar movimiento extendido)
- Breakout con volumen alto → TRAILING (momentum puede continuar)
- Oversold bounce (RSI < 30) → FIXED (el rebote suele ser limitado)
- Mercado lateral sin tendencia clara → FIXED

CRITERIOS DE ENTRADA:
- LONG: RSI < 30 (sobreventa) + cambio 24h > -3% + volatilidad > 0.5%
- SHORT: RSI > 70 (sobrecompra) + cambio 24h > 3% + volatilidad > 0.5%
- MOMENTUM: RSI 50-70 + cambio 24h > 2% para LONG, < -2% para SHORT

EVITAR SEÑALES EN:
- Tokens con positions distintas a "NONE"
- Tokens con volatilidad < 0.5%
- Tokens con market cap < $50M"""

    # Prompt del usuario
    # Get Fear & Greed for prompt
    fg_data = market.get("fear_greed", {})
    fg_value = fg_data.get("value", 50) if isinstance(fg_data, dict) else 50
    fg_label = fg_data.get("label", "Neutral") if isinstance(fg_data, dict) else "Neutral"
    
    # Determine how many signals to request based on sentiment
    if fg_value <= 20:
        target_signals = "2-3 LONGs selectivos (Extreme Fear — busca rebotes oversold)"
    elif fg_value <= 35:
        target_signals = "2-3 LONGs conservadores (Fear — tokens sobrevendidos)"
    elif fg_value >= 85:
        target_signals = "4-5 LONGs (Extreme Greed — APROVECHA la euforia)"
    elif fg_value >= 75:
        target_signals = "3-4 LONGs (Greed — mercado alcista)"
    else:
        target_signals = "2-3 señales mixtas (mercado neutral)"

    user_prompt = f"""Genera señales de trading basado en estos datos:

⚠️ SENTIMIENTO DEL MERCADO:
- Fear & Greed Index: {fg_value}/100 ({fg_label})
- OBJETIVO DE SEÑALES: {target_signals}

RESEARCH DEL MERCADO:
- Tendencia: {research.get('trend', 'NEUTRAL')}
- Confianza del análisis: {research.get('confidence', 0)}
- Recomendación general: {research.get('recommendation', 'NEUTRAL')}

TOKENS DISPONIBLES (Top 10):
"""

    for token in tokens_summary:
        status_emoji = "🟢" if token['position'] == "NONE" else "🔴"
        user_prompt += f"""
{status_emoji} {token['symbol']}:
- Precio: ${token['price']:.6f}
- Cambio 24h: {token['change_24h']:+.2f}%
- RSI: {token['rsi']:.1f}
- Volatilidad: {token['volatility']*100:.2f}%
- Market Cap: ${token['market_cap']:,.0f}
- Posición actual: {token['position']}"""

    user_prompt += """

INSTRUCCIONES:
1. Analiza cada token según los criterios de entrada
2. Genera MÁXIMO 3 señales de alta confianza
3. NO generar señales para tokens con posiciones abiertas
4. Calcula SL/TP según las reglas
5. Size USD = 2% del capital ($10 USD)
6. Proporciona justificación breve

Responde SOLO en JSON válido."""

    try:
        response = call_llm(user_prompt, system_prompt)
        
        # Intentar parsear JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            json_str = json_match.group()
            try:
                signals_data = json.loads(json_str)
                
                # Validar estructura
                if "signals" in signals_data:
                    return signals_data
                else:
                    log.warning("⚠️ LLM response no contiene 'signals'")
                    return {"signals": [], "error": "Invalid response structure"}
                    
            except json.JSONDecodeError as e:
                log.warning(f"⚠️ Error parseando JSON: {e}")
                return {"signals": [], "error": str(e)}
        else:
            log.warning("⚠️ No se encontró JSON en respuesta")
            return {"signals": [], "error": "No JSON found"}
            
    except Exception as e:
        log.error(f"❌ Error generando señales con LLM: {e}")
        return {"signals": [], "error": str(e)}

# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(debug: bool = False) -> dict:
    log.info("=" * 50)
    log.info("🎯 AI STRATEGY - Generación de Señales (Enhanced)")
    log.info("=" * 50)
    
    # Cargar datos
    market = load_market()
    research = load_research()
    portfolio = load_portfolio()
    
    # Validar datos
    if not market.get("tokens"):
        log.warning("⚠️ No hay datos de mercado disponibles")
        log.info("Ejecuta primero: python3 market_data.py")
        return {"signals": [], "error": "No market data"}
    
    capital = portfolio.get("capital_usd", INITIAL_CAPITAL)
    log.info(f"💰 Capital disponible: ${capital:.2f}")
    log.info(f"🧠 Research trend: {research.get('trend', 'N/A')}")
    
    # Generar señales con LLM
    log.info("🤖 Generando señales con LLM...")
    signals_data = generate_signals_with_llm(market, research, portfolio)
    
    signals = signals_data.get("signals", [])
    # SPOT MODE: Filter out any short signals the LLM might still generate
    signals = [s for s in signals if s.get("direction", "").lower() != "short"]

    summary = signals_data.get("summary", {})
    
    # Mostrar señales
    log.info(f"\n📊 Señales generadas: {len(signals)}")
    
    for i, signal in enumerate(signals, 1):
        symbol = signal.get("symbol", "N/A")
        direction = signal.get("direction", "none").upper()
        entry = signal.get("entry_price", 0)
        sl = signal.get("sl_price", 0)
        tp = signal.get("tp_price", 0)
        size = signal.get("size_usd", 0)
        confidence = signal.get("confidence", 0)
        strategy = signal.get("strategy", "N/A")
        reasoning = signal.get("reasoning", "")
        
        log.info(f"\n{i}. {symbol} ({direction}) - {strategy}")
        log.info(f"   Entry: ${entry:.6f} | SL: ${sl:.6f} | TP: ${tp:.6f}")
        log.info(f"   Size: ${size:.2f} | Confianza: {confidence:.2f}")
        
        if debug:
            log.info(f"   📝 {reasoning}")
    
    # Calcular summary
    long_signals = sum(1 for s in signals if s.get("direction") == "long")
    short_signals = sum(1 for s in signals if s.get("direction") == "short")
    avg_conf = sum(s.get("confidence", 0) for s in signals) / len(signals) if signals else 0
    
    summary["total_signals"] = len(signals)
    summary["long_signals"] = long_signals
    summary["short_signals"] = short_signals
    summary["avg_confidence"] = round(avg_conf, 3)
    
    signals_data["summary"] = summary
    
    # Guardar
    save_strategy(signals_data)
    log.info(f"\n💾 Señales guardadas en: {STRATEGY_FILE}")
    
    log.info("=" * 50)
    
    # Gemma 4 second opinion: adjust confidence
    if _HAS_GEMMA4 and signals_data.get("signals"):
        try:
            fg = market.get("fear_greed", {})
            fg_val = fg.get("value", 50) if isinstance(fg, dict) else int(fg or 50)
            for sig in signals_data["signals"]:
                v = _gemma4_validate(sig, f"FG:{fg_val}")
                if v and isinstance(v, dict):
                    adj = v.get("confidence_adj", 0)
                    old_c = sig.get("confidence", 0.5)
                    sig["confidence"] = round(max(0.1, min(0.99, old_c + adj)), 2)
                    sig["gemma4_agree"] = v.get("agree", True)
        except Exception:
            pass

    return signals_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Output detallado")
    args = parser.parse_args()
    
    run(debug=args.debug)
