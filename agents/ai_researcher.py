#!/usr/bin/env python3
"""
🧠 AI Researcher Agent - Solana Trading Bot
Análisis de mercado usando LLM (Claude Sonnet 4.6 → MiniMax M2.5)

Función:
- Análisis de tendencia (BULLISH/BEARISH/NEUTRAL)
- Niveles de soporte/resistencia
- Análisis de sentimiento (Fear & Greed)
- Eventos importantes que afectan el precio

Input:
- Precios actuales (market_latest.json)
- Noticias de crypto (web search)
- Sentimiento (Fear & Greed)

Output:
- research_latest.json
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuración ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MARKET_FILE = DATA_DIR / "market_latest.json"
RESEARCH_FILE = DATA_DIR / "research_latest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ai_researcher")

# Importar configuración LLM
sys.path.insert(0, str(BASE_DIR))
try:
    from llm_config import call_llm
except ImportError:
    # Fallback si llm_config.py no existe
    def call_llm(prompt, system=""):
        log.warning("⚠️ llm_config.py no encontrado, usando fallback simple")
        return {
            "trend": "NEUTRAL",
            "confidence": 0.5,
            "reasoning": "LLM no disponible"
        }

# ─── Carga / Guardado ─────────────────────────────────────────────────────────

def load_market() -> dict:
    """Carga datos de mercado."""
    if MARKET_FILE.exists():
        with open(MARKET_FILE, 'r') as f:
            return json.load(f)
    return {}

def load_research() -> dict:
    """Carga research anterior."""
    if RESEARCH_FILE.exists():
        with open(RESEARCH_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_research(research: dict):
    """Guarda research."""
    research["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(RESEARCH_FILE, 'w') as f:
        json.dump(research, f, indent=2)

# ─── APIs de Datos ────────────────────────────────────────────────────────────

def get_fear_greed_index() -> dict:
    """Obtiene el Fear & Greed Index de Alternative.me."""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and "data" in data and len(data["data"]) > 0:
            fng = data["data"][0]
            return {
                "value": fng.get("value", 50),
                "classification": fng.get("value_classification", "Neutral"),
                "timestamp": fng.get("timestamp", "")
            }
    except Exception as e:
        log.warning(f"⚠️ Error obteniendo Fear & Greed: {e}")
    
    return {"value": 50, "classification": "Neutral", "timestamp": ""}

# ─── RSS News Feed ───────────────────────────────────────────────────────────

def fetch_crypto_news() -> list:
    """Fetch latest crypto news from RSS feeds (CoinDesk + CoinTelegraph)."""
    import urllib.request
    import xml.etree.ElementTree as ET

    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    headlines = []
    for url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                tree = ET.parse(resp)
                root = tree.getroot()
                items = root.findall(".//item")[:5]
                for item in items:
                    title = item.findtext("title", "").strip()
                    desc = item.findtext("description", "").strip()
                    if title:
                        # Clean HTML from description
                        import re as _re
                        desc_clean = _re.sub(r'<[^>]+>', '', desc)[:200]
                        headlines.append(f"- {title}: {desc_clean}")
        except Exception as e:
            log.warning(f"⚠️ RSS fetch failed for {url}: {e}")
    return headlines[:10]  # Max 10 headlines total

# ─── Análisis con LLM ───────────────────────────────────────────────────────────

def analyze_market_with_llm(market: dict) -> dict:
    """Usa LLM para analizar el mercado."""
    
    # Obtener Fear & Greed
    fng = get_fear_greed_index()
    
    # Obtener tokens principales
    tokens_data = market.get("tokens", {})
    top_tokens = ["SOL", "BTC", "ETH", "JUP"]
    
    # Crear resumen de precios
    price_summary = []
    for token in top_tokens:
        if token in tokens_data:
            t = tokens_data[token]
            price_24h_change = t.get("price_24h_change_pct", 0)
            price_summary.append(f"{token}: {price_24h_change:+.2f}% 24h")
    
    price_text = "\n".join(price_summary)
    
    # Prompt para el LLM
    system_prompt = """Eres un analista de mercado cripto experto. Tu tarea es analizar el mercado de Solana y tokens relacionados.

Responde en formato JSON con esta estructura:
{
  "trend": "BULLISH|BEARISH|NEUTRAL",
  "confidence": 0.0-1.0,
  "key_levels": {
    "SOL": {"support": 0.0, "resistance": 0.0},
    "BTC": {"support": 0.0, "resistance": 0.0},
    "ETH": {"support": 0.0, "resistance": 0.0}
  },
  "factors": [
    "Factor 1 que afecta el precio",
    "Factor 2 que afecta el precio"
  ],
  "recommendation": "CAUTIOUS|AGGRESSIVE|NEUTRAL",
  "reasoning": "Explicación breve del análisis"
}

TIPS:
- Usa los cambios de 24h para determinar la tendencia
- El Fear & Greed Index indica sentimiento del mercado (<50 miedo, >50 codicia)
- Considera múltiples factores: precio, sentimiento, volatilidad"""

    # Fetch crypto news
    news = fetch_crypto_news()
    news_text = "\n".join(news) if news else "No news available"

    user_prompt = f"""Analiza el mercado de Solana basado en estos datos:

CAMBIOS 24H:
{price_text}

FEAR & GREED INDEX:
- Valor: {fng['value']}
- Clasificación: {fng['classification']}

NOTICIAS RECIENTES:
{news_text}

INSTRUCCIONES:
1. Determina la tendencia general (BULLISH/BEARISH/NEUTRAL)
2. Identifica niveles de soporte y resistencia clave
3. Lista factores que están afectando el precio
4. Da una recomendación de CAUTIOUS/AGGRESSIVE/NEUTRAL
5. Proporciona una breve explicación

Responde SOLO en formato JSON válido."""

    try:
        response = call_llm(user_prompt, system_prompt)
        
        # Intentar parsear JSON de la respuesta
        import re
        
        # Buscar bloque JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group()
            try:
                analysis = json.loads(json_str)
                
                # Añadir metadata
                analysis.update({
                    "fng_index": fng,
                    "price_changes": {token: tokens_data[token].get("price_24h_change_pct", 0) 
                                   for token in top_tokens if token in tokens_data},
                    "tokens_analyzed": top_tokens,
                    "llm_used": "claude_sonnet_or_minimax",
                    "generated_at": datetime.now(timezone.utc).isoformat()
                })
                
                return analysis
            except json.JSONDecodeError as e:
                log.warning(f"⚠️ Error parseando JSON: {e}")
        
        # Fallback si no se puede parsear
        return {
            "trend": "NEUTRAL",
            "confidence": 0.5,
            "key_levels": {},
            "factors": ["No se pudo parsear respuesta del LLM"],
            "recommendation": "NEUTRAL",
            "reasoning": response[:200],
            "fng_index": fng,
            "llm_used": "fallback",
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        log.error(f"❌ Error en análisis con LLM: {e}")
        return {
            "trend": "NEUTRAL",
            "confidence": 0.3,
            "error": str(e),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

# ─── Entry Point ───────────────────────────────────────────────────────────────

def run(debug: bool = False):
    log.info("=" * 50)
    log.info("🧠 AI RESEARCHER - Análisis de Mercado")
    log.info("=" * 50)
    
    # Cargar datos
    market = load_market()
    
    if not market:
        log.warning("⚠️ No hay datos de mercado disponibles")
        log.info("Ejecuta primero: python3 market_data.py")
        return None
    
    log.info(f"📊 Tokens disponibles: {len(market.get('tokens', {}))}")
    
    # Análisis con LLM
    log.info("🤖 Analizando mercado con LLM...")
    research = analyze_market_with_llm(market)
    
    # Mostrar resultados
    trend = research.get("trend", "NEUTRAL")
    confidence = research.get("confidence", 0)
    recommendation = research.get("recommendation", "NEUTRAL")
    
    log.info(f"\n📈 Tendencia: {trend}")
    log.info(f"🎯 Confianza: {confidence:.2f}")
    log.info(f"💡 Recomendación: {recommendation}")
    
    if debug:
        fng = research.get("fng_index", {})
        log.info(f"\n📊 Fear & Greed: {fng.get('value', 50)} ({fng.get('classification', 'N/A')})")
        
        factors = research.get("factors", [])
        if factors:
            log.info(f"\n🔍 Factores identificados:")
            for factor in factors[:5]:  # Máximo 5
                log.info(f"   • {factor}")
    
    # Guardar research
    save_research(research)
    log.info(f"\n💾 Research guardado en: {RESEARCH_FILE}")
    
    log.info("=" * 50)
    
    return research

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Output detallado")
    args = parser.parse_args()
    
    run(debug=args.debug)
