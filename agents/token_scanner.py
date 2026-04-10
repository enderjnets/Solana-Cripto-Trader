#!/usr/bin/env python3
"""
TOKEN SCANNER AGENT — Descubrimiento dinámico de oportunidades

Escanea el mercado de Solana para encontrar tokens con:
- Alto volumen / momentum
- Grandes movimientos de precio (pump o dump)
- Nuevos listings con tracción
- Tokens trending en redes sociales

Actualiza automáticamente la lista de tokens a tradear.
"""

import json
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# ─── Configuración ────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
TRACKED_TOKENS_FILE = DATA_DIR / "tracked_tokens.json"
SCANNER_REPORT_FILE = DATA_DIR / "scanner_report.json"

# APIs gratuitas para descubrimiento
BIRDEYE_TRENDING = "https://public-api.birdeye.so/defi/token_trending"
JUPITER_STRICT_LIST = "https://token.jup.ag/strict"
DEXSCREENER_SOLANA = "https://api.dexscreener.com/latest/dex/search?q=solana"
DEXSCREENER_GAINERS = "https://api.dexscreener.com/token-boosts/top/v1"

# Criterios mínimos para agregar un token
MIN_LIQUIDITY_USD = 500_000      # $500K - safe for trading mínimo de liquidez (bajado para más oportunidades)
MIN_VOLUME_24H = 200_000         # $200K - meaningful volume volumen mínimo
MIN_MARKET_CAP = 2_000_000       # $2M - established tokens only market cap mínimo
MAX_TOKENS_TRACKED = 30          # Máximo de tokens a trackear
MIN_PRICE_CHANGE_ALERT = 8       # % cambio para alertar (pump o dump)

# Tokens base que siempre se mantienen
CORE_TOKENS = ["SOL", "BTC", "ETH", "JUP", "RAY", "BONK", "WIF"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("token_scanner")

# ─── Funciones de API ─────────────────────────────────────────────────────────

def fetch_dexscreener_trending() -> list:
    """Obtiene tokens trending de DexScreener (Solana)."""
    results = []
    
    try:
        # Top volume pairs on Solana
        resp = requests.get(
            "https://api.dexscreener.com/latest/dex/search?q=SOL",
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs", [])
            # Filtrar solo Solana
            solana_pairs = [p for p in pairs if p.get("chainId") == "solana"]
            results.extend(solana_pairs[:30])
    except Exception as e:
        log.warning(f"DexScreener search error: {e}")
    
    try:
        # Top gainers - buscar tokens con movimiento en varias categorías
        queries = ["PUMP", "AI", "MEME", "TRUMP", "PEPE", "CAT", "DOG", "FROG", "COIN"]
        for query in queries:
            try:
                resp = requests.get(
                    f"https://api.dexscreener.com/latest/dex/search?q={query}",
                    timeout=8
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pairs = data.get("pairs", [])
                    solana_pairs = [p for p in pairs if p.get("chainId") == "solana"]
                    results.extend(solana_pairs[:15])
            except:
                pass
    except Exception as e:
        log.warning(f"DexScreener trending error: {e}")
    
    # Deduplicar por baseToken address
    seen = set()
    unique = []
    for p in results:
        addr = p.get("baseToken", {}).get("address", "")
        if addr and addr not in seen:
            seen.add(addr)
            unique.append(p)
    
    return unique


def fetch_jupiter_verified_tokens() -> dict:
    """Obtiene lista de tokens verificados de Jupiter."""
    try:
        resp = requests.get(JUPITER_STRICT_LIST, timeout=15)
        if resp.status_code == 200:
            tokens = resp.json()
            # Convertir a dict por símbolo
            return {t.get("symbol", ""): t for t in tokens if t.get("symbol")}
    except Exception as e:
        log.warning(f"Jupiter token list error: {e}")
    return {}


def fetch_solana_top_movers() -> list:
    """Obtiene los mayores movimientos en Solana via CoinGecko."""
    try:
        # Top gainers Solana ecosystem
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "category": "solana-ecosystem",
                "order": "volume_desc",
                "per_page": 50,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h"
            },
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        log.warning(f"CoinGecko error: {e}")
    return []


def search_birdeye_trending() -> list:
    """Busca tokens trending en Birdeye (requiere API key para producción)."""
    # Birdeye tiene API pública limitada, usamos alternativas
    return []


# ─── Análisis y Scoring ───────────────────────────────────────────────────────

def score_token_opportunity(token_data: dict) -> tuple[float, list]:
    """
    Puntúa un token como oportunidad de trading.
    Retorna (score 0-1, razones)
    """
    score = 0.0
    reasons = []
    
    liquidity = token_data.get("liquidity", 0)
    volume_24h = token_data.get("volume_24h", 0)
    price_change_24h = token_data.get("price_change_24h", 0)
    market_cap = token_data.get("market_cap", 0)
    
    # 1. Liquidez suficiente (0.15)
    if liquidity >= MIN_LIQUIDITY_USD:
        score += 0.15
        reasons.append(f"Liquidez ${liquidity/1e6:.2f}M ✅")
    elif liquidity >= MIN_LIQUIDITY_USD / 2:
        score += 0.08
        reasons.append(f"Liquidez ${liquidity/1e6:.2f}M ⚠️")
    else:
        reasons.append(f"Liquidez baja ${liquidity/1e6:.2f}M ❌")
        return score, reasons  # Descartado
    
    # 2. Volumen alto (0.20)
    if volume_24h >= MIN_VOLUME_24H * 10:
        score += 0.20
        reasons.append(f"Volumen muy alto ${volume_24h/1e6:.2f}M ✅✅")
    elif volume_24h >= MIN_VOLUME_24H:
        score += 0.12
        reasons.append(f"Volumen ${volume_24h/1e6:.2f}M ✅")
    else:
        score += 0.05
        reasons.append(f"Volumen bajo ${volume_24h/1e3:.0f}K ⚠️")
    
    # 3. Movimiento de precio significativo (0.25)
    abs_change = abs(price_change_24h)
    if abs_change >= 20:
        score += 0.25
        direction = "📈 PUMP" if price_change_24h > 0 else "📉 DUMP"
        reasons.append(f"{direction} {price_change_24h:+.1f}% ✅✅")
    elif abs_change >= MIN_PRICE_CHANGE_ALERT:
        score += 0.18
        direction = "subiendo" if price_change_24h > 0 else "bajando"
        reasons.append(f"Movimiento fuerte {price_change_24h:+.1f}% ✅")
    elif abs_change >= 5:
        score += 0.10
        reasons.append(f"Movimiento moderado {price_change_24h:+.1f}%")
    
    # 4. Market cap razonable (0.15)
    if market_cap >= MIN_MARKET_CAP * 10:
        score += 0.15
        reasons.append(f"MC ${market_cap/1e6:.1f}M — establecido ✅")
    elif market_cap >= MIN_MARKET_CAP:
        score += 0.10
        reasons.append(f"MC ${market_cap/1e6:.1f}M ✅")
    elif market_cap > 0:
        score += 0.05
        reasons.append(f"MC bajo ${market_cap/1e3:.0f}K ⚠️")
    
    # 5. Ratio volumen/liquidez (actividad) (0.15)
    if liquidity > 0:
        vol_liq_ratio = volume_24h / liquidity
        if vol_liq_ratio >= 2:
            score += 0.15
            reasons.append(f"Alta actividad (vol/liq={vol_liq_ratio:.1f}x) ✅✅")
        elif vol_liq_ratio >= 0.5:
            score += 0.10
            reasons.append(f"Actividad normal (vol/liq={vol_liq_ratio:.1f}x) ✅")
    
    # 6. Verificado en Jupiter (0.10)
    if token_data.get("verified"):
        score += 0.10
        reasons.append("Verificado en Jupiter ✅")
    
    return min(score, 1.0), reasons


def identify_opportunities(market_data: list, jupiter_verified: dict) -> list:
    """
    Analiza datos del mercado e identifica las mejores oportunidades.
    """
    opportunities = []
    
    for token in market_data:
        # DexScreener format
        base_token = token.get("baseToken", {})
        symbol = base_token.get("symbol", token.get("symbol", ""))
        address = base_token.get("address", token.get("address", ""))
        
        # CoinGecko format fallback
        if not symbol:
            symbol = token.get("symbol", "")
        if not address:
            # CoinGecko no tiene address, buscar en plataformas
            platforms = token.get("platforms", {})
            address = platforms.get("solana", "")
        
        if not symbol:
            continue
            
        # Preparar datos normalizados
        # Manejar tanto DexScreener como CoinGecko
        price_change = token.get("priceChange", {})
        if isinstance(price_change, dict):
            change_24h = float(price_change.get("h24", 0) or 0)
        else:
            change_24h = float(token.get("price_change_percentage_24h", 0) or 0)
        
        volume = token.get("volume", {})
        if isinstance(volume, dict):
            vol_24h = float(volume.get("h24", 0) or 0)
        else:
            vol_24h = float(token.get("total_volume", 0) or 0)
        
        liquidity = token.get("liquidity", {})
        if isinstance(liquidity, dict):
            liq_usd = float(liquidity.get("usd", 0) or 0)
        else:
            liq_usd = float(token.get("market_cap", 0) or 0)
        
        token_info = {
            "symbol": symbol.upper(),
            "name": base_token.get("name", token.get("name", "")),
            "address": address,
            "price": float(token.get("priceUsd", token.get("current_price", 0)) or 0),
            "price_change_24h": change_24h,
            "volume_24h": vol_24h,
            "liquidity": liq_usd,
            "market_cap": float(token.get("fdv", token.get("market_cap", 0)) or 0),
            "verified": symbol.upper() in jupiter_verified,
        }
        
        # Filtrar tokens sin datos suficientes
        if token_info["price"] <= 0:
            continue
        if token_info["liquidity"] < MIN_LIQUIDITY_USD / 2:
            continue
            
        # Puntuar oportunidad
        score, reasons = score_token_opportunity(token_info)
        
        if score >= 0.6:  # Raised from 0.4 - only quality opportunities
            token_info["opportunity_score"] = round(score, 3)
            token_info["reasons"] = reasons
            opportunities.append(token_info)
    
    # Ordenar por score
    opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
    
    return opportunities


# ─── Gestión de Lista de Tokens ───────────────────────────────────────────────

def load_tracked_tokens() -> dict:
    """Carga la lista actual de tokens trackeados."""
    if TRACKED_TOKENS_FILE.exists():
        return json.loads(TRACKED_TOKENS_FILE.read_text())
    return {"tokens": {}, "last_scan": None}


def save_tracked_tokens(data: dict):
    """Guarda la lista de tokens trackeados."""
    DATA_DIR.mkdir(exist_ok=True)
    TRACKED_TOKENS_FILE.write_text(json.dumps(data, indent=2))


def update_token_list(opportunities: list, current_tracked: dict) -> dict:
    """
    Actualiza la lista de tokens basándose en las oportunidades encontradas.
    - Mantiene tokens core siempre
    - Agrega nuevos tokens con alto score
    - Remueve tokens con bajo rendimiento
    """
    tokens = current_tracked.get("tokens", {})
    
    # Asegurar que tokens core estén siempre
    # (sus mints ya están en market_data.py)
    
    # Agregar nuevas oportunidades
    added = []
    for opp in opportunities[:MAX_TOKENS_TRACKED]:
        symbol = opp["symbol"]
        if symbol not in tokens and symbol not in CORE_TOKENS:
            if opp.get("address") and opp["opportunity_score"] >= 0.6:
                tokens[symbol] = {
                    "address": opp["address"],
                    "added_at": datetime.now(timezone.utc).isoformat(),
                    "score": opp["opportunity_score"],
                    "reasons": opp["reasons"][:3],
                    "price_at_add": opp["price"],
                }
                added.append(symbol)
                log.info(f"➕ Agregado {symbol} (score={opp['opportunity_score']:.2f})")
    
    # Remove expired tokens (older than 7 days)
    now = datetime.now(timezone.utc)
    expired = []
    for sym, data in list(tokens.items()):
        if sym in CORE_TOKENS:
            continue
        added_str = data.get("added_at", "")
        if added_str:
            try:
                added = datetime.fromisoformat(added_str)
                if added.tzinfo is None:
                    added = added.replace(tzinfo=timezone.utc)
                if (now - added).days > 7:
                    expired.append(sym)
            except:
                pass
    for sym in expired:
        del tokens[sym]
        log.info(f"⏰ Expired {sym} (older than 7 days)")

    # Limitar a MAX_TOKENS_TRACKED (excluyendo core)
    non_core = {k: v for k, v in tokens.items() if k not in CORE_TOKENS}
    if len(non_core) > MAX_TOKENS_TRACKED - len(CORE_TOKENS):
        # Ordenar por score y mantener los mejores
        sorted_tokens = sorted(non_core.items(), key=lambda x: x[1].get("score", 0), reverse=True)
        to_remove = sorted_tokens[MAX_TOKENS_TRACKED - len(CORE_TOKENS):]
        for symbol, _ in to_remove:
            del tokens[symbol]
            log.info(f"➖ Removido {symbol} (bajo score)")
    
    return {
        "tokens": tokens,
        "last_scan": datetime.now(timezone.utc).isoformat(),
        "added_this_scan": added,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def scan(debug: bool = False) -> dict:
    """Ejecuta un escaneo completo del mercado."""
    log.info("=" * 50)
    log.info("🔍 TOKEN SCANNER — Buscando oportunidades")
    log.info("=" * 50)
    
    # 1. Obtener datos de múltiples fuentes
    log.info("📡 Obteniendo datos de DexScreener...")
    dex_data = fetch_dexscreener_trending()
    log.info(f"   → {len(dex_data)} pares encontrados")
    
    log.info("📡 Obteniendo lista verificada de Jupiter...")
    jupiter_verified = fetch_jupiter_verified_tokens()
    log.info(f"   → {len(jupiter_verified)} tokens verificados")
    
    log.info("📡 Obteniendo top movers de CoinGecko...")
    cg_data = fetch_solana_top_movers()
    log.info(f"   → {len(cg_data)} tokens Solana")
    
    # 2. Combinar y analizar
    all_data = dex_data + cg_data
    log.info(f"🔬 Analizando {len(all_data)} tokens...")
    
    opportunities = identify_opportunities(all_data, jupiter_verified)
    log.info(f"🎯 {len(opportunities)} oportunidades identificadas")
    
    # 3. Mostrar top oportunidades
    if opportunities:
        log.info("\n📊 TOP OPORTUNIDADES:")
        for i, opp in enumerate(opportunities[:10], 1):
            direction = "📈" if opp.get("price_change_24h", 0) > 0 else "📉"
            log.info(f"   {i}. {opp['symbol']:10} | Score: {opp['opportunity_score']:.2f} | "
                    f"{direction} {opp.get('price_change_24h', 0):+.1f}% | "
                    f"Vol: ${opp.get('volume_24h', 0)/1e6:.2f}M")
    
    # 4. Actualizar lista de tokens
    current = load_tracked_tokens()
    updated = update_token_list(opportunities, current)
    save_tracked_tokens(updated)
    
    # 5. Guardar reporte
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "opportunities_found": len(opportunities),
        "top_opportunities": opportunities[:15],
        "tokens_added": updated.get("added_this_scan", []),
        "total_tracked": len(updated.get("tokens", {})) + len(CORE_TOKENS),
    }
    SCANNER_REPORT_FILE.write_text(json.dumps(report, indent=2))
    
    log.info(f"\n✅ Escaneo completado")
    log.info(f"   Oportunidades: {len(opportunities)}")
    log.info(f"   Tokens agregados: {updated.get('added_this_scan', [])}")
    log.info(f"   Total trackeados: {report['total_tracked']}")
    
    return report


def get_dynamic_token_mints() -> dict:
    """
    Retorna un diccionario de tokens para usar en market_data.py
    Combina tokens core + tokens descubiertos dinámicamente.
    """
    from market_data import TOKEN_MINTS as CORE_MINTS
    
    tracked = load_tracked_tokens()
    dynamic_mints = dict(CORE_MINTS)  # Empezar con core
    
    for symbol, data in tracked.get("tokens", {}).items():
        if symbol not in dynamic_mints and data.get("address"):
            dynamic_mints[symbol] = data["address"]
    
    return dynamic_mints


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    scan(debug=args.debug)
