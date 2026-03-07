#!/usr/bin/env python3
"""
BitTrader Content Strategist
Genera guiones optimizados basados en insights + trending crypto
"""
import json, random
from datetime import datetime, timezone
from pathlib import Path

INSIGHTS_FILE = Path("/home/enderj/.openclaw/workspace/bittrader/analytics/insights_latest.json")

# Plantillas de Shorts virales basadas en patrones probados
SHORT_TEMPLATES = {
    "pregunta_educativa": {
        "format": "¿{pregunta}? {explicacion}. {cta}",
        "hook_type": "pregunta",
        "style": "educativo",
        "target_duration": "25-35s"
    },
    "error_comun": {
        "format": "El error que {grupo} cometen: {error}. La solución: {solucion}. {cta}",
        "hook_type": "negativo",
        "style": "controversial",
        "target_duration": "30-40s"
    },
    "lista_rapida": {
        "format": "Las {num} {cosas} que {beneficio}: {lista}. {cta}",
        "hook_type": "numero",
        "style": "educativo",
        "target_duration": "35-50s"
    },
    "secreto_revelado": {
        "format": "Lo que nadie te dice sobre {tema}: {revelacion}. {cta}",
        "hook_type": "secreto",
        "style": "controversial",
        "target_duration": "25-35s"
    },
    "noticia_caliente": {
        "format": "{noticia}. ¿Qué significa para ti? {analisis}. {cta}",
        "hook_type": "urgencia",
        "style": "noticias",
        "target_duration": "30-45s"
    },
}

# Plantillas de Videos Largos
LONG_TEMPLATES = {
    "tutorial": {
        "format": "Paso a paso: {titulo}",
        "sections": ["Intro/Hook", "Problema", "Herramientas", "Paso 1-5", "Resultado", "CTA"],
        "target_duration": "3-5min"
    },
    "analisis_noticia": {
        "format": "{noticia}: Análisis completo",
        "sections": ["Hook/Noticia", "Contexto", "Impacto", "Qué hacer", "Predicción", "CTA"],
        "target_duration": "5-8min"
    },
    "comparativa": {
        "format": "{A} vs {B}: ¿Cuál es mejor?",
        "sections": ["Hook", "Qué es A", "Qué es B", "Comparación", "Veredicto", "CTA"],
        "target_duration": "4-6min"
    },
}

# Temas educativos (evergreen)
EVERGREEN_TOPICS = [
    "Qué es la capitalización de mercado y por qué importa",
    "Cómo leer un gráfico de velas japonesas en 60 segundos",
    "La diferencia entre mercado spot y futuros",
    "Qué es una wallet y cuál necesitas",
    "Por qué la diversificación te protege",
    "Qué son las stablecoins y para qué sirven",
    "Cómo calcular tu riesgo por operación",
    "Qué es el apalancamiento y por qué es peligroso",
    "RSI explicado: el indicador que todos usan",
    "Medias móviles: la herramienta más simple del trading",
    "Qué es el FOMO y cómo evitarlo",
    "Soporte y resistencia: los niveles que mueven el mercado",
    "Qué es un exchange descentralizado",
    "Cómo funciona la blockchain en 60 segundos",
    "La regla del 2% que usan los hedge funds",
    "Qué es el análisis fundamental en crypto",
    "Fibonacci en trading: ¿funciona o es magia?",
    "Cómo sobrevivir a un bear market",
    "Qué es yield farming y staking",
    "Los 5 errores fatales del trader principiante",
]

def generate_news_shorts(trending, market):
    """Genera guiones basados en noticias actuales"""
    guiones = []
    
    # Market movement
    if market:
        change = market.get("market_cap_change_24h", 0)
        if abs(change) > 3:
            direction = "SUBE" if change > 0 else "CAE"
            guiones.append({
                "tipo": "short",
                "template": "noticia_caliente",
                "titulo": f"El mercado crypto {direction} {abs(change):.1f}% hoy 📊 #crypto #shorts",
                "guion": f"El mercado crypto {'subió' if change > 0 else 'cayó'} {abs(change):.1f} porciento en las últimas 24 horas. "
                        f"La capitalización total {'supera' if change > 0 else 'se acerca a'} los {market.get('total_market_cap_usd', 0)/1e12:.1f} trillones de dólares. "
                        f"Bitcoin domina con el {market.get('btc_dominance', 0):.1f} porciento del mercado. "
                        f"{'Es momento de tomar ganancias con cuidado.' if change > 5 else 'No entres en pánico. Revisa tu plan.' if change < -5 else 'Mantente atento a los niveles clave.'} "
                        f"¿Qué opinas? Comenta abajo.",
                "clip_prompt": f"Cryptocurrency market {'going up green candles' if change > 0 else 'crashing red candles'}, dramatic financial charts, cinematic lighting",
                "priority": "high"
            })
    
    # Trending coins
    if trending:
        top3 = trending[:3]
        names = ", ".join(c["name"] for c in top3)
        guiones.append({
            "tipo": "short",
            "template": "lista_rapida",
            "titulo": f"Las 3 cryptos TRENDING hoy: {', '.join(c['symbol'] for c in top3)} 🔥 #crypto #shorts",
            "guion": f"Estas son las 3 criptomonedas que más están buscando hoy. "
                    f"Número 1: {top3[0]['name']}. {'Rank ' + str(top3[0].get('rank', '?')) + ' por capitalización. ' if top3[0].get('rank') else ''}"
                    f"Número 2: {top3[1]['name']}. "
                    f"Número 3: {top3[2]['name']}. "
                    f"Trending no significa que debes comprar. Significa que hay interés. Investiga antes de invertir. "
                    f"¿Cuál tienes tú? Comenta.",
            "clip_prompt": "Cryptocurrency coins floating in digital space, trending chart, golden glow, futuristic aesthetic",
            "priority": "high"
        })
    
    return guiones

def generate_evergreen_shorts(insights, count=5):
    """Genera shorts educativos basados en insights"""
    guiones = []
    
    # Evitar temas que ya fallaron
    avoid = insights.get("recommendations", {}).get("avoid_topics", [])
    best_pattern = insights.get("recommendations", {}).get("best_title_pattern", "pregunta")
    
    # Seleccionar temas aleatorios que no hemos cubierto
    available = [t for t in EVERGREEN_TOPICS]
    random.shuffle(available)
    
    for topic in available[:count]:
        # Elegir template basado en mejor patrón
        if "pregunta" in best_pattern:
            template = "pregunta_educativa"
        elif "negativo" in best_pattern or "error" in best_pattern:
            template = "error_comun"
        else:
            template = random.choice(list(SHORT_TEMPLATES.keys()))
        
        guiones.append({
            "tipo": "short",
            "template": template,
            "titulo": f"{topic} #crypto #trading #shorts",
            "guion": f"{topic}. Explicación breve y concisa para principiantes. Terminar con pregunta para engagement.",
            "clip_prompt": f"Financial education concept, {topic[:30]}, modern digital aesthetic",
            "priority": "medium",
            "needs_expansion": True  # Marcar para que el LLM expanda el guión
        })
    
    return guiones

def generate_long_video(insights, trending, market):
    """Genera 1-2 ideas de videos largos"""
    guiones = []
    
    # Video largo basado en noticia
    if market and abs(market.get("market_cap_change_24h", 0)) > 2:
        guiones.append({
            "tipo": "long",
            "template": "analisis_noticia",
            "titulo": f"Análisis del Mercado Crypto HOY — ¿Qué está pasando?",
            "sections": [
                "Hook: El mercado se mueve fuerte hoy",
                "Datos: Capitalización, dominancia BTC, trending coins",
                "Análisis técnico: Niveles de soporte y resistencia",
                "Qué hacer: Estrategias para este escenario",
                "Predicción: Hacia dónde puede ir",
                "CTA: Suscríbete para análisis diarios"
            ],
            "clip_prompts": [
                "Financial analyst looking at multiple monitors with crypto charts",
                "Bitcoin price chart with technical analysis, support resistance lines",
                "Trader making strategic decisions, calm professional environment"
            ],
            "priority": "high"
        })
    
    # Tutorial evergreen
    best_topic = insights.get("recommendations", {}).get("best_topic", "trading")
    guiones.append({
        "tipo": "long",
        "template": "tutorial",
        "titulo": f"Guía Completa de {best_topic.title()} para Principiantes 📚",
        "sections": [
            f"Hook: Por qué necesitas aprender {best_topic}",
            "Conceptos básicos explicados",
            "Herramientas que necesitas",
            "Tutorial paso a paso",
            "Errores comunes a evitar",
            "CTA: Comenta tus dudas"
        ],
        "clip_prompts": [
            "Person learning about cryptocurrency on laptop, modern desk setup",
            "Step by step tutorial interface, clean modern design"
        ],
        "priority": "medium"
    })
    
    return guiones

def main():
    print("🎯 BitTrader Strategist\n")
    
    # Load insights
    if INSIGHTS_FILE.exists():
        insights = json.loads(INSIGHTS_FILE.read_text())
        print("✅ Insights cargados")
    else:
        insights = {"recommendations": {}}
        print("⚠️ Sin insights previos — usando defaults")
    
    trending = insights.get("crypto_trending", [])
    market = insights.get("market", {})
    
    all_guiones = []
    
    # 1. Noticias crypto → 2-3 shorts
    news_shorts = generate_news_shorts(trending, market)
    all_guiones.extend(news_shorts)
    print(f"📰 {len(news_shorts)} shorts de noticias")
    
    # 2. Evergreen educativos → 5-6 shorts
    edu_shorts = generate_evergreen_shorts(insights, count=6)
    all_guiones.extend(edu_shorts)
    print(f"📚 {len(edu_shorts)} shorts educativos")
    
    # 3. Wildcards experimentales → 2 shorts
    wildcards = generate_evergreen_shorts({"recommendations": {}}, count=2)
    for w in wildcards:
        w["priority"] = "low"
        w["experimental"] = True
    all_guiones.extend(wildcards)
    print(f"🎲 {len(wildcards)} shorts experimentales")
    
    # 4. Videos largos → 1-2
    long_vids = generate_long_video(insights, trending, market)
    all_guiones.extend(long_vids)
    print(f"🎬 {len(long_vids)} videos largos")
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_guiones.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))
    
    print(f"\n📋 Total: {len(all_guiones)} contenidos generados")
    print(f"   Shorts: {len([g for g in all_guiones if g['tipo'] == 'short'])}")
    print(f"   Largos: {len([g for g in all_guiones if g['tipo'] == 'long'])}")
    
    # Save
    now = datetime.now(timezone.utc)
    week = now.strftime("semana_%Y-%m-%d")
    output_dir = Path(f"/home/enderj/.openclaw/workspace/bittrader/{week}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = output_dir / "guiones.json"
    output.write_text(json.dumps(all_guiones, indent=2, ensure_ascii=False))
    print(f"\n✅ Guardado: {output}")
    
    return all_guiones

if __name__ == "__main__":
    main()
