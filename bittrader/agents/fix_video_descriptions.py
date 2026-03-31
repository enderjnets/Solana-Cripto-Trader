#!/usr/bin/env python3
"""
Fix: Actualizar descripciones, tags y metadata de videos ya subidos a YouTube.
Genera descripciones ricas para videos que fueron subidos sin ellas.
"""
import json
import sys
import time
import glob
from pathlib import Path

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
DATA_DIR = BITTRADER / "agents/data"

sys.path.insert(0, str(WORKSPACE / "youtube_env/lib/python3.13/site-packages"))
sys.path.insert(0, str(BITTRADER / "agents"))

from publisher import get_youtube_client, build_description

# ── Cargar guiones históricos ────────────────────────────────────────────
def load_all_scripts() -> dict:
    """Indexa todos los guiones por título."""
    all_scripts = {}
    for f in sorted(glob.glob(str(DATA_DIR / "guiones*.json"))):
        try:
            data = json.load(open(f))
            scripts = data.get("scripts", data) if isinstance(data, dict) else data
            for s in scripts:
                title = s.get("title", "")
                if title and s.get("description"):
                    all_scripts[title] = {
                        "description": s.get("description", ""),
                        "tags": s.get("tags", []),
                        "hook": s.get("hook", ""),
                        "cta": s.get("cta", ""),
                        "type": s.get("type", "short"),
                    }
        except Exception:
            pass
    return all_scripts


# ── Generar descripción a partir del título (para los que no tienen guión) ──
def generate_description_from_title(title: str, video_type: str) -> dict:
    """Genera una descripción profesional basada en el título del video.
    
    Cada descripción debe:
    - Tener un párrafo introductorio que enganche (no repetir el título)
    - Explicar qué va a aprender/ver el espectador
    - Incluir un CTA conversacional al final
    - Sonar como escrita por un humano, no genérica
    """
    title_lower = title.lower()
    clean_title = title.rstrip("💰🚀💸😱📈📉 ")
    
    tags = []
    
    # Detectar tokens cripto mencionados
    tokens_map = {
        "bitcoin": "Bitcoin", "btc": "Bitcoin", "sol": "Solana", "solana": "Solana",
        "ethereum": "Ethereum", "eth": "Ethereum", "pi network": "PI Network",
        "pi ": "PI Network", "core": "CORE", "siren": "SIREN", "tao": "TAO",
        "bittensor": "Bittensor", "circle": "Circle", "blackrock": "BlackRock",
        "rain": "RAIN", "popcat": "POPCAT", "trump": "Trump Coin",
        "neiro": "NEIRO", "sui": "SUI", "dot": "DOT",
    }
    detected_tokens = []
    for key, name in tokens_map.items():
        if key in title_lower:
            detected_tokens.append(name)
            tags.append(name)
    
    # Detectar categoría para generar descripción contextual
    is_trading = any(w in title_lower for w in ["trading", "trader", "trade", "opera", "cuenta", "rentable"])
    is_crypto = any(w in title_lower for w in ["crypto", "cripto", "criptomoneda", "bitcoin", "btc", "sol", "eth"])
    is_ai = any(w in title_lower for w in ["bot", "ia", "inteligencia artificial", "claude", "gpt", "agente", "automatiz"])
    is_news = any(w in title_lower for w in ["circle", "blackrock", "cathie", "ark", "billones", "regulación"])
    is_pump = any(w in title_lower for w in ["explota", "pump", "sube", "rompe", "gainer", "+"])
    is_education = any(w in title_lower for w in ["qué es", "cómo", "secreto", "verdad", "error", "pierde", "aprende"])
    is_ftmo = any(w in title_lower for w in ["ftmo", "prop firm", "funded", "fondeo"])
    
    # Tags por categoría
    if is_trading: tags.extend(["trading", "day trading", "gestión de riesgo", "estrategia trading"])
    if is_crypto: tags.extend(["criptomonedas", "crypto", "inversión"])
    if is_ai: tags.extend(["inteligencia artificial", "bot trading", "automatización", "IA trading"])
    if is_news: tags.extend(["noticias financieras", "institucionales", "mercados"])
    if is_pump: tags.extend(["análisis técnico", "señales cripto", "altcoins"])
    if is_education: tags.extend(["educación financiera", "aprender trading"])
    if is_ftmo: tags.extend(["FTMO", "prop firm", "funded trader"])
    tags.extend(["BitTrader", "finanzas"])
    
    # Deduplicar tags
    seen = set()
    unique_tags = []
    for t in tags:
        if t.lower() not in seen:
            unique_tags.append(t)
            seen.add(t.lower())
    
    # ── Generar descripción contextual ────────────────────────────────────
    
    if detected_tokens and is_pump:
        token_list = ", ".join(detected_tokens[:3])
        description = (
            f"{token_list} está dando de qué hablar. ¿Hay fundamentos detrás de este "
            f"movimiento o es pura especulación?\n\n"
            f"Analizo la acción de precio, volumen, y qué lo diferencia de otros pumps "
            f"que duran 24 horas. Todo lo que necesitas saber antes de tomar una decisión.\n\n"
            f"⚠️ Esto NO es consejo financiero. Siempre haz tu propia investigación (DYOR)."
        )
    elif detected_tokens and is_education:
        token_list = ", ".join(detected_tokens[:3])
        description = (
            f"Te explico qué es {token_list}, cómo funciona, y si tiene potencial real "
            f"o es solo narrativa del momento.\n\n"
            f"Análisis técnico + fundamentales para que tomes decisiones informadas, "
            f"no basadas en hype.\n\n"
            f"💡 La educación es tu mejor inversión antes de meter capital."
        )
    elif detected_tokens and is_news:
        token_list = ", ".join(detected_tokens[:3])
        description = (
            f"Noticias importantes sobre {token_list} que pueden impactar el mercado. "
            f"Te explico qué está pasando, qué significa, y cómo puede afectar tu portfolio.\n\n"
            f"Los movimientos institucionales son la señal más importante del mercado. "
            f"Aquí te doy el contexto que necesitas.\n\n"
            f"📊 ¿Qué opinas? Déjame tu análisis en los comentarios."
        )
    elif is_ai and is_trading:
        description = (
            f"La inteligencia artificial está cambiando las reglas del trading. "
            f"¿Pero realmente funciona o es puro marketing?\n\n"
            f"Te muestro resultados reales, sin filtros. Lo bueno, lo malo, y lo que "
            f"nadie te dice sobre usar IA para operar en los mercados.\n\n"
            f"🤖 El futuro del trading es la automatización. ¿Tú ya la usas?"
        )
    elif is_education and is_trading:
        description = (
            f"La mayoría de traders pierde dinero. Pero no es por falta de indicadores "
            f"o estrategias — es por algo que nadie te enseña.\n\n"
            f"En este video te explico lo que realmente marca la diferencia entre "
            f"un trader que pierde todo y uno que es consistentemente rentable.\n\n"
            f"💬 ¿Ya sabías esto? Cuéntame tu experiencia en los comentarios."
        )
    elif is_trading:
        description = (
            f"Lecciones reales de trading que solo aprendes operando. No teoría de libro — "
            f"experiencia directa del mercado.\n\n"
            f"Te comparto exactamente qué hice, por qué lo hice, y qué puedes "
            f"aprender de esto para mejorar tus propias operaciones.\n\n"
            f"💬 ¿Cuál ha sido tu mejor lección del mercado? Comenta abajo."
        )
    elif detected_tokens:
        token_list = ", ".join(detected_tokens[:3])
        description = (
            f"Análisis completo de {token_list}: qué está pasando, por qué importa, "
            f"y qué deberías tener en cuenta antes de tomar acción.\n\n"
            f"No te quedes solo con el titular — aquí tienes el contexto completo "
            f"para tomar decisiones inteligentes.\n\n"
            f"📊 ¿Tú qué opinas? Déjalo en los comentarios."
        )
    else:
        description = (
            f"Análisis y perspectiva directa del mercado. Sin rodeos, sin hype — "
            f"solo la información que necesitas para tomar mejores decisiones.\n\n"
            f"Cada video en este canal está diseñado para darte valor real, ya sea "
            f"que estés empezando o ya tengas experiencia en trading y cripto.\n\n"
            f"💬 Déjame tu pregunta en los comentarios. Leo todos."
        )
    
    return {
        "title": title,
        "description": description,
        "tags": unique_tags,
        "hook": "",
        "cta": "",
    }


def main():
    print("🔧 Fix Video Descriptions — Actualizando videos en YouTube\n")
    
    # Cargar datos
    scripts_db = load_all_scripts()
    print(f"📚 Guiones indexados: {len(scripts_db)}")
    
    with open(DATA_DIR / "upload_queue.json") as f:
        queue = json.load(f)
    
    # Identificar videos que necesitan fix
    to_fix = []
    for v in queue:
        if v.get("status") != "uploaded" or not v.get("video_id"):
            continue
        desc = (v.get("script_data") or {}).get("description", "") or v.get("description", "")
        if not desc or desc.startswith("#") or len(desc) < 30:
            to_fix.append(v)
    
    print(f"❌ Videos sin descripción adecuada: {len(to_fix)}\n")
    
    if not to_fix:
        print("✅ Todos los videos tienen descripción. Nada que hacer.")
        return
    
    # Conectar a YouTube
    print("🔗 Conectando a YouTube API...")
    yt = get_youtube_client()
    print("✅ Conectado\n")
    
    fixed = 0
    errors = 0
    
    for v in to_fix:
        vid_id = v["video_id"]
        title = v.get("title", "")
        vtype = v.get("type", "short")
        
        print(f"{'─' * 60}")
        print(f"📹 {vid_id} — {title[:55]}")
        
        # Buscar guión real primero
        guion = scripts_db.get(title, {})
        if guion.get("description"):
            script_data = guion
            print(f"   ✅ Guión encontrado — usando descripción real")
        else:
            script_data = generate_description_from_title(title, vtype)
            print(f"   ⚠️ Sin guión — generando descripción desde título")
        
        # Generar descripción completa
        full_desc = build_description(script_data, vtype)
        tags = script_data.get("tags", [])
        
        print(f"   📝 Descripción: {len(full_desc)} chars")
        print(f"   🏷️ Tags: {len(tags)}")
        
        # Actualizar en YouTube
        try:
            # Primero obtener el snippet actual para no perder datos
            current = yt.videos().list(part="snippet,status", id=vid_id).execute()
            items = current.get("items", [])
            if not items:
                print(f"   ❌ Video no encontrado en YouTube — skip")
                errors += 1
                continue
            
            snippet = items[0]["snippet"]
            
            # Actualizar solo descripción y tags (no tocar título, categoría, etc.)
            snippet["description"] = full_desc
            if tags:
                snippet["tags"] = [t.replace(" ", "") if len(t) > 30 else t for t in tags[:15]]
            
            # Asegurar que categoryId está presente (requerido por la API)
            if "categoryId" not in snippet:
                snippet["categoryId"] = "27" if vtype == "short" else "27"
            
            yt.videos().update(
                part="snippet",
                body={
                    "id": vid_id,
                    "snippet": snippet
                }
            ).execute()
            
            print(f"   ✅ Actualizado en YouTube")
            fixed += 1
            
            # Rate limit: esperar entre updates
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:150]}")
            errors += 1
            time.sleep(2)
    
    print(f"\n{'═' * 60}")
    print(f"📊 RESULTADO")
    print(f"   ✅ Actualizados: {fixed}")
    print(f"   ❌ Errores: {errors}")
    print(f"   📹 Total procesados: {len(to_fix)}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
