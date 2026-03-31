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
    """Genera una descripción razonable basada solo en el título del video."""
    
    # Detectar tema principal del título para generar tags relevantes
    title_lower = title.lower()
    
    tags = []
    desc_parts = []
    
    # Crypto tokens mencionados
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
    
    # Temas
    if any(w in title_lower for w in ["trading", "trader", "trade", "opera"]):
        tags.extend(["Trading", "Day Trading"])
        desc_parts.append("trading y mercados financieros")
    if any(w in title_lower for w in ["crypto", "cripto", "criptomoneda"]):
        tags.extend(["Crypto", "Criptomonedas"])
        desc_parts.append("criptomonedas")
    if any(w in title_lower for w in ["bot", "ia", "inteligencia artificial", "claude", "gpt", "agente"]):
        tags.extend(["IA", "Inteligencia Artificial", "Bot Trading", "Automatización"])
        desc_parts.append("inteligencia artificial y automatización")
    if any(w in title_lower for w in ["ftmo", "prop firm", "funded"]):
        tags.extend(["FTMO", "Prop Firm", "Funded Trader"])
        desc_parts.append("prop firms y funded trading")
    if any(w in title_lower for w in ["explota", "pump", "sube", "rompe", "gainer"]):
        tags.extend(["Análisis Técnico", "Señales Cripto"])
    if any(w in title_lower for w in ["tendencia", "trending"]):
        tags.extend(["Tendencia", "Trending"])
    if any(w in title_lower for w in ["circle", "blackrock", "cathie", "ark"]):
        tags.extend(["Noticias Financieras", "Institucionales"])
        desc_parts.append("noticias financieras institucionales")
    
    # Base tags siempre
    tags.extend(["BitTrader", "Finanzas", "Inversión"])
    
    # Deduplicar tags
    seen = set()
    unique_tags = []
    for t in tags:
        if t.lower() not in seen:
            unique_tags.append(t)
            seen.add(t.lower())
    
    # Construir descripción
    clean_title = title.rstrip("💰🚀💸😱📈📉")
    if detected_tokens:
        token_text = ", ".join(detected_tokens[:3])
        description = f"{clean_title}\n\nEn este video hablamos de {token_text}"
        if desc_parts:
            description += f" — {desc_parts[0]}"
        description += ". ¿Qué opinas tú?"
    elif desc_parts:
        description = f"{clean_title}\n\nAnálisis y perspectiva sobre {desc_parts[0]}."
    else:
        description = f"{clean_title}\n\nAnálisis y estrategias de trading y criptomonedas."
    
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
