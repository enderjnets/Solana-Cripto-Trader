#!/usr/bin/env python3
"""
🎯 MrBeast Optimizer Agent - Especializado en YouTube Virality Tactics

Implementa las estrategias del Playbook de MrBeast:
- Títulos optimizados (<50 chars, números, dinero)
- Thumbnails con rostros expresivos
- Estructura de retención con re-enganches
- Testing A/B de thumbnails
- Subtítulos estilo MrBeast (2 palabras, verde)
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
sys.path.insert(0, str(BITTRADER / "agents"))

from llm_config import call_llm

DATA_DIR = BITTRADER / "agents/data"
PRODUCTION_FILE = DATA_DIR / "production_latest.json"

# ── MrBeast Rules ─────────────────────────────────────────────────────────
MAX_TITLE_LENGTH = 50
MONEY_KEYWORDS = ["$", "dólares", "USD", "euros", "millones"]
SUPERLATIVES = ["largest", "biggest", "insane", "crazy", "enormous", "extreme"]
GREEN_WORDS = ["dinero", "millones", "gratis", "gratis", "gana", "profit", "éxito", "record"]


# ════════════════════════════════════════════════════════════════════════
# TITLE OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════

def optimize_title(title: str, description: str = "") -> Dict[str, Any]:
    """Optimize title using MrBeast tactics"""
    
    prompt = f"""Eres un experto en viral titles estilo MrBeast. Optimiza este título.

TÍTULO ORIGINAL:
{title}

DESCRIPCIÓN:
{description}

REGLAS:
1. Menos de 50 caracteres
2. Incluir número específico si es posible
3. Mencionar cantidad de dinero si aplica
4. Usar superlativos (largest, insane, crazy)
5. Primera persona para desafíos
6. Sin palabras complejas

Responde con JSON:
{{
  "optimized_title": "título optimizado",
  "tactics_used": ["lista de tácticas aplicadas"],
  "character_count": 0,
  "score": 0-100
}}
"""

    response = call_llm(prompt, "Eres un experto en titles virales.", max_tokens=300)
    
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response[json_start:json_end])
    except:
        pass
    
    # Fallback: truncate title
    optimized = title[:MAX_TITLE_LENGTH]
    return {
        "optimized_title": optimized,
        "tactics_used": ["truncate"],
        "character_count": len(optimized),
        "score": 50
        }


def calculate_ctr_score(thumbnail_path: Path) -> Dict[str, Any]:
    """Calculate predicted CTR based on MrBeast rules"""
    
    if not thumbnail_path.exists():
        return {"error": "Thumbnail not found", "score": 0}
    
    try:
        from PIL import Image
        img = Image.open(thumbnail_path)
    except:
        return {"error": "Could not load image", "score": 0}
    
    score = 100
    issues = []
    
    # Check dimensions (1080x1920 for shorts, 1280x720 for longs)
    w, h = img.size
    if h > w:  # Vertical (short)
        if (w, h) != (1080, 1920):
            issues.append("Non-optimal dimensions for short")
            score -= 10
    else:  # Horizontal (long)
        if (w, h) != (1280, 720):
            issues.append("Non-optimal dimensions for long")
            score -= 10
    
    # Check for face (simplified check - look for skin tones)
    # In real implementation, would use face detection
    img_array = list(img.getdata())
    has_warm_tones = any(200 < sum(pixel[:3]) < 255 for pixel in img_array if len(pixel) >= 3)
    
    if not has_warm_tones:
        issues.append("No face/skin tones detected")
        score -= 20
    
    # Check contrast
    # Simplified contrast check without ImageStat
    # Convert to grayscale and check std dev
    gray = img.convert("L")
    import statistics
    pixels = list(gray.getdata())
    contrast = statistics.stdev(pixels)
    
    if contrast < 50:
        issues.append("Low contrast")
        score -= 15
    
    return {
        "score": max(0, score),
        "issues": issues,
        "recommendations": get_thumbnail_recommendations(issues)
    }


def get_thumbnail_recommendations(issues: List[str]) -> List[str]:
    """Generate recommendations based on issues"""
    recs = []
    
    if "No face/skin tones detected" in issues:
        recs.append("Add expressive face with shock/joy expression")
    if "Low contrast" in issues:
        recs.append("Increase contrast with darker background")
    if "Non-optimal dimensions" in issues:
        recs.append("Resize to 1080x1920 (short) or 1280x720 (long)")
    
    return recs if recs else ["Thumbnail looks good!"]


def generate_retention_hooks(duration: int) -> List[Dict[str, Any]]:
    """Generate re-engagement hooks for video"""
    
    # MrBeast rule: re-engage every 3 minutes
    num_hooks = duration // 180  # 180 seconds = 3 minutes
    
    hooks = []
    for i in range(num_hooks):
        hook_time = (i + 1) * 180  # seconds
        
        hook_type = ["plot_twist", "surprise_reward", "new_challenge", "dramatic_reveal"][i % 4]
        
        hooks.append({
            "timestamp": hook_time,
            "type": hook_type,
            "suggestion": f"At {hook_time//60}:{hook_time%60:02d}, insert {hook_type.replace('_', ' ')}"
        })
    
    return hooks


def validate_subtitle_style(subs_file: Path) -> Dict[str, Any]:
    """Check if subtitles follow MrBeast style"""
    
    if not subs_file.exists():
        return {"error": "Subtitle file not found", "score": 0}
    
    content = subs_file.read_text()
    lines = content.split('\n')
    
    score = 100
    issues = []
    
    # Check for MrBeast-style formatting
    # Look for green color codes or highlighted words
    has_green = "\\c&H0000FF00" in content or "\\c&H00FF00" in content or "green" in content.lower()
    
    if not has_green:
        issues.append("No green highlighting for keywords")
        score -= 20
    
    # Check for word grouping (MrBeast uses 2 words at a time)
    # This is a simplified check
    has_short_lines = any(len(line.split()) <= 3 for line in lines if line.strip() and not line.isdigit())
    
    if not has_short_lines:
        issues.append("Consider shorter subtitle lines (2-3 words)")
        score -= 15
    
    return {
        "score": max(0, score),
        "issues": issues,
    }


def run_analysis(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run full MrBeast optimization analysis"""
    
    print("=" * 80)
    print("🎯 MrBEAST OPTIMIZER AGENT")
    print("=" * 80)
    print(f"  Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "optimizations": []
    }
    
    # Analyze titles
    if "title" in video_data:
        print("📝 Analizando título...")
        title_opt = optimize_title(video_data["title"], video_data.get("description", ""))
        results["title_optimization"] = title_opt
        print(f"  Score: {title_opt['score']}/100")
        print(f"  Tácticas: {', '.join(title_opt['tactics_used'])}")
        print()
    
    # Analyze thumbnail
    if "thumbnail_path" in video_data:
        print("🖼️ Analizando thumbnail...")
        thumb_score = calculate_ctr_score(Path(video_data["thumbnail_path"]))
        results["thumbnail_score"] = thumb_score
        print(f"  Score: {thumb_score['score']}/100")
        if thumb_score['issues']:
            print(f"  Issues: {', '.join(thumb_score['issues'])}")
        print()
    
    # Generate retention hooks
    if "duration" in video_data:
        print("🎬 Generando retención hooks...")
        hooks = generate_retention_hooks(video_data["duration"])
        results["retention_hooks"] = hooks
        print(f"  Hooks generados: {len(hooks)}")
        for hook in hooks:
            print(f"    • {hook['timestamp']}s: {hook['suggestion']}")
        print()
    
    # Overall score
    scores = [
        results.get("title_optimization", {}).get("score", 0),
        results.get("thumbnail_score", {}).get("score", 0)
    ]
    results["overall_score"] = sum(scores) / len(scores) if scores else 0
    
    print("=" * 80)
    print(f"✅ SCORE TOTAL: {results['overall_score']:.1f}/100")
    print("=" * 80)
    
    # Save results
    output_file = DATA_DIR / f"mrbeast_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\n💾 Resultados guardados: {output_file}")
    
    return results


if __name__ == "__main__":
    # Test with sample video data
    sample = {
        "title": "Bitcoin cae un 1.4% mientras AKT explota al alza",
        "description": "Análisis del mercado crypto hoy",
        "duration": 60,
        "thumbnail_path": "/home/enderj/.openclaw/workspace/bittrader/agents/output/rhino_v1/akt/thumbnail.jpg"
    }
    
    run_analysis(sample)
