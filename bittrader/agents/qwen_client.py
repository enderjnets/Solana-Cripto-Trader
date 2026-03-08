"""
qwen_client.py — Wrapper para Qwen2.5 14B local via Ollama
Uso: para tareas repetitivas/monitoreo sin gastar tokens de Claude
"""

import requests
import json
import time
import logging

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:14b"

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Verifica si Ollama está corriendo y Qwen disponible."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(DEFAULT_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


def ask(prompt: str, system: str = None, model: str = DEFAULT_MODEL,
        temperature: float = 0.3, max_tokens: int = 2048,
        timeout: int = 60) -> str:
    """
    Llama a Qwen local y retorna la respuesta como string.
    
    Args:
        prompt: El mensaje/pregunta
        system: System prompt opcional
        model: Modelo a usar (default: qwen2.5:14b)
        temperature: 0.0-1.0 (0.3 para monitoreo, 0.7 para creativo)
        max_tokens: Máximo de tokens en respuesta
        timeout: Segundos de espera máximo
    
    Returns:
        str: Respuesta del modelo, o "" si falla
    """
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            timeout=timeout
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    except requests.exceptions.Timeout:
        logger.error(f"Qwen timeout ({timeout}s)")
        return ""
    except Exception as e:
        logger.error(f"Qwen error: {e}")
        return ""


def ask_json(prompt: str, system: str = None, retries: int = 2, **kwargs) -> dict:
    """
    Llama a Qwen esperando respuesta JSON válida.
    Reintenta si el JSON está malformado.
    
    Returns:
        dict: JSON parseado, o {} si falla
    """
    system_json = (system or "") + "\nResponde ÚNICAMENTE con JSON válido, sin texto adicional, sin markdown."

    for attempt in range(retries + 1):
        raw = ask(prompt, system=system_json, **kwargs)
        # Limpiar markdown si viene con ```json
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt < retries:
                logger.warning(f"JSON inválido (intento {attempt+1}), reintentando...")
                time.sleep(1)
            else:
                logger.error(f"Qwen no devolvió JSON válido: {raw[:200]}")
                return {}


def summarize(text: str, max_words: int = 100) -> str:
    """Resume un texto largo en N palabras."""
    return ask(
        f"Resume el siguiente texto en máximo {max_words} palabras en español:\n\n{text}",
        temperature=0.2,
        max_tokens=256
    )


def analyze_news(headline: str, body: str) -> dict:
    """
    Analiza una noticia de cripto/trading.
    Retorna: sentimiento, impacto, activos afectados, resumen.
    """
    prompt = f"""Analiza esta noticia de trading/cripto y responde en JSON:

TITULAR: {headline}
CONTENIDO: {body[:1000]}

Responde con este formato exacto:
{{
  "sentimiento": "positivo|negativo|neutro",
  "impacto": "alto|medio|bajo",
  "activos": ["BTC", "ETH"],
  "resumen": "Una frase del punto clave",
  "es_relevante": true
}}"""

    return ask_json(prompt, temperature=0.1)


def check_video_status(video_info: dict) -> dict:
    """
    Evalúa si un video tiene problemas basado en su metadata.
    Retorna: tiene_problema, descripcion, accion_requerida
    """
    prompt = f"""Analiza el estado de este video de YouTube y determina si hay problemas:

{json.dumps(video_info, ensure_ascii=False, indent=2)}

Responde en JSON:
{{
  "tiene_problema": true,
  "descripcion": "qué problema hay",
  "accion_requerida": "qué hacer",
  "urgencia": "alta|media|baja"
}}"""

    return ask_json(prompt, temperature=0.1)


def generate_title_variations(title: str, n: int = 5) -> list:
    """Genera variaciones de título para YouTube A/B testing."""
    prompt = f"""Genera {n} variaciones del siguiente título de YouTube para trading/cripto.
Mantén el mismo tema pero varía el gancho. En español. Incluye emojis.

TÍTULO ORIGINAL: {title}

Responde en JSON:
{{"variaciones": ["título 1", "título 2", "título 3", "título 4", "título 5"]}}"""

    result = ask_json(prompt, temperature=0.7)
    return result.get("variaciones", [])


def draft_script_section(topic: str, section: str, duration_sec: int = 15) -> str:
    """
    Genera un borrador de sección de guión para video BitTrader.
    Sections: intro|desarrollo|conclusion
    """
    system = """Eres un guionista de videos de YouTube sobre trading y cripto.
Estilo: directo, energético, español latinoamericano.
El presentador es un rinoceronte 3D (mascota BitTrader).
NO incluyas headers ni etiquetas, solo el texto del guión."""

    words_per_sec = 2.5
    target_words = int(duration_sec * words_per_sec)

    prompt = f"""Escribe la sección "{section}" de un video sobre: {topic}
Duración objetivo: {duration_sec} segundos (~{target_words} palabras).
Sección: {section}"""

    return ask(prompt, system=system, temperature=0.7, max_tokens=512)


if __name__ == "__main__":
    # Test rápido
    print("🔍 Verificando Qwen local...")
    if not is_available():
        print("❌ Qwen no disponible — ¿está corriendo Ollama?")
        exit(1)

    print("✅ Qwen disponible\n")

    print("📝 Test 1: Pregunta simple")
    resp = ask("¿Cuál es la capitalización de mercado de Bitcoin hoy aproximadamente? Una frase.")
    print(f"   → {resp}\n")

    print("📊 Test 2: Análisis de noticia")
    result = analyze_news(
        "Bitcoin supera $90,000 tras anuncio de reserva estratégica de EE.UU.",
        "El presidente firmó una orden ejecutiva para crear una reserva estratégica de Bitcoin..."
    )
    print(f"   → {json.dumps(result, ensure_ascii=False, indent=2)}\n")

    print("🎬 Test 3: Variaciones de título")
    titles = generate_title_variations("Bitcoin sube 10% en un día")
    for i, t in enumerate(titles, 1):
        print(f"   {i}. {t}")
