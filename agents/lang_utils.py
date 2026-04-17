"""
lang_utils — helper para leer preferencia de idioma del usuario y generar
directivas LLM bilingues. Usado por martingale_engine y risk_manager para
que los campos 'reasoning' del LLM salgan en el idioma seleccionado en
el dashboard.

Fail-safe: si el archivo user_profile.json no existe o esta corrupto,
retorna 'es' (comportamiento historico del bot).
"""
from pathlib import Path
import json

PROFILE_FILE = Path(__file__).parent / 'data' / 'user_profile.json'


def get_user_language() -> str:
    """Retorna 'es' o 'en'. Default 'es' si hay cualquier error."""
    try:
        lang = json.loads(PROFILE_FILE.read_text()).get('language', 'es')
        return lang if lang in ('es', 'en') else 'es'
    except Exception:
        return 'es'


def lang_directive(lang: str) -> str:
    """
    Directiva que se apendiza al prompt para pedir al LLM que responda
    el campo 'reasoning' en el idioma solicitado.

    Para mayor fiabilidad, la instruccion se repite dos veces (top/bottom
    de uso) — la funcion devuelve solo el bloque para apend al final.
    Si lang == 'es' retorna cadena vacia (cero cambio, comportamiento actual).
    """
    if lang == 'en':
        return (
            "\n\n=============================================================\n"
            "IMPORTANT OUTPUT LANGUAGE INSTRUCTION:\n"
            "- The 'reasoning' field of your JSON response MUST be written "
            "in natural, fluent English.\n"
            "- Keep ALL JSON keys, action codes (HOLD, OPEN_LEVEL, CLOSE, "
            "CLOSE_CHAIN, REDUCE, TIGHTEN, ABANDON_ALL) and symbol tickers "
            "exactly as specified — only the 'reasoning' text changes language.\n"
            "- Do not translate numeric values, percentages, or dollar amounts.\n"
            "============================================================="
        )
    return ''
