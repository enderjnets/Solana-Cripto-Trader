#!/usr/bin/env python3
"""
outreach_caller.py — Eco AI Automation LLC
==========================================
Ejecuta llamadas outbound automatizadas a prospectos usando VAPI.

Modos de uso:
    1. PROSPECCIÓN (pitch 30s + oferta de demo):
       python3 outreach_caller.py --mode prospect

    2. DEMO DIRECTA (llama con assistant personalizado):
       python3 outreach_caller.py --mode demo --name "Salon Sorella" --type salon --phone +13035551234

    3. LLAMADA INDIVIDUAL:
       python3 outreach_caller.py --single --phone +13035551234 --business "Denver Smiles Dental" --industry dental

    4. DRY RUN (sin llamadas reales):
       python3 outreach_caller.py --dry-run

    5. VER ESTADO DE LLAMADAS:
       python3 outreach_caller.py --status

REQUISITOS:
    - pip install requests
    - Ejecutar prospecting_assistant.py primero para obtener el PROSPECTING_ASSISTANT_ID
    - Tener números de teléfono en crm_prospects.md

NOTAS IMPORTANTES:
    - VAPI cobra por minuto de llamada
    - Las llamadas salientes requieren que el número tenga llamadas salientes habilitadas
    - Máximo recomendado: 5 llamadas simultáneas
    - El intervalo entre llamadas es de 60 segundos por defecto (evita spam)
"""

import argparse
import json
import time
import sys
import re
import os
import requests
from datetime import datetime
from pathlib import Path

# ─── VAPI CONFIG ────────────────────────────────────────────────────────────
VAPI_API_KEY = "f361bb66-8274-403a-8c0c-b984d7dd1cee"
VAPI_BASE_URL = "https://api.vapi.ai"
PHONE_NUMBER_ID = "64fcd5de-ab68-4ae0-93f6-846ce1209cce"
HEADERS = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json",
}

# ─── ASSISTANT IDs ───────────────────────────────────────────────────────────
# Ejecuta primero: python3 prospecting_assistant.py
# y pega el ID generado aquí:
PROSPECTING_ASSISTANT_ID = "REEMPLAZAR_CON_ID_DE_prospecting_assistant.py"

# El assistant principal de Eko (demos genéricas al número directo)
EKO_MAIN_ASSISTANT_ID = "225a9f9f-5d58-412a-b8df-81b72c799a4a"

# ─── PATHS ───────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/enderj/.openclaw/workspace/eco")
CRM_FILE = WORKSPACE / "crm_prospects.md"
CALLS_LOG = WORKSPACE / "calls_log.json"

# ─── LISTA DE PROSPECTOS (desde voice_agent_sales_plan.md) ──────────────────
# Actualizar con teléfonos reales después de investigación
PROSPECTS = [
    {"id": 1,  "business": "Salon Sorella",               "industry": "salon",       "city": "Denver",  "lang": "es", "phone": None, "contact": None},
    {"id": 2,  "business": "Aria Spa & Salon",            "industry": "spa",         "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 3,  "business": "Bliss Nail & Spa",            "industry": "spa",         "city": "Aurora",  "lang": "en", "phone": None, "contact": None},
    {"id": 4,  "business": "Denver Smiles Dental",        "industry": "dental",      "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 5,  "business": "Colorado Family Dental",      "industry": "dental",      "city": "Aurora",  "lang": "en", "phone": None, "contact": None},
    {"id": 6,  "business": "Aspen Chiropractic",          "industry": "chiropractic","city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 7,  "business": "Peak Performance Chiro",      "industry": "chiropractic","city": "Aurora",  "lang": "en", "phone": None, "contact": None},
    {"id": 8,  "business": "Allstate - Maria Torres",     "industry": "insurance",   "city": "Denver",  "lang": "es", "phone": None, "contact": "Maria Torres"},
    {"id": 9,  "business": "State Farm - Carlos Mendez",  "industry": "insurance",   "city": "Aurora",  "lang": "es", "phone": None, "contact": "Carlos Mendez"},
    {"id": 10, "business": "Farmers Insurance - Denise Park", "industry": "insurance","city": "Denver", "lang": "en", "phone": None, "contact": "Denise Park"},
    {"id": 11, "business": "Glow Med Spa",                "industry": "medspa",      "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 12, "business": "Rejuvenate Skin Studio",      "industry": "esthetics",   "city": "Aurora",  "lang": "en", "phone": None, "contact": None},
    {"id": 13, "business": "Serenity Massage & Wellness", "industry": "massage",     "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 14, "business": "The Lash Lounge Denver",      "industry": "lash",        "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 15, "business": "Radiance Esthetics",          "industry": "esthetics",   "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 16, "business": "Colorado Dermatology",        "industry": "dermatology", "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 17, "business": "Aurora Animal Hospital",      "industry": "veterinary",  "city": "Aurora",  "lang": "en", "phone": None, "contact": None},
    {"id": 18, "business": "Mile High Pet Clinic",        "industry": "veterinary",  "city": "Denver",  "lang": "en", "phone": None, "contact": None},
    {"id": 19, "business": "Prestige Auto Detailing",     "industry": "auto",        "city": "Aurora",  "lang": "en", "phone": None, "contact": None},
    {"id": 20, "business": "Rocky Mountain Tax Services", "industry": "tax",         "city": "Aurora",  "lang": "es", "phone": None, "contact": None},
]


# ─── LOGGING ─────────────────────────────────────────────────────────────────

def load_calls_log() -> list:
    """Carga el historial de llamadas."""
    if CALLS_LOG.exists():
        with open(CALLS_LOG) as f:
            return json.load(f)
    return []


def save_calls_log(log: list) -> None:
    """Guarda el historial de llamadas."""
    with open(CALLS_LOG, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def log_call(prospect: dict, call_result: dict, mode: str) -> None:
    """Registra una llamada en el log."""
    log = load_calls_log()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "business": prospect.get("business"),
        "phone": prospect.get("phone"),
        "industry": prospect.get("industry"),
        "call_id": call_result.get("id"),
        "status": call_result.get("status", "initiated"),
        "success": bool(call_result.get("id")),
    }
    log.append(entry)
    save_calls_log(log)


# ─── VAPI CALL FUNCTIONS ──────────────────────────────────────────────────────

def make_outbound_call(
    phone: str,
    assistant_id: str,
    business_name: str,
    contact_name: str = None,
    dry_run: bool = False,
) -> dict:
    """Inicia una llamada outbound via VAPI."""
    customer = {"number": phone}
    if contact_name:
        customer["name"] = contact_name
    elif business_name:
        customer["name"] = business_name

    payload = {
        "phoneNumberId": PHONE_NUMBER_ID,
        "assistantId": assistant_id,
        "customer": customer,
        # Variables de contexto inyectadas en el assistant
        "assistantOverrides": {
            "variableValues": {
                "business_name": business_name,
                "contact_name": contact_name or business_name,
            }
        },
    }

    if dry_run:
        print(f"  [DRY RUN] Llamaría a: {phone} ({business_name})")
        print(f"  [DRY RUN] Payload: {json.dumps(payload, indent=6)}")
        return {"id": "dry-run-fake-id", "status": "dry_run"}

    resp = requests.post(
        f"{VAPI_BASE_URL}/call",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    if resp.status_code in (200, 201):
        return resp.json()
    else:
        print(f"  ❌ Error API: {resp.status_code} — {resp.text}")
        return {"error": resp.text, "status_code": resp.status_code}


def get_call_status(call_id: str) -> dict:
    """Obtiene el estado de una llamada por ID."""
    resp = requests.get(
        f"{VAPI_BASE_URL}/call/{call_id}",
        headers=HEADERS,
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()
    return {}


def list_recent_calls(limit: int = 20) -> list:
    """Lista las llamadas recientes en VAPI."""
    resp = requests.get(
        f"{VAPI_BASE_URL}/call",
        headers=HEADERS,
        params={"limit": limit},
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()
    return []


# ─── DEMO ASSISTANT CREATOR ───────────────────────────────────────────────────

def create_temp_demo_assistant(business_name: str, industry: str, lang: str = "es") -> str:
    """Crea un assistant de demo personalizado y devuelve su ID."""
    try:
        # Importar desde el mismo directorio
        sys.path.insert(0, str(WORKSPACE))
        from create_demo_assistant import create_vapi_assistant
        result = create_vapi_assistant(business_name, industry, lang)
        if result.get("success"):
            return result["assistant_id"]
    except ImportError:
        pass

    # Fallback: usar el assistant principal de Eko
    print(f"  ⚠️  No se pudo crear demo personalizada, usando Eko principal")
    return EKO_MAIN_ASSISTANT_ID


# ─── MAIN OUTREACH FLOWS ──────────────────────────────────────────────────────

def run_prospecting_campaign(
    prospects: list = None,
    max_calls: int = 5,
    interval_seconds: int = 60,
    dry_run: bool = False,
) -> None:
    """Ejecuta campaña de prospección outbound."""
    if PROSPECTING_ASSISTANT_ID == "REEMPLAZAR_CON_ID_DE_prospecting_assistant.py":
        print("❌ ERROR: Primero ejecuta prospecting_assistant.py para obtener el ID.")
        print("   python3 prospecting_assistant.py")
        print("   Luego actualiza PROSPECTING_ASSISTANT_ID en este archivo.")
        sys.exit(1)

    targets = prospects or [p for p in PROSPECTS if p.get("phone")]

    if not targets:
        print("⚠️  No hay prospectos con número de teléfono en la lista.")
        print("   Actualiza los teléfonos en PROSPECTS o en crm_prospects.md")
        print("\n   Para hacer una llamada de prueba:")
        print("   python3 outreach_caller.py --single --phone +1TUTELEFONO --business 'Test' --industry salon")
        return

    print(f"\n📞 CAMPAÑA DE PROSPECCIÓN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Prospectos con teléfono: {len(targets)}")
    print(f"   Límite de llamadas: {max_calls}")
    print(f"   Intervalo: {interval_seconds}s")
    print(f"   Dry run: {dry_run}")
    print(f"   Assistant ID: {PROSPECTING_ASSISTANT_ID}")
    print("─" * 50)

    called = 0
    for i, prospect in enumerate(targets):
        if called >= max_calls:
            print(f"\n✋ Límite de {max_calls} llamadas alcanzado.")
            break

        phone = prospect.get("phone")
        if not phone:
            print(f"  [{i+1}] ⏭️  {prospect['business']} — sin teléfono, saltando")
            continue

        print(f"\n  [{called+1}/{max_calls}] 📞 {prospect['business']}")
        print(f"       Teléfono: {phone} | Industria: {prospect['industry']}")

        result = make_outbound_call(
            phone=phone,
            assistant_id=PROSPECTING_ASSISTANT_ID,
            business_name=prospect["business"],
            contact_name=prospect.get("contact"),
            dry_run=dry_run,
        )

        if result.get("id"):
            print(f"       ✅ Llamada iniciada — ID: {result['id']}")
            log_call(prospect, result, "prospecting")
            called += 1
        else:
            print(f"       ❌ Fallo al iniciar llamada")

        if called < max_calls and i < len(targets) - 1 and not dry_run:
            print(f"       ⏳ Esperando {interval_seconds}s antes de la siguiente...")
            time.sleep(interval_seconds)

    print(f"\n📊 RESUMEN: {called} llamadas iniciadas")
    if not dry_run:
        print(f"   Log guardado en: {CALLS_LOG}")


def run_demo_call(
    phone: str,
    business_name: str,
    industry: str,
    lang: str = "es",
    dry_run: bool = False,
) -> None:
    """Hace una sola llamada de demo a un prospecto específico."""
    print(f"\n🎯 LLAMADA DE DEMO")
    print(f"   Negocio: {business_name}")
    print(f"   Industria: {industry}")
    print(f"   Teléfono: {phone}")
    print(f"   Idioma: {lang}")

    print(f"\n   Creando assistant personalizado...")
    assistant_id = create_temp_demo_assistant(business_name, industry, lang)
    print(f"   Assistant ID: {assistant_id}")

    result = make_outbound_call(
        phone=phone,
        assistant_id=assistant_id,
        business_name=business_name,
        dry_run=dry_run,
    )

    if result.get("id"):
        print(f"\n✅ Llamada de demo iniciada!")
        print(f"   Call ID: {result['id']}")
        print(f"   El prospecto escuchará al asistente de '{business_name}'")
    else:
        print(f"\n❌ Error al iniciar la llamada de demo")


def run_single_call(
    phone: str,
    business_name: str,
    industry: str,
    assistant_id: str = None,
    dry_run: bool = False,
) -> None:
    """Hace una sola llamada de prospección a un número específico."""
    aid = assistant_id or PROSPECTING_ASSISTANT_ID

    if aid == "REEMPLAZAR_CON_ID_DE_prospecting_assistant.py":
        print("❌ ERROR: Especifica --assistant-id o configura PROSPECTING_ASSISTANT_ID")
        sys.exit(1)

    print(f"\n📞 LLAMADA INDIVIDUAL")
    print(f"   Negocio: {business_name}")
    print(f"   Teléfono: {phone}")
    print(f"   Assistant ID: {aid}")

    result = make_outbound_call(
        phone=phone,
        assistant_id=aid,
        business_name=business_name,
        dry_run=dry_run,
    )

    if result.get("id"):
        print(f"\n✅ Llamada iniciada — ID: {result['id']}")
        prospect = {"business": business_name, "phone": phone, "industry": industry}
        log_call(prospect, result, "single")
    else:
        print(f"\n❌ Fallo: {result.get('error', 'Error desconocido')}")


def show_call_status() -> None:
    """Muestra el estado de llamadas recientes."""
    print("\n📊 LLAMADAS RECIENTES EN VAPI:")
    calls = list_recent_calls(20)

    if not calls:
        print("   No hay llamadas recientes.")
        return

    for call in calls:
        created = call.get("createdAt", "")[:16].replace("T", " ")
        status = call.get("status", "?")
        duration = call.get("duration", 0)
        customer = call.get("customer", {})
        phone = customer.get("number", "?")
        name = customer.get("name", "-")

        status_icon = {
            "ended": "✅", "in-progress": "🔵", "failed": "❌",
            "queued": "⏳", "ringing": "📞"
        }.get(status, "⚪")

        print(f"  {status_icon} [{created}] {phone} ({name})")
        print(f"       Status: {status} | Duración: {duration}s | ID: {call.get('id', '?')[:20]}...")

    # También mostrar log local
    local_log = load_calls_log()
    if local_log:
        print(f"\n📁 LOG LOCAL ({CALLS_LOG.name}): {len(local_log)} entradas")
        for entry in local_log[-5:]:
            ts = entry.get("timestamp", "")[:16]
            biz = entry.get("business", "?")
            mode = entry.get("mode", "?")
            ok = "✅" if entry.get("success") else "❌"
            print(f"  {ok} [{ts}] {biz} — modo: {mode}")


def show_prospects_status() -> None:
    """Muestra el estado de los prospectos (con/sin teléfono)."""
    with_phone = [p for p in PROSPECTS if p.get("phone")]
    without_phone = [p for p in PROSPECTS if not p.get("phone")]

    print(f"\n📋 ESTADO DE PROSPECTOS ({len(PROSPECTS)} total):")
    print(f"   ✅ Con teléfono: {len(with_phone)}")
    print(f"   ❌ Sin teléfono: {len(without_phone)}")

    if without_phone:
        print("\n   Sin teléfono (necesitan investigación):")
        for p in without_phone:
            print(f"     [{p['id']:2d}] {p['business']} ({p['city']})")

    if with_phone:
        print("\n   Con teléfono (listos para llamar):")
        for p in with_phone:
            print(f"     [{p['id']:2d}] {p['business']} — {p['phone']}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta llamadas outbound de prospección y demo via VAPI"
    )

    # Modos principales
    parser.add_argument(
        "--mode",
        choices=["prospect", "demo"],
        help="prospect=campaña de prospección | demo=demo personalizada",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Hacer una sola llamada de prueba",
    )

    # Opciones para llamada individual / demo
    parser.add_argument("--phone", help="Número de teléfono (formato: +13031234567)")
    parser.add_argument("--business", help="Nombre del negocio")
    parser.add_argument("--industry", default="salon", help="Tipo de industria")
    parser.add_argument("--lang", choices=["es", "en"], default="es", help="Idioma")
    parser.add_argument("--assistant-id", help="ID de assistant VAPI a usar")

    # Opciones de campaña
    parser.add_argument("--max-calls", type=int, default=5, help="Máximo de llamadas (default: 5)")
    parser.add_argument("--interval", type=int, default=60, help="Segundos entre llamadas (default: 60)")

    # Utilidades
    parser.add_argument("--dry-run", action="store_true", help="Simula sin hacer llamadas reales")
    parser.add_argument("--status", action="store_true", help="Muestra estado de llamadas recientes")
    parser.add_argument("--prospects", action="store_true", help="Muestra estado de prospectos")

    args = parser.parse_args()

    if args.status:
        show_call_status()
        return

    if args.prospects:
        show_prospects_status()
        return

    if args.single or (args.phone and not args.mode):
        if not args.phone:
            print("❌ --phone es requerido para llamadas individuales")
            sys.exit(1)
        if not args.business:
            print("❌ --business es requerido para llamadas individuales")
            sys.exit(1)
        run_single_call(
            phone=args.phone,
            business_name=args.business,
            industry=args.industry,
            assistant_id=args.assistant_id,
            dry_run=args.dry_run,
        )
        return

    if args.mode == "demo":
        if not args.phone or not args.business:
            print("❌ --phone y --business son requeridos para el modo demo")
            sys.exit(1)
        run_demo_call(
            phone=args.phone,
            business_name=args.business,
            industry=args.industry,
            lang=args.lang,
            dry_run=args.dry_run,
        )
        return

    if args.mode == "prospect" or not args.mode:
        run_prospecting_campaign(
            max_calls=args.max_calls,
            interval_seconds=args.interval,
            dry_run=args.dry_run,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
