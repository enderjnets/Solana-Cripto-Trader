#!/usr/bin/env python3
"""
create_demo_assistant.py — Eco AI Automation LLC
================================================
Crea un VAPI assistant personalizado por negocio para demos en vivo.

USO:
    python3 create_demo_assistant.py --name "Salon Sorella" --type salon
    python3 create_demo_assistant.py --name "Denver Smiles Dental" --type dental --lang es
    python3 create_demo_assistant.py --list-types

El script genera un system prompt personalizado e inserta un assistant
temporal en VAPI. Devuelve el Assistant ID para usarlo en llamadas outbound.

TIPOS SOPORTADOS:
    salon, spa, dental, chiropractic, insurance, medspa, massage,
    lash, esthetics, dermatology, veterinary, auto, tax
"""

import argparse
import json
import sys
import requests
from datetime import datetime

# ─── VAPI CONFIG ────────────────────────────────────────────────────────────
VAPI_API_KEY = "f361bb66-8274-403a-8c0c-b984d7dd1cee"
VAPI_BASE_URL = "https://api.vapi.ai"
HEADERS = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json",
}

# ─── VOICE CONFIG ────────────────────────────────────────────────────────────
# Fernando (ElevenLabs) — voz configurada en el assistant principal de Eko
VOICE_CONFIG = {
    "provider": "11labs",
    "voiceId": "dlGxemPxFMTY7iXagmOj",  # Fernando
    "stability": 0.5,
    "similarityBoost": 0.75,
    "style": 0.0,
    "useSpeakerBoost": True,
}

# ─── TEMPLATES DE SYSTEM PROMPT POR INDUSTRIA ───────────────────────────────
INDUSTRY_TEMPLATES = {
    "salon": {
        "emoji": "✂️",
        "services": "cortes de cabello, coloración, peinados, tratamientos capilares, y manicure",
        "role": "recepcionista virtual del salón",
        "pain_point": "perder clientes cuando el equipo está ocupado atendiendo",
        "cta": "agendar una cita",
        "hours_example": "lunes a sábado de 9am a 7pm",
    },
    "spa": {
        "emoji": "🧖",
        "services": "masajes, faciales, tratamientos corporales, aromaterapia, y servicios de relajación",
        "role": "recepcionista virtual del spa",
        "pain_point": "no poder contestar el teléfono durante las sesiones de tratamiento",
        "cta": "reservar un servicio",
        "hours_example": "martes a domingo de 10am a 8pm",
    },
    "dental": {
        "emoji": "🦷",
        "services": "limpiezas dentales, revisiones, ortodoncia, blanqueamiento, y urgencias dentales",
        "role": "asistente virtual de la clínica dental",
        "pain_point": "pacientes que no pueden comunicarse cuando la recepción está ocupada",
        "cta": "agendar una cita o resolver dudas",
        "hours_example": "lunes a viernes de 8am a 5pm",
    },
    "chiropractic": {
        "emoji": "🦴",
        "services": "ajustes quiroprácticos, terapia de masajes, rehabilitación, y consultas para dolor de espalda y cuello",
        "role": "asistente virtual de la clínica quiropráctica",
        "pain_point": "perder pacientes nuevos cuando el doctor está en consulta",
        "cta": "agendar una consulta o primera visita",
        "hours_example": "lunes a viernes de 8am a 6pm, sábados de 9am a 1pm",
    },
    "insurance": {
        "emoji": "🛡️",
        "services": "cotizaciones de seguro de auto, hogar, vida y salud, y atención a pólizas existentes",
        "role": "asistente virtual de la agencia de seguros",
        "pain_point": "perder leads cuando el agente está en una reunión o llamada",
        "cta": "obtener una cotización o hablar con un agente",
        "hours_example": "lunes a viernes de 9am a 6pm",
    },
    "medspa": {
        "emoji": "💉",
        "services": "Botox, fillers, tratamientos láser, rejuvenecimiento facial, y consultas estéticas",
        "role": "coordinadora virtual del med spa",
        "pain_point": "perder clientes de alto valor cuando el staff está en procedimientos",
        "cta": "agendar una consulta o cotizar un tratamiento",
        "hours_example": "martes a sábado de 9am a 6pm",
    },
    "massage": {
        "emoji": "💆",
        "services": "masajes terapéuticos, masajes deportivos, masajes de relajación, y terapia de tejidos profundos",
        "role": "recepcionista virtual del centro de masajes",
        "pain_point": "no poder contestar el teléfono mientras se da una sesión de masaje",
        "cta": "reservar una sesión",
        "hours_example": "lunes a domingo de 10am a 8pm",
    },
    "lash": {
        "emoji": "👁️",
        "services": "extensiones de pestañas, lifting de pestañas, tinte, y mantenimiento",
        "role": "recepcionista virtual del salón de lashes",
        "pain_point": "perder citas cuando el técnico está trabajando con un cliente",
        "cta": "agendar una cita",
        "hours_example": "martes a sábado de 9am a 6pm",
    },
    "esthetics": {
        "emoji": "✨",
        "services": "faciales, limpiezas de piel, exfoliaciones, tratamientos anti-edad, y consultas de skincare",
        "role": "asistente virtual del estudio de estética",
        "pain_point": "perder clientes cuando la esteticista está en medio de un tratamiento",
        "cta": "agendar un facial o consulta de skincare",
        "hours_example": "martes a sábado de 9am a 5pm",
    },
    "dermatology": {
        "emoji": "🏥",
        "services": "consultas dermatológicas, tratamiento de acné, dermatología estética, y biopsias de piel",
        "role": "asistente virtual de la clínica de dermatología",
        "pain_point": "la larga lista de espera y pacientes que no logran comunicarse",
        "cta": "agendar una cita o preguntar sobre disponibilidad",
        "hours_example": "lunes a viernes de 8am a 5pm",
    },
    "veterinary": {
        "emoji": "🐾",
        "services": "consultas veterinarias, vacunas, cirugías, emergencias, y cuidado preventivo para mascotas",
        "role": "asistente virtual de la clínica veterinaria",
        "pain_point": "llamadas urgentes que no se pueden atender cuando el doctor está en cirugía",
        "cta": "agendar una cita o reportar una emergencia",
        "hours_example": "lunes a viernes de 8am a 6pm, sábados de 9am a 2pm",
    },
    "auto": {
        "emoji": "🚗",
        "services": "detailing de autos, lavado, pulido, protección de pintura, y tintado de ventanas",
        "role": "asistente virtual del negocio de auto detailing",
        "pain_point": "perder cotizaciones cuando el equipo está trabajando en los vehículos",
        "cta": "obtener una cotización o agendar un servicio",
        "hours_example": "lunes a sábado de 8am a 6pm",
    },
    "tax": {
        "emoji": "📊",
        "services": "preparación de impuestos personales y de negocios, contabilidad, planificación fiscal, y ITIN",
        "role": "asistente virtual de la oficina de impuestos",
        "pain_point": "el caos de la temporada de impuestos cuando todos llaman al mismo tiempo",
        "cta": "agendar una cita o preguntar sobre servicios",
        "hours_example": "lunes a viernes de 9am a 6pm, sábados en temporada de impuestos",
    },
}

DEFAULT_TEMPLATE = {
    "emoji": "🏢",
    "services": "nuestros servicios",
    "role": "asistente virtual del negocio",
    "pain_point": "perder clientes importantes por no poder contestar el teléfono",
    "cta": "agendar una cita o resolver dudas",
    "hours_example": "lunes a viernes de 9am a 6pm",
}


def generate_system_prompt(business_name: str, industry: str, lang: str = "es") -> str:
    """Genera un system prompt personalizado para VAPI basado en el negocio."""
    t = INDUSTRY_TEMPLATES.get(industry.lower(), DEFAULT_TEMPLATE)

    if lang == "en":
        return f"""You are a professional virtual receptionist for {business_name}.

Your name is Alex and you work for {business_name} as their {t['role']}.

IMPORTANT: You are NOT a demo — you ARE the actual assistant for {business_name}. Speak as if you've been working here for months and know the business well.

YOUR ROLE:
- Answer incoming calls professionally and warmly
- Help customers book appointments for {t['services']}
- Answer common questions about hours, location, pricing, and availability
- Collect customer information (name, phone, service requested, preferred time)
- Transfer urgent calls to the team when needed

BUSINESS INFO (use as examples during the demo):
- Business: {business_name}
- Services: {t['services']}
- Hours: {t['hours_example']} (adjust based on what the caller mentions)
- Location: Denver/Aurora area, Colorado

PERSONALITY:
- Warm, professional, and efficient
- Bilingual (English/Spanish) — match the caller's language
- Never say you're an AI unless directly asked
- If asked if you're AI: "I'm a virtual assistant for {business_name} — how can I help you today?"

BOOKING FLOW:
1. Greet: "Thank you for calling {business_name}, how can I help you?"
2. Listen to their need
3. Collect: name, service, preferred date/time, phone number
4. Confirm: "Perfect, I have you scheduled for [service] on [date] at [time]"
5. Close: "Is there anything else I can help you with?"

This is a LIVE DEMO for the owner of {business_name} — they are experiencing what their customers would hear. Be impressive."""

    else:  # Spanish (default)
        return f"""Eres la recepcionista virtual profesional de {business_name}.

Tu nombre es Alex y trabajas para {business_name} como {t['role']}.

IMPORTANTE: No eres una demo — ERES el asistente real de {business_name}. Habla como si llevaras meses trabajando aquí y conocieras bien el negocio.

TU ROL:
- Contestar llamadas de manera profesional y cálida
- Ayudar a los clientes a agendar citas para {t['services']}
- Responder preguntas sobre horarios, ubicación, precios y disponibilidad
- Recopilar información del cliente (nombre, teléfono, servicio deseado, horario preferido)
- Transferir llamadas urgentes al equipo cuando sea necesario

INFORMACIÓN DEL NEGOCIO (usa como ejemplos durante la demo):
- Negocio: {business_name}
- Servicios: {t['services']}
- Horarios: {t['hours_example']} (ajusta según lo que mencione quien llama)
- Ubicación: Área de Denver/Aurora, Colorado

PERSONALIDAD:
- Cálida, profesional y eficiente
- Bilingüe (español/inglés) — responde en el idioma del cliente
- Nunca digas que eres IA a menos que te pregunten directamente
- Si preguntan si eres IA: "Soy el asistente virtual de {business_name} — ¿en qué puedo ayudarte hoy?"

FLUJO DE AGENDAMIENTO:
1. Saluda: "Gracias por llamar a {business_name}, ¿en qué le puedo ayudar?"
2. Escucha su necesidad
3. Recopila: nombre, servicio, fecha y hora preferida, número de teléfono
4. Confirma: "Perfecto, lo tengo agendado para [servicio] el [fecha] a las [hora]"
5. Cierra: "¿Hay algo más en lo que pueda ayudarle?"

Esta es una DEMO EN VIVO para el dueño de {business_name} — están experimentando lo que escucharían sus clientes. Sé impresionante."""


def build_assistant_payload(business_name: str, industry: str, lang: str = "es") -> dict:
    """Construye el payload completo para crear el assistant en VAPI."""
    system_prompt = generate_system_prompt(business_name, industry, lang)
    safe_name = business_name.replace(" ", "-").lower()
    timestamp = datetime.now().strftime("%m%d-%H%M")

    return {
        "name": f"Demo-{safe_name}-{timestamp}",
        "model": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": system_prompt}
            ],
            "temperature": 0.7,
            "maxTokens": 250,
        },
        "voice": VOICE_CONFIG,
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-3",
            "language": "es" if lang == "es" else "en",
        },
        "firstMessage": (
            f"Gracias por llamar a {business_name}, le atiende Alex. ¿En qué le puedo ayudar el día de hoy?"
            if lang == "es"
            else f"Thank you for calling {business_name}, this is Alex. How can I help you today?"
        ),
        "silenceTimeoutSeconds": 30,
        "maxDurationSeconds": 600,
        "backgroundSound": "office",
        "backchannelingEnabled": True,
        "metadata": {
            "business": business_name,
            "industry": industry,
            "type": "demo",
            "created_by": "eco-ai-automation",
            "created_at": datetime.now().isoformat(),
        },
    }


def create_vapi_assistant(business_name: str, industry: str, lang: str = "es") -> dict:
    """Crea el assistant en VAPI y devuelve el resultado."""
    payload = build_assistant_payload(business_name, industry, lang)

    print(f"\n🚀 Creando assistant para: {business_name}")
    print(f"   Industria: {industry} | Idioma: {lang}")
    print(f"   Nombre VAPI: {payload['name']}")

    resp = requests.post(
        f"{VAPI_BASE_URL}/assistant",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    if resp.status_code in (200, 201):
        data = resp.json()
        assistant_id = data.get("id")
        print(f"\n✅ Assistant creado exitosamente!")
        print(f"   Assistant ID: {assistant_id}")
        print(f"\n📞 Para hacer una llamada de demo, usa outreach_caller.py:")
        print(f"   python3 outreach_caller.py --demo --assistant-id {assistant_id} --phone +1XXXXXXXXXX")
        return {"success": True, "assistant_id": assistant_id, "data": data}
    else:
        print(f"\n❌ Error al crear assistant: {resp.status_code}")
        print(f"   {resp.text}")
        return {"success": False, "error": resp.text, "status_code": resp.status_code}


def list_assistants() -> None:
    """Lista todos los assistants existentes en VAPI."""
    resp = requests.get(f"{VAPI_BASE_URL}/assistant", headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        assistants = resp.json()
        print(f"\n📋 Assistants en VAPI ({len(assistants)} total):\n")
        for a in assistants:
            created = a.get("createdAt", "")[:10] if a.get("createdAt") else "?"
            meta = a.get("metadata", {})
            biz = meta.get("business", "-")
            t = meta.get("type", "-")
            print(f"  [{created}] {a.get('name', 'Sin nombre')}")
            print(f"           ID: {a.get('id')} | Tipo: {t} | Negocio: {biz}")
    else:
        print(f"❌ Error al listar: {resp.status_code} — {resp.text}")


def delete_assistant(assistant_id: str) -> None:
    """Elimina un assistant de VAPI por ID."""
    resp = requests.delete(
        f"{VAPI_BASE_URL}/assistant/{assistant_id}",
        headers=HEADERS,
        timeout=30,
    )
    if resp.status_code in (200, 204):
        print(f"✅ Assistant {assistant_id} eliminado.")
    else:
        print(f"❌ Error al eliminar: {resp.status_code} — {resp.text}")


def show_prompt_preview(business_name: str, industry: str, lang: str = "es") -> None:
    """Muestra el system prompt que se generaría sin crear el assistant."""
    prompt = generate_system_prompt(business_name, industry, lang)
    print(f"\n📝 PREVIEW del system prompt para '{business_name}' ({industry}, {lang}):")
    print("─" * 60)
    print(prompt)
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Crea assistants VAPI personalizados para demos de Eco AI Automation"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Comando: create (default)
    create_p = parser.add_argument_group("Crear assistant")
    parser.add_argument("--name", "-n", help="Nombre del negocio (ej: 'Salon Sorella')")
    parser.add_argument(
        "--type", "-t",
        help="Tipo de industria (salon, dental, spa, etc.)",
        default="salon",
    )
    parser.add_argument(
        "--lang", "-l",
        help="Idioma del assistant: es (default) o en",
        choices=["es", "en"],
        default="es",
    )

    # Comandos adicionales
    parser.add_argument("--list", action="store_true", help="Lista todos los assistants")
    parser.add_argument("--list-types", action="store_true", help="Lista tipos de industria disponibles")
    parser.add_argument("--delete", metavar="ASSISTANT_ID", help="Elimina un assistant por ID")
    parser.add_argument("--preview", action="store_true", help="Solo muestra el prompt sin crear")
    parser.add_argument("--output-json", action="store_true", help="Salida en JSON (para scripting)")

    args = parser.parse_args()

    if args.list_types:
        print("\n📋 Tipos de industria disponibles:\n")
        for key, val in INDUSTRY_TEMPLATES.items():
            print(f"  {val['emoji']}  {key:<15} — {val['services'][:50]}...")
        return

    if args.list:
        list_assistants()
        return

    if args.delete:
        delete_assistant(args.delete)
        return

    if not args.name:
        parser.print_help()
        print("\n💡 Ejemplo: python3 create_demo_assistant.py --name 'Salon Sorella' --type salon")
        sys.exit(1)

    if args.preview:
        show_prompt_preview(args.name, args.type, args.lang)
        return

    result = create_vapi_assistant(args.name, args.type, args.lang)

    if args.output_json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
