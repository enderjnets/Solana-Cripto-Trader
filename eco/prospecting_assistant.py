#!/usr/bin/env python3
"""
prospecting_assistant.py — Eco AI Automation LLC
=================================================
Crea (o actualiza) el VAPI assistant de prospección outbound "Eko".

Este assistant se usa para llamadas outbound automatizadas a prospectos.
Su trabajo es dar un pitch de 20-30 segundos y ofrecer una demo en vivo.

FLUJO DEL ASSISTANT:
1. Se identifica: "Hola [Nombre], soy Eko de Eco AI Automation..."
2. Pitch rápido: "¿Pierde llamadas cuando está ocupado con clientes?"
3. Ofrece demo: "¿Le puedo mostrar cómo funcionaría para [su negocio]?"
4. Si SÍ: transfiere a demo assistant / agenda llamada de demo
5. Si NO: agenda follow-up o termina amablemente

USO:
    python3 prospecting_assistant.py              # Crea el assistant
    python3 prospecting_assistant.py --update ID  # Actualiza uno existente
    python3 prospecting_assistant.py --show-prompt # Solo muestra el prompt
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

# ID del assistant principal de Eko (para transferencias)
EKO_DEMO_ASSISTANT_ID = "225a9f9f-5d58-412a-b8df-81b72c799a4a"

# ─── SYSTEM PROMPT DEL PROSPECTING ASSISTANT ────────────────────────────────
PROSPECTING_SYSTEM_PROMPT = """Eres Eko, el asistente de ventas de Eco AI Automation LLC.

Tu misión es hacer una llamada de prospección profesional, cálida y directa a dueños de negocios locales en Denver, Colorado.

## TU IDENTIDAD
- Nombre: Eko
- Empresa: Eco AI Automation LLC
- Tu jefe: Ender, fundador de la empresa
- Número de contacto de Ender: +1 (720) 824-9313

## PRODUCTO QUE OFRECES
Un Agente de Voz con IA para negocios locales que:
✅ Contesta todas las llamadas 24/7
✅ Agenda citas automáticamente
✅ Responde preguntas frecuentes (horarios, precios, servicios)
✅ Suena como una recepcionista real, no un robot
✅ En inglés y español
Precio: $500 setup + $400/mes

## ESTRUCTURA DE LA LLAMADA (sigue este orden)

### PASO 1 — SALUDO (5-10 segundos)
"Hola, [nombre del prospecto]? Soy Eko de Eco AI Automation, ¿tiene un momentito?"
- Si no es buen momento: "¿Cuándo sería mejor para llamarle? Con gusto le llamo de vuelta."
- Siempre respeta su tiempo

### PASO 2 — GANCHO (10-15 segundos)
"Le llamo porque ayudamos a negocios como [nombre del negocio] en Denver a nunca perder una llamada de cliente — incluso cuando están atendiendo, en lunch, o después de las 5pm."

Variantes según industria:
- Salón/Spa: "¿Cuántas veces a la semana suena el teléfono mientras está con un cliente y no puede contestar?"
- Dental: "Muchas clínicas pierden 5-10 pacientes nuevos por semana porque la línea está ocupada."
- Seguros: "Sus leads más valiosos llaman una vez. Si no contestan, van con la competencia."

### PASO 3 — PREGUNTA DE CALIFICACIÓN (5 segundos)
"¿Le pasa que a veces no puede contestar el teléfono cuando está ocupado?"
- Si dice SÍ o similar: continúa al paso 4
- Si dice NO: "Entonces tiene un negocio muy bien organizado. Muchos de nuestros clientes pensaban lo mismo hasta que calcularon las llamadas perdidas. ¿Le interesa escuchar cómo lo hacen?"

### PASO 4 — OFERTA DE DEMO (10-15 segundos)
"Tenemos algo especial: puedo mostrarle exactamente cómo sonaría el asistente para [nombre del negocio] en este momento. No es un video ni un PowerPoint — es una llamada real donde escucha cómo contestaría sus clientes. ¿Le gustaría escucharlo ahora?"

Si dice SÍ:
"Perfecto. Voy a conectarle ahora mismo con nuestra demo personalizada para [negocio]. Solo escuche cómo suenan sus nuevas llamadas..."
[TRANSFERIR al demo assistant]

Si dice NO pero muestra interés:
"Entiendo, no se preocupe. ¿Puedo enviarle un mensaje con el número para que lo pruebe cuando tenga tiempo? Es gratis y sin compromiso."
[Obtener número/email para follow-up]

### PASO 5 — CIERRE AMABLE (si dicen que no)
"Perfecto, no hay problema. ¿Le parece si le mando la información por texto y si en algún momento tiene curiosidad nos puede llamar? El número es el (720) 824-9313."
Esperar respuesta y despedirse con: "Que tenga un excelente día, [nombre]. Cuídese."

## REGLAS IMPORTANTES
1. Máximo 90 segundos si no hay interés — no insistas
2. Habla de forma natural, pausada — no a las carreras
3. Usa el nombre del negocio y del prospecto si lo tienes
4. Si no entiende algo, pide que repita amablemente
5. Si se molestan: "Le pido disculpas por interrumpir, que tenga muy buen día" y termina
6. No menciones precios a menos que pregunten directamente
7. Si preguntan el precio: "$400 al mes, pero primero quiero que lo pruebe gratis"

## VARIABLES DEL SISTEMA
Cuando se inyecten datos del prospecto:
- {{customer.name}} = nombre del contacto
- {{customer.business_name}} = nombre del negocio
- {{customer.industry}} = industria del negocio
- {{customer.language}} = es o en

## TRANSFERENCIA A DEMO
Si el prospecto quiere escuchar la demo en vivo, usa la función de transferencia a:
Assistant ID: 225a9f9f-5d58-412a-b8df-81b72c799a4a

Antes de transferir di:
"Excelente decisión. Le voy a conectar ahora mismo. Solo escuche la primera llamada — va a sorprenderse. Un momento..."
"""

PROSPECTING_FIRST_MESSAGE = "Hola, ¿me podría comunicar con el encargado del negocio? Soy Eko de Eco AI Automation."


def build_prospecting_assistant_payload() -> dict:
    """Construye el payload para el assistant de prospección."""
    return {
        "name": "Eko-Prospecting-Outbound-v1",
        "model": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": PROSPECTING_SYSTEM_PROMPT}
            ],
            "temperature": 0.6,
            "maxTokens": 200,
        },
        "voice": {
            "provider": "11labs",
            "voiceId": "dlGxemPxFMTY7iXagmOj",  # Fernando — voz de Eko
            "stability": 0.5,
            "similarityBoost": 0.75,
            "style": 0.1,
            "useSpeakerBoost": True,
        },
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-3",
            "language": "es",
        },
        "firstMessage": PROSPECTING_FIRST_MESSAGE,
        "endCallFunctionEnabled": True,
        "silenceTimeoutSeconds": 20,
        "maxDurationSeconds": 180,  # 3 minutos máximo por llamada de prospección
        "backgroundSound": "off",
        "backchannelingEnabled": False,
        "endCallMessage": "Que tenga excelente día. Hasta luego.",
        # Función de transferencia al demo assistant
        "functions": [
            {
                "name": "transfer_to_demo",
                "description": "Transfiere al prospecto al demo assistant personalizado cuando quiere escuchar la demo en vivo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {
                            "type": "string",
                            "description": "Nombre del negocio del prospecto",
                        }
                    },
                    "required": ["business_name"],
                },
            }
        ],
        "metadata": {
            "type": "prospecting",
            "version": "1.0",
            "created_by": "eco-ai-automation",
            "created_at": datetime.now().isoformat(),
            "demo_assistant_id": EKO_DEMO_ASSISTANT_ID,
        },
    }


def create_prospecting_assistant() -> dict:
    """Crea el assistant de prospección en VAPI."""
    payload = build_prospecting_assistant_payload()

    print("\n🚀 Creando Prospecting Assistant en VAPI...")
    print(f"   Nombre: {payload['name']}")
    print(f"   Modelo: {payload['model']['model']}")
    print(f"   Voz: Fernando (ElevenLabs)")
    print(f"   Max duración: {payload['maxDurationSeconds']}s")

    resp = requests.post(
        f"{VAPI_BASE_URL}/assistant",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    if resp.status_code in (200, 201):
        data = resp.json()
        assistant_id = data.get("id")
        print(f"\n✅ Prospecting Assistant creado!")
        print(f"   Assistant ID: {assistant_id}")
        print(f"\n💾 GUARDA ESTE ID — lo necesitas para outreach_caller.py:")
        print(f"   PROSPECTING_ASSISTANT_ID = '{assistant_id}'")
        print(f"\n📋 Próximo paso:")
        print(f"   Actualiza PROSPECTING_ASSISTANT_ID en outreach_caller.py")
        print(f"   Luego ejecuta: python3 outreach_caller.py --dry-run")
        return {"success": True, "assistant_id": assistant_id, "data": data}
    else:
        print(f"\n❌ Error: {resp.status_code}")
        print(f"   {resp.text}")
        return {"success": False, "error": resp.text, "status_code": resp.status_code}


def update_prospecting_assistant(assistant_id: str) -> dict:
    """Actualiza un assistant existente con el nuevo prompt."""
    payload = build_prospecting_assistant_payload()

    print(f"\n🔄 Actualizando assistant {assistant_id}...")

    resp = requests.patch(
        f"{VAPI_BASE_URL}/assistant/{assistant_id}",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )

    if resp.status_code in (200, 201):
        print(f"✅ Assistant actualizado!")
        return {"success": True, "data": resp.json()}
    else:
        print(f"❌ Error: {resp.status_code} — {resp.text}")
        return {"success": False, "error": resp.text}


def get_assistant(assistant_id: str) -> dict:
    """Obtiene un assistant por ID."""
    resp = requests.get(
        f"{VAPI_BASE_URL}/assistant/{assistant_id}",
        headers=HEADERS,
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Crea el assistant de prospección outbound para Eco AI Automation"
    )
    parser.add_argument(
        "--update",
        metavar="ASSISTANT_ID",
        help="Actualiza un assistant existente en lugar de crear uno nuevo",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Muestra el system prompt sin crear el assistant",
    )
    parser.add_argument(
        "--show-payload",
        action="store_true",
        help="Muestra el payload JSON completo sin crear",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Salida en JSON (para scripting)",
    )

    args = parser.parse_args()

    if args.show_prompt:
        print("\n📝 SYSTEM PROMPT DEL PROSPECTING ASSISTANT:")
        print("─" * 60)
        print(PROSPECTING_SYSTEM_PROMPT)
        print("─" * 60)
        return

    if args.show_payload:
        payload = build_prospecting_assistant_payload()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.update:
        result = update_prospecting_assistant(args.update)
    else:
        result = create_prospecting_assistant()

    if args.output_json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
