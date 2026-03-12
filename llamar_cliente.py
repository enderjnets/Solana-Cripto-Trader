#!/usr/bin/env python3
"""
📞 Llamada automática a clientes Uber
Usa VAPI para llamar al cliente y avisar que estás en camino
"""
import sys
import requests
from datetime import datetime, timezone

VAPI_KEY = "f361bb66-8274-403a-8c0c-b984d7dd1cee"
ASSISTANT_ID = "225a9f9f-5d58-412a-b8df-81b72c799a4a"
PHONE_ID = "64fcd5de-ab68-4ae0-93f6-846ce1209cce"

def call_cliente(telefono: str, nombre: str, hora_salida: str, hora_pickup: str, direccion: str):
    """Llamar al cliente para avisar que estás en camino"""
    
    mensaje = f"Hola {nombre}, soy el conductor de Uber. Te llamo para confirmar que estoy en camino hacia tu ubicación en {direccion}. Salí a las {hora_salida} y llegaré puntual a las {hora_pickup}. ¿Estás listo? Gracias."
    
    response = requests.post(
        "https://api.vapi.ai/call",
        headers={
            "Authorization": f"Bearer {VAPI_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "phoneNumberId": PHONE_ID,
            "assistantId": ASSISTANT_ID,
            "customer": {
                "number": telefono,
                "name": nombre
            }
        },
        timeout=30
    )
    
    if response.status_code == 200:
        print(f"✅ Llamada iniciada a {nombre}")
        return response.json()
    else:
        print(f"❌ Error: {response.text}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Uso: python3 llamar_cliente.py <telefono> <nombre> <hora_salida> <hora_pickup> <direccion>")
        sys.exit(1)
    
    call_cliente(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
