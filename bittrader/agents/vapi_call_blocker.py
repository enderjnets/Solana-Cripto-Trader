#!/usr/bin/env python3
"""
VAPI Call Blocker — Webhook para rechazar números spammers automáticamente
Bloquea llamadas entrantes antes de conectar al asistente
"""

import json
import requests
from datetime import datetime
from pathlib import Path

# Configuración
BLOCKLIST_FILE = Path("/home/enderj/.openclaw/workspace/memory/VAPI_BLOCKLIST.json")
VAPI_API_KEY = "f361bb66-8274-403a-8c0c-b984d7dd1cee"
VAPI_PHONE_ID = "64fcd5de-ab68-4ae0-93f6-846ce1209cce"

# Números bloqueados hardcoded (fallback)
BLOCKED_NUMBERS = {
    "+17205121753": "Bitcoin/Crypto automated spammer script",
}

WHITELIST = {
    "+16159751056": "Yonathan Luzardo (Yona)",
    "+17208387940": "Maggie (Esposa)",
    "+17208249313": "Eko (VAPI itself)",
}

def load_blocklist():
    """Carga la lista de números bloqueados desde archivo"""
    try:
        if BLOCKLIST_FILE.exists():
            with open(BLOCKLIST_FILE) as f:
                data = json.load(f)
                blocked = {n["number"]: n["reason"] for n in data.get("blocked_spammers", [])}
                whitelist = {n["number"]: n["name"] for n in data.get("whitelist", [])}
                return blocked, whitelist
    except Exception as e:
        print(f"⚠️ Error loading blocklist: {e}")
    
    return BLOCKED_NUMBERS, WHITELIST

def should_block_call(incoming_number):
    """Determina si una llamada debe ser bloqueada"""
    blocked, whitelist = load_blocklist()
    
    # Si está en whitelist, permitir
    if incoming_number in whitelist:
        return False, f"Whitelisted: {whitelist[incoming_number]}"
    
    # Si está en blocklist, bloquear
    if incoming_number in blocked:
        return True, f"Blocked: {blocked[incoming_number]}"
    
    # Por defecto, permitir
    return False, "Not in any list"

def reject_call(call_id, reason):
    """Rechaza una llamada en VAPI"""
    url = f"https://api.vapi.ai/call/{call_id}/reject"
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "reason": reason,
        "status": "rejected"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code in [200, 204]:
            print(f"✅ Call {call_id} rejected: {reason}")
            return True
        else:
            print(f"❌ Failed to reject call: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error rejecting call: {e}")
        return False

def log_blocked_call(incoming_number, call_id, reason):
    """Registra una llamada bloqueada"""
    log_file = Path("/home/enderj/.openclaw/workspace/memory/VAPI_BLOCKED_CALLS.log")
    
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] {incoming_number} | Call ID: {call_id} | Reason: {reason}\n"
    
    try:
        with open(log_file, "a") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"⚠️ Error logging blocked call: {e}")

def handle_webhook(event):
    """
    Maneja webhooks de VAPI
    Esperado: evento de llamada entrante con número del llamante
    """
    print(f"\n🔔 Webhook recibido: {json.dumps(event, indent=2)}")
    
    # Extrae información de la llamada
    incoming_number = event.get("customer", {}).get("number")
    call_id = event.get("id")
    call_type = event.get("type")
    
    if not incoming_number or call_type != "inboundPhoneCall":
        print("⚠️ Webhook inválido o no es llamada entrante")
        return {"status": "ok", "action": "ignored"}
    
    print(f"\n📱 Llamada entrante de {incoming_number}")
    
    # Chequea si debe bloquearse
    should_block, reason = should_block_call(incoming_number)
    
    if should_block:
        print(f"🚫 BLOQUEANDO: {reason}")
        log_blocked_call(incoming_number, call_id, reason)
        reject_call(call_id, reason)
        return {"status": "blocked", "number": incoming_number, "reason": reason}
    else:
        print(f"✅ PERMITIENDO: {reason}")
        return {"status": "allowed", "number": incoming_number, "reason": reason}

def monitor_recent_calls():
    """Monitorea llamadas recientes y bloquea spammers"""
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get("https://api.vapi.ai/call?limit=10", headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ Error fetching calls: {response.status_code}")
            return
        
        calls = response.json()
        blocked, whitelist = load_blocklist()
        
        print("\n🔍 Monitoreando últimas 10 llamadas...")
        
        for call in calls:
            incoming_number = call.get("customer", {}).get("number")
            call_id = call.get("id")
            status = call.get("status")
            
            if not incoming_number:
                continue
            
            # Si está bloqueado y ya fue processed, documentar
            if incoming_number in blocked and status == "ended":
                print(f"  🚫 {incoming_number} - {status} (BLOCKED: {blocked[incoming_number]})")
                log_blocked_call(incoming_number, call_id, blocked[incoming_number])
            elif incoming_number in whitelist:
                print(f"  ✅ {incoming_number} - {status} (WHITELISTED: {whitelist[incoming_number]})")
            else:
                print(f"  📞 {incoming_number} - {status}")
    
    except Exception as e:
        print(f"❌ Error monitoring calls: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # Modo monitoreo
        monitor_recent_calls()
    else:
        # Modo webhook (simulado)
        print("=" * 80)
        print("VAPI Call Blocker v1.0")
        print("=" * 80)
        print("\nUso:")
        print("  python3 vapi_call_blocker.py monitor  — Monitorea y bloquea spammers")
        print("\nPara usar como webhook en VAPI:")
        print("  1. Deploy este script en un servidor (ej: Flask)")
        print("  2. Configura URL en VAPI webhook settings")
        print("  3. VAPI enviará eventos POST a tu endpoint")
        print("\nActualmente bloqueando:")
        blocked, whitelist = load_blocklist()
        for num, reason in blocked.items():
            print(f"  🚫 {num}: {reason}")
        print("\nWhitelist:")
        for num, name in whitelist.items():
            print(f"  ✅ {num}: {name}")
