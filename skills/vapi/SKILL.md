# VAPI Voice Calls Skill

## Descripción
Hacer y recibir llamadas telefónicas vía VAPI.ai con voz IA.

## Credenciales
- **Private Key**: f361bb66-8274-403a-8c0c-b984d7dd1cee
- **Phone Number ID**: 64fcd5de-ab68-4ae0-93f6-846ce1209cce  
- **Assistant ID (Eko)**: 225a9f9f-5d58-412a-b8df-81b72c799a4a
- **Número**: +1 (720) 824-9313

## Hacer una llamada saliente

```bash
curl -s -X POST https://api.vapi.ai/call \
  -H "Authorization: Bearer f361bb66-8274-403a-8c0c-b984d7dd1cee" \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumberId": "64fcd5de-ab68-4ae0-93f6-846ce1209cce",
    "assistantId": "225a9f9f-5d58-412a-b8df-81b72c799a4a",
    "assistantOverrides": {
      "firstMessage": "TU MENSAJE AQUÍ",
      "model": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "system", "content": "INSTRUCCIONES PARA LA LLAMADA"}]
      }
    },
    "customer": {
      "number": "+1XXXXXXXXXX",
      "name": "NOMBRE"
    }
  }'
```

## Cambiar voz

```bash
curl -X PATCH https://api.vapi.ai/assistant/225a9f9f-5d58-412a-b8df-81b72c799a4a \
  -H "Authorization: Bearer f361bb66-8274-403a-8c0c-b984d7dd1cee" \
  -H "Content-Type: application/json" \
  -d '{"voice": {"provider": "azure", "voiceId": "es-VE-SebastianNeural"}}'
```

## Voces Azure en Español (gratis)
| VoiceId | Género | País |
|---------|--------|------|
| es-VE-SebastianNeural | M | Venezuela ✅ actual |
| es-MX-JorgeNeural | M | México |
| es-MX-DaliaNeural | F | México |
| es-CO-GonzaloNeural | M | Colombia |
| es-CO-SalomeNeural | F | Colombia |
| es-AR-TomasNeural | M | Argentina |
| es-VE-PaolaNeural | F | Venezuela |
| es-US-AlonsoNeural | M | Latino USA |

## Ver estado de llamada

```bash
curl https://api.vapi.ai/call/CALL_ID \
  -H "Authorization: Bearer f361bb66-8274-403a-8c0c-b984d7dd1cee"
```

## Ver historial de llamadas

```bash
curl "https://api.vapi.ai/call?limit=10" \
  -H "Authorization: Bearer f361bb66-8274-403a-8c0c-b984d7dd1cee"
```

## Notas importantes
- Llamadas entrantes: siempre funcionan (llamar al 720-824-9313)
- Llamadas salientes: pueden ser rechazadas por carrier detection de spam
- ElevenLabs sin créditos → usar Azure (gratis)
- Costo: ~$0.01/llamada corta, ~$0.05/2min
- Concurrencia: hasta 10 llamadas simultáneas
