# 🚀 Guía de Configuración Mainnet

## ⚠️ AVISO IMPORTANTE

**ESTE MODO USA DINERO REAL. LEE CUIDADOSAMENTE.**

---

## 📋 Checklist Antes de Usar Mainnet

- [ ] Wallet mainnet generada
- [ ] Fondos depositados en la wallet
- [ ] Jupiter API Key obtenida
- [ ] Private key encriptada
- [ ] Límites de riesgo configurados
- [ ] Notificaciones Telegram activadas

---

## 🔐 Generar Wallet Mainnet

```bash
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
source venv/bin/activate
python3 tools/solana_wallet.py --network mainnet
```

**Wallet generada:**
```
Address: Ht3J5crwQoMgJ77K2y2V7BPo6F4Ld6pRyMBCCCKGgSTw
Private Key: SWkK24YnSZGHwEt51WcXcRdqbHWCBnWfWa5nuzTQBko
```

**⚠️ IMPORTANTE:**
- Guarda la private key en un lugar seguro
- NUNCA compartas esta clave
- Usa encriptación en producción

---

## 💰 Depositar Fondos

### Opción 1: Desde Exchange (Coinbase, Binance)

1. Ve a tu exchange
2. Retira SOL a: `Ht3J5crwQoMgJ77K2y2V7BPo6F4Ld6pRyMBCCCKGgSTw`
3. Red: Solana Network
4. Espera confirmación (~30 segundos)

### Opción 2: Desde otra wallet Solana

```
Desde: Tu otra wallet
A: Ht3J5crwQoMgJ77K2y2V7BPo6F4Ld6pRyMBCCCKGgSTw
Monto: TU ELIGES
```

### Verificar Balance

```bash
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
source venv/bin/activate
python3 tools/solana_wallet.py
```

---

## 🔑 Obtener Jupiter API Key

### ¿Por qué?

- Mayor límite de requests
- Soporte prioritario
- Features exclusivos

### Cómo obtenerla:

1. Ve a: https://portal.jup.ag
2. Crea cuenta
3. Genera API Key
4. Copia la key

### Configurar en .env:

```env
JUPITER_API_KEY=tu-api-key-aquí
```

---

## 🔒 Encriptar Private Key

### Método Simple (base64 + password)

```python
# tools/encrypt_wallet.py
import base64
from cryptography.fernet import Fernet

# Generar key (solo una vez)
key = Fernet.generate_key()
with open("config/encryption.key", "wb") as f:
    f.write(key)

# Encriptar
with open("config/encryption.key", "rb") as f:
    key = f.read()

fernet = Fernet(key)
encrypted = fernet.encrypt(b"tu-private-key-aquí")

with open("config/wallet.enc", "wb") as f:
    f.write(encrypted)
```

---

## ⚙️ Configurar .env para Mainnet

Edita `.env`:

```env
# Network
NETWORK=mainnet

# RPC
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# Wallet
HOT_WALLET_ADDRESS=Ht3J5crwQoMgJ77K2y2V7BPo6F4Ld6pRyMBCCCKGgSTw
HOT_WALLET_PRIVATE_KEY=SWkK24YnSZGHwEt51WcXcRdqbHWCBnWfWa5nuzTQBko

# APIs
JUPITER_API_KEY=tu-jupiter-key
HELIUS_API_KEY=tu-helius-key

# Límites de riesgo
MAX_TRADE_AMOUNT=0.5
DAILY_LOSS_LIMIT=0.1
```

---

## 🛡️ Configurar Límites de Riesgo

### Recomendaciones:

| Setting | Valor | Descripción |
|---------|-------|-------------|
| Max Trade | 0.1-0.3 | 10-30% por trade |
| Daily Loss | 0.05-0.1 | 5-10% pérdida máxima |
| Min Reserve | 0.01 SOL | Para fees de transacción |

### Configurar en .env:

```env
MAX_TRADE_AMOUNT=0.2  # Max 20% por trade
DAILY_LOSS_LIMIT=0.1   # Max 10% pérdida diaria
```

---

## 📱 Configurar Notificaciones Telegram

1. Crea un bot: @BotFather
2. Obtén el token
3. Obtén tu Chat ID: @userinfobot
4. Configura en .env:

```env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=123456789
```

---

## 🧪 Probar Antes de Usar Dinero Real

### 1. Probar Conexión

```bash
cd /home/enderj/.openclaw/workspace/solana-jupiter-bot
source venv/bin/activate
python3 -c "
from solana.rpc.api import Client
from solders.pubkey import Pubkey

rpc = Client('https://api.mainnet-beta.solana.com')
wallet = 'Ht3J5crwQoMgJ77K2y2V7BPo6F4Ld6pRyMBCCCKGgSTw'
resp = rpc.get_balance(Pubkey.from_string(wallet))
print(f'Balance: {resp.value / 1e9} SOL')
"
```

### 2. Probar Quote (sin ejecutar)

```bash
source venv/bin/activate
python3 tools/jupiter_client.py --quote SOL USDC 0.001
```

### 3. Probar Dashboard

```bash
streamlit run dashboard/solana_dashboard.py
```

Verifica que todo funciona antes de hacer trades reales.

---

## 🔄 Cambiar entre Devnet/Mainnet

### Devnet (pruebas):
```env
NETWORK=devnet
SOLANA_RPC_URL=https://api.devnet.solana.com
```

### Mainnet (dinero real):
```env
NETWORK=mainnet
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
```

---

## 🚨 Si Ocurre un Error

### Error: "Insufficient funds"
- Deposita más SOL a tu wallet

### Error: "Slippage exceeded"
- Aumenta slippage en configuración
- O el mercado está muy volátil

### Error: "Transaction failed"
- Revisa el explorador: https://explorer.solana.com
- Verifica que tienes SOL para fees

---

## 📞 Recursos

- Explorador: https://explorer.solana.com
- Jupiter: https://jup.ag
- Helius RPC: https://helius.dev
- Solana Docs: https://docs.solana.com

---

**⚠️ DISCLAIMER: Trading involucra riesgos. Solo invierte lo que puedas permitirte perder.**
