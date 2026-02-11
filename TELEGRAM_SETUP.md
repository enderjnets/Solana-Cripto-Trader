# Telegram Bot Configuration
# ==========================

# Para configurar Telegram:

# 1. Crea un bot con @BotFather en Telegram
#    - Envía /newbot
#    - Nombre: Eko Trading Bot
#    - Username:eko_trading_bot (debe terminar en bot)
#    - Copia el TOKEN

# 2. Obtener Chat ID
#    - Envía /start a @userinfobot
#    - Copia el Chat ID (número negativo para grupos)

# 3. Agregar al .env:
TELEGRAM_BOT_TOKEN=tu_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui

# 4. Instalar dependencia:
#    pip install python-telegram-bot

# 5. Iniciar bot:
#    python3 telegram_bot.py

# ==========================================
# ESTADO ACTUAL:
# ==========================================
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# El bot está listo pero necesita configuración.
# Una vez configurado, podrás usar:

# Comandos disponibles:
# /status - Ver estado del sistema
# /balance - Ver balance de wallet
# /workers - Ver workers activos
# /trades - Ver trades recientes
# /help - Ayuda
