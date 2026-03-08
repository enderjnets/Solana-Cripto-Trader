# Plan: Alertas de Trading para Bot Solana

## Tareas

### Task 1: Crear módulo de alertas
- **File**: `Solana-Cripto-Trader/alerts.py`
- **Action**: Crear clase AlertManager con métodos para enviar alertas Telegram
- **Verify**: `python -c "from alerts import AlertManager; print('ok')"`

### Task 2: Agregar integración en bot
- **File**: `Solana-Cripto-Trader/bot_v7_hybrid.py`
- **Action**: Importar AlertManager, agregar en open_position(), close_position(), on_disconnect(), on_connect()
- **Verify**: `python -c "from bot_v7_hybrid import TradingBot; print('ok')"`

### Task 3: Probar alertas
- **File**: Terminal
- **Action**: Enviar alerta de prueba
- **Verify**: Recibir mensaje en Telegram

### Task 4: Reiniciar bot
- **File**: Sistema
- **Action**: Restart bot con nuevas alertas
- **Verify**: Bot corriendo + state actualizandose
