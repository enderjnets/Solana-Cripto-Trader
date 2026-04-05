# Implementación de SHORT Trading con Drift Protocol

## Resumen del Proyecto

Sistema de trading automatizado para Solana que incluye:
- Trading SPOT (compra/venta de tokens)
- Trading de PERPETUALS (SHORT/LONG con leverage)
- Precios en tiempo real
- Paper trading (simulado)

---

## Fuentes de Precios (APIs)

### 1. Jupiter API (Principal)
```python
import requests

# Obtener precios de tokens
ids = "So11111111111111111111111111111111111111112,9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"  # SOL, BTC
url = "https://lite-api.jup.ag/price/v3"
params = {"ids": ids}
response = requests.get(url, params=params, timeout=10)
data = response.json()

# Estructura de respuesta:
# {
#   "So11111111111111111111111111111111111111112": {
#       "usdPrice": 85.50,
#       "priceChange24h": -2.5
#   }
# }
```

### 2. CoinGecko API (Backup)
```python
url = "https://api.coingecko.com/api/v3/simple/price"
params = {
    "ids": "solana,bitcoin,ethereum",
    "vs_currencies": "usd",
    "include_24hr_change": "true"
}
response = requests.get(url, params=params, timeout=10)
data = response.json()
# Estructura: {"solana": {"usd": 85.50, "usd_24h_change": -2.5}}
```

### 3. DEX Screener (Descubrimiento de tokens)
```python
url = "https://api.dexscreener.com/latest/dex/tokens/solana"
response = requests.get(url, timeout=10)
data = response.json()
# Devuelve tokens trending en Solana
```

---

## Tokens Soportados (Direcciones Mint en Solana)

```python
TOKENS = {
    # Majors
    "SOL": "So11111111111111111111111111111111111111112",
    "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
    "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tFm7mkBZKaUUTwJWS3d",
    
    # DeFi
    "RAY": "4k3DyjzvzpLhG1hGLbo2duNZf1kWQqawqjJHbDkPkrm",
    "ORCA": "orcaEKTdK7LKz57ZfY8EfYsKKF9LRcUsQh3m",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2",
    
    # Memes
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP",
    "WIF": "85VBFQZC9TZkfaptBWqv14ALD9fJNUKtWA41kh69teRP",
    "PEPE": "HZ1JovNiVvGrGNiiYvEozEVgZ58xa3kPfYoBKRJiNfnh",
    "FLOKI": "FLEniGBX6aLQJ9JGC5m1N3xKmBYL3z6S4VqV7XWDTpo",
}
```

---

## Lógica de Señales SHORT

### Condiciones para SHORT (apostar a la baja):

```python
def calculate_short_signals(prices):
    """
    Detecta oportunidades de SHORT cuando el precio va a caer
    
    Condiciones:
    - price_change_24h < -3%: Fuerte caída → SHORT
    - price_change_24h < -1.5%: Caída moderada → SHORT débil
    """
    short_signals = []
    
    for symbol, data in prices.items():
        change_24h = data.get("change", 0)
        
        # SHORT señal fuerte
        if change_24h < -3:
            short_signals.append({
                "symbol": symbol,
                "action": "SHORT",
                "direction": "short",
                "strength": abs(change_24h),
                "reason": f"Caída fuerte: {change_24h:+.1f}%",
                "leverage": 2.0
            })
        
        # SHORT señal moderada
        elif change_24h < -1.5:
            short_signals.append({
                "symbol": symbol,
                "action": "SHORT",
                "direction": "short",
                "strength": abs(change_24h) * 0.7,
                "reason": f"Caída moderada: {change_24h:+.1f}%",
                "leverage": 1.5
            })
    
    return sorted(short_signals, key=lambda x: x["strength"], reverse=True)
```

### Condiciones para LONG (perp et al):

```python
def calculate_long_signals(prices):
    """
    Detecta oportunidades de LONG en perp et al
    
    Condiciones:
    - price_change_24h > 3%: Fuerte subida → LONG
    - price_change_24h > 1.5%: Subida moderada → LONG débil
    """
    long_signals = []
    
    for symbol, data in prices.items():
        change_24h = data.get("change", 0)
        
        if change_24h > 3:
            long_signals.append({
                "symbol": symbol,
                "action": "LONG",
                "direction": "long",
                "strength": change_24h,
                "reason": f"Subida fuerte: {change_24h:+.1f}%",
                "leverage": 2.0
            })
        
        elif change_24h > 1.5:
            long_signals.append({
                "symbol": symbol,
                "action": "LONG",
                "direction": "long",
                "strength": change_24h * 0.7,
                "reason": f"Subida moderada: {change_24h:+.1f}%",
                "leverage": 1.5
            })
    
    return sorted(long_signals, key=lambda x: x["strength"], reverse=True)
```

---

## Drift Protocol (Perpetual Futures)

### Dirección del Programa (Mainnet)
```
DRIFT_PROGRAM_ID: dRiftyHA39MWEi3m9G5DgqFvsE8z1D2vLwT6N7x
```

### Mercados Perpetuos Soportados

```python
PERP_MARKETS = {
    "SOL": {"address": "4ooGWrxVGQAP4D3Te8Ajxu8N8ueuAq6kYBWBRM3V4xc"},
    "BTC": {"address": "4Ah8BNbRaMtLV6e1MZW3E5x6e7v3KxqN8vK9jX5YmBz"},
    "ETH": {"address": "3x8mVGQAP4D3Te8Ajxu8N8ueuAq6kYBWBRM3V4xcE"},
}
```

### Cálculo de PnL para SHORT

```python
def calculate_short_pnl(entry_price, exit_price, leverage):
    """
    Calcula ganancias/pérdidas para posición SHORT
    
    SHORT: Ganas cuando el precio BAJA
    PnL = ((entry_price - exit_price) / entry_price) * leverage * 100
    """
    pnl_pct = ((entry_price - exit_price) / entry_price) * leverage * 100
    return pnl_pct
```

### Ejemplo:
- Entry: $100
- Exit: $90 (precio bajó)
- Leverage: 2x
- PnL = (100-90)/100 * 2 * 100 = +20%

---

## Estructura del Estado (JSON)

```python
state = {
    "capital_usd": 500.0,
    "positions": {
        "SOL": {
            "amount": 0.5,
            "entry_price": 85.0,
            "direction": "long",  # o "short"
            "leverage": 1.0,  # o 2.0 para perp
            "timestamp": "2026-02-16T10:00:00"
        },
        "BTC_SHORT": {
            "amount": 0.01,
            "entry_price": 43000.0,
            "direction": "short",
            "leverage": 2.0,
            "timestamp": "2026-02-16T10:00:00"
        }
    },
    "trades": [
        {
            "time": "2026-02-16T10:00:00",
            "action": "BUY",  # o "SHORT", "TAKE_PROFIT", "STOP_LOSS"
            "symbol": "SOL",
            "price": 85.0,
            "amount": 0.5,
            "cost": 42.5,
            "direction": "long"
        }
    ],
    "today_pnl": 0.0,
    "last_date": "2026-02-16"
}
```

---

## Parámetros de Riesgo (Ajustables)

```python
# Configuración de riesgo
TP_TAKE_PROFIT = 2.5  # % Take Profit
SL_STOP_LOSS = 1.0     # % Stop Loss
MAX_POSITIONS = 4      # Posiciones max
MAX_POS_PER_CYCLE = 2  # Entradas max por ciclo
TRADE_SIZE_PCT = 0.01  # 1% del capital por trade
COOLDOWN_MIN = 5       # Minutos entre entradas del mismo token
LEVERAGE_DEFAULT = 2.0 # Leverage para perp et al
```

---

## Ejemplo de Flujo Completo

```python
import requests
import time
import json

class TradingBot:
    def __init__(self):
        self.state = self.load_state()
        
    def get_prices(self):
        """Obtiene precios de Jupiter + CoinGecko"""
        prices = {}
        
        # Jupiter
        ids = "So11111111111111111111111111111111111111112,9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"
        resp = requests.get(f"https://lite-api.jup.ag/price/v3?ids={ids}", timeout=10)
        data = resp.json()
        
        for sym, mint in [("SOL", "So11111111111111111111111111111111111111112"), ("BTC", "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E")]:
            if mint in data:
                prices[sym] = {
                    "price": float(data[mint].get("usdPrice", 0)),
                    "change": float(data[mint].get("priceChange24h", 0))
                }
        
        return prices
    
    def calculate_signals(self, prices):
        """Genera señales de SHORT/LONG"""
        short_signals = []
        long_signals = []
        
        for sym, data in prices.items():
            change = data.get("change", 0)
            
            # SHORT cuando baja
            if change < -3:
                short_signals.append({"symbol": sym, "action": "SHORT", "strength": abs(change)})
            # LONG cuando sube
            elif change > 3:
                long_signals.append({"symbol": sym, "action": "LONG", "strength": change})
        
        return short_signals, long_signals
    
    def run_cycle(self):
        """Un ciclo de trading"""
        prices = self.get_prices()
        short_signals, long_signals = self.calculate_signals(prices)
        
        print(f"Precios: {prices}")
        print(f"SHORT signals: {[s['symbol'] for s in short_signals]}")
        print(f"LONG signals: {[s['symbol'] for s in long_signals]}")
        
        # Aquí implementar ejecución de trades...
    
    def load_state(self):
        """Carga estado desde archivo"""
        try:
            with open("state.json") as f:
                return json.load(f)
        except:
            return {"capital_usd": 500, "positions": {}, "trades": []}
    
    def save_state(self):
        """Guarda estado a archivo"""
        with open("state.json", "w") as f:
            json.dump(self.state, f, indent=2)

# Ejecutar
bot = TradingBot()
while True:
    bot.run_cycle()
    time.sleep(60)  # Cada minuto
```

---

## Notas Importantes

1. **Paper Trading**: Actualmente todo es simulado. Para dinero real, necesitas conectar wallet Solana.

2. **Drift API**: Para ejecución real, usa el SDK oficial:
   ```bash
   pip install drift-sdk
   ```

3. **Tasas de Funding**: Losperp et al tienen tasas de funding cada 1 hora. Considerar esto en estrategias largas.

4. **Liquidación**: Con leverage 2x y SL 1%, hay riesgo de liquidación si el precio move contra ti por ~50%.

5. **Deslizamiento (Slippage)**: Enperp et al puede haber deslizamiento. Considerar 0.5-1% adicional en SL.

---

## Referencias

- **Jupiter API**: https://docs.jup.ag
- **CoinGecko API**: https://www.coingecko.com/en/api
- **DEX Screener**: https://docs.dexscreener.com
- **Drift Protocol**: https://docs.drift.trade

---

*Documento generado para implementación de SHORT trading con datos reales y Drift Protocol.*
