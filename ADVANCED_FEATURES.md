# Advanced Trading Features - README

## Resumen de Implementación

Se han implementado 3 mejoras modulares para el sistema de trading Solana:

### 1. WebSocket Real-Time (`websocket_price_client.py`)
- Cliente WebSocket para precios en tiempo real (Binance)
- Buffer de precios para análisis técnico
- VWAP y volatilidad en tiempo real
- Reconnection automática
- **Flag**: `config.websocket.enabled`

### 2. Jito Bundles (`jito_bundle_client.py`)
- Integración con Jito Block Engine
- Transacciones priorizadas con tips
- Protección MEV
- Bundle de múltiples transacciones
- **Flag**: `config.jito.enabled`

### 3. Monte Carlo (`monte_carlo_integration.py`)
- Análisis probabilístico de estrategias
- Risk of Ruin calculator
- Simulaciones bootstrap
- Recomendaciones de trading
- **Flag**: `config.monte_carlo.enabled`

## Uso

```python
from config.config import Config, get_config
from unified_advanced_trading import AdvancedTradingSystem

# Cargar configuración
config = get_config()

# Habilitar features (en config/config.py o .env)
config.websocket.enabled = True
config.jito.enabled = True
config.jito.auth_key = "tu_jito_auth_key"
config.monte_carlo.enabled = True

# Crear sistema
system = AdvancedTradingSystem(config)

# Iniciar
await system.start()

# Obtener precio
price = system.get_price("SOL/USDT")

# Ejecutar trade con Jito
result = await system.execute_with_jito(tx, priority=True)

# Análisis Monte Carlo
system.add_trade(2.5)  # 2.5% profit
analysis = await system.analyze()
recommendation = system.get_recommendation(analysis)

# Detener
await system.stop()
```

## Archivos Creados

| Archivo | Descripción |
|---------|-------------|
| `websocket_price_client.py` | Cliente WebSocket para precios |
| `jito_bundle_client.py` | Integración Jito Bundles |
| `monte_carlo_integration.py` | Wrapper Monte Carlo |
| `unified_advanced_trading.py` | Sistema integrado |

## Configuración

Los flags están en `config/config.py`:

```python
@dataclass
class WebSocketConfig:
    enabled: bool = False
    provider: str = "binance"

@dataclass  
class JitoConfig:
    enabled: bool = False
    auth_key: str = ""
    tip_amount: int = 1000

@dataclass
class MonteCarloConfig:
    enabled: bool = False
    num_simulations: int = 10000
```
