# Sistema Existente: Enlace Simbólico

Este directorio es un enlace simbólico al proyecto base:
`/home/enderj/Documents/Coinbase Cripto Trader Claude/Coinbase Cripto Trader Claude`

## Archivos Clave Disponibles

### Core Trading
| Archivo | Función | Tamaño |
|---------|---------|--------|
| `coordinator_port5001.py` | Flask REST API + SQLite | 36 KB |
| `crypto_worker.py` | Worker distribuido | 13 KB |
| `strategy_miner.py` | Algoritmo genético | 19 KB |
| `numba_backtester.py` | Backtester JIT (4000x) | 12 KB |
| `interface.py` | Streamlit Dashboard | 27 KB |
| `telegram_bot.py` | Telegram Monitor | 6 KB |

### Datos
| Archivo | Función |
|---------|---------|
| `requirements.txt` | Dependencias Python |
| `config.py` | Configuraciones |
| `.env` | Variables de entorno |

## Uso

Los archivos del sistema existente están disponibles directamente:
```bash
# Desde el proyecto Solana
ls -la sistema_existente/*.py

# Usar como referencia
python sistema_existente/coordinator_port5001.py
```

## Adaptación Planificada

1. **Copiar** archivos necesarios
2. **Modificar** para Jupiter API
3. **Testear** en testnet
4. **Deploy** en mainnet

