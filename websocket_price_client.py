#!/usr/bin/env python3
"""
WebSocket Price Client for Real-Time Trading
=============================================
Cliente WebSocket para obtener precios en tiempo real de Jupiter/Solana.
Proveé menor latencia que las APIs REST tradicionales.

Características:
- Conexión WebSocket a Binance (precios SOL/USDC)
- Reconnection automática
- Callbacks para actualizaciones de precio
- Buffer de precios para análisis técnico
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import websockets
import hashlib
import hmac
import time

logger = logging.getLogger("websocket_client")


@dataclass
class PriceData:
    """Datos de precio en tiempo real"""
    symbol: str
    price: float
    volume_24h: float = 0.0
    change_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "change_24h": self.change_24h,
            "bid": self.bid,
            "ask": self.ask,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class WebSocketConfig:
    """Configuración del WebSocket"""
    # Proveedores
    provider: str = "binance"  # binance, jupiter
    
    # Binance WebSocket
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"
    binance_stream: str = "solusdt@trade@0.1s"  # SOL/USDT tickers
    
    # Jupiter WebSocket (para cuando esté disponible)
    jupiter_ws_url: str = "wss://api.jup.ag/ws"
    
    # Reconnection
    reconnect_delay: int = 5  # segundos
    max_reconnect_attempts: int = 10
    ping_interval: int = 30  # segundos
    
    # Buffer
    price_buffer_size: int = 100
    
    # Habilitar
    enabled: bool = False


class WebSocketPriceClient:
    """
    Cliente WebSocket para precios en tiempo real.
    
    Uso:
        client = WebSocketPriceClient(config)
        client.start()
        
        # Obtener precio actual
        price("SOL/USDC")
        
        # = client.get_price Suscribirse a callbacks
        client.on_price_update(callback_function)
    """
    
    def __init__(self, config: WebSocketConfig = None):
        self.config = config or WebSocketConfig()
        self.websocket = None
        self.is_connected = False
        self.should_reconnect = True
        
        # Price buffer (para análisis técnico)
        self.price_buffer: deque = deque(maxlen=self.config.price_buffer_size)
        self.latest_prices: Dict[str, PriceData] = {}
        
        # Callbacks
        self.price_callbacks: List[Callable[[PriceData], None]] = []
        
        # Metrics
        self.messages_received = 0
        self.last_update = None
        
        logger.info(f"📡 WebSocket Client initialized (provider: {self.config.provider})")
    
    def on_price_update(self, callback: Callable[[PriceData], None]):
        """Registrar callback para actualizaciones de precio"""
        self.price_callbacks.append(callback)
    
    async def connect(self) -> bool:
        """Conectar al WebSocket"""
        try:
            if self.config.provider == "binance":
                url = f"{self.config.binance_ws_url}/{self.config.binance_stream}"
            else:
                url = self.config.jupiter_ws_url
            
            logger.info(f"🔌 Connecting to {url}")
            self.websocket = await websockets.connect(url, ping_interval=self.config.ping_interval)
            self.is_connected = True
            logger.info("✅ WebSocket connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Desconectar del WebSocket"""
        self.should_reconnect = False
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("👋 WebSocket disconnected")
    
    async def listen(self):
        """Escuchar mensajes del WebSocket"""
        reconnect_attempts = 0
        
        while self.should_reconnect:
            try:
                if not self.is_connected:
                    success = await self.connect()
                    if not success:
                        reconnect_attempts += 1
                        if reconnect_attempts >= self.config.max_reconnect_attempts:
                            logger.error("❌ Max reconnection attempts reached")
                            break
                        await asyncio.sleep(self.config.reconnect_delay)
                        continue
                    reconnect_attempts = 0
                
                # Receive message
                message = await self.websocket.recv()
                self.messages_received += 1
                self.last_update = datetime.now()
                
                # Parse message
                price_data = self._parse_message(message)
                if price_data:
                    # Update buffers
                    self.price_buffer.append(price_data)
                    self.latest_prices[price_data.symbol] = price_data
                    
                    # Call callbacks
                    for callback in self.price_callbacks:
                        try:
                            callback(price_data)
                        except Exception as e:
                            logger.error(f"❌ Callback error: {e}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("⚠️ Connection closed, reconnecting...")
                self.is_connected = False
            except Exception as e:
                logger.error(f"❌ Listen error: {e}")
                await asyncio.sleep(1)
    
    def _parse_message(self, message: str) -> Optional[PriceData]:
        """Parsear mensaje del WebSocket"""
        try:
            data = json.loads(message)
            
            if self.config.provider == "binance":
                # Binance trade message
                return PriceData(
                    symbol="SOL/USDT",
                    price=float(data.get("p", 0)),
                    volume_24h=float(data.get("q", 0)),
                    timestamp=datetime.now()
                )
            elif self.config.provider == "jupiter":
                # Jupiter message format (when available)
                return PriceData(
                    symbol=data.get("symbol", "SOL/USDC"),
                    price=float(data.get("price", 0)),
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"❌ Parse error: {e}")
        
        return None
    
    def get_price(self, symbol: str = "SOL/USDT") -> Optional[float]:
        """Obtener precio actual de un símbolo"""
        if symbol in self.latest_prices:
            return self.latest_prices[symbol].price
        return None
    
    def get_price_data(self, symbol: str = "SOL/USDT") -> Optional[PriceData]:
        """Obtener datos completos del precio"""
        return self.latest_prices.get(symbol)
    
    def get_price_history(self, symbol: str = "SOL/USDT", count: int = 10) -> List[PriceData]:
        """Obtener historial de precios del buffer"""
        history = [p for p in self.price_buffer if p.symbol == symbol]
        return history[-count:] if len(history) > count else history
    
    def calculate_vwap(self, symbol: str = "SOL/USDT") -> Optional[float]:
        """Calcular VWAP (Volume Weighted Average Price) del buffer"""
        history = [p for p in self.price_buffer if p.symbol == symbol]
        if not history:
            return None
        
        total_volume = sum(p.volume_24h for p in history)
        if total_volume == 0:
            return history[-1].price
        
        weighted_sum = sum(p.price * p.volume_24h for p in history)
        return weighted_sum / total_volume
    
    def calculate_volatility(self, symbol: str = "SOL/USDT", window: int = 20) -> Optional[float]:
        """Calcular volatilidad (desviación estándar) del precio"""
        import numpy as np
        history = [p.price for p in self.price_buffer if p.symbol == symbol][-window:]
        
        if len(history) < 2:
            return None
        
        return float(np.std(history))
    
    def get_status(self) -> Dict:
        """Obtener estado del cliente WebSocket"""
        return {
            "connected": self.is_connected,
            "provider": self.config.provider,
            "messages_received": self.messages_received,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "buffer_size": len(self.price_buffer),
            "tracked_symbols": list(self.latest_prices.keys())
        }


class WebSocketManager:
    """
    Gestor de múltiples conexiones WebSocket.
    
    Uso:
        manager = WebSocketManager()
        manager.add_client("prices", ws_client)
        manager.start_all()
    """
    
    def __init__(self):
        self.clients: Dict[str, WebSocketPriceClient] = {}
        self.tasks: List[asyncio.Task] = []
    
    def add_client(self, name: str, client: WebSocketPriceClient):
        """Agregar cliente al manager"""
        self.clients[name] = client
    
    async def start_all(self):
        """Iniciar todos los clientes"""
        for name, client in self.clients.items():
            task = asyncio.create_task(client.listen())
            self.tasks.append(task)
            logger.info(f"✅ Started WebSocket client: {name}")
    
    async def stop_all(self):
        """Detener todos los clientes"""
        for client in self.clients.values():
            await client.disconnect()
        
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("✅ All WebSocket clients stopped")
    
    def get_client(self, name: str) -> Optional[WebSocketPriceClient]:
        """Obtener cliente por nombre"""
        return self.clients.get(name)


# ============================================================================
# CONFIGURATION INTEGRATION
# ============================================================================

def create_websocket_config(enabled: bool = False, **kwargs) -> WebSocketConfig:
    """Crear configuración del WebSocket desde kwargs"""
    return WebSocketConfig(
        enabled=enabled,
        provider=kwargs.get("provider", "binance"),
        reconnect_delay=kwargs.get("reconnect_delay", 5),
        price_buffer_size=kwargs.get("price_buffer_size", 100)
    )


# ============================================================================
# MAIN TEST
# ============================================================================

async def test_websocket():
    """Test del cliente WebSocket"""
    print("=" * 60)
    print("🧪 WebSocket Client Test")
    print("=" * 60)
    
    config = WebSocketConfig(
        enabled=True,
        provider="binance",
        price_buffer_size=50
    )
    
    client = WebSocketPriceClient(config)
    
    # Callback example
    def on_price(data: PriceData):
        print(f"📊 {data.symbol}: ${data.price:.4f}")
    
    client.on_price_update(on_price)
    
    # Start listening (will run in background)
    listen_task = asyncio.create_task(client.listen())
    
    # Wait for some data
    print("\n⏳ Waiting for price data...")
    await asyncio.sleep(10)
    
    # Show status
    print("\n📡 Client Status:")
    status = client.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Get current price
    price = client.get_price("SOL/USDT")
    print(f"\n💰 Current SOL price: ${price:.4f}" if price else "\n❌ No price data")
    
    # Stop
    await client.disconnect()
    listen_task.cancel()
    
    print("\n✅ Test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_websocket())
