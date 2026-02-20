#!/usr/bin/env python3
"""
Unified Trading System - Advanced Features Integration
========================================================
Módulo que integra las 3 mejoras avanzadas al sistema de trading:

1. WebSocket Real-Time Prices - Precios en tiempo real
2. Jito Bundles - Transacciones priorizadas con protección MEV
3. Monte Carlo Analysis - Análisis probabilístico

Cada feature puede habilitarse/deshabilitarse desde config.

Uso:
    from unified_advanced_trading import AdvancedTradingSystem
    
    system = AdvancedTradingSystem(config)
    
    # Iniciar
    await system.start()
    
    # Obtener precio
    price = system.get_price("SOL/USDT")
    
    # Ejecutar trade con Jito
    result = await system.execute_trade_with_jito(tx, priority=True)
    
    # Análisis Monte Carlo
    analysis = await system.run_monte_carlo_analysis()
    
    # Detener
    await system.stop()
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger("advanced_trading")

from pathlib import Path


# ============================================================================
# IMPORTS
# ============================================================================

# Importar los módulos creados
try:
    from websocket_price_client import (
        WebSocketPriceClient,
        WebSocketConfig as WSConfig,
        PriceData,
        WebSocketManager
    )
    WS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ WebSocket module not available: {e}")
    WS_AVAILABLE = False
    WebSocketPriceClient = None

try:
    from jito_bundle_client import (
        JitoBundleClient,
        JitoConfig as JitoModuleConfig,
        BundleResult
    )
    JITO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Jito module not available: {e}")
    JITO_AVAILABLE = False
    JitoBundleClient = None

try:
    from monte_carlo_integration import (
        MonteCarloAnalyzer,
        MonteCarloConfig as MCModuleConfig
    )
    MC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Monte Carlo module not available: {e}")
    MC_AVAILABLE = False
    MonteCarloAnalyzer = None


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class AdvancedTradingSystem:
    """
    Sistema de trading avanzado que integra:
    - WebSocket para precios en tiempo real
    - Jito para transacciones priorizadas
    - Monte Carlo para análisis probabilístico
    """
    
    def __init__(self, config=None):
        """
        Inicializar el sistema avanzado.
        
        Args:
            config: Objeto de configuración (de config/config.py)
                   Si es None, usa configuración por defecto
        """
        self.config = config
        self.is_running = False
        
        # Componentes
        self.ws_client: Optional[WebSocketPriceClient] = None
        self.ws_manager: Optional[WebSocketManager] = None
        self.jito_client: Optional[JitoBundleClient] = None
        self.mc_analyzer: Optional[MonteCarloAnalyzer] = None
        
        # Callbacks
        self.trade_callbacks: List[Callable] = []
        self.price_callbacks: List[Callable[[PriceData], None]] = []
        
        # Estado
        self.started_at: Optional[datetime] = None
        
        # Inicializar componentes
        self._init_components()
        
        logger.info("🚀 Advanced Trading System initialized")
    
    def _init_components(self):
        """Inicializar componentes basados en configuración"""
        
        # 1. WebSocket
        if WS_AVAILABLE and self.config:
            ws_enabled = getattr(self.config.websocket, 'enabled', False)
            if ws_enabled:
                try:
                    ws_cfg = WSConfig(
                        enabled=True,
                        provider=self.config.websocket.provider,
                        price_buffer_size=self.config.websocket.price_buffer_size,
                        reconnect_delay=self.config.websocket.reconnect_delay
                    )
                    self.ws_client = WebSocketPriceClient(ws_cfg)
                    logger.info("✅ WebSocket client initialized")
                except Exception as e:
                    logger.error(f"❌ WebSocket init failed: {e}")
        
        # 2. Jito
        if JITO_AVAILABLE and self.config:
            jito_enabled = getattr(self.config.jito, 'enabled', False)
            if jito_enabled:
                try:
                    jito_cfg = JitoModuleConfig(
                        enabled=True,
                        auth_key=self.config.jito.auth_key,
                        tip_amount=self.config.jito.tip_amount,
                        tip_amount_max=self.config.jito.tip_amount_max,
                        fallback_to_regular=self.config.jito.fallback_to_regular
                    )
                    self.jito_client = JitoBundleClient(jito_cfg)
                    logger.info("✅ Jito client initialized")
                except Exception as e:
                    logger.error(f"❌ Jito init failed: {e}")
        
        # 3. Monte Carlo
        if MC_AVAILABLE and self.config:
            mc_enabled = getattr(self.config.monte_carlo, 'enabled', False)
            if mc_enabled:
                try:
                    mc_cfg = MCModuleConfig(
                        enabled=True,
                        initial_balance=self.config.monte_carlo.initial_balance,
                        num_simulations=self.config.monte_carlo.num_simulations,
                        run_on_trade=self.config.monte_carlo.run_on_trade,
                        run_on_schedule=self.config.monte_carlo.run_on_schedule
                    )
                    self.mc_analyzer = MonteCarloAnalyzer(mc_cfg)
                    logger.info("✅ Monte Carlo analyzer initialized")
                except Exception as e:
                    logger.error(f"❌ Monte Carlo init failed: {e}")
    
    async def start(self):
        """Iniciar el sistema avanzado"""
        if self.is_running:
            logger.warning("⚠️ System already running")
            return
        
        self.started_at = datetime.now()
        
        # Iniciar WebSocket
        if self.ws_client:
            try:
                asyncio.create_task(self.ws_client.listen())
                logger.info("✅ WebSocket listening started")
            except Exception as e:
                logger.error(f"❌ WebSocket start failed: {e}")
        
        # Nota: Jito y MC no requieren start activo
        
        self.is_running = True
        logger.info("🚀 Advanced Trading System started")
    
    async def stop(self):
        """Detener el sistema"""
        if not self.is_running:
            return
        
        # Detener WebSocket
        if self.ws_client:
            await self.ws_client.disconnect()
        
        # Cerrar Jito
        if self.jito_client:
            await self.jito_client.close()
        
        self.is_running = False
        logger.info("👋 Advanced Trading System stopped")
    
    # =========================================================================
    # WEBSOCKET METHODS
    # =========================================================================
    
    def get_price(self, symbol: str = "SOL/USDT") -> Optional[float]:
        """Obtener precio actual (requiere WebSocket activo)"""
        if self.ws_client:
            return self.ws_client.get_price(symbol)
        return None
    
    def get_price_data(self, symbol: str = "SOL/USDT") -> Optional[PriceData]:
        """Obtener datos completos del precio"""
        if self.ws_client:
            return self.ws_client.get_price_data(symbol)
        return None
    
    def get_price_history(self, symbol: str = "SOL/USDT", count: int = 10) -> List[PriceData]:
        """Obtener historial de precios del buffer"""
        if self.ws_client:
            return self.ws_client.get_price_history(symbol, count)
        return []
    
    def calculate_vwap(self, symbol: str = "SOL/USDT") -> Optional[float]:
        """Calcular VWAP"""
        if self.ws_client:
            return self.ws_client.calculate_vwap(symbol)
        return None
    
    def calculate_volatility(self, symbol: str = "SOL/USDT") -> Optional[float]:
        """Calcular volatilidad"""
        if self.ws_client:
            return self.ws_client.calculate_volatility(symbol)
        return None
    
    def on_price_update(self, callback: Callable[[PriceData], None]):
        """Registrar callback para actualizaciones de precio"""
        if self.ws_client:
            self.ws_client.on_price_update(callback)
        self.price_callbacks.append(callback)
    
    def get_ws_status(self) -> Dict:
        """Obtener estado del WebSocket"""
        if self.ws_client:
            return self.ws_client.get_status()
        return {"enabled": False, "connected": False}
    
    # =========================================================================
    # JITO METHODS
    # =========================================================================
    
    async def send_bundle(
        self,
        transactions: List[str],
        tip_amount: int = None,
        wait_confirmation: bool = True
    ) -> BundleResult:
        """
        Enviar bundle de transacciones via Jito.
        
        Args:
            transactions: Lista de transacciones en base64
            tip_amount: Tip en lamports
            wait_confirmation: Esperar confirmación
            
        Returns:
            BundleResult
        """
        if not self.jito_client:
            return BundleResult(
                bundle_id="",
                success=False,
                error="Jito client not initialized"
            )
        
        return await self.jito_client.send_bundle(
            transactions,
            tip_amount,
            wait_confirmation
        )
    
    async def execute_with_jito(
        self,
        transaction: str,
        priority: bool = False
    ) -> Dict:
        """
        Ejecutar transacción con Jito (priorizada).
        
        Args:
            transaction: Transacción en base64
            priority: Si True, usa Jito para priorizar
            
        Returns:
            Dict con resultado
        """
        if not priority or not self.jito_client:
            # Transacción normal
            return {
                "success": True,
                "method": "regular",
                "transaction": transaction
            }
        
        # Con Jito
        result = await self.jito_client.send_bundle(
            [transaction],
            wait_confirmation=False
        )
        
        return {
            "success": result.success,
            "method": "jito",
            "bundle_id": result.bundle_id,
            "error": result.error
        }
    
    def estimate_priority_fee(self, desired_priority: float = 0.5) -> int:
        """Estimar fee de prioridad"""
        if self.jito_client:
            return self.jito_client.config.tip_amount
        return 1000  # Default
    
    def get_jito_status(self) -> Dict:
        """Obtener estado de Jito"""
        if self.jito_client:
            return self.jito_client.get_status()
        return {"enabled": False, "bundles_sent": 0}
    
    # =========================================================================
    # MONTE CARLO METHODS
    # =========================================================================
    
    def add_trade(self, pnl_percent: float):
        """Agregar trade al historial para análisis"""
        if self.mc_analyzer:
            self.mc_analyzer.add_trade(pnl_percent)
    
    def add_trades(self, trades: List[float]):
        """Agregar múltiples trades"""
        if self.mc_analyzer:
            self.mc_analyzer.add_trades(trades)
    
    async def analyze(self, method: str = "bootstrap") -> Dict:
        """Ejecutar análisis Monte Carlo"""
        if not self.mc_analyzer:
            return {"error": "Monte Carlo not enabled"}
        
        return await self.mc_analyzer.analyze(method)
    
    def get_recommendation(self, analysis: Dict = None) -> Dict:
        """Obtener recomendación de trading"""
        if not self.mc_analyzer:
            return {"recommendation": "UNAVAILABLE", "reason": "MC not enabled"}
        
        return self.mc_analyzer.get_recommendation(analysis)
    
    def get_mc_summary(self) -> Dict:
        """Obtener resumen de Monte Carlo"""
        if self.mc_analyzer:
            return self.mc_analyzer.get_summary()
        return {"enabled": False, "total_trades": 0}
    
    # =========================================================================
    # INTEGRATED METHODS
    # =========================================================================
    
    async def execute_trade_workflow(
        self,
        trade_command: Dict,
        use_jito: bool = False,
        priority: bool = False
    ) -> Dict:
        """
        Ejecutar workflow completo de trade con features avanzados.
        
        Args:
            trade_command: Comando de trade
            use_jito: Usar Jito para el trade
            priority: Priorizar transacción
            
        Returns:
            Dict con resultado completo
        """
        result = {
            "command": trade_command,
            "timestamp": datetime.now().isoformat(),
            "features": {}
        }
        
        # 1. Obtener precio en tiempo real (si disponible)
        if self.ws_client:
            symbol = f"{trade_command.get('from', 'SOL')}/{trade_command.get('quote', 'USDC')}"
            price_data = self.get_price_data(symbol)
            if price_data:
                result["features"]["price"] = price_data.to_dict()
                result["features"]["price"]["vwap"] = self.calculate_vwap(symbol)
                result["features"]["price"]["volatility"] = self.calculate_volatility(symbol)
        
        # 2. Ejecutar trade
        if trade_command.get("dry_run"):
            result["status"] = "dry_run"
        else:
            # Aquí iría la ejecución real con Jupiter
            # Por ahora simulamos
            result["status"] = "simulated"
            result["features"]["execution"] = {
                "method": "jito" if (use_jito and priority) else "regular"
            }
        
        # 3. Análisis post-trade
        if self.mc_analyzer and trade_command.get("record_trade"):
            pnl = trade_command.get("pnl", 0)
            if pnl != 0:
                self.add_trade(pnl)
                
                # Ejecutar análisis si hay suficientes trades
                if len(self.mc_analyzer.trade_history) >= 10:
                    analysis = await self.analyze()
                    result["features"]["monte_carlo"] = analysis
                    result["features"]["recommendation"] = self.get_recommendation(analysis)
        
        return result
    
    def get_system_status(self) -> Dict:
        """Obtener estado completo del sistema"""
        return {
            "running": self.is_running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "websocket": self.get_ws_status(),
            "jito": self.get_jito_status(),
            "monte_carlo": self.get_mc_summary(),
            "modules": {
                "websocket_available": WS_AVAILABLE,
                "jito_available": JITO_AVAILABLE,
                "monte_carlo_available": MC_AVAILABLE
            }
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_advanced_system(config=None) -> AdvancedTradingSystem:
    """Crear sistema avanzado con configuración"""
    return AdvancedTradingSystem(config)


async def create_from_config_path(config_path: str = None) -> AdvancedTradingSystem:
    """Crear sistema desde archivo de configuración"""
    config = None
    
    if config_path and Path(config_path).exists():
        try:
            from config.config import get_config
            config = get_config(config_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not load config: {e}")
    
    return AdvancedTradingSystem(config)


# ============================================================================
# MAIN TEST
# ============================================================================

async def test_advanced_system():
    """Test del sistema avanzado"""
    print("=" * 70)
    print("🚀 ADVANCED TRADING SYSTEM TEST")
    print("=" * 70)
    
    # Create system (without config - uses defaults)
    system = AdvancedTradingSystem()
    
    # Show status
    print("\n📊 Initial Status:")
    status = system.get_system_status()
    print(f"   Running: {status['running']}")
    print(f"   Modules Available:")
    for mod, available in status['modules'].items():
        print(f"      {mod}: {'✅' if available else '❌'}")
    
    # Start system
    print("\n▶️  Starting system...")
    await system.start()
    
    # Show started status
    print("\n📊 Started Status:")
    status = system.get_system_status()
    print(f"   Running: {status['running']}")
    print(f"   WebSocket: {status['websocket'].get('connected', False)}")
    
    # Wait for some price data
    print("\n⏳ Waiting for price data (5s)...")
    await asyncio.sleep(5)
    
    # Check price
    price = system.get_price("SOL/USDT")
    print(f"\n💰 Current SOL price: ${price:.4f}" if price else "\n❌ No price yet")
    
    # Show all statuses
    print("\n📊 WebSocket Status:")
    ws_status = system.get_ws_status()
    for key, value in ws_status.items():
        print(f"   {key}: {value}")
    
    print("\n📊 Jito Status:")
    jito_status = system.get_jito_status()
    for key, value in jito_status.items():
        print(f"   {key}: {value}")
    
    print("\n📊 Monte Carlo Status:")
    mc_status = system.get_mc_summary()
    for key, value in mc_status.items():
        print(f"   {key}: {value}")
    
    # Test Monte Carlo
    print("\n🎲 Testing Monte Carlo...")
    system.add_trades([2.5, -1.2, 3.1, -0.8, 1.5, 4.2, -2.0, 0.5, 1.8, -0.3])
    analysis = await system.analyze()
    if "error" not in analysis:
        print(f"   Mean Balance: ${analysis.get('mean_balance', 0):,.2f}")
        print(f"   P95 Balance: ${analysis.get('percentile_95', 0):,.2f}")
    
    # Get recommendation
    rec = system.get_recommendation(analysis)
    print(f"\n🎯 Recommendation: {rec.get('recommendation')}")
    print(f"   Action: {rec.get('action')}")
    
    # Stop system
    print("\n⬛ Stopping system...")
    await system.stop()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_advanced_system())
