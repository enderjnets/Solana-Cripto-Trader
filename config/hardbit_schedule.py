#!/usr/bin/env python3
"""
HARDBIT NIGHT SCHEDULE - Solana Jupiter Trading Bot
=================================================
Configuración estricta para horario nocturno de trading.

HORARIO: 23:00 - 06:00 MST (UTC-7)
RIESGO: HIGH (más agresivo)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, time


# ============ HORARIO HARDBIT ============
HARDBIT_NIGHT_START = time(23, 0)    # 11:00 PM MST
HARDBIT_NIGHT_END = time(5, 0)       # 5:00 AM MST
TIMEZONE = "America/Denver"         # MST


# ============ PERFILES DE RIESGO ============
@dataclass
class HardbitRiskProfile:
    """Perfil de riesgo hardbit para horario nocturno"""
    
    # Tamaño de posición
    max_position_pct: float = 0.10      # 10% del capital (conservative)
    
    # Stop Loss ajustado — ratio 2.75:1
    stop_loss_pct: float = 0.02         # 2% SL
    
    # Take Profit — ratio 2.75:1
    take_profit_pct: float = 0.055      # 5.5% TP
    
    # Trailing stop
    trailing_stop: bool = True          # Enable trailing stop
    trailing_stop_pct: float = 0.015    # 1.5% trailing distance
    
    # Límites diarios
    max_daily_loss_pct: float = 0.08    # 8% daily loss limit
    max_daily_trades: int = 10           # Máximo 10 trades/noche
    
    # Concurrent positions
    max_concurrent: int = 3             # Máximo 3 posiciones abiertas
    
    # Tiempo de espera entre trades
    cooldown_seconds: int = 30           # 30s entre trades


# ============ PERFIL DEFAULT (DÍA) ============
@dataclass
class DayTimeProfile:
    """Perfil diurno por defecto"""
    
    max_position_pct: float = 0.10      # 10% del capital
    stop_loss_pct: float = 0.025        # 2.5% SL — ratio 2.6:1
    take_profit_pct: float = 0.065      # 6.5% TP — ratio 2.6:1
    trailing_stop: bool = True          # Enable trailing stop
    trailing_stop_pct: float = 0.015    # 1.5% trailing distance
    max_daily_loss_pct: float = 0.10    # 10% daily loss limit
    max_daily_trades: int = 20           # Máximo 20 trades/día
    max_concurrent: int = 5               # Máximo 5 posiciones
    cooldown_seconds: int = 60            # 60s entre trades


# ============ CONFIGURACIÓN ACTIVA ============
HARDBIT_CONFIG = {
    "enabled": True,
    "night_start": "23:00 MST",   # 11 PM
    "night_end": "05:00 MST",    # 5 AM
    "timezone": "America/Denver",
    
    "night_profile": {
        "risk_level": "HIGH",
        "max_position_pct": 0.15,
        "stop_loss_pct": 0.02,           # 2% SL (was 1%)
        "take_profit_pct": 0.055,         # 5.5% TP → ratio 2.75:1
        "max_daily_loss_pct": 0.08,
        "max_daily_trades": 10,
        "max_concurrent_positions": 3,
        "cooldown_seconds": 30,
        "trailing_stop": True,            # Enable trailing stop
        "trailing_stop_pct": 0.015,       # 1.5% trailing distance
    },
    
    "day_profile": {
        "risk_level": "MEDIUM",
        "max_position_pct": 0.10,
        "stop_loss_pct": 0.025,           # 2.5% SL (was 1%)
        "take_profit_pct": 0.065,         # 6.5% TP → ratio 2.6:1
        "max_daily_loss_pct": 0.10,
        "max_daily_trades": 20,
        "max_concurrent_positions": 5,
        "cooldown_seconds": 60,
        "trailing_stop": True,            # Enable trailing stop
        "trailing_stop_pct": 0.015,       # 1.5% trailing distance
    },
    
    # Tokens favoritos para noche hardbit
    "night_tokens": [
        "SOL",
        "JUP", 
        "BONK",
        "WIF",
        "POPCAT",
    ],
    
    # Mercados a EVITAR durante noche
    "avoid_markets": [
        "illiquid_pairs",
        "new_listings",
    ],
}


def is_night_time() -> bool:
    """Verificar si es horario nocturno hardbit"""
    now = datetime.now()
    
    # Convertir a hora local
    local_hour = now.hour
    
    # Hardbit night: 23:00 - 05:00
    return local_hour >= 23 or local_hour < 5


def get_active_profile() -> Dict:
    """Obtener perfil activo según horario"""
    if is_night_time():
        return HARDBIT_CONFIG["night_profile"]
    return HARDBIT_CONFIG["day_profile"]


def print_hardbit_status():
    """Mostrar estado actual del horario hardbit"""
    import datetime
    
    now = datetime.datetime.now()
    is_night = is_night_time()
    
    print("\n" + "="*60)
    print("🦞 HARDBIT NIGHT SCHEDULE STATUS")
    print("="*60)
    print(f"📅 Fecha: {now.strftime('%Y-%m-%d')}")
    print(f"⏰ Hora: {now.strftime('%H:%M:%S')} MST")
    print(f"🌙 Modo: {'HARDBIT NIGHT' if is_night else 'DAY TRADING'}")
    print("-"*60)
    
    profile = get_active_profile()
    print(f"⚠️  Risk Level: {profile['risk_level']}")
    print(f"💰 Max Position: {profile['max_position_pct']*100}%")
    print(f"🛑 Stop Loss: {profile['stop_loss_pct']*100}%")
    print(f"🎯 Take Profit: {profile['take_profit_pct']*100}%")
    print(f"📊 Max Daily Trades: {profile['max_daily_trades']}")
    print(f"🔒 Max Concurrent: {profile['max_concurrent_positions']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_hardbit_status()
