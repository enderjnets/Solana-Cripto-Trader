#!/usr/bin/env python3
"""
Unified Trading System v3 - IMPROVED
=====================================
Complete trading system with ML signals, risk management, and Redis caching.

Architecture:
- Market Scanner (10 tokens: SOL, ETH, cbBTC, JUP, BONK, WIF, RAY, JTO + trending)
- ML Signal Generator (RSI 30% + EMA 25% + Momentum 25% + Trend 20%)
- Risk Agent (validates size, confidence, limits, R/R)
- Trader (Jito + Jupiter integration)
- Redis Cache Layer (PriceCache, TradeState, MarketData)
- SQLite DB (trades), Webhooks, Telegram, Dashboard

Features:
- HARDBIT Night Schedule (22:00-09:00 MST)
- Confidence-based position sizing
- Paper trading mode (default)

Usage:
    python3 unified_trading_system.py --start          # Start trading
    python3 unified_trading_system.py --status        # Show status
    python3 unified_trading_system.py --scan          # Scan market
    python3 unified_trading_system.py --test-signal   # Test signal
    python3 unified_trading_system.py --paper-reset   # Reset paper trading
"""

import sys
import os
import json
import logging
import asyncio
import argparse
import uuid
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import json
import time as time_module

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing modules
from config.config import Config
from config.hardbit_schedule import HARDBIT_CONFIG, is_night_time, get_active_profile
from agents.risk_agent import RiskAgent, RiskLimits
from agents.market_scanner_agent import MarketScannerAgent, Opportunity
from paper_trading_engine import PaperTradingEngine
from api.kraken_price import get_kraken_price
from api.price_feed import get_price_feed

# =============================================================================
# CONFIGURATION
# =============================================================================

# Logging setup
LOG_FILE = PROJECT_ROOT / "unified_trading_system.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("unified_trading_system")

# =============================================================================
# REDIS CACHE MANAGER (from original v3)
# =============================================================================

class RedisCacheManager:
    """
    Redis Cache Manager from original v3 design.
    
    Features:
    - PriceCache with TTL
    - TradeStateManager
    - MarketDataCache
    - File-based fallback
    """
    
    def __init__(self):
        self.redis_available = False
        self._init_redis()
        self.cache_dir = PROJECT_ROOT / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}. Using file fallback.")
            self.redis_available = False
    
    # ==================== PRICE CACHE ====================
    
    def set_price(self, symbol: str, price: float, ttl: int = 60):
        """Cache token price with TTL"""
        data = {"price": price, "timestamp": datetime.now().isoformat()}
        
        if self.redis_available:
            try:
                self.redis_client.setex(f"price:{symbol}", ttl, json.dumps(data))
                return
            except:
                pass
        
        # File fallback
        cache_file = self.cache_dir / f"price_{symbol}.json"
        cache_file.write_text(json.dumps(data))
    
    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get cached price"""
        if self.redis_available:
            try:
                data = self.redis_client.get(f"price:{symbol}")
                if data:
                    return json.loads(data)
            except:
                pass
        
        # File fallback
        cache_file = self.cache_dir / f"price_{symbol}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
    
    # ==================== TRADE STATE ====================
    
    def set_trade_state(self, trade_id: str, state: Dict, ttl: int = 3600):
        """Cache trade state"""
        data = {**state, "timestamp": datetime.now().isoformat()}
        
        if self.redis_available:
            try:
                self.redis_client.setex(f"trade:{trade_id}", ttl, json.dumps(data))
                return
            except:
                pass
        
        # File fallback
        cache_file = self.cache_dir / f"trade_{trade_id}.json"
        cache_file.write_text(json.dumps(data))
    
    def get_trade_state(self, trade_id: str) -> Optional[Dict]:
        """Get cached trade state"""
        if self.redis_available:
            try:
                data = self.redis_client.get(f"trade:{trade_id}")
                if data:
                    return json.loads(data)
            except:
                pass
        
        # File fallback
        cache_file = self.cache_dir / f"trade_{trade_id}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
    
    def delete_trade_state(self, trade_id: str):
        """Delete trade state"""
        if self.redis_available:
            try:
                self.redis_client.delete(f"trade:{trade_id}")
            except:
                pass
        
        cache_file = self.cache_dir / f"trade_{trade_id}.json"
        if cache_file.exists():
            cache_file.unlink()
    
    # ==================== MARKET DATA ====================
    
    def set_market_data(self, key: str, data: Dict, ttl: int = 300):
        """Cache market data"""
        cache_data = {**data, "timestamp": datetime.now().isoformat()}
        
        if self.redis_available:
            try:
                self.redis_client.setex(f"market:{key}", ttl, json.dumps(cache_data))
                return
            except:
                pass
        
        # File fallback
        cache_file = self.cache_dir / f"market_{key}.json"
        cache_file.write_text(json.dumps(cache_data))
    
    def get_market_data(self, key: str) -> Optional[Dict]:
        """Get cached market data"""
        if self.redis_available:
            try:
                data = self.redis_client.get(f"market:{key}")
                if data:
                    return json.loads(data)
            except:
                pass
        
        # File fallback
        cache_file = self.cache_dir / f"market_{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
    
    # ==================== SIGNAL CACHE ====================
    
    def set_signal(self, symbol: str, signal: Dict, ttl: int = 300):
        """Cache ML signal"""
        cache_data = {**signal, "timestamp": datetime.now().isoformat()}
        
        if self.redis_available:
            try:
                self.redis_client.setex(f"signal:{symbol}", ttl, json.dumps(cache_data))
                return
            except:
                pass
        
        cache_file = self.cache_dir / f"signal_{symbol}.json"
        cache_file.write_text(json.dumps(cache_data))
    
    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get cached ML signal"""
        if self.redis_available:
            try:
                data = self.redis_client.get(f"signal:{symbol}")
                if data:
                    return json.loads(data)
            except:
                pass
        
        cache_file = self.cache_dir / f"signal_{symbol}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
    
    # ==================== UTILS ====================
    
    def clear_expired(self):
        """Clear expired cache files"""
        import glob
        
        pattern = str(self.cache_dir / "*.json")
        for file in glob.glob(pattern):
            f = Path(file)
            try:
                data = json.loads(f.read_text())
                ts = datetime.fromisoformat(data.get("timestamp", "2020-01-01"))
                age = (datetime.now() - ts).total_seconds()
                if age > 600:  # 10 min old
                    f.unlink()
            except:
                pass
    
    def flush_all(self):
        """Clear all cache"""
        import glob
        
        pattern = str(self.cache_dir / "*.json")
        for file in glob.glob(pattern):
            Path(file).unlink()
        
        if self.redis_available:
            try:
                self.redis_client.flushdb()
            except:
                pass


# =============================================================================
# ML SIGNAL GENERATOR
# =============================================================================

class MLSignalGenerator:
    """
    ML Signal Generator with ensemble approach.
    
    Weights:
    - RSI (30%)
    - EMA Crossover (25%)
    - Momentum (25%)
    - Trend (20%)
    
    Returns confidence score (0-95%)
    """
    
    def __init__(self, cache_manager: RedisCacheManager):
        self.cache = cache_manager
        self.price_history: Dict[str, List[float]] = {}
        self.history_length = 50  # Keep 50 price points
    
    def update_price(self, symbol: str, price: float):
        """Update price history"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only last N prices
        if len(self.price_history[symbol]) > self.history_length:
            self.price_history[symbol] = self.price_history[symbol][-self.history_length:]
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index)
        Returns value 0-100
        """
        prices = self.price_history.get(symbol, [])
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        if len(changes) < period:
            return 50.0
        
        # Get last N changes
        changes = changes[-period:]
        
        # Separate gains and losses
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]
        
        # Calculate averages
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_ema(self, symbol: str, period: int) -> float:
        """Calculate Exponential Moving Average"""
        prices = self.price_history.get(symbol, [])
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        prices = prices[-period:]
        
        # EMA calculation
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_ema_crossover(self, symbol: str, fast: int = 9, slow: int = 21) -> float:
        """
        Calculate EMA Crossover signal
        Returns: -1 (bearish) to +1 (bullish)
        """
        ema_fast = self.calculate_ema(symbol, fast)
        ema_slow = self.calculate_ema(symbol, slow)
        
        if ema_fast == 0 or ema_slow == 0:
            return 0
        
        # Normalize: (fast - slow) / slow
        crossover = (ema_fast - ema_slow) / ema_slow
        
        # Clamp to -1 to 1
        return max(-1, min(1, crossover * 10))
    
    def calculate_momentum(self, symbol: str, period: int = 10) -> float:
        """
        Calculate Momentum
        Returns: -1 to +1
        """
        prices = self.price_history.get(symbol, [])
        if len(prices) < period + 1:
            return 0
        
        current = prices[-1]
        past = prices[-period-1]
        
        if past == 0:
            return 0
        
        momentum = (current - past) / past
        
        # Clamp to -1 to 1
        return max(-1, min(1, momentum * 5))
    
    def calculate_trend(self, symbol: str) -> float:
        """
        Calculate 24h Trend
        Returns: -1 (bearish) to +1 (bullish)
        """
        prices = self.price_history.get(symbol, [])
        
        if len(prices) < 24:  # Assuming hourly data
            return 0
        
        current = prices[-1]
        day_ago = prices[0]
        
        if day_ago == 0:
            return 0
        
        change_pct = (current - day_ago) / day_ago
        
        # Clamp to -1 to 1
        return max(-1, min(1, change_pct))
    
    def generate_signal(self, symbol: str) -> Dict:
        """
        Generate ensemble ML signal
        
        Weights:
        - RSI: 30%
        - EMA: 25%
        - Momentum: 25%
        - Trend: 20%
        
        Returns:
            Dict with signal components and confidence (0-95%)
        """
        # Get individual signals
        rsi = self.calculate_rsi(symbol)  # 0-100
        ema_crossover = self.calculate_ema_crossover(symbol)  # -1 to 1
        momentum = self.calculate_momentum(symbol)  # -1 to 1
        trend = self.calculate_trend(symbol)  # -1 to 1
        
        # Add market volatility simulation (for paper trading)
        # This simulates real market noise
        import time
        hash_val = hash(f"{symbol}{int(time.time() / 60)}")  # Changes every minute
        noise = ((hash_val % 100) - 50) / 100  # -0.5 to 0.5
        
        # Apply noise to RSI (bounded 20-80 to not break indicator logic)
        rsi = max(20, min(80, rsi + noise * 30))
        
        # Convert RSI to -1 to 1 scale (50 = neutral)
        rsi_signal = (rsi - 50) / 50  # -1 to 1
        
        # Add noise to momentum/trend
        ema_crossover = max(-1, min(1, ema_crossover + noise * 0.3))
        momentum = max(-1, min(1, momentum + noise * 0.3))
        trend = max(-1, min(1, trend + noise * 0.3))
        
        # Weighted ensemble
        # RSI: 30%, EMA: 25%, Momentum: 25%, Trend: 20%
        ensemble = (
            rsi_signal * 0.30 +
            ema_crossover * 0.25 +
            momentum * 0.25 +
            trend * 0.20
        )
        
        # Convert to confidence (0-95%)
        # ensemble is -1 to 1, confidence is absolute value scaled
        raw_confidence = abs(ensemble) * 95
        confidence = min(95, max(0, raw_confidence))
        
        # Determine direction (lowered threshold from 0.1 to 0.05 for more signals)
        direction = "bullish" if ensemble > 0.05 else "bearish" if ensemble < -0.05 else "neutral"
        
        signal = {
            "symbol": symbol,
            "direction": direction,
            "confidence": round(confidence, 1),
            "ensemble_score": round(ensemble, 4),
            "components": {
                "rsi": {
                    "value": round(rsi, 2),
                    "signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
                    "weight": 0.30
                },
                "ema_crossover": {
                    "fast_ema": round(self.calculate_ema(symbol, 9), 4),
                    "slow_ema": round(self.calculate_ema(symbol, 21), 4),
                    "signal": "bullish" if ema_crossover > 0 else "bearish" if ema_crossover < 0 else "neutral",
                    "weight": 0.25
                },
                "momentum": {
                    "value": round(momentum, 4),
                    "signal": "strong" if abs(momentum) > 0.5 else "weak",
                    "weight": 0.25
                },
                "trend": {
                    "value": round(trend, 4),
                    "signal": "bullish" if trend > 0 else "bearish",
                    "weight": 0.20
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the signal
        self.cache.set_signal(symbol, signal)
        
        logger.info(f"📊 ML Signal for {symbol}: {direction} ({confidence:.1f}% confidence)")
        
        return signal


# =============================================================================
# SQLITE TRADES DATABASE
# =============================================================================

class TradesDatabase:
    """SQLite database for storing trades"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = PROJECT_ROOT / "data" / "trades.db"
        
        self.db_path = str(db_path)
        self._conn = None  # Persistent connection
        self._init_db()
    
    def _get_conn(self):
        """Get or create persistent connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn
    
    def _init_db(self):
        """Initialize database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                pnl_pct REAL,
                status TEXT,
                reason TEXT,
                confidence REAL,
                entry_time TEXT,
                exit_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
    
    def add_trade(self, trade: Dict):
        """Add a trade"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trades 
            (id, symbol, direction, entry_price, exit_price, size, pnl, pnl_pct, 
             status, reason, confidence, entry_time, exit_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get("id"),
            trade.get("symbol"),
            trade.get("direction"),
            trade.get("entry_price"),
            trade.get("exit_price"),
            trade.get("size"),
            trade.get("pnl", 0),
            trade.get("pnl_pct", 0),
            trade.get("status", "open"),
            trade.get("reason"),
            trade.get("confidence"),
            trade.get("entry_time"),
            trade.get("exit_time")
        ))
        
        conn.commit()
    
    def get_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        
        if not rows:
            return []
        
        columns = ["id", "symbol", "direction", "entry_price", "exit_price", "size", 
                   "pnl", "pnl_pct", "status", "reason", "confidence", "entry_time", 
                   "exit_time", "created_at"]
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_open_trades(self) -> List[Dict]:
        """Get open trades"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        rows = cursor.fetchall()
        
        
        if not rows:
            return []
        
        columns = ["id", "symbol", "direction", "entry_price", "exit_price", "size", 
                   "pnl", "pnl_pct", "status", "reason", "confidence", "entry_time", 
                   "exit_time", "created_at"]
        
        return [dict(zip(columns, row)) for row in rows]
    
    def update_trade_pnl(self, trade_id: str, current_price: float):
        """Update trade P&L"""
        trade = self.get_trade_by_id(trade_id)
        if not trade:
            return
        
        if trade["direction"] == "long":
            pnl_pct = (current_price - trade["entry_price"]) / trade["entry_price"]
        else:
            pnl_pct = (trade["entry_price"] - current_price) / trade["entry_price"]
        
        pnl = trade["size"] * pnl_pct
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE trades SET pnl = ?, pnl_pct = ? WHERE id = ?", 
                      (pnl, pnl_pct * 100, trade_id))
        
        conn.commit()
    
    def close_trade(self, trade_id: str, exit_price: float, reason: str = "Close"):
        """Close a trade"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        columns = ["id", "symbol", "direction", "entry_price", "exit_price", "size", 
                   "pnl", "pnl_pct", "status", "reason", "confidence", "entry_time", 
                   "exit_time", "created_at"]
        trade = dict(zip(columns, row))
        
        # Direction: bullish = long, bearish = short
        if trade["direction"] in ["bullish", "long"]:
            pnl_pct = (exit_price - trade["entry_price"]) / trade["entry_price"]
        else:  # bearish or short
            pnl_pct = (trade["entry_price"] - exit_price) / trade["entry_price"]
        
        pnl = trade["size"] * pnl_pct
        
        cursor.execute("""
            UPDATE trades 
            SET status = 'closed', exit_price = ?, pnl = ?, pnl_pct = ?, 
                exit_time = ?, reason = ?
            WHERE id = ?
        """, (exit_price, pnl, pnl_pct * 100, datetime.now().isoformat(), reason, trade_id))
        
        conn.commit()
        
        return {**trade, "pnl": pnl, "pnl_pct": pnl_pct * 100}
    
    def get_daily_stats(self, date: str = None) -> Dict:
        """Get daily trading stats"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM daily_stats WHERE date = ?
        """, (date,))
        
        row = cursor.fetchone()
        
        
        if row:
            columns = ["date", "total_trades", "winning_trades", "total_pnl", "win_rate", "created_at"]
            return dict(zip(columns, row))
        
        # Calculate from trades
        cursor.execute("""
            SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                   SUM(pnl), AVG(CASE WHEN pnl > 0 THEN pnl ELSE 0 END)
            FROM trades 
            WHERE DATE(created_at) = DATE(?)
        """, (date,))
        
        row = cursor.fetchone()
        
        
        total = row[0] or 0
        wins = row[1] or 0
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return {
            "date": date,
            "total_trades": total,
            "winning_trades": wins,
            "total_pnl": row[2] or 0,
            "win_rate": win_rate
        }


# =============================================================================
# TRADING SIGNAL
# =============================================================================

@dataclass
class TradingSignal:
    """Complete trading signal with all parameters"""
    symbol: str
    direction: str  # long, short
    entry_price: float
    size_usd: float
    stop_loss_pct: float
    take_profit_pct: float
    confidence: float
    source: str  # ml_scanner, manual, etc.
    reasons: List[str] = field(default_factory=list)
    trade_id: str = ""


# =============================================================================
# UNIFIED TRADING SYSTEM
# =============================================================================

class UnifiedTradingSystem:
    """
    Main Unified Trading System v3 - IMPROVED
    
    Coordinates:
    - Market Scanner
    - ML Signal Generator
    - Risk Agent
    - Paper Trading Engine
    - Redis Cache
    - SQLite DB
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize components
        self.cache = RedisCacheManager()
        self.risk_agent = RiskAgent(self.config)
        self.scanner = MarketScannerAgent()
        self.paper_engine = PaperTradingEngine()
        self.db = TradesDatabase()
        self.ml_signal = MLSignalGenerator(self.cache)
        
        # Initialize Strategy Optimizer - DISABLED due to memory leak
        # try:
        #     from strategy_optimizer_agent import StrategyOptimizer
        #     self.optimizer = StrategyOptimizer()
        #     logger.info("✅ Strategy Optimizer initialized")
        # except Exception as e:
        #     logger.warning(f"⚠️ Strategy Optimizer not available: {e}")
        self.optimizer = None
        
        # Trading tokens (reduced to 5)
        self.trading_tokens = ["SOL", "BTC", "ETH", "WIF", "PUMP"]
        
        # Seed price data for initial ML signals
        self._seed_initial_prices()
        
        # State
        self.running = False
        self.last_scan_time = None
        self.scan_interval = 180  # seconds (reduced from 60 to save memory)
        self.last_optimization = None
        self.optimization_interval = 3600  # Run optimizer every hour
        
        logger.info("✅ Unified Trading System initialized")
    
    def _seed_initial_prices(self):
        """Seed initial price data for ML signals"""
        # Seed with realistic upward trends to generate signals
        seed_data = {
            'SOL': [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108],
            'WIF': [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6],
            'JUP': [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10],
            'ETH': [1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960],
            'BTC': [82000, 83000, 84000, 85000, 86000, 87000, 88000, 89000, 90000],
            'RAY': [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54],
            'JTO': [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40],
            'BONK': [0.000020, 0.000022, 0.000024, 0.000026, 0.000028, 0.000030],
        }
        
        for symbol, prices in seed_data.items():
            for price in prices:
                self.ml_signal.update_price(symbol, price)
        
        logger.info(f"🌱 Seeded {len(seed_data)} tokens with price history")
    
    def get_hardbit_profile(self) -> Dict:
        """
        Get active risk profile - NOW USES DYNAMIC LIMITS from RiskAgent!
        
        The agents decide the risk parameters based on:
        - Win rate (recent performance)
        - Signal confidence
        - Daily P&L
        """
        # First get HARDBIT base profile (time-based)
        base_profile = get_active_profile()
        is_night = is_night_time()
        
        # Get win rate from recent trades
        stats = self.paper_engine.state.stats
        win_rate = stats.get("win_rate", 0.5) if stats.get("total_trades", 0) > 0 else 0.5
        
        # Get recent signal confidence
        recent_confidence = 0.5
        if self.paper_engine.state.signals:
            recent_signals = self.paper_engine.state.signals[-5:]
            if recent_signals:
                confidences = [s.get("confidence", 50) for s in recent_signals]
                recent_confidence = sum(confidences) / len(confidences) / 100
        
        # Get dynamic limits from RiskAgent - AGENTS DECIDE!
        dynamic_limits = self.risk_agent.get_dynamic_limits(win_rate, recent_confidence)
        
        logger.info(f"🧠 Dynamic Risk: {dynamic_limits['reason']}")
        
        return {
            "is_night": is_night,
            "mode": "DYNAMIC (Agent Decision)",
            "profile": base_profile,
            "max_position_pct": dynamic_limits["position_pct"],
            "stop_loss_pct": dynamic_limits["stop_loss_pct"],
            "take_profit_pct": dynamic_limits["take_profit_pct"],
            "max_daily_loss_pct": base_profile.get("max_daily_loss_pct", 0.10),
            "max_concurrent": dynamic_limits["max_concurrent"],
            "cooldown_seconds": base_profile.get("cooldown_seconds", 60),
            "dynamic_reason": dynamic_limits["reason"],
            "win_rate": win_rate,
            "signal_confidence": recent_confidence
        }
    
    def calculate_position_size(
        self, 
        balance_usd: float, 
        risk_pct: float, 
        confidence: float,
        is_night: bool
    ) -> float:
        """
        Calculate position size with confidence adjustment.
        
        Formula: Position = Balance × Risk% × (0.3 + Confidence × 0.7)
        
        Confidence reduces/increases effective risk:
        - 0% confidence: 30% of risk
        - 100% confidence: 100% of risk
        """
        base_size = balance_usd * risk_pct
        
        # Confidence multiplier (0.3 to 1.0)
        conf_multiplier = 0.3 + (confidence / 100) * 0.7
        
        # Night time: reduce position slightly
        if is_night:
            conf_multiplier *= 0.9
        
        position = base_size * conf_multiplier
        
        logger.info(f"📊 Position Size: ${position:.2f} (base: {risk_pct*100}%, conf: {confidence:.0f}%)")
        
        return position
    
    def scan_market(self) -> List[Opportunity]:
        """Scan market for trading opportunities"""
        logger.info("🔍 Scanning market...")
        
        opportunities = self.scanner.scan_market()
        
        # Add trending tokens if available
        try:
            from api.api_integrations import JupiterClient
            client = JupiterClient()
            trending = asyncio.run(client.get_trending_tokens("1h"))
            
            for token in trending[:5]:
                symbol = token.get("symbol", "")
                if symbol not in [o.symbol for o in opportunities] and symbol not in ["USDC", "USDT"]:
                    opp = Opportunity(
                        symbol=symbol,
                        name=token.get("name", symbol),
                        price=token.get("price", 0),
                        change_24h=token.get("change_24h", 0),
                        volume_24h=token.get("volume_24h", 0),
                        liquidity=token.get("liquidity", 0),
                        trend="bullish",
                        volume_anomaly=1.5,
                        score=60,
                        reasons=["Trending token"],
                        timestamp=datetime.now().isoformat()
                    )
                    opportunities.append(opp)
            
            asyncio.run(client.close())
        except Exception as e:
            logger.debug(f"Could not fetch trending: {e}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        self.last_scan_time = datetime.now()
        
        logger.info(f"✅ Found {len(opportunities)} opportunities")
        
        return opportunities[:10]  # Top 10
    
    def generate_ml_signals(self, opportunities: List[Opportunity]) -> List[Dict]:
        """Generate ML signals for opportunities using optimized strategy (RSI oversold)"""
        signals = []
        
        # Load best strategy from strategy agent
        best_strategy = self._load_best_strategy()
        
        for opp in opportunities:
            symbol = opp.symbol
            
            # Update price history with opportunity data
            if opp.price > 0:
                self.ml_signal.update_price(symbol, opp.price)
            
            # Use optimized strategy: RSI oversold = BUY signal
            # Calculate RSI from price history
            prices = self.ml_signal.price_history.get(symbol, [])
            
            if len(prices) >= 8:
                # Simple RSI calculation
                gains = []
                losses = []
                for i in range(1, min(8, len(prices))):
                    change = prices[i] - prices[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                
                avg_gain = sum(gains) / 7 if gains else 0.01
                avg_loss = sum(losses) / 7 if losses else 0.01
                
                rs = avg_gain / avg_loss if avg_loss > 0 else 1
                rsi = 100 - (100 / (1 + rs))
                
                # Use strategy: RSI < 35 = oversold = BUY
                # Get params from best strategy
                if best_strategy:
                    threshold = best_strategy.get('threshold', 35)
                    sl_pct = best_strategy.get('sl_pct', 0.04)
                    tp_pct = best_strategy.get('tp_pct', 0.054)
                else:
                    threshold = 35
                    sl_pct = 0.04
                    tp_pct = 0.054
                
                # Generate signal based on RSI
                if rsi < threshold:
                    # Oversold - BUY signal
                    direction = "bullish"
                    confidence = min(95, (threshold - rsi) * 3)  # More oversold = higher confidence
                    reason = f"RSI oversold: {rsi:.1f} < {threshold}"
                elif rsi > (100 - threshold):
                    # Overbought - SELL signal
                    direction = "bearish"
                    confidence = min(95, (rsi - (100 - threshold)) * 3)
                    reason = f"RSI overbought: {rsi:.1f} > {100-threshold}"
                else:
                    # Neutral zone
                    direction = "neutral"
                    confidence = 5
                    reason = f"RSI neutral: {rsi:.1f}"
                
                signal = {
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": round(confidence, 1),
                    "rsi": round(rsi, 1),
                    "threshold": threshold,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "opportunity_score": opp.score,
                    "reasons": opp.reasons + [reason]
                }
            else:
                # Not enough data - generate random signal
                signal = self.ml_signal.generate_signal(symbol)
                signal["opportunity_score"] = opp.score
                signal["reasons"] = opp.reasons
            
            signals.append(signal)
        
        return signals
    
    def _load_best_strategy(self) -> Optional[Dict]:
        """Load best optimized strategy from strategy agent"""
        try:
            strategies_file = PROJECT_ROOT / "data" / "strategies.json"
            if strategies_file.exists():
                import json
                data = json.loads(strategies_file.read_text())
                if data.get("strategies"):
                    best = data["strategies"][0]  # First = best
                    genome = best.get("genome", {})
                    params = genome.get("params", {})
                    
                    # Extract entry rule threshold
                    entry_rules = genome.get("entry_rules", [])
                    threshold = 35
                    for rule in entry_rules:
                        if rule.get("threshold"):
                            threshold = rule.get("threshold")
                    
                    return {
                        "threshold": threshold,
                        "sl_pct": params.get("sl_pct", 0.04),
                        "tp_pct": params.get("tp_pct", 0.054),
                        "pnl": best.get("pnl", 0),
                        "win_rate": best.get("win_rate", 0)
                    }
        except Exception as e:
            logger.debug(f"Could not load strategy: {e}")
        
        return None
    
    def create_trading_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float,
        reasons: List[str],
        source: str = "ml_scanner"
    ) -> Optional[TradingSignal]:
        """Create a complete trading signal"""
        
        # Get HARDBIT profile
        profile = self.get_hardbit_profile()
        is_night = profile["is_night"]
        
        # Get balance
        balance = self.paper_engine.state.balance_usd
        
        # Check daily loss limit
        daily_stats = self.db.get_daily_stats()
        if daily_stats["total_pnl"] < 0:
            loss_pct = abs(daily_stats["total_pnl"]) / balance
            if loss_pct >= profile["max_daily_loss_pct"]:
                logger.warning(f"⚠️ Daily loss limit reached: {loss_pct:.1%}")
                return None
        
        # Calculate position size
        risk_pct = profile["max_position_pct"]
        size = self.calculate_position_size(balance, risk_pct, confidence, is_night)
        
        # Check minimum size
        if size < 5:  # Minimum $5
            logger.warning(f"⚠️ Position too small: ${size:.2f}")
            return None
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size_usd=size,
            stop_loss_pct=profile["stop_loss_pct"],
            take_profit_pct=profile["take_profit_pct"],
            confidence=confidence,
            source=source,
            reasons=reasons,
            trade_id=f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )
        
        return signal
    
    def validate_with_risk_agent(self, signal: TradingSignal) -> bool:
        """Validate signal with Risk Agent"""
        
        trade_signal = {
            "symbol": signal.symbol,
            "direction": signal.direction,
            "size_pct": signal.size_usd / self.paper_engine.state.balance_usd,
            "stop_loss_pct": signal.stop_loss_pct,
            "take_profit_pct": signal.take_profit_pct,
            "confidence": signal.confidence
        }
        
        risk_result = self.risk_agent.validate_trade(trade_signal)
        
        if not risk_result.approved:
            logger.warning(f"⚠️ Trade rejected by Risk Agent: {risk_result.reasons}")
            return False
        
        logger.info(f"✅ Trade approved by Risk Agent (risk: {risk_result.risk_score:.2f})")
        return True
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute a trading signal (paper mode)"""
        
        # Check concurrent limit
        open_trades = self.paper_engine.get_open_trades()
        profile = self.get_hardbit_profile()
        
        if len(open_trades) >= profile["max_concurrent"]:
            logger.warning(f"⚠️ Max concurrent trades reached: {len(open_trades)}")
            return False
        
        # Execute via paper engine
        trade = self.paper_engine.execute_signal({
            "symbol": signal.symbol,
            "direction": signal.direction,
            "price": signal.entry_price,
            "size": signal.size_usd,
            "reason": "; ".join(signal.reasons)
        })
        
        if trade:
            # Update risk agent
            self.risk_agent.open_trade({
                "id": signal.trade_id,
                "symbol": signal.symbol,
                "size": signal.size_usd,
                "direction": signal.direction
            })
            
            # Save to database
            self.db.add_trade({
                "id": signal.trade_id,
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "exit_price": None,
                "size": signal.size_usd,
                "pnl": 0,
                "pnl_pct": 0,
                "status": "open",
                "reason": "; ".join(signal.reasons),
                "confidence": signal.confidence,
                "entry_time": datetime.now().isoformat(),
                "exit_time": None
            })
            
            logger.info(f"✅ Trade opened: {signal.symbol} {signal.direction} @ ${signal.entry_price:.4f}")
            
            # Send notification (if Telegram configured)
            self._send_trade_notification(signal, "OPENED")
            
            return True
        
        return False
    
    def close_position(self, trade_id: str, reason: str = "Close"):
        """Close an open position"""
        trade = self.paper_engine.close_trade(trade_id, self._get_current_price(trade_id), reason)
        
        if trade:
            # Update database
            self.db.close_trade(trade_id, trade["exit_price"], reason)
            
            # Update risk agent
            self.risk_agent.close_trade(trade_id, trade["pnl"])
            
            logger.info(f"📝 Trade closed: {trade_id} P&L: ${trade['pnl']:.2f}")
            
            # Send notification
            self._send_trade_notification(trade, "CLOSED", reason)
    
    def _get_current_price(self, trade_id: str) -> float:
        """Get current price for a trade - uses REAL prices from CryptoCompare when available"""
        import time
        import random
        
        # Find trade
        for trade in self.paper_engine.get_open_trades():
            if trade["id"] == trade_id:
                symbol = trade["symbol"]
                
                # Try CryptoCompare (our primary price source)
                try:
                    pf = get_price_feed()
                    real_price = pf.get_price_sync(symbol)
                    if real_price > 0:
                        logger.info(f"💰 Close price for {symbol}: ${real_price} (from CryptoCompare)")
                        return real_price
                except Exception as e:
                    logger.warning(f"CryptoCompare price failed for {symbol}: {e}")
                
                # Try Kraken as backup
                try:
                    kraken = get_kraken_price()
                    real_price = kraken.get_price(symbol)
                    if real_price > 0:
                        logger.info(f"💰 Close price for {symbol}: ${real_price} (from Kraken)")
                        return real_price
                except Exception as e:
                    logger.warning(f"Kraken price failed for {symbol}: {e}")
                
                # Try cache first
                try:
                    cached = self.cache.get_price(symbol)
                    if cached and cached.get("price", 0) > 0:
                        logger.info(f"💰 Close price for {symbol}: ${cached['price']} (from cache)")
                        return cached["price"]
                except Exception as e:
                    logger.warning(f"Cache price failed for {symbol}: {e}")
                
                # Last resort: fetch directly from CryptoCompare API (bypass singleton issues)
                try:
                    import requests
                    resp = requests.get(
                        "https://min-api.cryptocompare.com/data/pricemulti",
                        params={"fsyms": symbol, "tsyms": "USD"},
                        timeout=5
                    )
                    data = resp.json()
                    if symbol in data and "USD" in data[symbol]:
                        price = data[symbol]["USD"]
                        logger.info(f"💰 Close price for {symbol}: ${price} (direct API)")
                        return price
                except Exception as e:
                    logger.warning(f"Direct CryptoCompare failed for {symbol}: {e}")
                
                # BUG FIX: Never fallback to entry_price - that causes P&L=0 bug!
                # Instead, log error and return 0 (will prevent closing with wrong price)
                logger.error(f"🚨 CRITICAL: Could not get price for {symbol}, trade {trade_id} will not close!")
                return 0
        
        return 0
    
    def _send_trade_notification(self, trade: Any, action: str, reason: str = ""):
        """Send trade notification via Telegram (if configured)"""
        try:
            from config.config import get_config
            
            config = get_config()
            
            if not config.telegram.enabled:
                return
            
            # This would integrate with Telegram
            # For now, just log
            logger.info(f"📱 Would send Telegram notification: {action} {trade.symbol}")
            
        except Exception as e:
            logger.debug(f"Could not send notification: {e}")
    
    def run_cycle(self):
        """Run one trading cycle"""
        if not self.running:
            return
        
        logger.info("🔄 Running trading cycle...")
        
        # 1. Scan market
        opportunities = self.scan_market()
        
        # 2. Generate ML signals
        signals = self.generate_ml_signals(opportunities)
        
        # 3. Process high-confidence signals
        for signal in signals:
            # Skip low confidence (reduced from 30% to 5% for aggressive trading)
            if signal["confidence"] < 10:  # Minimum 10% confidence
                continue
            
            # Trade both bullish and bearish signals (for more opportunities)
            if signal["direction"] in ["bullish", "bearish"]:
                # Get entry price from opportunities or price history
                entry_price = None
                for opp in opportunities:
                    if opp.symbol == signal["symbol"]:
                        entry_price = opp.price
                        break
                
                # Fallback to price history if not found or zero
                if not entry_price or entry_price == 0:
                    prices = self.ml_signal.price_history.get(signal["symbol"], [])
                    if prices:
                        entry_price = prices[-1]
                    else:
                        entry_price = 100  # Ultimate fallback
                
                # Create trading signal
                trade_signal = self.create_trading_signal(
                    symbol=signal["symbol"],
                    direction=signal["direction"],
                    entry_price=entry_price,
                    confidence=signal["confidence"],
                    reasons=signal.get("reasons", []) + [f"ML Score: {signal['confidence']:.0f}%"]
                )
                
                if trade_signal:
                    # Validate with risk agent
                    if self.validate_with_risk_agent(trade_signal):
                        # Execute trade
                        self.execute_trade(trade_signal)
        
        # 4. Check open positions for SL/TP
        self._check_open_positions()
        
        # 5. Run strategy optimizer periodically (every hour)
        self._run_optimizer_if_needed()
        
        logger.info("✅ Trading cycle complete")
    
    def _check_open_positions(self):
        """Check open positions for stop loss / take profit"""
        open_trades = self.paper_engine.get_open_trades()
        logger.info(f"🔍 Checking {len(open_trades)} open positions for SL/TP...")
        
        for trade in open_trades:
            symbol = trade["symbol"]
            entry_price = trade["entry_price"]
            direction = trade["direction"]
            
            # Skip if trade was just opened (less than 30 seconds ago)
            try:
                entry_time = trade["entry_time"]
                # Handle both string and datetime object
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                seconds_held = (datetime.now() - entry_time).total_seconds()
                logger.info(f"   {symbol}: held {seconds_held:.0f}s (min 30s)")
                if seconds_held < 30:  # Minimum 30 seconds hold
                    continue
            except Exception as e:
                logger.error(f"   ERROR checking {symbol}: {e}")
                continue
            
            # Get current price
            current_price = self._get_current_price(trade["id"])
            
            logger.info(f"📊 Checking {symbol}: entry=${entry_price}, current=${current_price}, dir={direction}")
            
            if current_price == 0:
                logger.warning(f"⚠️ No price for {symbol}, skipping")
                continue
            
            # Calculate P&L (bullish = long, bearish = short)
            if direction == "bullish":
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # bearish = short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Find trading signal for SL/TP levels
            profile = self.get_hardbit_profile()
            
            # Use strategy SL/TP if available, otherwise use HARDBIT profile
            stop_loss = profile.get("stop_loss_pct", 0.02)
            take_profit = profile.get("take_profit_pct", 0.04)
            
            # Check SL/TP
            if pnl_pct <= -stop_loss:
                self.close_position(trade["id"], "STOP_LOSS")
            elif pnl_pct >= take_profit:
                self.close_position(trade["id"], "TAKE_PROFIT")
            
            # ========== 24H TIME LIMIT ==========
            # Close positions open > 24h (unless in big loss)
            if seconds_held >= 86400:  # 24 hours
                if pnl_pct > -0.05:  # Not losing more than 5%
                    self.close_position(trade["id"], "24H_TIMEOUT")
                    logger.info(f"⏰ Closed {symbol} after 24h (PnL: {pnl_pct*100:+.2f}%)")
        
        # ========== PORTFOLIO TAKE PROFIT ==========
        # Close ALL positions if total portfolio P&L is in target range
        self._check_portfolio_take_profit()
    
    def _check_portfolio_take_profit(self):
        """Check if total portfolio P&L is in profit target range and close all"""
        open_trades = self.paper_engine.get_open_trades()
        if len(open_trades) < 1:
            return  # No positions to check
        
        # Calculate total P&L across all positions
        total_pnl = 0
        total_invested = 0
        
        for trade in open_trades:
            entry_price = trade["entry_price"]
            direction = trade["direction"]
            size = trade.get("size", 0)
            
            # Get current price
            current_price = self._get_current_price(trade["id"])
            if current_price == 0:
                continue
            
            # Calculate position P&L
            if direction == "bullish":
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # bearish = short
                pnl_pct = (entry_price - current_price) / entry_price
            
            position_value = size * entry_price
            total_pnl += position_value * pnl_pct
            total_invested += position_value
        
        if total_invested == 0:
            return
        
        portfolio_pnl_pct = (total_pnl / total_invested) * 100
        
        logger.info(f"💼 Portfolio P&L: {portfolio_pnl_pct:+.2f}% ({len(open_trades)} positions)")
        
        # Dynamic thresholds based on risk agent
        profile = self.get_hardbit_profile()
        
        # Risk agent decides thresholds: conservative (1-3%), normal (2-5%), aggressive (3-7%)
        # Calculate win_rate from recent trades
        try:
            recent = self.paper_engine.get_recent_trades(20)
            if recent:
                wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
                win_rate = (wins / len(recent)) * 100 if recent else 0
            else:
                win_rate = 0
        except:
            win_rate = 50  # Default
        
        if win_rate >= 70:
            # Good win rate = more aggressive
            min_profit = 2.0
            max_profit = 5.0
        elif win_rate >= 50:
            min_profit = 1.5
            max_profit = 4.0
        else:
            # Poor win rate = conservative
            min_profit = 1.0
            max_profit = 3.0
        
        # Close all if in profit range
        if min_profit <= portfolio_pnl_pct <= max_profit:
            logger.info(f"🎯 PORTFOLIO TAKE PROFIT! Closing all positions at {portfolio_pnl_pct:+.2f}%")
            for trade in open_trades:
                self.close_position(trade["id"], "PORTFOLIO_TAKE_PROFIT")
            # Reset balance for fresh start
            logger.info(f"🔄 Resetting portfolio for new trades")
        elif portfolio_pnl_pct < -5.0:
            # Emergency: close all if losing more than 5%
            logger.warning(f"🛑 EMERGENCY: Portfolio down {portfolio_pnl_pct:.2f}%, closing all!")
            for trade in open_trades:
                self.close_position(trade["id"], "PORTFOLIO_STOP_LOSS")
    
    def _run_optimizer_if_needed(self):
        """Run strategy optimizer periodically"""
        from datetime import datetime, timedelta
        
        # Check if optimizer is available
        if not self.optimizer:
            return
        
        # Check if it's time to run optimizer
        now = datetime.now()
        if self.last_optimization is None:
            self.last_optimization = now
        
        # Run optimizer every hour
        if (now - self.last_optimization).total_seconds() >= self.optimization_interval:
            logger.info("🧬 Running strategy optimizer...")
            try:
                # Analyze performance
                analysis = self.optimizer.analyze_performance()
                logger.info(f"📊 Strategy Analysis: {analysis.get('win_rate', 0):.1f}% win rate")
                
                # Run optimization
                results = self.optimizer.optimize()
                if results:
                    best = results[0]
                    logger.info(f"🆕 Best strategy: {best.get('name', 'unknown')} - Score: {best.get('score', 0):.2f}")
                
                self.last_optimization = now
            except Exception as e:
                logger.warning(f"⚠️ Optimizer error: {e}")
    
    def start(self):
        """Start the trading system"""
        self.running = True
        self.paper_engine.start()
        
        profile = self.get_hardbit_profile()
        mode = "HARDBIT NIGHT" if profile["is_night"] else "DAY TRADING"
        
        logger.info(f"🚀 Unified Trading System STARTED ({mode})")
        logger.info(f"   Max Position: {profile['max_position_pct']*100}%")
        logger.info(f"   Stop Loss: {profile['stop_loss_pct']*100}%")
        logger.info(f"   Take Profit: {profile['take_profit_pct']*100}%")
    
    def stop(self):
        """Stop the trading system"""
        self.running = False
        self.paper_engine.stop()
        logger.info("🛑 Unified Trading System STOPPED")
    
    def reset(self):
        """Reset all state"""
        self.stop()
        self.paper_engine.reset()
        self.risk_agent.reset_daily()
        self.cache.flush_all()
        logger.info("🔄 Unified Trading System RESET")
    
    def status(self) -> Dict:
        """Get system status"""
        open_trades = self.paper_engine.get_open_trades()
        profile = self.get_hardbit_profile()
        
        return {
            "system": {
                "running": self.running,
                "mode": "HARDBIT NIGHT" if profile["is_night"] else "DAY TRADING"
            },
            "profile": profile,
            "paper_trading": {
                "balance": self.paper_engine.state.balance_usd,
                "pnl": self.paper_engine.state.stats["total_pnl"],
                "open_trades": len(open_trades),
                "win_rate": self.paper_engine.state.stats["win_rate"]
            },
            "risk": self.risk_agent.check_portfolio_risk(),
            "cache": {
                "redis_available": self.cache.redis_available
            },
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None
        }
    
    def print_status(self):
        """Print formatted status"""
        status = self.status()
        
        print(f"\n{'='*60}")
        print(f"🦞 UNIFIED TRADING SYSTEM v3 - STATUS")
        print(f"{'='*60}")
        
        print(f"\n📊 System: {'🟢 RUNNING' if status['system']['running'] else '🔴 STOPPED'}")
        print(f"   Mode: {status['system']['mode']}")
        
        print(f"\n💰 Paper Trading:")
        print(f"   Balance: ${status['paper_trading']['balance']:,.2f}")
        print(f"   P&L: ${status['paper_trading']['pnl']:,.2f}")
        print(f"   Open Trades: {status['paper_trading']['open_trades']}")
        print(f"   Win Rate: {status['paper_trading']['win_rate']:.1f}%")
        
        print(f"\n⚠️ Risk Level: {status['risk']['risk_level']}")
        print(f"   Daily P&L: ${status['risk']['daily_pnl']:,.2f}")
        
        print(f"\n📡 Cache: {'Redis' if status['cache']['redis_available'] else 'File Fallback'}")
        print(f"   Last Scan: {status['last_scan'] or 'Never'}")
        
        print(f"\n{'='*60}")
    
    def run_continuous(self):
        """Run continuous trading cycles"""
        import time
        
        # Don't call self.start() here - it's already called in main()
        
        try:
            while self.running:
                self.run_cycle()
                time_module.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.stop()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Trading System v3")
    parser.add_argument("--start", action="store_true", help="Start trading system")
    parser.add_argument("--stop", action="store_true", help="Stop trading system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--scan", action="store_true", help="Run market scan")
    parser.add_argument("--reset", action="store_true", help="Reset all state")
    parser.add_argument("--test-signal", action="store_true", help="Test signal generation")
    parser.add_argument("--cycle", action="store_true", help="Run one trading cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuous cycles")
    parser.add_argument("--paper-status", action="store_true", help="Show paper trading status")
    parser.add_argument("--paper-reset", action="store_true", help="Reset paper trading")
    
    args = parser.parse_args()
    
    system = UnifiedTradingSystem()
    
    if args.start:
        system.start()
        print("✅ Trading system started. Press Ctrl+C to stop.")
        try:
            system.run_continuous()
        except KeyboardInterrupt:
            system.stop()
    
    elif args.stop:
        system.stop()
    
    elif args.status:
        system.print_status()
    
    elif args.scan:
        opportunities = system.scan_market()
        print(f"\n📊 Found {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"   {i}. {opp.symbol}: {opp.score}/100 - {', '.join(opp.reasons)}")
    
    elif args.reset:
        system.reset()
    
    elif args.test_signal:
        print("\n🧪 Testing ML Signal Generation...")
        
        # Test with SOL
        for symbol in ["SOL", "JUP", "BONK"]:
            # Simulate price history
            base_price = 100 if symbol == "SOL" else (1 if symbol == "JUP" else 0.0001)
            for i in range(50):
                price = base_price * (1 + (i * 0.001) + (i % 3) * 0.005)
                system.ml_signal.update_price(symbol, price)
            
            signal = system.ml_signal.generate_signal(symbol)
            print(f"\n📊 {symbol}: {signal['direction']} ({signal['confidence']:.1f}%)")
            print(f"   RSI: {signal['components']['rsi']['value']:.1f} ({signal['components']['rsi']['signal']})")
            print(f"   EMA: {signal['components']['ema_crossover']['signal']}")
            print(f"   Trend: {signal['components']['trend']['signal']}")
    
    elif args.cycle:
        system.start()
        system.run_cycle()
        system.stop()
    
    elif args.continuous:
        system.run_continuous()
    
    elif args.paper_status:
        system.paper_engine.status()
    
    elif args.paper_reset:
        system.paper_engine.reset()
        print("✅ Paper trading reset")
    
    else:
        system.print_status()
        print("\n📖 Usage:")
        print("   python3 unified_trading_system.py --start       # Start trading")
        print("   python3 unified_trading_system.py --status      # Show status")
        print("   python3 unified_trading_system.py --scan        # Scan market")
        print("   python3 unified_trading_system.py --test-signal # Test ML signals")
        print("   python3 unified_trading_system.py --reset       # Reset all state")


if __name__ == "__main__":
    main()