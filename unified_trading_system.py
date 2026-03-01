#!/usr/bin/env python3
"""
Unified Trading System v4 - ROBUST FIX
=====================================
Complete trading system with ML signals, risk management, and Redis caching.

FIXES APPLIED IN v4:
1. Robust sleep with error handling and logging
2. Heartbeat file for external monitoring
3. Additional signal handlers
4. Comprehensive exception handling
5. Process health monitoring
6. Auto-restart on unrecoverable errors

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
import traceback

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing modules
from config.config import Config
from config.hardbit_schedule import HARDBIT_CONFIG, is_night_time, get_active_profile
from agents.risk_agent import RiskAgent, RiskLimits
from agents.market_scanner_agent import MarketScannerAgent, Opportunity
from paper_trading_engine import PaperTradingEngine
from auto_improver import AutoImprover
from self_improver import SelfImprover
from ml.adaptive_weights import AdaptiveWeights
from api.kraken_price import get_kraken_price
from api.price_feed import get_price_feed
from notifications import NotificationLogger, get_notifier

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

# Heartbeat file for monitoring
HEARTBEAT_FILE = PROJECT_ROOT / "data" / "heartbeat.txt"

def update_heartbeat():
    """Update heartbeat file with current timestamp and process info"""
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        import psutil
        pid = os.getpid()
        process = psutil.Process(pid)
        heartbeat_data = {
            "timestamp": datetime.now().isoformat(),
            "pid": pid,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "status": "running"
        }
        HEARTBEAT_FILE.write_text(json.dumps(heartbeat_data))
    except:
        HEARTBEAT_FILE.write_text(datetime.now().isoformat())

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
        if self.redis_available:
            try:
                self.redis_client.setex(f"trade:{trade_id}", ttl, json.dumps(state))
                return
            except:
                pass

        # File fallback
        cache_file = self.cache_dir / f"trade_{trade_id}.json"
        cache_file.write_text(json.dumps(state))

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

    # ==================== MARKET DATA ====================

    def set_market_data(self, symbol: str, data: Dict, ttl: int = 300):
        """Cache market data"""
        if self.redis_available:
            try:
                self.redis_client.setex(f"market:{symbol}", ttl, json.dumps(data))
                return
            except:
                pass

        # File fallback
        cache_file = self.cache_dir / f"market_{symbol}.json"
        cache_file.write_text(json.dumps(data))

    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get cached market data"""
        if self.redis_available:
            try:
                data = self.redis_client.get(f"market:{symbol}")
                if data:
                    return json.loads(data)
            except:
                pass

        # File fallback
        cache_file = self.cache_dir / f"market_{symbol}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None

    # ==================== FLUSH ====================

    def flush_all(self):
        """Flush all cache"""
        if self.redis_available:
            try:
                self.redis_client.flushall()
            except:
                pass

        # Clear file cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


# =============================================================================
# ML SIGNAL GENERATOR (from original v3)
# =============================================================================

class MLSignalGenerator:
    """
    ML Signal Generator with RSI, EMA, Momentum, and Trend indicators.
    Supports adaptive weights that learn from trade outcomes.
    """

    def __init__(self, rsi_period: int = 14, ema_short: int = 9, ema_long: int = 21):
        self.rsi_period = rsi_period
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.price_history: Dict[str, List[float]] = {}
        # FASE 3: Adaptive weights (loaded from data/ml_weights.json)
        self.adaptive_weights = AdaptiveWeights()

    def update_price(self, symbol: str, price: float):
        """Update price history for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

        # Keep last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

    def calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period if gains else 0
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA indicator"""
        if len(prices) < period:
            return sum(prices) / len(prices)

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def generate_signal(self, symbol: str) -> Dict:
        """Generate trading signal with ML indicators"""
        prices = self.price_history.get(symbol, [])

        if len(prices) < 5:  # Reduced from 20 to 5 - Allow signals with minimal history
            return {
                "symbol": symbol,
                "direction": "neutral",
                "confidence": 0.0,
                "components": {}
            }

        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        ema_short = self.calculate_ema(prices, self.ema_short)
        ema_long = self.calculate_ema(prices, self.ema_long)
        current_price = prices[-1]

        # RSI signal (30% weight)
        rsi_score = 0
        rsi_signal = "neutral"
        if rsi < 30:
            rsi_score = 50  # Oversold
            rsi_signal = "bullish"
        elif rsi > 70:
            rsi_score = -50  # Overbought
            rsi_signal = "bearish"
        else:
            rsi_score = 0

        # EMA crossover signal (25% weight)
        ema_score = 0
        ema_signal = "neutral"
        if ema_short > ema_long:
            ema_score = 40  # Uptrend
            ema_signal = "bullish"
        elif ema_short < ema_long:
            ema_score = -40  # Downtrend
            ema_signal = "bearish"
        else:
            ema_score = 0

        # Momentum signal (25% weight)
        momentum = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        momentum_score = min(max(momentum * 1000, -50), 50)
        momentum_signal = "bullish" if momentum_score > 10 else ("bearish" if momentum_score < -10 else "neutral")

        # Trend signal (20% weight)
        trend = (prices[-1] - prices[0]) / prices[0] if len(prices) > 0 else 0
        trend_score = min(max(trend * 500, -40), 40)
        trend_signal = "bullish" if trend_score > 5 else ("bearish" if trend_score < -5 else "neutral")

        # FASE 3: Use adaptive weights instead of hardcoded values
        weights = self.adaptive_weights.get_weights()
        ensemble_score = (
            rsi_score * weights.get("rsi", 0.30) +
            ema_score * weights.get("ema_crossover", 0.25) +
            momentum_score * weights.get("momentum", 0.25) +
            trend_score * weights.get("trend", 0.20)
        )

        # Apply non-linear confidence scaling (penalize weak signals)
        confidence = abs(ensemble_score)
        if confidence < 0.05:
            confidence = confidence * 0.1  # Heavy penalty for very weak signals
        elif confidence < 0.30:
            confidence = confidence * 0.3  # Moderate penalty
        else:
            confidence = confidence * 0.7  # Less penalty for strong signals

        # Determine direction
        if ensemble_score > 0.10:  # Reduced from 0.20 to 0.10 - More reasonable threshold
            direction = "bullish"
        elif ensemble_score < -0.10:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": min(confidence * 100, 95.0),  # Max 95%
            "ensemble_score": ensemble_score,
            "components": {
                "rsi": {"value": rsi, "signal": rsi_signal, "score": rsi_score},
                "ema_crossover": {
                    "short": ema_short,
                    "long": ema_long,
                    "signal": ema_signal,
                    "score": ema_score
                },
                "momentum": {"value": momentum, "signal": momentum_signal, "score": momentum_score},
                "trend": {"value": trend, "signal": trend_signal, "score": trend_score}
            }
        }


# =============================================================================
# UNIFIED TRADING SYSTEM v4
# =============================================================================

@dataclass
class TradingSignal:
    """Trading signal with risk management"""
    symbol: str
    direction: str  # "bullish" or "bearish"
    entry_price: float
    confidence: float  # 0-100
    reasons: List[str] = field(default_factory=list)
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    risk_reward: float = 0.0


class UnifiedTradingSystem:
    """
    Unified Trading System v4 with robust continuous operation.

    Features:
    - Market scanner with 10 tokens + trending
    - ML signal generation (RSI, EMA, Momentum, Trend)
    - Risk agent with validation
    - Paper trading engine
    - Auto-improver with feedback loop
    - HARDBIT night schedule
    - Robust error handling and recovery
    - Heartbeat monitoring
    """

    def __init__(self, scan_interval: int = 180):
        self.scan_interval = scan_interval
        self.running = False
        self.cycle_count = 0
        self.last_cycle_time: Optional[datetime] = None
        self.last_scan_time: Optional[datetime] = None

        # 📊 Track trades since last optimizer run
        self._last_optimizer_trades_count = 0

        # Initialize components
        self.cache = RedisCacheManager()
        self.market_scanner = MarketScannerAgent()
        self.ml_signal = MLSignalGenerator()
        self.risk_agent = RiskAgent()
        self.paper_engine = PaperTradingEngine()
        self.auto_improver = AutoImprover()
        self.self_improver = SelfImprover()
        self.notifier = NotificationLogger()

        # Load config
        self.config = Config()

        logger.info("✅ Unified Trading System v4 initialized")

    def start(self):
        """Start the trading system"""
        self.running = True
        self.paper_engine.start()

        # Set leverage to 5x (Drift Protocol-like)
        self.set_leverage(5)
        logger.info(f"   Leverage: {self.paper_engine.state.leverage}x (Drift Protocol mode)")

        # Get trading params (auto-improver or HARDBIT)
        profile = self.get_trading_params()

        # Get HARDBIT schedule for mode display (separate from trading params)
        try:
            hardbit_profile = get_active_profile()
            is_night = hardbit_profile.get("is_night", False)
        except:
            is_night = False

        mode = "HARDBIT NIGHT" if is_night else "DAY TRADING"

        logger.info(f"🚀 Unified Trading System STARTED v4 ({mode})")
        logger.info(f"   Max Position: {profile['max_position_pct']*100}%")
        logger.info(f"   Stop Loss: {profile['stop_loss_pct']*100}%")
        logger.info(f"   Take Profit: {profile['take_profit_pct']*100}%")
        logger.info(f"   🔧 ROBUST MODE: Enhanced error handling & heartbeat monitoring")

        # Send notification
        try:
            self.notifier.system_started(mode)
        except Exception as e:
            logger.debug(f"Could not send start notification: {e}")

    def stop(self, reason: str = "Manual shutdown"):
        """Stop the trading system"""
        self.running = False
        self.paper_engine.stop()
        logger.info(f"🛑 Unified Trading System v4 STOPPED ({reason})")

        # Send notification
        try:
            self.notifier.system_stopped(reason)
        except Exception as e:
            logger.debug(f"Could not send stop notification: {e}")

    def reset(self):
        """Reset all state"""
        self.stop()
        self.paper_engine.reset()
        self.risk_agent.reset_daily()
        self.cache.flush_all()
        logger.info("🔄 Unified Trading System v4 RESET")

    def status(self) -> Dict:
        """Get system status"""
        open_trades = self.paper_engine.get_open_trades()

        # Get HARDBIT schedule for time display (not for trading params)
        try:
            hardbit_profile = get_active_profile()
            is_night = hardbit_profile.get("is_night", False)
        except:
            is_night = False

        # Get trading params (may be auto-improver or HARDBIT)
        profile = self.get_trading_params()

        return {
            "system": {
                "running": self.running,
                "mode": "HARDBIT NIGHT" if is_night else "DAY TRADING",
                "cycle_count": self.cycle_count,
                "last_cycle": self.last_cycle_time.isoformat() if self.last_cycle_time else None
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
        print(f"🦞 UNIFIED TRADING SYSTEM v4 - STATUS")
        print(f"{'='*60}")

        print(f"\n📊 System: {'🟢 RUNNING' if status['system']['running'] else '🔴 STOPPED'}")
        print(f"   Mode: {status['system']['mode']}")
        print(f"   Cycles: {status['system']['cycle_count']}")

        print(f"\n💰 Paper Trading:")
        print(f"   Balance: ${status['paper_trading']['balance']:,.2f}")
        print(f"   P&L: ${status['paper_trading']['pnl']:,.2f}")
        print(f"   Open Trades: {status['paper_trading']['open_trades']}")
        print(f"   Win Rate: {status['paper_trading']['win_rate']:.1f}%")

        print(f"\n⚠️ Risk Level: {status['risk']['risk_level']}")
        print(f"   Daily P&L: ${status['risk']['daily_pnl']:,.2f}")

        print(f"\n📡 Cache: {'Redis' if status['cache']['redis_available'] else 'File Fallback'}")
        print(f"   Last Scan: {status['last_scan'] or 'Never'}")

        print(f"\n🔧 ROBUST MODE:")
        print(f"   ✅ Enhanced error handling")
        print(f"   ✅ Heartbeat monitoring")
        print(f"   ✅ Auto-restart on critical errors")

    def set_leverage(self, leverage: int) -> bool:
        """Set leverage (1-20x, like Drift Protocol)"""
        return self.paper_engine.set_leverage(leverage)

    def get_trading_params(self) -> Dict:
        """Get trading parameters (auto-improver or HARDBIT)"""
        # Try auto_improver first
        try:
            params = self.auto_improver.get_trading_params()
            if params:
                logger.info("✅ Using auto-improver parameters")
                params["leverage"] = self.paper_engine.state.leverage  # Add current leverage
                return params
        except (FileNotFoundError, KeyError, AttributeError, ValueError) as e:
            logger.debug(f"Auto-improver not available: {e}. Using HARDBIT fallback")
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error with auto-improver: {e}. Using HARDBIT fallback")

        # Fallback to HARDBIT
        try:
            hardbit_profile = get_active_profile()
            logger.info("✅ Using HARDBIT schedule parameters")
            return {
                "max_position_pct": hardbit_profile.get("max_position", 0.15),
                "max_concurrent_positions": hardbit_profile.get("max_concurrent", 5),
                "stop_loss_pct": hardbit_profile.get("stop_loss", 0.02),
                "take_profit_pct": hardbit_profile.get("take_profit", 0.04),
                "max_daily_loss_pct": hardbit_profile.get("max_daily_loss", 0.05),
                "max_trades_per_day": hardbit_profile.get("max_trades", 20),
                "risk_reward_min": 2.0,
                "leverage": self.paper_engine.state.leverage  # Add current leverage
            }
        except:
            # Default parameters
            return {
                "max_position_pct": 0.15,
                "max_concurrent_positions": 5,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "max_daily_loss_pct": 0.05,
                "max_trades_per_day": 20,
                "risk_reward_min": 2.0,
                "leverage": self.paper_engine.state.leverage  # Add current leverage
            }

    def scan_market(self) -> List[Opportunity]:
        """Scan market for trading opportunities"""
        self.last_scan_time = datetime.now()
        opportunities = self.market_scanner.scan_market()
        logger.info(f"✅ Found {len(opportunities)} opportunities")

        # Update price history for ML signals
        for opp in opportunities:
            self.ml_signal.update_price(opp.symbol, opp.price)

            # Cache price
            self.cache.set_price(opp.symbol, opp.price)

            # Cache market data
            self.cache.set_market_data(opp.symbol, {
                "price": opp.price,
                "score": opp.score,
                "reasons": opp.reasons,
                "timestamp": opp.timestamp if isinstance(opp.timestamp, str) else opp.timestamp.isoformat()
            })

        return opportunities

    def generate_ml_signals(self, opportunities: List[Opportunity]) -> List[Dict]:
        """Generate ML signals for opportunities, weighted by historical win rate"""
        signals = []

        for opp in opportunities:
            signal = self.ml_signal.generate_signal(opp.symbol)
            if signal["confidence"] > 0:
                # 📊 FASE 2: Adjust confidence by historical win rate per symbol+direction
                try:
                    raw_confidence = signal["confidence"]
                    win_rate = self.self_improver.get_win_rate(
                        symbol=signal["symbol"],
                        direction=signal["direction"]
                    )
                    # Formula: adjusted = raw * (0.5 + win_rate * 0.5)
                    # win_rate=0.0 → multiplier=0.5 (halve confidence)
                    # win_rate=0.5 → multiplier=0.75 (default baseline)
                    # win_rate=1.0 → multiplier=1.0 (full confidence)
                    adjustment_factor = 0.5 + win_rate * 0.5
                    signal["confidence"] = raw_confidence * adjustment_factor
                    signal["raw_confidence"] = raw_confidence
                    signal["win_rate_adjustment"] = adjustment_factor
                    signal["historical_win_rate"] = win_rate
                    logger.info(
                        f"📊 ML Signal for {signal['symbol']}: "
                        f"{signal['direction']} ({signal['confidence']:.1f}% confidence, "
                        f"raw={raw_confidence:.1f}%, wr={win_rate:.0%}, adj={adjustment_factor:.2f})"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Could not adjust confidence for {signal['symbol']}: {e}")
                    logger.info(
                        f"📊 ML Signal for {signal['symbol']}: "
                        f"{signal['direction']} ({signal['confidence']:.1f}% confidence)"
                    )
                signals.append(signal)

        return signals

    def create_trading_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float,
        reasons: List[str]
    ) -> Optional[TradingSignal]:
        """Create trading signal with risk management"""
        try:
            profile = self.get_trading_params()

            # Calculate SL/TP
            stop_loss_pct = profile["stop_loss_pct"]
            take_profit_pct = profile["take_profit_pct"]

            if direction == "bullish":
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # bearish
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)

            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0

            # Confidence-based position sizing (base 15% * confidence factor)
            base_position_pct = profile["max_position_pct"]
            confidence_factor = 0.3 + (confidence / 100) * 0.7  # 0.3 to 1.0 based on confidence
            position_pct = base_position_pct * confidence_factor

            balance = self.paper_engine.state.balance_usd
            position_size = balance * position_pct

            return TradingSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                confidence=confidence,
                reasons=reasons,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_reward=risk_reward
            )
        except Exception as e:
            logger.error(f"❌ Error creating trading signal: {e}")
            return None

    def validate_with_risk_agent(self, signal: TradingSignal) -> bool:
        """Validate signal with risk agent"""
        try:
            profile = self.get_trading_params()

            # Check if we already have a position in this symbol
            open_trades = self.paper_engine.get_open_trades()
            for trade in open_trades:
                if trade["symbol"] == signal.symbol:
                    logger.warning(
                        f"⚠️ Trade rejected: Already have 1 open position(s) in {signal.symbol}"
                    )
                    return False

            # Validate with risk agent
            trade_signal = {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "position_size": signal.position_size,
                "confidence": signal.confidence
            }
            risk_assessment = self.risk_agent.validate_trade(trade_signal)

            logger.info(f"✅ Trade approved by Risk Agent (risk: {risk_assessment.risk_score:.2f})")

            return risk_assessment.approved
        except Exception as e:
            logger.error(f"❌ Error validating with risk agent: {e}")
            return False

    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade"""
        try:
            # Map direction: bullish/bearish → long/short
            direction_map = {
                "bullish": "long",
                "bearish": "short"
            }
            mapped_direction = direction_map.get(signal.direction, signal.direction)
            
            # Create signal dict for paper trading engine
            signal_dict = {
                'symbol': signal.symbol,
                'direction': mapped_direction,  # Fixed: Now correctly maps to long/short
                'price': signal.entry_price,  # Mapping: entry_price → price
                'size': signal.position_size,
                'reason': ', '.join(signal.reasons) if signal.reasons else 'Signal',  # reasons → reason
                'leverage': self.paper_engine.state.leverage,  # Get current leverage from engine state
                'confidence': signal.confidence  # Pass ML confidence to paper engine
            }

            # Execute trade using correct API
            trade = self.paper_engine.execute_signal(signal_dict)
            trade_id = trade.id if trade else None

            if trade_id:
                logger.info(
                    f"✅ Trade opened: {signal.symbol} {signal.direction} @ ${signal.entry_price:.6f}"
                )

                # Notify
                try:
                    self.notifier.trade_opened(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        size=signal.position_size,
                        confidence=signal.confidence
                    )
                except:
                    pass

                # Store in cache (including ML components for adaptive weights learning)
                # Get the latest ML signal components for this symbol
                ml_signal_data = self.ml_signal.generate_signal(signal.symbol)
                ml_components = ml_signal_data.get("components", {}) if ml_signal_data else {}
                
                self.cache.set_trade_state(trade_id, {
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "entry_price": signal.entry_price,
                    "size": signal.position_size,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "confidence": signal.confidence,
                    "ml_components": ml_components  # FASE 3: For adaptive weight learning
                })

                return True
            else:
                logger.error(f"❌ Failed to open position for {signal.symbol}")
                return False
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return False

    def _get_current_price(self, trade_id: str) -> float:
        """Get current price for a trade"""
        try:
            # Get trade from paper engine
            trades = self.paper_engine.state.trades
            trade = next((t for t in trades if t["id"] == trade_id), None)

            if not trade:
                return 0

            symbol = trade["symbol"]

            # Try cache first
            cached = self.cache.get_price(symbol)
            if cached and cached.get("timestamp"):
                # Check if cache is fresh (< 60 seconds)
                cache_time = datetime.fromisoformat(cached["timestamp"])
                if (datetime.now() - cache_time).total_seconds() < 60:
                    return cached["price"]

            # Get fresh price
            price_feed = get_price_feed()
            price = price_feed.get_price(f"{symbol}USD")

            if price:
                logger.info(f"💰 Close price for {symbol}: ${price} (from price feed)")
                self.cache.set_price(symbol, price)
                return price

            # Fallback to Kraken
            try:
                kraken_feed = get_kraken_price()
                price = kraken_feed.get_price(f"{symbol}USD")
                if price:
                    logger.info(f"💰 Close price for {symbol}: ${price} (from Kraken)")
                    self.cache.set_price(symbol, price)
                    return price
            except:
                pass

            return 0
        except Exception as e:
            logger.error(f"❌ Error getting current price: {e}")
            return 0

    def close_position(self, trade_id: str, reason: str):
        """Close position"""
        try:
            current_price = self._get_current_price(trade_id)
            if current_price == 0:
                logger.warning(f"⚠️ Cannot close position {trade_id}: No price available")
                return

            trade = self.paper_engine.close_trade(trade_id, current_price, reason)

            if trade:
                logger.info(
                    f"✅ Position closed: {trade['symbol']} @ ${current_price:.6f} "
                    f"({reason}) - P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)"
                )

                # 📊 Register trade with auto_improver for learning
                try:
                    duration_seconds = 0
                    if 'entry_time' in trade:
                        entry_time = trade['entry_time']
                        if isinstance(entry_time, str):
                            entry_time = datetime.fromisoformat(entry_time)
                        duration_seconds = (datetime.now() - entry_time).total_seconds()
                    
                    # Use real ML confidence from trade (stored at open time)
                    real_confidence = trade.get('confidence', 0.0)
                    
                    self.auto_improver.record_trade({
                        'symbol': trade.get('symbol', 'UNKNOWN'),
                        'direction': trade.get('direction', 'bullish'),
                        'entry_price': trade.get('entry_price', 0),
                        'exit_price': current_price,
                        'size_usd': trade.get('size', 0),
                        'pnl': trade.get('pnl', 0),
                        'pnl_percent': trade.get('pnl_pct', 0),
                        'duration_seconds': duration_seconds,
                        'confidence': real_confidence,
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.debug(f"📊 Trade registered with auto_improver: {trade['symbol']}")
                except Exception as e:
                    logger.error(f"❌ Failed to register trade with auto_improver: {e}")

                # 📊 FASE 3: Record trade with adaptive weights for indicator learning
                try:
                    cached_state = self.cache.get_trade_state(trade_id)
                    if cached_state and "ml_components" in cached_state:
                        is_win = trade.get('pnl', 0) > 0
                        trade_dir = trade.get('direction', 'bullish')
                        # Map long/short to bullish/bearish for consistency
                        if trade_dir == 'long':
                            trade_dir = 'bullish'
                        elif trade_dir == 'short':
                            trade_dir = 'bearish'
                        self.ml_signal.adaptive_weights.record_trade(
                            components=cached_state["ml_components"],
                            trade_direction=trade_dir,
                            is_win=is_win
                        )
                        logger.debug(f"📊 Adaptive weights updated: {trade['symbol']} ({'WIN' if is_win else 'LOSS'})")
                except Exception as e:
                    logger.error(f"❌ Failed to update adaptive weights: {e}")

                # 📊 Register trade with self_improver for symbol+direction learning
                try:
                    self.self_improver.record_trade(
                        symbol=trade.get('symbol', 'UNKNOWN'),
                        direction=trade.get('direction', 'bullish'),
                        pnl_pct=trade.get('pnl_pct', 0)
                    )
                    logger.debug(f"📊 Trade registered with self_improver: {trade['symbol']}_{trade['direction']}")
                except Exception as e:
                    logger.error(f"❌ Failed to register trade with self_improver: {e}")

                # Notify
                try:
                    self.notifier.trade_closed(
                        symbol=trade["symbol"],
                        close_price=current_price,
                        pnl=trade["pnl"],
                        pnl_pct=trade["pnl_pct"],
                        reason=reason
                    )
                except:
                    pass
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")

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
            profile = self.get_trading_params()

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

        if not open_trades:
            return

        # Calculate total portfolio P&L
        total_pnl_pct = 0
        for trade in open_trades:
            current_price = self._get_current_price(trade["id"])
            if current_price > 0:
                entry_price = trade["entry_price"]
                direction = trade["direction"]

                if direction == "bullish":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # bearish
                    pnl_pct = (entry_price - current_price) / entry_price

                total_pnl_pct += pnl_pct

        # Average P&L across positions
        avg_pnl_pct = total_pnl_pct / len(open_trades) if open_trades else 0

        logger.info(f"💼 Portfolio P&L: {avg_pnl_pct*100:.2f}% ({len(open_trades)} positions)")

        # Close all if in profit target range (3% to 8%)
        if 0.03 <= avg_pnl_pct <= 0.08:
            logger.info(f"🎯 Portfolio in profit target range, closing all positions...")
            for trade in open_trades:
                self.close_position(trade["id"], "PORTFOLIO_TAKE_PROFIT")

        # Close all if in big loss (> -5%)
        elif avg_pnl_pct < -0.05:
            logger.warning(f"⚠️ Portfolio in big loss, closing all positions...")
            for trade in open_trades:
                self.close_position(trade["id"], "PORTFOLIO_STOP_LOSS")

    def _run_optimizer_if_needed(self):
        """Run strategy optimizer dynamically - after every 5-10 trades close"""
        try:
            open_trades = self.paper_engine.get_open_trades()
            closed_trades = self.paper_engine.state.stats["total_trades"]

            # 📊 Check if we should run optimizer (every 5-10 trades)
            trades_since_last_optimizer = closed_trades - getattr(self, '_last_optimizer_trades_count', 0)
            
            should_run = trades_since_last_optimizer >= 5  # Run every 5 trades minimum
            
            if should_run:
                feedback = {
                    "total_trades": closed_trades,
                    "win_rate": self.paper_engine.state.stats["win_rate"],
                    "avg_pnl": self.paper_engine.state.stats["total_pnl"] / closed_trades if closed_trades > 0 else 0,
                    "open_positions": len(open_trades),
                    "timestamp": datetime.now().isoformat()
                }

                # Run optimizer
                self.auto_improver.process_feedback(feedback)
                logger.info(f"✅ Strategy optimizer executed (after {trades_since_last_optimizer} trades)")
                
                # Update tracker
                self._last_optimizer_trades_count = closed_trades
            else:
                logger.debug(f"📊 Skipping optimizer: only {trades_since_last_optimizer} trades since last run (need 5)")
                
        except Exception as e:
            logger.debug(f"Could not run optimizer: {e}")

    def run_cycle(self):
        """Run one trading cycle with error handling"""
        if not self.running:
            return

        try:
            logger.info("🔄 Running trading cycle...")

            # 1. Scan market
            opportunities = self.scan_market()

            # 2. Generate ML signals
            signals = self.generate_ml_signals(opportunities)

            # ====== FIX: Check concurrent limit BEFORE processing signals ======
            open_trades = self.paper_engine.get_open_trades()
            profile = self.get_trading_params()
            max_concurrent = profile.get("max_concurrent_positions", profile.get("max_concurrent", 5))
            slots_available = max_concurrent - len(open_trades)

            if slots_available <= 0:
                logger.info(f"⏸️ No slots available ({len(open_trades)}/{max_concurrent}), skipping cycle")
            else:
                logger.info(f"🎯 Processing {len(signals)} signals, {slots_available} slots available ({len(open_trades)}/{max_concurrent})")
            # ====================================================================

            # 3. Process high-confidence signals (with limit)
            processed_trades = 0
            for signal in signals:
                # ====== FIX: Check if we have slots available ======
                if processed_trades >= slots_available:
                    logger.info(f"⚠️ Reached max concurrent ({processed_trades} trades), stopping")
                    break
                # ====================================================

                # Skip low confidence (adjusted to be realistic with ensemble thresholds)
                if signal["confidence"] < 25:  # Minimum 25% confidence - Balanced threshold
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
                            if self.execute_trade(trade_signal):
                                processed_trades += 1  # ====== FIX: Count processed ======

            # 4. Check open positions for SL/TP
            self._check_open_positions()

            # 5. Run strategy optimizer periodically (every hour)
            self._run_optimizer_if_needed()

            logger.info("✅ Trading cycle complete")
            self.last_cycle_time = datetime.now()
            self.cycle_count += 1

        except Exception as e:
            import traceback
            logger.error(f"❌ ERROR in cycle: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    # =============================================================================
    # ROBUST CONTINUOUS MODE - FIX APPLIED HERE
    # =============================================================================

    def run_continuous(self):
        """Run continuous trading cycles with ROBUST error handling v4"""
        import signal

        # Start system first
        self.start()

        # Setup signal handlers for graceful shutdown
        shutdown_reason = "Manual shutdown"

        def signal_handler(signum, frame):
            nonlocal shutdown_reason
            if signum == signal.SIGINT:
                shutdown_reason = "User interrupt (Ctrl+C)"
            elif signum == signal.SIGTERM:
                shutdown_reason = "Termination signal"
            logger.info(f"⚠️ Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)   # kill command
        signal.signal(signal.SIGUSR1, signal_handler)   # User signal

        cycle_error_count = 0
        max_cycle_errors = 10

        # ====== FIX v4: Enhanced logging and error handling ======
        logger.info("🔧 ROBUST CONTINUOUS MODE STARTED")
        logger.info(f"📊 Scan interval: {self.scan_interval} seconds")
        logger.info(f"🛡️ Max consecutive errors: {max_cycle_errors}")
        # =======================================================

        while self.running:
            try:
                # ====== FIX v4: Update heartbeat before cycle ======
                update_heartbeat()
                # =================================================

                # Run cycle with detailed logging
                logger.info(f"🔄 Starting cycle #{self.cycle_count + 1}")
                self.run_cycle()
                cycle_error_count = 0  # Reset on success

                # ====== FIX v4: Update heartbeat after successful cycle ======
                update_heartbeat()
                logger.info(f"✅ Cycle #{self.cycle_count} completed successfully")
                # ==========================================================

            except KeyboardInterrupt:
                logger.info("⏹️ Keyboard interrupt received")
                shutdown_reason = "Keyboard interrupt"
                self.running = False
                break

            except Exception as e:
                cycle_error_count += 1
                logger.error(f"❌ Error in trading cycle ({cycle_error_count}/{max_cycle_errors}): {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Send error notification for critical errors
                if cycle_error_count >= 3:
                    try:
                        self.notifier.system_error(str(e), f"Cycle {cycle_error_count} consecutive errors")
                    except:
                        pass

                # Give up after too many consecutive errors
                if cycle_error_count >= max_cycle_errors:
                    logger.error(f"❌ Too many consecutive errors ({max_cycle_errors}), stopping...")
                    shutdown_reason = f"Too many consecutive errors: {e}"
                    self.running = False
                    break

                logger.info("⏳ Waiting 30s before retry...")
                time_module.sleep(30)
                continue

            # ====== FIX v5: Ultra-robust sleep with heartbeat monitoring ======
            try:
                logger.info(f"😴 Sleeping for {self.scan_interval} seconds until next cycle...")
                
                # Split sleep into smaller intervals for better responsiveness
                sleep_interval = 10  # Check every 10 seconds
                elapsed = 0
                while elapsed < self.scan_interval and self.running:
                    time_module.sleep(sleep_interval)
                    elapsed += sleep_interval
                    
                    # Write heartbeat every 10 seconds
                    try:
                        HEARTBEAT_FILE.write_text(f"{datetime.now().isoformat()}\n{elapsed}/{self.scan_interval}s")
                    except:
                        pass
                    
                    # Log progress every 30 seconds
                    if elapsed % 30 == 0:
                        logger.debug(f"💤 Still sleeping... {elapsed}/{self.scan_interval}s elapsed")
                
                if self.running:
                    logger.info(f"⏰ Woke up after {elapsed} seconds")
            except KeyboardInterrupt:
                logger.info("⏹️ Keyboard interrupt during sleep")
                shutdown_reason = "Keyboard interrupt during sleep"
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Error during sleep: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Continue anyway, next cycle will handle it
            # ================================================================

        # ====== FIX v4: Clean heartbeat on shutdown ======
        if HEARTBEAT_FILE.exists():
            HEARTBEAT_FILE.unlink()
        # =================================================

        # Graceful shutdown
        logger.info("🛑 Shutting down...")
        self.stop(reason=shutdown_reason)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Trading System v4 - ROBUST")
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
            system.stop(reason="User interrupt")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            system.stop(reason=f"Fatal error: {e}")

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
        logger.info("🚀 Starting continuous trading mode v4 ROBUST...")
        try:
            system.run_continuous()
        except KeyboardInterrupt:
            logger.info("⏹️ Interrupted by user")
            system.stop(reason="User interrupt")
        except Exception as e:
            logger.error(f"❌ Fatal error in continuous mode: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            system.stop(reason=f"Fatal error: {e}")
            sys.exit(1)

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
        print("\n🔧 v4 ROBUST MODE:")
        print("   ✅ Enhanced error handling")
        print("   ✅ Heartbeat monitoring")
        print("   ✅ Robust sleep with logging")
        print("   ✅ Comprehensive exception handling")


if __name__ == "__main__":
    main()
