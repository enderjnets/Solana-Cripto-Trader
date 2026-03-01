"""
ML Model for Trading Signal Generation
=======================================
Real machine learning model using GradientBoostingClassifier.

Features:
- RSI value
- EMA cross ratio (short_ema / long_ema)
- Momentum (5-period price change %)
- Volatility (std of returns)
- Hour of day (cyclical encoding)
- Symbol (label encoded)

Target: 1 (win) / 0 (loss)

Retrains every 50 trades closed.
Falls back to technical indicators if <30 trades available.
Persists model to data/ml_model.pkl.
"""

import json
import logging
import pickle
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ml_model")

MODEL_FILE = Path(__file__).parent.parent / "data" / "ml_model.pkl"
MODEL_STATE_FILE = Path(__file__).parent.parent / "data" / "ml_model_state.json"
MIN_TRADES_FOR_MODEL = 30
RETRAIN_INTERVAL = 50


class TradingMLModel:
    """
    Gradient Boosting model for predicting trade success.
    
    Features (per trade):
    - rsi: RSI indicator value (0-100)
    - ema_ratio: short_ema / long_ema ratio
    - momentum: 5-period price change percentage
    - volatility: standard deviation of recent returns
    - hour_sin: sin(hour * 2pi / 24) - cyclical time encoding
    - hour_cos: cos(hour * 2pi / 24) - cyclical time encoding
    - symbol_encoded: integer label for the trading symbol
    
    Target: 1 (profitable trade), 0 (losing trade)
    """

    def __init__(self):
        self.model = None
        self.symbol_encoder: Dict[str, int] = {}
        self.next_symbol_id = 0
        self.training_data: List[Dict] = []
        self.trades_since_retrain = 0
        self.total_retrains = 0
        self.model_accuracy = 0.0
        self.is_ready = False
        
        self._load_state()
        self._load_model()

    def _load_state(self):
        """Load model state (training data, symbol encoder, etc.)."""
        if MODEL_STATE_FILE.exists():
            try:
                data = json.loads(MODEL_STATE_FILE.read_text())
                self.training_data = data.get("training_data", [])
                self.symbol_encoder = data.get("symbol_encoder", {})
                self.next_symbol_id = data.get("next_symbol_id", 0)
                self.trades_since_retrain = data.get("trades_since_retrain", 0)
                self.total_retrains = data.get("total_retrains", 0)
                self.model_accuracy = data.get("model_accuracy", 0.0)
                logger.info(f"📊 ML Model state loaded: {len(self.training_data)} training samples")
            except Exception as e:
                logger.warning(f"Could not load ML model state: {e}")

    def _save_state(self):
        """Save model state."""
        MODEL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "training_data": self.training_data[-500:],  # Keep last 500 samples
            "symbol_encoder": self.symbol_encoder,
            "next_symbol_id": self.next_symbol_id,
            "trades_since_retrain": self.trades_since_retrain,
            "total_retrains": self.total_retrains,
            "model_accuracy": self.model_accuracy,
            "last_update": datetime.now().isoformat()
        }
        MODEL_STATE_FILE.write_text(json.dumps(data, indent=2))

    def _load_model(self):
        """Load trained model from pickle."""
        if MODEL_FILE.exists():
            try:
                with open(MODEL_FILE, "rb") as f:
                    self.model = pickle.load(f)
                self.is_ready = len(self.training_data) >= MIN_TRADES_FOR_MODEL
                if self.is_ready:
                    logger.info(f"✅ ML Model loaded (accuracy: {self.model_accuracy:.0%})")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
                self.model = None
                self.is_ready = False

    def _save_model(self):
        """Save trained model to pickle."""
        if self.model is not None:
            MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(self.model, f)

    def _encode_symbol(self, symbol: str) -> int:
        """Encode symbol as integer."""
        if symbol not in self.symbol_encoder:
            self.symbol_encoder[symbol] = self.next_symbol_id
            self.next_symbol_id += 1
        return self.symbol_encoder[symbol]

    def _extract_features(self, rsi: float, ema_short: float, ema_long: float,
                          momentum: float, volatility: float,
                          hour: int, symbol: str) -> List[float]:
        """Extract feature vector from indicators."""
        # EMA cross ratio
        ema_ratio = ema_short / ema_long if ema_long != 0 else 1.0
        
        # Cyclical hour encoding
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # Symbol encoding
        symbol_id = self._encode_symbol(symbol)
        
        return [
            rsi,
            ema_ratio,
            momentum,
            volatility,
            hour_sin,
            hour_cos,
            float(symbol_id)
        ]

    def record_trade(self, rsi: float, ema_short: float, ema_long: float,
                     momentum: float, volatility: float,
                     hour: int, symbol: str, direction: str, is_win: bool):
        """
        Record a completed trade for training data.
        
        Args:
            rsi, ema_short, ema_long, momentum, volatility: indicator values at entry
            hour: hour of day at entry (0-23)
            symbol: trading symbol
            direction: 'bullish' or 'bearish'
            is_win: whether the trade was profitable
        """
        features = self._extract_features(rsi, ema_short, ema_long,
                                          momentum, volatility, hour, symbol)
        
        self.training_data.append({
            "features": features,
            "target": 1 if is_win else 0,
            "direction": direction,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        })
        
        self.trades_since_retrain += 1
        
        # Check if we should retrain
        if (len(self.training_data) >= MIN_TRADES_FOR_MODEL and 
            self.trades_since_retrain >= RETRAIN_INTERVAL):
            self._train_model()
        
        self._save_state()

    def _train_model(self):
        """Train/retrain the Gradient Boosting model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score
            
            # Prepare data
            X = np.array([d["features"] for d in self.training_data])
            y = np.array([d["target"] for d in self.training_data])
            
            # Train model (lightweight config)
            self.model = GradientBoostingClassifier(
                n_estimators=50,       # Small number of trees (fast)
                max_depth=3,           # Shallow trees
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42
            )
            
            # Cross-validation to estimate accuracy
            if len(self.training_data) >= 10:
                n_splits = min(5, len(self.training_data) // 5)
                if n_splits >= 2:
                    scores = cross_val_score(self.model, X, y, cv=n_splits, scoring='accuracy')
                    self.model_accuracy = float(np.mean(scores))
                else:
                    self.model_accuracy = 0.5
            
            # Train on full dataset
            self.model.fit(X, y)
            
            self.is_ready = True
            self.trades_since_retrain = 0
            self.total_retrains += 1
            
            # Save model and state
            self._save_model()
            self._save_state()
            
            # Log feature importances
            feature_names = ["RSI", "EMA_ratio", "Momentum", "Volatility", 
                           "Hour_sin", "Hour_cos", "Symbol"]
            importances = self.model.feature_importances_
            importance_str = ", ".join(
                f"{name}={imp:.0%}" 
                for name, imp in sorted(zip(feature_names, importances), 
                                       key=lambda x: x[1], reverse=True)
            )
            
            logger.info(
                f"🧠 ML Model retrained #{self.total_retrains}: "
                f"accuracy={self.model_accuracy:.0%}, "
                f"samples={len(self.training_data)}, "
                f"importances: {importance_str}"
            )
            
        except Exception as e:
            logger.error(f"❌ ML Model training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def predict_confidence(self, rsi: float, ema_short: float, ema_long: float,
                           momentum: float, volatility: float,
                           hour: int, symbol: str) -> Optional[float]:
        """
        Predict trade success probability.
        
        Returns:
            Confidence score (0-100) or None if model not ready
        """
        if not self.is_ready or self.model is None:
            return None
        
        try:
            features = self._extract_features(rsi, ema_short, ema_long,
                                              momentum, volatility, hour, symbol)
            X = np.array([features])
            
            # Get probability of win (class 1)
            proba = self.model.predict_proba(X)[0]
            
            # proba is [prob_loss, prob_win]
            win_probability = proba[1] if len(proba) > 1 else proba[0]
            
            # Scale to 0-100 confidence
            confidence = win_probability * 100
            
            return confidence
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            "is_ready": self.is_ready,
            "training_samples": len(self.training_data),
            "min_samples_required": MIN_TRADES_FOR_MODEL,
            "trades_since_retrain": self.trades_since_retrain,
            "retrain_interval": RETRAIN_INTERVAL,
            "total_retrains": self.total_retrains,
            "model_accuracy": self.model_accuracy,
            "symbols_known": list(self.symbol_encoder.keys()),
            "model_file_exists": MODEL_FILE.exists()
        }


def seed_from_paper_trading(model: TradingMLModel, paper_state_file: str = "data/paper_trading_state.json"):
    """
    Seed the ML model with existing closed trades from paper trading.
    
    Extracts approximate features from trade data (some features are estimated
    since we don't have the exact indicator values at entry time).
    """
    state_file = Path(__file__).parent.parent / paper_state_file
    if not state_file.exists():
        logger.warning(f"Paper trading state file not found: {state_file}")
        return
    
    data = json.loads(state_file.read_text())
    closed_trades = [t for t in data.get("trades", []) if t.get("status") == "closed"]
    
    logger.info(f"Seeding ML model with {len(closed_trades)} closed trades...")
    
    for trade in closed_trades:
        symbol = trade.get("symbol", "UNKNOWN")
        direction = trade.get("direction", "long")
        pnl = trade.get("pnl", 0)
        is_win = pnl > 0
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        
        # Parse entry time for hour
        entry_time_str = trade.get("entry_time", "")
        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            hour = entry_time.hour
        except:
            hour = 12  # Default noon
        
        # Estimate features from trade data (approximate since we don't have exact indicators)
        # RSI: estimate from price direction
        if direction in ["long", "bullish"]:
            estimated_rsi = 35.0 if is_win else 55.0  # Oversold worked
        else:
            estimated_rsi = 65.0 if is_win else 45.0  # Overbought worked
        
        # EMA ratio: estimate from price movement
        price_change = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        estimated_ema_short = entry_price * (1 + price_change * 0.3)
        estimated_ema_long = entry_price * (1 - price_change * 0.1)
        
        # Momentum: estimate from PnL
        pnl_pct = trade.get("pnl_pct", 0)
        estimated_momentum = pnl_pct * 0.5  # Rough estimate
        
        # Volatility: estimate from trade duration and PnL magnitude
        duration = trade.get("duration_seconds", 3600)
        estimated_volatility = abs(pnl_pct) / (duration / 3600) * 0.01 if duration > 0 else 0.02
        estimated_volatility = max(0.005, min(0.1, estimated_volatility))
        
        # Map direction
        if direction == "long":
            direction = "bullish"
        elif direction == "short":
            direction = "bearish"
        
        model.record_trade(
            rsi=estimated_rsi,
            ema_short=estimated_ema_short,
            ema_long=estimated_ema_long,
            momentum=estimated_momentum,
            volatility=estimated_volatility,
            hour=hour,
            symbol=symbol,
            direction=direction,
            is_win=is_win
        )
    
    # Force training if we have enough data
    if len(model.training_data) >= MIN_TRADES_FOR_MODEL:
        model._train_model()
    
    logger.info(f"✅ ML Model seeded with {len(closed_trades)} trades, ready={model.is_ready}")


# Test / Seed
if __name__ == "__main__":
    import sys
    
    model = TradingMLModel()
    
    if "--seed" in sys.argv:
        seed_from_paper_trading(model)
        print(f"\nModel stats: {json.dumps(model.get_stats(), indent=2)}")
    elif "--test" in sys.argv:
        # Quick prediction test
        if model.is_ready:
            conf = model.predict_confidence(
                rsi=35.0, ema_short=87.5, ema_long=86.0,
                momentum=2.5, volatility=0.03,
                hour=14, symbol="SOL"
            )
            print(f"Prediction confidence: {conf:.1f}%")
        else:
            print(f"Model not ready. Need {MIN_TRADES_FOR_MODEL} trades, have {len(model.training_data)}")
    else:
        print(f"Stats: {json.dumps(model.get_stats(), indent=2)}")
