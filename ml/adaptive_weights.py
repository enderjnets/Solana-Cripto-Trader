"""
Adaptive ML Weights System
==========================
Tracks performance of each indicator (RSI, EMA, Momentum, Trend) vs trade outcomes.
Periodically recalculates optimal weights based on which indicators predicted correctly.

Persists weights to data/ml_weights.json between restarts.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("adaptive_weights")

WEIGHTS_FILE = Path(__file__).parent.parent / "data" / "ml_weights.json"
DEFAULT_WEIGHTS = {
    "rsi": 0.30,
    "ema_crossover": 0.25,
    "momentum": 0.25,
    "trend": 0.20
}
# How many trades before recalculating weights
RECALC_INTERVAL = 20


class AdaptiveWeights:
    """
    Tracks each indicator's contribution to trade outcomes and adjusts weights.
    
    For each closed trade, we record:
    - Each indicator's signal direction (bullish/bearish/neutral) and score
    - Whether the trade was a win or loss
    
    Indicators that correctly predicted the outcome get higher weights.
    """

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load weights and tracking data from file."""
        if WEIGHTS_FILE.exists():
            try:
                data = json.loads(WEIGHTS_FILE.read_text())
                # Ensure all keys exist
                if "weights" not in data:
                    data["weights"] = DEFAULT_WEIGHTS.copy()
                if "indicator_scores" not in data:
                    data["indicator_scores"] = self._init_scores()
                if "trades_since_recalc" not in data:
                    data["trades_since_recalc"] = 0
                if "total_trades_tracked" not in data:
                    data["total_trades_tracked"] = 0
                if "recalc_history" not in data:
                    data["recalc_history"] = []
                return data
            except Exception as e:
                logger.warning(f"Could not load weights file: {e}")
        
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "indicator_scores": self._init_scores(),
            "trades_since_recalc": 0,
            "total_trades_tracked": 0,
            "recalc_history": [],
            "last_update": None
        }

    def _init_scores(self) -> Dict:
        """Initialize indicator score tracking."""
        return {
            "rsi": {"correct": 0, "incorrect": 0, "neutral": 0},
            "ema_crossover": {"correct": 0, "incorrect": 0, "neutral": 0},
            "momentum": {"correct": 0, "incorrect": 0, "neutral": 0},
            "trend": {"correct": 0, "incorrect": 0, "neutral": 0}
        }

    def _save_state(self):
        """Save state to file."""
        self.state["last_update"] = datetime.now().isoformat()
        WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_FILE.write_text(json.dumps(self.state, indent=2))

    def get_weights(self) -> Dict[str, float]:
        """Get current adaptive weights."""
        return self.state["weights"].copy()

    def record_trade(self, components: Dict, trade_direction: str, is_win: bool):
        """
        Record a trade outcome and track which indicators were correct.
        
        Args:
            components: Dict from MLSignalGenerator.generate_signal()["components"]
                        Each has 'signal' (bullish/bearish/neutral) and 'score'
            trade_direction: 'bullish'/'bearish' (the direction the trade was taken)
            is_win: Whether the trade was profitable
        """
        for indicator_name in ["rsi", "ema_crossover", "momentum", "trend"]:
            if indicator_name not in components:
                continue
            
            component = components[indicator_name]
            indicator_signal = component.get("signal", "neutral")
            
            if indicator_name not in self.state["indicator_scores"]:
                self.state["indicator_scores"][indicator_name] = {
                    "correct": 0, "incorrect": 0, "neutral": 0
                }
            
            if indicator_signal == "neutral":
                self.state["indicator_scores"][indicator_name]["neutral"] += 1
            elif is_win:
                # Indicator agreed with trade direction AND trade won
                if indicator_signal == trade_direction:
                    self.state["indicator_scores"][indicator_name]["correct"] += 1
                else:
                    # Indicator disagreed but trade still won (indicator was wrong)
                    self.state["indicator_scores"][indicator_name]["incorrect"] += 1
            else:
                # Trade lost
                if indicator_signal == trade_direction:
                    # Indicator agreed with trade direction but it lost (indicator was wrong)
                    self.state["indicator_scores"][indicator_name]["incorrect"] += 1
                else:
                    # Indicator disagreed and trade lost (indicator was right to disagree)
                    self.state["indicator_scores"][indicator_name]["correct"] += 1
        
        self.state["trades_since_recalc"] += 1
        self.state["total_trades_tracked"] += 1
        
        # Check if we should recalculate weights
        if self.state["trades_since_recalc"] >= RECALC_INTERVAL:
            self._recalculate_weights()
        
        self._save_state()

    def _recalculate_weights(self):
        """Recalculate indicator weights based on accumulated scores."""
        scores = self.state["indicator_scores"]
        
        # Calculate accuracy for each indicator
        accuracies = {}
        for name, data in scores.items():
            total = data["correct"] + data["incorrect"]
            if total > 0:
                accuracies[name] = data["correct"] / total
            else:
                accuracies[name] = 0.5  # Default 50% if no data
        
        # Convert accuracies to weights (normalize to sum to 1.0)
        # Apply softmax-like transformation: better accuracy → higher weight
        # But clamp minimum weight to 0.10 so no indicator is completely ignored
        MIN_WEIGHT = 0.10
        MAX_WEIGHT = 0.40
        
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            raw_weights = {k: v / total_accuracy for k, v in accuracies.items()}
        else:
            raw_weights = {k: 0.25 for k in accuracies}
        
        # Iteratively clamp and re-normalize until stable
        new_weights = raw_weights.copy()
        for _ in range(10):  # Max iterations to converge
            clamped = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for k, v in new_weights.items()}
            total = sum(clamped.values())
            new_weights = {k: v / total for k, v in clamped.items()}
            # Check if all within bounds (with float tolerance)
            if all(v >= MIN_WEIGHT - 1e-9 and v <= MAX_WEIGHT + 1e-9 for v in new_weights.values()):
                break
        
        # Final clamp to ensure strict bounds despite float precision
        new_weights = {k: round(max(MIN_WEIGHT, min(MAX_WEIGHT, v)), 6) for k, v in new_weights.items()}
        # Re-normalize after rounding
        total = sum(new_weights.values())
        new_weights = {k: round(v / total, 6) for k, v in new_weights.items()}
        
        old_weights = self.state["weights"].copy()
        self.state["weights"] = new_weights
        
        # Record history
        self.state["recalc_history"].append({
            "timestamp": datetime.now().isoformat(),
            "trades_used": self.state["trades_since_recalc"],
            "accuracies": accuracies,
            "old_weights": old_weights,
            "new_weights": new_weights
        })
        
        # Keep last 10 recalculations
        if len(self.state["recalc_history"]) > 10:
            self.state["recalc_history"] = self.state["recalc_history"][-10:]
        
        # Reset counter (but keep accumulated scores for continued learning)
        self.state["trades_since_recalc"] = 0
        
        logger.info(
            f"🔄 Weights recalculated: "
            f"RSI={new_weights.get('rsi', 0):.0%}, "
            f"EMA={new_weights.get('ema_crossover', 0):.0%}, "
            f"Mom={new_weights.get('momentum', 0):.0%}, "
            f"Trend={new_weights.get('trend', 0):.0%} "
            f"(from {self.state['total_trades_tracked']} trades)"
        )

    def get_stats(self) -> Dict:
        """Get adaptive weights stats."""
        scores = self.state["indicator_scores"]
        accuracies = {}
        for name, data in scores.items():
            total = data["correct"] + data["incorrect"]
            accuracies[name] = data["correct"] / total if total > 0 else 0.5
        
        return {
            "current_weights": self.state["weights"],
            "indicator_accuracies": accuracies,
            "indicator_scores": scores,
            "trades_since_recalc": self.state["trades_since_recalc"],
            "total_trades_tracked": self.state["total_trades_tracked"],
            "recalc_interval": RECALC_INTERVAL,
            "recalc_history_count": len(self.state.get("recalc_history", []))
        }


# Test
if __name__ == "__main__":
    aw = AdaptiveWeights()
    print(f"Current weights: {aw.get_weights()}")
    print(f"Stats: {json.dumps(aw.get_stats(), indent=2)}")
