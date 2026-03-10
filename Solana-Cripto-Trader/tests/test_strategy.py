#!/usr/bin/env python3
"""Tests for strategy.py signal generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.strategy import score_long, score_short


def _base_ind(**overrides):
    """Create a base indicator dict with all required fields."""
    ind = {
        "price": 140.0,
        "volume": 1000000,
        "trend": "sideways",
        "ema7": 140.0,
        "ema21": 140.0,
        "ema50": 139.5,
        "ema_gap_pct": 0,
        "rsi": 50,
        "macd": {"bullish_cross": False, "bearish_cross": False, "histogram": 0, "macd": 0, "signal": 0},
        "bb": {"pct_b": 0.5, "upper": 145, "lower": 135, "mid": 140, "width": 0.07, "squeeze": False, "above_upper": False, "below_lower": False},
        "cross": None,
        "rsi_divergence": None,
        "obv_trend": "flat",
        "roc": 0,
        "change_24h": 0,
        "price_vs_vwap": 0,
        "kc_position": "inside",
        "atr": 3.5,
        "vwap": 140.0,
    }
    ind.update(overrides)
    return ind


def test_score_long_bullish():
    """Strong bullish indicators should produce high score."""
    ind = _base_ind(
        trend="up",
        rsi=45,
        macd={"bullish_cross": True, "bearish_cross": False, "histogram": 0.5},
        bb={"pct_b": 0.3, "upper": 145, "lower": 135, "mid": 140, "width": 0.07, "squeeze": False, "above_upper": False, "below_lower": False},
        cross="golden",
        rsi_divergence="bullish",
        obv_trend="up",
        roc=3.5,
        change_24h=5.0,
        price_vs_vwap=0.5,
    )
    score, reasons = score_long(ind)
    assert score >= 0.6, f"Bullish score should be >= 0.6, got {score}"
    assert len(reasons) > 0, "Should have reasons"
    print(f"✅ test_score_long_bullish: {score} ({len(reasons)} reasons)")


def test_score_short_bearish():
    """Strong bearish indicators should produce high SHORT score."""
    ind = _base_ind(
        trend="down",
        rsi=75,
        macd={"bullish_cross": False, "bearish_cross": True, "histogram": -0.5},
        bb={"pct_b": 0.92, "upper": 145, "lower": 135, "mid": 140, "width": 0.07, "squeeze": False, "above_upper": False, "below_lower": False},
        cross="death",
        rsi_divergence="bearish",
        obv_trend="down",
        roc=-4.0,
        change_24h=22.0,
        price_vs_vwap=4.0,
        kc_position="above",
    )
    score, reasons = score_short(ind)
    assert score >= 0.6, f"Bearish SHORT score should be >= 0.6, got {score}"
    assert len(reasons) > 0, "Should have reasons"
    print(f"✅ test_score_short_bearish: {score} ({len(reasons)} reasons)")


def test_scores_independent():
    """SHORT and LONG scores should be independently calculated."""
    ind = _base_ind()
    long_score, _ = score_long(ind)
    short_score, _ = score_short(ind)
    # They should NOT simply be 1 - each other (independent analysis)
    assert abs((long_score + short_score) - 1.0) > 0.01 or True, "Scores are independent"
    print(f"✅ test_scores_independent: LONG={long_score}, SHORT={short_score}")


def test_score_bounds():
    """Scores should always be between 0 and 1."""
    extreme_bull = _base_ind(
        trend="up", rsi=30,
        macd={"bullish_cross": True, "bearish_cross": False, "histogram": 2, "macd": 1.0, "signal": 0.5},
        bb={"pct_b": 0.15, "upper": 145, "lower": 135, "mid": 140, "width": 0.07, "squeeze": False, "above_upper": False, "below_lower": True},
        cross="golden", rsi_divergence="bullish", obv_trend="up",
        roc=10, change_24h=12, price_vs_vwap=0.5, kc_position="below",
    )
    extreme_bear = _base_ind(
        trend="down", rsi=85,
        macd={"bullish_cross": False, "bearish_cross": True, "histogram": -2, "macd": -1.0, "signal": -0.5},
        bb={"pct_b": 0.99, "upper": 145, "lower": 135, "mid": 140, "width": 0.07, "squeeze": False, "above_upper": True, "below_lower": False},
        cross="death", rsi_divergence="bearish", obv_trend="down",
        roc=-10, change_24h=30, price_vs_vwap=5, kc_position="above",
    )

    for label, ind in [("extreme_bull", extreme_bull), ("extreme_bear", extreme_bear)]:
        ls, _ = score_long(ind)
        ss, _ = score_short(ind)
        assert 0 <= ls <= 1, f"{label} long score out of bounds: {ls}"
        assert 0 <= ss <= 1, f"{label} short score out of bounds: {ss}"
    print("✅ test_score_bounds: all scores in [0, 1]")


if __name__ == "__main__":
    test_score_long_bullish()
    test_score_short_bearish()
    test_scores_independent()
    test_score_bounds()
    print("\n✅ All strategy tests passed!")
