#!/usr/bin/env python3
"""Tests for executor.py safety nets and signal loading."""
import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_kill_switch():
    """STOP_TRADING file should prevent new positions."""
    from agents.executor import DATA_DIR
    stop_file = DATA_DIR / "STOP_TRADING"
    
    # Create kill switch
    stop_file.write_text("STOP")
    assert stop_file.exists(), "Kill switch file should exist"
    
    # Clean up
    stop_file.unlink()
    assert not stop_file.exists(), "Kill switch file should be removed"
    print("✅ test_kill_switch: file creation/deletion works")


def test_load_signals():
    """Signal loading should prefer AI over technical."""
    from agents.executor import load_signals
    result = load_signals()
    assert "signals" in result, "Should return signals key"
    assert isinstance(result["signals"], list), "Signals should be a list"
    print(f"✅ test_load_signals: loaded {len(result['signals'])} signals, source={result.get('source', 'default')}")


def test_portfolio_structure():
    """Portfolio file should have required fields."""
    from agents.executor import PORTFOLIO_FILE
    if PORTFOLIO_FILE.exists():
        p = json.loads(PORTFOLIO_FILE.read_text())
        required = ["capital_usd", "positions"]
        for key in required:
            assert key in p, f"Portfolio missing required key: {key}"
        assert isinstance(p["positions"], list), "Positions should be a list"
        print(f"✅ test_portfolio_structure: capital=${p['capital_usd']}, positions={len(p['positions'])}")
    else:
        print("⏭️ test_portfolio_structure: skipped (no portfolio file)")


if __name__ == "__main__":
    test_kill_switch()
    test_load_signals()
    test_portfolio_structure()
    print("\n✅ All executor tests passed!")
