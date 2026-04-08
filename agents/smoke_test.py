#!/usr/bin/env python3
"""
Smoke Test for Solana Trading Bot Runtime
==========================================
Fails if:
  - Two or more orchestrator processes are running simultaneously
  - State files are being read from legacy root paths
  - The orchestrator lock is not properly set up

Run manually:
    python3 agents/smoke_test.py

Run in CI/automated context (exit code 0 = pass, 1 = fail):
    python3 agents/smoke_test.py && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED"
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Add workspace/ to path so we can import executor for unit tests
AGENTS_DIR = Path(__file__).parent
WORKSPACE_DIR = AGENTS_DIR.parent
sys.path.insert(0, str(WORKSPACE_DIR))

LEGACY_STATE_FILES = [
    WORKSPACE_DIR / "paper_trading_state.json",
    WORKSPACE_DIR / "unified_brain_state.json",
]

ORCHESTRATOR_LOCK_FILE = Path("/tmp/solana_jupiter_orchestrator.lock")
ORCHESTRATOR_SCRIPT = AGENTS_DIR / "orchestrator.py"


def get_orchestrator_pids() -> list[int]:
    """Return PIDs of all running orchestrator.py processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "orchestrator\\.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = []
            for pid in result.stdout.strip().split("\n"):
                if not pid:
                    continue
                cmdline_path = f"/proc/{pid}/cmdline"
                try:
                    cmdline = Path(cmdline_path).read_text()
                    if "smoke_test" in cmdline or "smoke-test" in cmdline:
                        continue
                except Exception:
                    pass
                pids.append(int(pid))
            return pids
        return []
    except Exception:
        return []


def check_duplicate_orchestrators() -> tuple[bool, str]:
    """Check for multiple orchestrator instances."""
    pids = get_orchestrator_pids()
    if len(pids) == 0:
        return False, "No orchestrator process running"
    if len(pids) > 1:
        return False, f"DUPLICATE ORCHESTRATORS: PIDs {pids} — only 1 allowed"
    return True, f"OK: 1 orchestrator running (PID {pids[0]})"


def check_lock_file() -> tuple[bool, str]:
    """Check that the orchestrator lock file is consistent with running process."""
    pids = get_orchestrator_pids()
    if not pids:
        return True, "No orchestrator running, lock file not checked"

    running_pid = pids[0]

    if not ORCHESTRATOR_LOCK_FILE.exists():
        return False, f"LOCK MISSING: orchestrator running (PID {running_pid}) but lock file {ORCHESTRATOR_LOCK_FILE} does not exist"

    try:
        lock_pid = int(ORCHESTRATOR_LOCK_FILE.read_text().strip())
    except (ValueError, IOError) as e:
        return False, f"LOCk CORRUPT: could not read PID from lock file: {e}"

    if lock_pid != running_pid:
        return False, f"LOCk MISMATCH: lock file has PID {lock_pid} but running PID is {running_pid}"

    return True, f"OK: lock file holds PID {lock_pid} matching running process"


def check_canonical_state() -> tuple[bool, str]:
    """Verify that the canonical state directory exists and has required files."""
    data_dir = AGENTS_DIR / "data"
    if not data_dir.exists():
        return False, f"CANONICAL STATE DIR MISSING: {data_dir} does not exist"

    required = ["portfolio.json", "trade_history.json"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        return False, f"CANONICAL STATE INCOMPLETE: missing {missing} in {data_dir}"

    return True, f"OK: canonical state dir {data_dir} has required files"


def check_legacy_files_not_used_for_health() -> tuple[bool, str]:
    """Check that legacy root files are stale and small (not actively written)."""
    now = datetime.now(timezone.utc).timestamp()
    issues = []
    for lf in LEGACY_STATE_FILES:
        if not lf.exists():
            continue
        age_hours = (now - lf.stat().st_mtime) / 3600
        size = lf.stat().st_size
        if age_hours < 24 and size > 100:
            issues.append(f"{lf.name}: age={age_hours:.1f}h size={size}B (may be active)")

    if issues:
        return False, f"LEGACY FILES POSSIBLY ACTIVE: {', '.join(issues)}"
    return True, "OK: legacy root files are stale/ignored (not used for health checks)"


def check_lock_is_activated_in_code() -> tuple[bool, str]:
    """Verify that _acquire_lock() is called in the orchestrator __main__ loop section."""
    try:
        content = ORCHESTRATOR_SCRIPT.read_text()
    except IOError as e:
        return False, f"Cannot read orchestrator.py: {e}"

    if "_acquire_lock()" not in content:
        return False, "LOCK DEFINITION MISSING: _acquire_lock() not found in orchestrator.py"

    lines = content.split("\n")
    in_main = False
    in_else_block = False
    found_acquire_lock = False
    found_loop = False

    for i, line in enumerate(lines):
        if "__main__" in line:
            in_main = True
        if in_main and "else:" in line and ":" in line:
            in_else_block = True
        if in_else_block and "_acquire_lock()" in line:
            found_acquire_lock = True
        if in_else_block and "while True" in line:
            found_loop = True
            break

    if not found_acquire_lock:
        return False, "LOCK NOT ACTIVATED: _acquire_lock() defined but not called in loop mode __main__"

    return True, "OK: _acquire_lock() is activated in orchestrator.py loop entry"


def _run_test(test_fn) -> tuple[bool, str]:
    """Wrapper that calls a test function and returns (ok, message)."""
    try:
        return test_fn()
    except AssertionError as e:
        return False, f"ASSERTION FAILED: {e}"
    except Exception as e:
        return False, f"ERROR: {e}"


def _restore_env(key, value):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


# ─── Anti-Rebound SHORT Filter Tests (SOLAA-23) ────────────────────────────────

def test_short_rebound_filter_blocked():
    """SHORT + RSI 28 + trend bullish => bloqueado."""
    from agents import executor as ex

    orig_enabled = os.environ.get("SHORT_REBOUND_FILTER_ENABLED")
    orig_dry_run = os.environ.get("SHORT_REBOUND_FILTER_DRY_RUN")

    try:
        os.environ["SHORT_REBOUND_FILTER_ENABLED"] = "true"
        os.environ["SHORT_REBOUND_FILTER_DRY_RUN"] = "false"

        signal = {
            "symbol": "FARTCOIN",
            "direction": "short",
            "rsi": 28,
            "trend": "up",
        }
        market = {"tokens": {"FARTCOIN": {}}}

        blocked, reason = ex._should_block_short_rebound(signal, market)
        assert blocked, f"Expected SHORT to be blocked but reason={reason}"
        assert "RSI=" in reason and "bullish" in reason
        return True, f"OK: SHORT+RSI28+bullish blocked ({reason})"
    finally:
        _restore_env("SHORT_REBOUND_FILTER_ENABLED", orig_enabled)
        _restore_env("SHORT_REBOUND_FILTER_DRY_RUN", orig_dry_run)


def test_short_rebound_filter_permitted_downtrend():
    """SHORT + RSI 28 + trend down => PERMITIDO."""
    from agents import executor as ex

    orig_enabled = os.environ.get("SHORT_REBOUND_FILTER_ENABLED")
    orig_dry_run = os.environ.get("SHORT_REBOUND_FILTER_DRY_RUN")

    try:
        os.environ["SHORT_REBOUND_FILTER_ENABLED"] = "true"
        os.environ["SHORT_REBOUND_FILTER_DRY_RUN"] = "false"

        signal = {
            "symbol": "FARTCOIN",
            "direction": "short",
            "rsi": 28,
            "trend": "down",
        }
        market = {"tokens": {"FARTCOIN": {}}}

        blocked, reason = ex._should_block_short_rebound(signal, market)
        assert not blocked, f"Expected SHORT to be permitted but blocked=True"
        return True, "OK: SHORT+RSI28+downtrend permitted"
    finally:
        _restore_env("SHORT_REBOUND_FILTER_ENABLED", orig_enabled)
        _restore_env("SHORT_REBOUND_FILTER_DRY_RUN", orig_dry_run)


def test_short_rebound_filter_permitted_high_rsi():
    """SHORT + RSI 35 + trend bullish => PERMITIDO."""
    from agents import executor as ex

    orig_enabled = os.environ.get("SHORT_REBOUND_FILTER_ENABLED")
    orig_dry_run = os.environ.get("SHORT_REBOUND_FILTER_DRY_RUN")

    try:
        os.environ["SHORT_REBOUND_FILTER_ENABLED"] = "true"
        os.environ["SHORT_REBOUND_FILTER_DRY_RUN"] = "false"

        signal = {
            "symbol": "FARTCOIN",
            "direction": "short",
            "rsi": 35,
            "trend": "up",
        }
        market = {"tokens": {"FARTCOIN": {}}}

        blocked, reason = ex._should_block_short_rebound(signal, market)
        assert not blocked, f"Expected SHORT to be permitted (RSI>=30) but blocked=True"
        return True, "OK: SHORT+RSI35+bullish permitted (RSI not in rebound zone)"
    finally:
        _restore_env("SHORT_REBOUND_FILTER_ENABLED", orig_enabled)
        _restore_env("SHORT_REBOUND_FILTER_DRY_RUN", orig_dry_run)


def test_short_rebound_filter_dry_run():
    """Modo dry-run: no bloquea pero deja log/flag de decision."""
    from agents import executor as ex

    orig_enabled = os.environ.get("SHORT_REBOUND_FILTER_ENABLED")
    orig_dry_run = os.environ.get("SHORT_REBOUND_FILTER_DRY_RUN")

    try:
        os.environ["SHORT_REBOUND_FILTER_ENABLED"] = "true"
        os.environ["SHORT_REBOUND_FILTER_DRY_RUN"] = "true"

        signal = {
            "symbol": "FARTCOIN",
            "direction": "short",
            "rsi": 28,
            "trend": "up",
        }
        market = {"tokens": {"FARTCOIN": {}}}

        blocked, reason = ex._should_block_short_rebound(signal, market)
        assert not blocked, "In dry-run mode, should_block should be False"
        assert reason, "In dry-run mode, reason should be returned"
        return True, f"OK: dry-run mode returns reason without blocking"
    finally:
        _restore_env("SHORT_REBOUND_FILTER_ENABLED", orig_enabled)
        _restore_env("SHORT_REBOUND_FILTER_DRY_RUN", orig_dry_run)


def test_long_not_affected():
    """LONGs no deben ser afectados por el filtro anti-rebound."""
    from agents import executor as ex

    orig_enabled = os.environ.get("SHORT_REBOUND_FILTER_ENABLED")

    try:
        os.environ["SHORT_REBOUND_FILTER_ENABLED"] = "true"

        signal = {
            "symbol": "FARTCOIN",
            "direction": "long",
            "rsi": 20,
            "trend": "up",
        }
        market = {"tokens": {"FARTCOIN": {}}}

        blocked, reason = ex._should_block_short_rebound(signal, market)
        assert not blocked, "LONGs should never be blocked by anti-rebound filter"
        return True, "OK: LONGs unaffected by anti-rebound filter"
    finally:
        _restore_env("SHORT_REBOUND_FILTER_ENABLED", orig_enabled)


def run_smoke_tests():
    """Run all smoke tests and print results."""
    print("=" * 60)
    print("SMOKE TEST — Solana Trading Bot Runtime")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    tests = [
        ("Duplicate Orchestrator Check", check_duplicate_orchestrators),
        ("Lock File Consistency", check_lock_file),
        ("Canonical State Directory", check_canonical_state),
        ("Lock Activation in Code", check_lock_is_activated_in_code),
        ("Legacy Files Not Used for Health", check_legacy_files_not_used_for_health),
        # Anti-Rebound SHORT Filter tests (SOLAA-23)
        ("[SOLAA-23] SHORT+RSI28+bullish blocked", lambda: _run_test(test_short_rebound_filter_blocked)),
        ("[SOLAA-23] SHORT+RSI28+downtrend permitted", lambda: _run_test(test_short_rebound_filter_permitted_downtrend)),
        ("[SOLAA-23] SHORT+RSI35+bullish permitted", lambda: _run_test(test_short_rebound_filter_permitted_high_rsi)),
        ("[SOLAA-23] dry-run mode", lambda: _run_test(test_short_rebound_filter_dry_run)),
        ("[SOLAA-23] LONGs unaffected", lambda: _run_test(test_long_not_affected)),
    ]

    all_passed = True
    for name, fn in tests:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        symbol = "✅" if ok else "❌"
        print(f"[{status}] {symbol} {name}: {msg}")
        if not ok:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("RESULT: ALL SMOKE TESTS PASSED ✅")
        return 0
    else:
        print("RESULT: SMOKE TESTS FAILED — review failures above ❌")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
