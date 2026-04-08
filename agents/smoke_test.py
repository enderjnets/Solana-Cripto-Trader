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

# Paths
AGENTS_DIR = Path(__file__).parent
WORKSPACE_DIR = AGENTS_DIR.parent
DATA_DIR = AGENTS_DIR / "data"
LEGACY_ROOT = WORKSPACE_DIR

LEGACY_STATE_FILES = [
    LEGACY_ROOT / "paper_trading_state.json",
    LEGACY_ROOT / "unified_brain_state.json",
]

ORCHESTRATOR_LOCK_FILE = DATA_DIR / "orchestrator.lock"
ORCHESTRATOR_SCRIPT = AGENTS_DIR / "orchestrator.py"


def get_orchestrator_pids() -> list[int]:
    """Return PIDs of all running orchestrator.py processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python3.*agents/orchestrator.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return [int(pid) for pid in result.stdout.strip().split("\n") if pid]
        return []
    except Exception:
        return []


def check_duplicate_orchestrators() -> tuple[bool, str]:
    """Check for multiple orchestrator instances. Returns (ok, message)."""
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
    if not DATA_DIR.exists():
        return False, f"CANONICAL STATE DIR MISSING: {DATA_DIR} does not exist"

    required = ["portfolio.json", "trade_history.json"]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        return False, f"CANONICAL STATE INCOMPLETE: missing {missing} in {DATA_DIR}"

    return True, f"OK: canonical state dir {DATA_DIR} has required files"


def check_legacy_files_not_used_for_health() -> tuple[bool, str]:
    """
    Check that legacy root files are stale and small (not actively written).
    These files should NOT be used for operational health checks.
    """
    # Check if legacy files are stale (>24h old) or very small
    now = datetime.now(timezone.utc).timestamp()
    issues = []
    for lf in LEGACY_STATE_FILES:
        if not lf.exists():
            continue  # Missing is fine, means not being written
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

    # Check that _acquire_lock is defined
    if "_acquire_lock()" not in content:
        return False, "LOCK DEFINITION MISSING: _acquire_lock() not found in orchestrator.py"

    # Check that it's called in __main__ before the loop
    # Look for the pattern: else: block with _acquire_lock() before while True
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
            break  # we've gone past the relevant section

    if not found_acquire_lock:
        return False, "LOCK NOT ACTIVATED: _acquire_lock() defined but not called in loop mode __main__"

    return True, "OK: _acquire_lock() is activated in orchestrator.py loop entry"


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