#!/usr/bin/env python3
"""tools/health_check.py — cron-friendly health probe for the live bot.

Checks:
  1. Heartbeat age < 300s
  2. Orchestrator process alive with CWD = live workspace
  3. Kill switch flag absent
  4. close_failures < 3 (kill switch threshold)
  5. Watchdog process alive

Run as cron:
    */5 * * * * /usr/bin/python3 /home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live/tools/health_check.py

Flags:
    --verbose    print full diagnostic table
    --alert      on unhealthy state, create a Paperclip issue in SOLAAA
                 (deduplicated: only 1 alert per 15-min rolling window)

Exit codes: 0 healthy, 1 unhealthy, 2 check error.

v2.12.19 — Phase 1 observability.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LIVE_DIR = Path("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live")
HB_FILE = Path("/tmp/solana_live_heartbeat")
KILLSWITCH_FILE = Path("/tmp/solana_live_killswitch")
CLOSE_FAILURES_FILE = Path("/tmp/solana_live_close_failures")
ALERT_DEDUP_FILE = Path("/tmp/solana_live_health_alert_last")
ALERT_DEDUP_WINDOW_SEC = 15 * 60  # 15 minutes


def _load_env() -> None:
    """Load .env so paperclip_client has API key."""
    env_path = LIVE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def check_heartbeat() -> dict:
    if not HB_FILE.exists():
        return {"ok": False, "detail": "heartbeat file missing"}
    try:
        age = time.time() - float(HB_FILE.read_text().strip())
    except (ValueError, OSError) as e:
        return {"ok": False, "detail": f"read error: {e}"}
    return {"ok": age < 300, "detail": f"age={age:.0f}s (threshold 300s)", "age": age}


def check_orch() -> dict:
    try:
        pids = subprocess.check_output(["pgrep", "-f", "orchestrator.py"], text=True).strip().splitlines()
    except subprocess.CalledProcessError:
        return {"ok": False, "detail": "no orchestrator processes"}

    live_pids = []
    for pid in pids:
        pid = pid.strip()
        if not pid:
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            if cwd == str(LIVE_DIR):
                live_pids.append(pid)
        except (FileNotFoundError, PermissionError):
            pass

    if not live_pids:
        return {"ok": False, "detail": "no live orch (CWD match)"}
    return {"ok": True, "detail": f"live orch PID={live_pids[0]}", "pids": live_pids}


def check_watchdog() -> dict:
    try:
        lines = subprocess.check_output(["pgrep", "-af", "run_watchdog.sh"], text=True).splitlines()
    except subprocess.CalledProcessError:
        return {"ok": False, "detail": "no watchdog processes"}

    live_wd = []
    for line in lines:
        pid = line.split()[0]
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            if cwd == str(LIVE_DIR):
                live_wd.append(pid)
        except (FileNotFoundError, PermissionError):
            pass

    if not live_wd:
        return {"ok": False, "detail": "no live watchdog (CWD match)"}
    return {"ok": True, "detail": f"watchdog PID={live_wd[0]}", "pids": live_wd}


def check_kill_switch() -> dict:
    if KILLSWITCH_FILE.exists():
        content = KILLSWITCH_FILE.read_text(errors="ignore")[:100]
        return {"ok": False, "detail": f"ACTIVE: {content.strip()}"}
    return {"ok": True, "detail": "clear"}


def check_close_failures() -> dict:
    if not CLOSE_FAILURES_FILE.exists():
        return {"ok": True, "detail": "counter=0 (file absent)", "count": 0}
    try:
        count = int(CLOSE_FAILURES_FILE.read_text().strip() or 0)
    except ValueError:
        return {"ok": False, "detail": "counter file unreadable", "count": -1}
    return {
        "ok": count < 3,
        "detail": f"counter={count} (threshold 3)",
        "count": count,
    }


def should_alert() -> bool:
    """Dedup: only alert once per 15min window."""
    if not ALERT_DEDUP_FILE.exists():
        return True
    try:
        last = float(ALERT_DEDUP_FILE.read_text().strip())
    except (ValueError, OSError):
        return True
    return (time.time() - last) > ALERT_DEDUP_WINDOW_SEC


def record_alert() -> None:
    ALERT_DEDUP_FILE.write_text(str(time.time()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--alert", action="store_true",
                    help="Create Paperclip issue on unhealthy (deduplicated 15min)")
    args = ap.parse_args()

    checks = {
        "heartbeat": check_heartbeat(),
        "orchestrator": check_orch(),
        "watchdog": check_watchdog(),
        "kill_switch": check_kill_switch(),
        "close_failures": check_close_failures(),
    }

    all_ok = all(c["ok"] for c in checks.values())

    if args.verbose or not all_ok:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"=== HEALTH CHECK {ts} — {'✅ HEALTHY' if all_ok else '❌ UNHEALTHY'} ===")
        for name, c in checks.items():
            status = "✅" if c["ok"] else "❌"
            print(f"  {status} {name:15s}: {c['detail']}")

    if not all_ok and args.alert:
        if should_alert():
            _load_env()
            try:
                sys.path.insert(0, str(LIVE_DIR / "agents"))
                import paperclip_client as pc
                failed = [f"{k}: {v['detail']}" for k, v in checks.items() if not v["ok"]]
                title = f"🚨 Live bot UNHEALTHY — {len(failed)} check(s) failed"
                body = (
                    f"**Automated health check alert** (cron */5min)\n\n"
                    f"Time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n"
                    f"**Failed checks:**\n" + "\n".join(f"- {f}" for f in failed) + "\n\n"
                    "**Runbook**:\n"
                    "1. SSH to ROG: `ssh enderj@10.0.0.240`\n"
                    "2. Inspect: `tail -50 /tmp/solana_live_watchdog.log`\n"
                    "3. If kill_switch active: check reconcile criterion; may need manual sweep\n"
                    "4. If orch dead + no watchdog: relaunch per playbook\n"
                    "5. If close_failures>=3: imminent kill_switch — investigate Jupiter slippage\n\n"
                    "Auto-dedup: next alert suppressed for 15 min."
                )
                issue_id = pc._create_issue(title, body, priority="urgent", status="todo")
                if issue_id:
                    print(f"📋 Paperclip alert posted: {issue_id}")
                    record_alert()
            except Exception as e:
                print(f"⚠️ Failed to post Paperclip alert: {e}", file=sys.stderr)
        else:
            if args.verbose:
                print("(alert suppressed — last alert within 15min dedup window)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
