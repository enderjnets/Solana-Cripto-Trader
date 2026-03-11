#!/usr/bin/env python3
"""
🔍 BitTrader Auto-Auditor
Audita el pipeline completo y genera un reporte con score.
Si el score < 10/10, guarda las fallas en audit_failures.json
para que el agente reparador las corrija.

Ejecutar: python3 auditor.py
"""
import json
import os
import subprocess
import sys
import importlib
from pathlib import Path
from datetime import datetime, timezone

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"
AGENTS_DIR = BITTRADER / "agents"
DATA_DIR = AGENTS_DIR / "data"
AUDIT_FILE = DATA_DIR / "audit_result.json"
FAILURES_FILE = DATA_DIR / "audit_failures.json"

# Add agents dir to path
sys.path.insert(0, str(AGENTS_DIR))


def check_file_exists(path: str, label: str) -> dict:
    """Check if a file exists."""
    p = AGENTS_DIR / path if not Path(path).is_absolute() else Path(path)
    exists = p.exists()
    return {
        "test": f"file_exists:{label}",
        "label": f"File: {label}",
        "passed": exists,
        "error": None if exists else f"Missing: {p}",
        "fix_hint": f"Create or restore file: {p}",
        "severity": "error",
    }


def check_import(module: str, func: str = None) -> dict:
    """Check if a Python module imports correctly."""
    try:
        m = importlib.import_module(module)
        if func and not hasattr(m, func):
            return {
                "test": f"import:{module}.{func}",
                "label": f"Import {module}.{func}",
                "passed": False,
                "error": f"Function '{func}' not found in {module}",
                "fix_hint": f"Add or rename function '{func}' in {module}.py",
                "severity": "error",
            }
        return {
            "test": f"import:{module}",
            "label": f"Import {module}",
            "passed": True,
            "error": None,
            "severity": "ok",
        }
    except Exception as e:
        return {
            "test": f"import:{module}",
            "label": f"Import {module}",
            "passed": False,
            "error": str(e)[:200],
            "fix_hint": f"Fix import error in {module}.py: {str(e)[:100]}",
            "severity": "error",
        }


def check_command(cmd: list, label: str, expect_in: str = None) -> dict:
    """Check if a CLI command works."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            if expect_in and expect_in not in result.stdout:
                return {
                    "test": f"cmd:{label}",
                    "label": label,
                    "passed": False,
                    "error": f"Command ran but output missing '{expect_in}'",
                    "fix_hint": f"Check {label} configuration",
                    "severity": "error",
                }
            return {"test": f"cmd:{label}", "label": label, "passed": True, "error": None, "severity": "ok"}
        return {
            "test": f"cmd:{label}",
            "label": label,
            "passed": False,
            "error": result.stderr[:200],
            "fix_hint": f"Fix {label}: {result.stderr[:100]}",
            "severity": "error",
        }
    except Exception as e:
        return {
            "test": f"cmd:{label}",
            "label": label,
            "passed": False,
            "error": str(e)[:200],
            "fix_hint": f"Install or fix {label}",
            "severity": "error",
        }


def check_rss_feeds() -> dict:
    """Check RSS news feeds work."""
    try:
        from scout import fetch_rss_news, filter_breaking_news
        news = fetch_rss_news()
        if len(news) >= 10:
            return {"test": "rss_feeds", "label": "RSS News Feeds", "passed": True,
                    "error": None, "severity": "ok", "detail": f"{len(news)} articles"}
        elif len(news) > 0:
            return {"test": "rss_feeds", "label": "RSS News Feeds", "passed": True,
                    "error": None, "severity": "warning", "detail": f"Only {len(news)} articles (some feeds down?)"}
        else:
            return {"test": "rss_feeds", "label": "RSS News Feeds", "passed": False,
                    "error": "0 articles fetched", "fix_hint": "Check RSS feed URLs in scout.py",
                    "severity": "error"}
    except Exception as e:
        return {"test": "rss_feeds", "label": "RSS News Feeds", "passed": False,
                "error": str(e)[:200], "fix_hint": "Fix fetch_rss_news() in scout.py",
                "severity": "error"}


def check_scout_data_fresh() -> dict:
    """Check scout data is fresh (< 14h)."""
    try:
        data = json.loads((DATA_DIR / "scout_latest.json").read_text())
        gen = data.get("generated_at", "")
        gen_dt = datetime.fromisoformat(gen)
        age_h = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 3600
        if age_h < 14:
            return {"test": "scout_freshness", "label": "Scout Data Freshness",
                    "passed": True, "error": None, "severity": "ok",
                    "detail": f"{age_h:.1f}h old"}
        else:
            return {"test": "scout_freshness", "label": "Scout Data Freshness",
                    "passed": False, "error": f"Data is {age_h:.1f}h old (stale)",
                    "fix_hint": "Run: python3 orchestrator.py --scout-only --no-telegram",
                    "severity": "warning"}
    except Exception as e:
        return {"test": "scout_freshness", "label": "Scout Data Freshness",
                "passed": False, "error": str(e)[:200],
                "fix_hint": "Run scout to generate data",
                "severity": "warning"}


def check_youtube_token() -> dict:
    """Check YouTube credentials are valid."""
    try:
        import requests
        creds = json.loads(Path(WORKSPACE / "memory/youtube_credentials.json").read_text())
        access = creds.get("access_token", "")
        if not access:
            return {"test": "youtube_token", "label": "YouTube Token",
                    "passed": False, "error": "No access token",
                    "fix_hint": "Re-authenticate YouTube OAuth",
                    "severity": "error"}
        # Test the token
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/channels?part=id&mine=true",
            headers={"Authorization": f"Bearer {access}"}, timeout=10
        )
        if r.status_code == 200:
            return {"test": "youtube_token", "label": "YouTube Token",
                    "passed": True, "error": None, "severity": "ok"}
        else:
            return {"test": "youtube_token", "label": "YouTube Token",
                    "passed": False, "error": f"HTTP {r.status_code}",
                    "fix_hint": "YouTube access token expired — needs refresh or re-auth. HUMAN ACTION REQUIRED.",
                    "severity": "warning"}
    except Exception as e:
        return {"test": "youtube_token", "label": "YouTube Token",
                "passed": False, "error": str(e)[:200],
                "fix_hint": "Check youtube_credentials.json",
                "severity": "warning"}


def check_api_key(name: str, key_file: str, key_field: str = None) -> dict:
    """Check an API key exists and is non-empty."""
    try:
        p = BITTRADER / "keys" / key_file
        if not p.exists():
            return {"test": f"api_key:{name}", "label": f"API Key: {name}",
                    "passed": False, "error": f"File missing: {p}",
                    "fix_hint": f"Create {p} with valid API key",
                    "severity": "error"}
        data = json.loads(p.read_text())
        has_key = any(v for v in data.values() if isinstance(v, str) and len(v) > 10)
        if has_key:
            return {"test": f"api_key:{name}", "label": f"API Key: {name}",
                    "passed": True, "error": None, "severity": "ok"}
        else:
            return {"test": f"api_key:{name}", "label": f"API Key: {name}",
                    "passed": False, "error": "Key empty or too short",
                    "fix_hint": f"Add valid API key to {p}",
                    "severity": "error"}
    except Exception as e:
        return {"test": f"api_key:{name}", "label": f"API Key: {name}",
                "passed": False, "error": str(e)[:200],
                "fix_hint": f"Fix {key_file}",
                "severity": "error"}


def check_pollinations() -> dict:
    """Check Pollinations image API is reachable."""
    try:
        import requests
        r = requests.head("https://image.pollinations.ai/prompt/test", timeout=10)
        if r.status_code in (200, 301, 302):
            return {"test": "pollinations", "label": "Pollinations Image API",
                    "passed": True, "error": None, "severity": "ok"}
        return {"test": "pollinations", "label": "Pollinations Image API",
                "passed": False, "error": f"HTTP {r.status_code}",
                "fix_hint": "Pollinations API down — wait or use alternative",
                "severity": "warning"}
    except Exception as e:
        return {"test": "pollinations", "label": "Pollinations Image API",
                "passed": False, "error": str(e)[:100],
                "fix_hint": "Network issue reaching Pollinations",
                "severity": "warning"}


def run_audit() -> dict:
    """Run the full audit and return results."""
    print("\n" + "=" * 60)
    print("🔍 BitTrader Auto-Auditor v1.0")
    print("=" * 60)
    
    results = []
    
    # ── Core Files ──
    print("\n📁 Core Files...")
    core_files = {
        "scout.py": "scout",
        "creator.py": "creator",
        "producer.py": "producer",
        "publisher.py": "publisher",
        "orchestrator.py": "orchestrator",
        "llm_config.py": "llm_config",
        "ken_burns_producer.py": "ken_burns_producer",
        "quality_checker.py": "quality_checker",
        "karaoke_subs.py": "karaoke_subs",
        "calendar_guard.py": "calendar_guard",
        "youtube_stats.py": "youtube_stats",
    }
    for fname, label in core_files.items():
        results.append(check_file_exists(fname, label))
    
    # ── Python Imports ──
    print("🐍 Python Imports...")
    imports = [
        ("scout", "run_scout"),
        ("creator", "run_creator"),
        ("orchestrator", "main"),
        ("llm_config", "call_llm"),
        ("publisher", "run_publisher"),
    ]
    for mod, func in imports:
        results.append(check_import(mod, func))
    
    # ── CLI Tools ──
    print("🔧 CLI Tools...")
    results.append(check_command(["edge-tts", "--list-voices"], "Edge TTS"))
    results.append(check_command(["ffmpeg", "-version"], "FFmpeg"))
    
    # ── Google Services ──
    print("🔑 Google Services...")
    results.append(check_command(
        ["gog", "calendar", "events", "primary", "--max", "1", "--plain"],
        "Google Calendar", "2026"
    ))
    results.append(check_command(
        ["gog", "gmail", "search", "is:inbox", "--max", "1", "--plain"],
        "Gmail"
    ))
    
    # ── API Keys ──
    print("🔑 API Keys...")
    results.append(check_api_key("Z.ai (GLM-5)", "zai.json"))
    results.append(check_api_key("MiniMax", "minimax.json"))
    results.append(check_api_key("Pexels", "pexels.json"))
    
    # ── YouTube ──
    print("🎥 YouTube...")
    results.append(check_youtube_token())
    
    # ── RSS ──
    print("📰 RSS Feeds...")
    results.append(check_rss_feeds())
    
    # ── Scout Data ──
    print("🔭 Scout Data...")
    results.append(check_scout_data_fresh())
    
    # ── External APIs ──
    print("🌐 External APIs...")
    results.append(check_pollinations())
    
    # ── Calculate Score ──
    passed = [r for r in results if r["passed"]]
    failed_errors = [r for r in results if not r["passed"] and r.get("severity") == "error"]
    failed_warnings = [r for r in results if not r["passed"] and r.get("severity") == "warning"]
    
    total = len(results)
    score = len(passed) / total * 10 if total > 0 else 0
    
    # ── Print Results ──
    print("\n" + "=" * 60)
    print("📋 AUDIT RESULTS")
    print("=" * 60)
    for r in passed:
        print(f"  ✅ {r['label']}")
    for r in failed_warnings:
        print(f"  ⚠️ {r['label']}: {r['error']}")
    for r in failed_errors:
        print(f"  ❌ {r['label']}: {r['error']}")
    
    print(f"\n📊 SCORE: {score:.1f}/10  |  ✅ {len(passed)}  |  ⚠️ {len(failed_warnings)}  |  ❌ {len(failed_errors)}")
    
    # ── Save Results ──
    audit_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": round(score, 1),
        "total_checks": total,
        "passed": len(passed),
        "warnings": len(failed_warnings),
        "errors": len(failed_errors),
        "perfect": score >= 9.5,
        "results": results,
    }
    AUDIT_FILE.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False))
    
    # ── Save Failures for Repair Agent ──
    failures = [r for r in results if not r["passed"] and r.get("severity") == "error"]
    if failures:
        FAILURES_FILE.write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": round(score, 1),
            "failures": failures,
        }, indent=2, ensure_ascii=False))
        print(f"\n⚠️ {len(failures)} failures saved to {FAILURES_FILE.name} for repair agent")
    else:
        # Clear failures file if all good
        if FAILURES_FILE.exists():
            FAILURES_FILE.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "score": round(score, 1),
                "failures": [],
            }, indent=2))
        print("\n🎉 All critical checks passed!")
    
    return audit_result


if __name__ == "__main__":
    result = run_audit()
    sys.exit(0 if result["errors"] == 0 else 1)
