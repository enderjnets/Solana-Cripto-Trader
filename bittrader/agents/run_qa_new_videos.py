#!/usr/bin/env python3
"""
Run QA on all 6 new/fixed videos and update upload_queue.json
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

AGENTS = Path("/home/enderj/.openclaw/workspace/bittrader/agents")
QUEUE_FILE = AGENTS / "data/upload_queue.json"
DATA_DIR = AGENTS / "data"
THUMBS_DIR = DATA_DIR / "thumbnails"
OUTPUT_BASE = AGENTS / "output/2026-03-28"

# Import QA agent
sys.path.insert(0, str(AGENTS))
from qa_agent import QAAgent

qa = QAAgent()

# Load guiones for script text
guiones = json.loads((DATA_DIR / "guiones_latest.json").read_text())
scripts_by_id = {s["id"]: s for s in guiones.get("scripts", [])}

# Target videos for QA
TARGET_IDS = [
    "short_1774708351_945",
    "short_1774708455_613",
    "short_1774708563_481",
    "long_1774708225_575",
    "long_1774708531_991",
    "short_1774708280_155",  # SOL - regenerated thumbnail
    "short_1774708608_208",  # RAIN - newly produced
    "short_1774708646_182",  # 90% traders - newly produced
]

# Load queue
queue = json.loads(QUEUE_FILE.read_text())
queue_by_id = {item.get("script_id"): (i, item) for i, item in enumerate(queue)}

print("=" * 70)
print("🔬 QA Agent — Checking 6 new/fixed videos")
print("=" * 70)

results = []

for script_id in TARGET_IDS:
    entry = queue_by_id.get(script_id)
    if not entry:
        print(f"\n⚠️  {script_id}: NOT IN QUEUE — skipping")
        continue
    
    idx, item = entry
    title = item.get("title", "")
    description = item.get("description", "")
    video_type = item.get("type", "short")
    video_path = item.get("output_file", "")
    thumb_path = item.get("thumbnail_path", item.get("thumbnail", ""))
    
    # Get script text from guiones
    script_data = scripts_by_id.get(script_id, {})
    script_text = script_data.get("script", "")
    
    print(f"\n📹 {title[:55]}")
    print(f"   video: {video_path}")
    print(f"   thumb: {thumb_path}")
    print(f"   type:  {video_type}")
    
    # Verify files exist before QA
    if not Path(video_path).exists():
        print(f"   ❌ VIDEO FILE NOT FOUND: {video_path}")
        queue[idx]["status"] = "blocked"
        queue[idx]["qa_issues"] = ["NO_VIDEO_FILE"]
        results.append({"id": script_id, "passed": False, "issues": ["NO_VIDEO_FILE"]})
        continue
    
    if not Path(thumb_path).exists():
        print(f"   ❌ THUMBNAIL NOT FOUND: {thumb_path}")
        queue[idx]["status"] = "blocked"
        queue[idx]["qa_issues"] = ["NO_THUMBNAIL"]
        results.append({"id": script_id, "passed": False, "issues": ["NO_THUMBNAIL"]})
        continue
    
    # Run QA
    result = qa.run_all_checks(
        video_path=video_path,
        thumb_path=thumb_path,
        title=title,
        description=description,
        script_text=script_text,
        video_type=video_type,
    )
    
    passed = result["passed"]
    issues = result.get("issues", [])
    checks = result.get("checks", {})
    
    # Show check results
    for check_name, check_data in checks.items():
        ok = check_data.get("passed", False)
        icon = "✅" if ok else "❌"
        detail = check_data.get("issue", "") or ""
        print(f"   {icon} {check_name}: {detail if not ok else 'OK'}")
    
    if passed:
        print(f"   ✅ QA PASSED — marking as ready")
        queue[idx]["qa_passed"] = True
        queue[idx]["qa_passed_at"] = result["timestamp"]
        queue[idx]["qa_checks"] = f"{len(checks)}/{len(checks)} passed"
        # Keep status as ready so queue_processor can pick it up
        if queue[idx].get("status") not in ("uploaded", "published"):
            queue[idx]["status"] = "ready"
    else:
        print(f"   ❌ QA FAILED — blocking. Issues: {issues}")
        queue[idx]["qa_passed"] = False
        queue[idx]["qa_issues"] = issues
        queue[idx]["qa_checked_at"] = result["timestamp"]
        queue[idx]["status"] = "qa_failed"
    
    results.append({"id": script_id, "passed": passed, "issues": issues, "title": title})

# Save queue
QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))

# Summary
print("\n" + "=" * 70)
print("📊 QA SUMMARY")
print("=" * 70)
passed_count = sum(1 for r in results if r["passed"])
failed_count = len(results) - passed_count
print(f"  ✅ Passed: {passed_count}/{len(results)}")
print(f"  ❌ Failed: {failed_count}/{len(results)}")
if failed_count:
    print("\n  Failed videos:")
    for r in results:
        if not r["passed"]:
            print(f"    ❌ {r['id']}: {r.get('issues', [])}")
print("=" * 70)
