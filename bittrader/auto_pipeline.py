#!/usr/bin/env python3
"""
BitTrader Auto Pipeline — Orquestador Principal
Ejecuta: Collector → Analyzer → Strategist → Producer → Publisher
"""
import sys, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("/home/enderj/.openclaw/workspace")
BITTRADER = WORKSPACE / "bittrader"

def run_step(name, script):
    """Ejecuta un paso del pipeline"""
    print(f"\n{'='*60}")
    print(f"🔄 {name}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BITTRADER),
        capture_output=True, text=True, timeout=300
    )
    
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[-500:]}")
    
    if result.returncode != 0:
        print(f"❌ {name} falló (exit code {result.returncode})")
        return False
    
    print(f"✅ {name} completado")
    return True

def main():
    now = datetime.now(timezone.utc)
    print(f"🚀 BitTrader Auto Pipeline — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"   Mode: {'full' if '--full' in sys.argv else 'collect-analyze'}")
    
    # Step 1: Collect data
    ok = run_step("COLLECTOR", BITTRADER / "analytics/collector.py")
    if not ok:
        return
    
    # Step 2: Analyze patterns
    ok = run_step("ANALYZER", BITTRADER / "analytics/analyzer.py")
    if not ok:
        return
    
    # Step 3: Generate content strategy (only on full runs)
    if "--full" in sys.argv:
        ok = run_step("STRATEGIST", BITTRADER / "analytics/strategist.py")
        if not ok:
            return
        
        print(f"\n{'='*60}")
        print("📋 Guiones generados — revisa y confirma producción")
        print(f"{'='*60}")
    
    print(f"\n🏁 Pipeline completado — {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

if __name__ == "__main__":
    main()
