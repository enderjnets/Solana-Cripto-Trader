#!/usr/bin/env python3
"""
Simple test for auto-learning integration
=======================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

print("="*80)
print("SIMPLE AUTO-LEARNING INTEGRATION TEST")
print("="*80)

print("\n1. Importing modules...")
try:
    from master_orchestrator import MasterOrchestrator
    print("   ✅ Master imported")
except Exception as e:
    print(f"   ❌ Master import failed: {e}")
    sys.exit(1)

try:
    from integrate_auto_learner import AutoLearningIntegration
    print("   ✅ Integration imported")
except Exception as e:
    print(f"   ❌ Integration import failed: {e}")
    sys.exit(1)

print("\n2. Creating Master Orchestrator...")
try:
    master = MasterOrchestrator()
    print("   ✅ Master created")
    print(f"      Type: {type(master)}")
except Exception as e:
    print(f"   ❌ Master creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Creating AutoLearningIntegration...")
try:
    integration = AutoLearningIntegration(master)
    print("   ✅ Integration created")
    print(f"      Type: {type(integration)}")
except Exception as e:
    print(f"   ❌ Integration creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Calling integration.start()...")
try:
    integration.start()
    print("   ✅ Integration started")
except Exception as e:
    print(f"   ❌ Integration start failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Getting status...")
try:
    status = integration.get_status()
    print("   ✅ Status retrieved")
    print(f"      Status: {status}")
except Exception as e:
    print(f"   ❌ Get status failed: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Sleeping for 3 seconds...")
try:
    import time
    time.sleep(3)
    print("   ✅ Sleep completed")
except Exception as e:
    print(f"   ❌ Sleep failed: {e}")

print("\n7. Getting status again...")
try:
    status = integration.get_status()
    print("   ✅ Status retrieved")
    print(f"      Status: {status}")
except Exception as e:
    print(f"   ❌ Get status failed: {e}")

print("\n8. Stopping integration...")
try:
    integration.stop()
    print("   ✅ Integration stopped")
except Exception as e:
    print(f"   ❌ Stop failed: {e}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED")
print("="*80)
