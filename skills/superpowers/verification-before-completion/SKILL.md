---
name: verification-before-completion
description: "Use when about to claim work is complete, fixed, or passing. Requires running verification commands and confirming output before making any success claims."
---

# Verification Before Completion

## Overview

**Evidence before assertions.** Don't claim success without proof.

## ⚠️ HARD-GATE

**Do NOT:**
- Say "it's fixed" without running verification
- Claim "tests pass" without running them
- Declare completion without evidence

## Verification Checklist

Before declaring done, verify:

### 1. Run Tests
```bash
python -m pytest
# Must show all passing
```

### 2. Check Syntax/Lint
```bash
python -m py_compile file.py
# No syntax errors
```

### 3. Manual Verification
```bash
# Run the actual command
python script.py
# Check output matches expected
```

### 4. Review Changes
```bash
git diff
# Confirm only intended changes
```

## Evidence Template

When you claim completion, include evidence:

```
✅ VERIFIED:
- Tests: `pytest tests/ -v` → 15 passed
- Syntax: `python -m py_compile` → No errors
- Manual: Ran script → Output matches expected
```

## Anti-Patterns

❌ **Don't:**
- Assume it works because it looks right
- Skip running tests
- Trust old test results

✅ **Do:**
- Run verification commands fresh
- Check actual output
- Show evidence in response

## When Verification Fails

1. **Don't hide the failure**
2. **Report exact error**
3. **Fix and re-verify**
4. **Then claim success**

## After Verification

Include verification output in your completion message so the user knows you actually checked.
