---
name: systematic-debugging
description: "Use when encountering any bug, test failure, or unexpected behavior. Follows 4-phase root cause analysis."
---

# Systematic Debugging

## Overview

Debug systematically using evidence, not guesses. Follow the 4-phase process.

## ⚠️ HARD-GATE

**Do NOT:**
- Propose fixes based on guesses
- Change code before identifying root cause
- Declare "fixed" without verification

## 4-Phase Process

### Phase 1: Gather Evidence

1. **Reproduce the bug** - Get exact steps
2. **Collect logs** - Check relevant log files
3. **Check state** - What was the system state?
4. **Identify inputs** - What data triggered the error?

```bash
# Commands to gather evidence
ls -la *.log
tail -100 error.log
git log --oneline -5
```

### Phase 2: Root Cause Tracing

1. **Find the error source** - Trace back from symptoms
2. **Identify the mechanism** - Why did it fail?
3. **Check for side effects** - What else broke?
4. **Form hypothesis** - What's the actual cause?

Questions to ask:
- What did I expect to happen?
- What actually happened?
- What's different from the last time it worked?

### Phase 3: Design Solution

1. **Fix the root cause** - Not just symptoms
2. **Consider edge cases** - What else could break?
3. **Plan verification** - How will you confirm it's fixed?
4. **Consider alternatives** - Is there a better fix?

### Phase 4: Verify & Document

1. **Test the fix** - Does it work now?
2. **Check for regressions** - Did anything else break?
3. **Document findings** - What did you learn?
4. **Update tests** - Add test for this case

## Tools to Use

| Claude Tool | OpenClaw Equivalent |
|------------|-------------------|
| grep | exec + grep |
| Read | read |
| Bash | exec |
| web_search | web_search |

## Anti-Patterns

❌ **Don't:**
- Change code randomly hoping it works
- Assume you know the cause without evidence
- Skip reproduction steps
- Fix symptoms not root cause

✅ **Do:**
- Reproduce first
- Gather evidence
- Form hypothesis
- Test systematically

## After Debugging

Update relevant files with findings:
- Add to MEMORY.md if it's a pattern to remember
- Update docs if the bug revealed documentation gap
- Add tests to prevent regression
