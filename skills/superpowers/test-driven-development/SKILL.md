---
name: test-driven-development
description: "Use when implementing any feature or bugfix, before writing implementation code. Enforces RED-GREEN-REFACTOR cycle."
---

# Test-Driven Development (TDD)

## Overview

Write tests FIRST, then implementation. Follow RED-GREEN-REFACTOR cycle.

## ⚠️ HARD-GATE

**Do NOT:**
- Write implementation code before tests
- Write tests that pass immediately (they should fail first)
- Skip the RED phase

## RED-GREEN-REFACTOR Cycle

### RED: Write Failing Test

1. **Write a test** for the smallest piece of functionality
2. **Run it** — it MUST fail
3. **Verify failure** — confirm it fails for the right reason

```bash
# Example: Test should fail because feature doesn't exist yet
python -m pytest tests/test_new_feature.py
# Expected: FAIL (feature not implemented)
```

### GREEN: Write Minimal Code

1. **Write just enough code** to make the test pass
2. **No extras** — skip elegance, optimization
3. **Run test** — it MUST pass
4. **If it fails** — fix the code, not the test

### REFACTOR: Improve Code

1. **Clean up** the implementation
2. **Keep tests passing**
3. **Commit** after refactoring

## Rules

| Rule | Description |
|------|-------------|
| One test at a time | Focus on smallest unit |
| Test first | Always RED before GREEN |
| Minimal code | Just enough to pass |
| No premature optimization | YAGNI |
| Keep tests passing | Never commit broken tests |

## When to Write Tests

- **New feature** → Write test first
- **Bug fix** → Write test that reproduces bug, then fix
- **Refactor** → Tests should already exist
- **Regression** → Add test after finding bug

## Tools

```bash
# Run tests
python -m pytest

# Run specific test
python -m pytest tests/test_file.py::test_name

# Run with coverage
python -m pytest --cov
```

## Anti-Patterns

❌ **Don't:**
- Write implementation before tests
- Write tests that pass immediately
- Test too much at once
- Skip RED phase

✅ **Do:**
- Write smallest possible test
- Watch it fail
- Write minimal code
- Watch it pass

## After TDD

1. All tests pass
2. Code is minimal
3. Commit with message: "Add [feature], tests pass"
