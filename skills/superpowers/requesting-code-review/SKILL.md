---
name: requesting-code-review
description: "Use when completing tasks, implementing major features, or before merging to verify work meets requirements."
---

# Requesting Code Review

## Overview

Before merging or declaring done, review your work against requirements.

## Review Checklist

### 1. Requirements Check
- [ ] Does code match the design/spec?
- [ ] Are all tasks complete?
- [ ] Did I miss anything?

### 2. Code Quality
- [ ] Code is readable?
- [ ] No obvious bugs?
- [ ] Error handling in place?

### 3. Testing
- [ ] Tests pass?
- [ ] Edge cases covered?
- [ ] No regressions?

### 4. Security
- [ ] No secrets exposed?
- [ ] Input validation?
- [ ] Safe defaults?

## Reporting

When requesting review, report:

```
## Code Review: [Feature]

### Summary
Brief description of what was done

### Changes
- file1.py: Added User model
- file2.py: Updated API endpoints

### Testing
- Ran pytest: 15 passed
- Manual test: Verified login works

### Issues
List any known issues or questions
```

## Severity Levels

| Severity | Description | Action |
|----------|-------------|--------|
| Critical | Security, data loss | Fix before merge |
| Major | Breaks functionality | Fix before merge |
| Minor | Style, preferences | Can address later |
| Nitpick | Optional improvements | Skip or address later |

## Anti-Patterns

❌ **Don't:**
- Skip review entirely
- Review only style
- Ignore critical issues

✅ **Do:**
- Check requirements
- Test thoroughly
- Report honest issues
