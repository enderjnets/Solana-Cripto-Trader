---
name: writing-plans
description: "Use when you have a spec or requirements for a multi-step task, before touching code. Breaks work into bite-sized tasks."
---

# Writing Implementation Plans

## Overview

Break approved design into executable tasks. Each task should take 2-5 minutes.

## ⚠️ HARD-GATE

**Do NOT:**
- Start implementing before having a plan
- Skip tasks and jump to code
- Have tasks longer than 5 minutes

## Plan Structure

```
# Plan: [Project Name]

## Tasks

### Task 1: [Description]
- **File**: `path/to/file.py`
- **Action**: What to do
- **Verify**: How to confirm it works

### Task 2: [Description]
- **File**: `path/to/file.py`  
- **Action**: What to do
- **Verify**: How to confirm it works

...
```

## Task Format

Each task must have:
1. **Clear description** - What exactly to do
2. **Exact file paths** - Don't make agent guess
3. **Verification step** - How to confirm it works

## Principles

- **Bite-sized** - 2-5 minutes each
- **Independent** - Tasks should not depend on each other
- **Verifiable** - Each task has a clear pass/fail
- **Sequential** - Order matters for dependent tasks

## Tools

| OpenClaw Tool | Use |
|---------------|-----|
| exec | Run commands, tests |
| read | Check files |
| write | Create files |
| edit | Modify files |

## Example

```
# Plan: Add User Authentication

## Tasks

### Task 1: Create user model
- **File**: `models/user.py`
- **Action**: Create User class with id, email, password_hash
- **Verify**: `python -c "from models.user import User; print('ok')"`

### Task 2: Add login endpoint
- **File**: `api/auth.py`
- **Action**: Add POST /login endpoint
- **Verify**: `curl -X POST localhost:8000/login`
```

## After Planning

1. Present plan to user for approval
2. Once approved, invoke subagent-driven-development or execute tasks yourself
