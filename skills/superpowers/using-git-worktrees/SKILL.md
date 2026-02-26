---
name: using-git-worktrees
description: "Use when starting feature work that needs isolation from current workspace or before executing implementation plans. Creates isolated git worktrees."
---

# Using Git Worktrees

## Overview

Create isolated git worktrees for feature development. Keeps main workspace clean.

## When to Use

- Feature work that might break things
- Working on multiple features in parallel
- Need isolation from current state
- Running tests that might modify files

## Process

### 1. Create Worktree

```bash
# Create new branch with worktree
git worktree add -b feature/my-feature /tmp/my-feature-workspace main

# Or from specific commit
git worktree add -b hotfix/issue-123 /tmp/issue-123 abc123
```

### 2. Work in Isolation

```bash
# Work in the worktree directory
cd /tmp/my-feature-workspace

# Make changes, run tests, etc.
```

### 3. Cleanup

```bash
# Remove worktree when done
git worktree remove /tmp/my-feature-workspace

# Delete branch if merged
git branch -d feature/my-feature
```

## Tools

| OpenClaw Tool | Use |
|---------------|-----|
| exec | git worktree commands |
| sessions_spawn | Run parallel worktrees |

## Safety

✅ **Do:**
- Use /tmp for worktrees
- Clean up after done
- Check branch name is unique

❌ **Don't:**
- Overwrite existing directories
- Use paths that matter

## Example

```bash
# Start feature work
git worktree add -b feature/new-bot /tmp/solana-bot-feature main

# Work in isolation...
# Then merge and cleanup
cd /home/enderj/.openclaw/workspace
git merge feature/new-bot
git worktree remove /tmp/solana-bot-feature
git branch -d feature/new-bot
```
