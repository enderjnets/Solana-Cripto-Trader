---
name: using-superpowers
description: "Use when starting any conversation. Establishes how to find and use skills, requiring Skill tool invocation before ANY response including clarifying questions."
---

# Using Superpowers in OpenClaw

## Overview

This skill activates automatically when relevant. Use the skill system for structured development work.

## Available Skills

| Skill | When to Use |
|-------|-------------|
| brainstorming | Before creating anything new |
| systematic-debugging | When something breaks |
| test-driven-development | When implementing features |
| writing-plans | When you have a spec |
| verification-before-completion | Before claiming done |
| using-git-worktrees | For isolated work |
| requesting-code-review | Before merging/done |
| receiving-code-review | When you get feedback |

## How to Use

### 1. Check Available Skills

When the user asks for something that involves:
- Building something new → brainstorming
- Fixing a bug → systematic-debugging
- Adding features → test-driven-development

### 2. Read the Skill First

```bash
# Read skill before responding
read path:/home/enderj/.openclaw/workspace/skills/superpowers/[skill-name]/SKILL.md
```

### 3. Follow the Process

Each skill has a structured process. Follow it.

## Workflow Example

```
User: "Build a dashboard for my trading bot"

→ Invoke brainstorming skill
  - Ask what they want
  - Propose designs
  - Get approval

→ Invoke writing-plans skill  
  - Break into tasks

→ Execute tasks
  - Use test-driven-development for each

→ Invoke verification-before-completion
  - Verify it works

→ Invoke requesting-code-review
  - Review against requirements
```

## Anti-Patterns

❌ **Don't:**
- Jump straight to code
- Skip the design phase
- Claim "done" without verification

✅ **Do:**
- Use structured skills
- Follow processes
- Verify before declaring success

## Skills Location

```
/home/enderj/.openclaw/workspace/skills/superpowers/
├── brainstorming/
├── systematic-debugging/
├── test-driven-development/
├── writing-plans/
├── verification-before-completion/
├── using-git-worktrees/
├── requesting-code-review/
└── receiving-code-review/
```
