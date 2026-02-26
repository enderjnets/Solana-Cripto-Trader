---
name: brainstorming
description: "You MUST use this before any creative work - creating features, building components, adding functionality, or modifying behavior. Explores user intent, requirements and design before implementation."
---

# Brainstorming Ideas Into Designs

## Overview

Help turn ideas into fully formed designs and specs through natural collaborative dialogue.

Start by understanding the current project context, then ask questions one at a time to refine the idea. Once you understand what you're building, present the design and get user approval.

## ⚠️ HARD-GATE

**Do NOT:**
- Write any code
- Create any files (except docs)
- Take any implementation action

**Until you have presented a design and the user has approved it.**

Every project goes through this process. Even "simple" projects.

## Checklist

Complete these in order:

1. **Explore project context** — check files, docs, recent commits
2. **Ask clarifying questions** — one at a time, understand purpose/constraints/success criteria
3. **Propose 2-3 approaches** — with trade-offs and your recommendation
4. **Present design** — in sections scaled to their complexity, get user approval after each section
5. **Write design doc** — save to `memory/plans/YYYY-MM-DD-<topic>-design.md` 
6. **Transition to implementation** — invoke writing-plans skill

## Process

### 1. Explore Context
Check the project state first:
```bash
# Check recent files, git log, structure
ls -la
git log --oneline -10
```

### 2. Ask Questions (ONE AT A TIME)
- Focus on: purpose, constraints, success criteria
- Prefer multiple choice when possible
- Wait for answer before next question

### 3. Propose 2-3 Approaches
- Present options with trade-offs
- Lead with your recommendation
- Explain why

### 4. Present Design
- Scale to complexity (few sentences = simple, 200-300 words = nuanced)
- Ask after each section: "Does this look right?"
- Cover: architecture, components, data flow, error handling

### 5. Write Design Doc
Save to: `memory/plans/YYYY-MM-DD-<topic>-design.md`

### 6. Invoke writing-plans
After design approval, move to implementation planning.

## Key Principles

- **One question at a time** - Don't overwhelm
- **Multiple choice preferred** - Easier for user
- **YAGNI** - Remove unnecessary features
- **Explore alternatives** - Always propose options
- **Incremental validation** - Get approval before moving on
