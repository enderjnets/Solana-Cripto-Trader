---
name: receiving-code-review
description: "Use when receiving code review feedback, before implementing suggestions. Requires technical rigor and verification, not performative agreement."
---

# Receiving Code Review

## Overview

When you get feedback, verify before implementing. Not all suggestions are right.

## ⚠️ HARD-GATE

**Do NOT:**
- Blindly implement every suggestion
- Ignore feedback
- Take feedback personally

## Process

### 1. Read Feedback Carefully

Understand what the reviewer is asking. Ask for clarification if unclear.

### 2. Evaluate Each Suggestion

For each point, ask:

| Question | If Yes | If No |
|----------|--------|-------|
| Is it technically correct? | Implement | Discuss |
| Does it match requirements? | Implement | Explain |
| Is it better than current? | Consider | Skip |
| Is it a preference/style? | Consider | Skip |

### 3. Respond to Feedback

```
## Review Response

### Point 1: [Reviewer's comment]
**Decision**: Implement
**Reason**: Agreed, improves readability

### Point 2: [Reviewer's comment]
**Decision**: Won't implement
**Reason**: Doesn't match our use case because...

### Point 3: [Reviewer's comment]
**Decision**: Need clarification
**Question**: Can you explain what you mean by X?
```

### 4. Implement What Makes Sense

- Fix legitimate issues
- Discuss disagreements professionally
- Explain your reasoning

## Anti-Patterns

❌ **Don't:**
- "Sure, will fix" without understanding
- Argue about everything
- Ignore feedback because you wrote it

✅ **Do:**
- Verify before implementing
- Ask clarifying questions
- Explain your reasoning
- Accept when you're wrong

## When You Disagree

1. **Explain your reasoning** - Why you did it this way
2. **Provide evidence** - Show why your approach works
3. **Be open** - Sometimes reviewer is right
4. **Escalate if needed** - If can't agree, ask user

## Technical Rigor

- Verify suggestions with tests
- Don't assume reviewer is always right
- Question "this is how it's done" without explanation
