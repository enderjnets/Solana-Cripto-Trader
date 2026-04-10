# Auto-Learner Knowledge Base

## Current Lessons (Post-Fix 2026-04-10)

**Context**: All previous lessons were from a broken system (liquidations, wrong SL/TP,
oversized positions). The system has been completely overhauled with:
- Capital-proportional sizing (20% max margin per position)
- Progressive trailing stops
- TIME_EXIT respects breakeven + SL
- Partial take records trades
- MiniMax M2.7 for all decisions

**Active Rules**:
- BONK consistently loses → AVOID
- RAY, FARTCOIN, WIF, PENGU, POPCAT → historically best performers
- In extreme fear (FG<15): shorts are high probability
- Trailing stop should tighten as profit grows (progressive trailing active)
- Partial takes at 50% of TP distance lock in profits

**Parameters**: SL=2.5%, TP=8%, leverage=3-7x (confidence-based), max 6 positions
