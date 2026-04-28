# Auto-Learner Knowledge Base

## Active Rules (post-reset 2026-04-10)

- **AVOID**: BONK (consistently loses)
- **PREFER**: RAY, FARTCOIN, WIF, PENGU, POPCAT (moderado profile validated tokens)
- **In extreme fear (FG<15)**: shorts have higher probability
- **Progressive trailing**: tightens with profit
- **Partial takes**: at 50% of TP distance to lock in profits

## Risk Parameters
- SL: 2.5%
- TP: 8.0%
- Max positions: 4
- Risk per trade: 1.5% of capital
- Leverage tier: 1 (5x default for moderado profile)

## Historical Note
Pre-reset trades invalidated due to equity accounting bug fixed 2026-04-09.
Auto-learner starts fresh from this point with clean $100 capital.

## Wild Mode Integration
Wild mode (martingale_engine.py) maintains its own knowledge base at
wild_mode_knowledge.jsonl with 2000 bootstrap simulations + live sessions.
Insights are injected into LLM prompts during wild mode active sessions.

## Cycle 2026-04-10 06:13
**Insight**: 100% short failure across all tokens indicates a strong directional bias against short positions. The bot continues shorting despite consecutive losses, suggesting a structural issue with directional bias selection or market regime mismatch.
**Analysis**: All 6 trades are shorts and all 6 lost money across 3 different tokens (GOAT, MOODENG, WIF). The 0% win rate suggests either a strong bullish market environment making shorts systematically fail, or the entry timing is consistently bad. The PNL_TARGET_$1.50 close reason (3 trades) indicates losses exceeded $1.50 per trade despite the 2.5% SL suggesting the positions were held too long or had adverse price action beyond the stop loss threshold.
**Lessons**:
- All-short bias is systematically failing - bot should consider alternating direction or wait for downtrend confirmation before shorting
- WIF appears in PREFER list but both WIF trades lost - remove from prefer or require additional confirmation before trading it
- MOODENG and GOAT both showing consistent short losses - avoid these tokens for now
**Avoid**: GOAT, MOODENG, WIF
**Confidence**: 72%

## Cycle 2026-04-10 06:55
**Insight**: The bot has a structural directional bias problem - it's exclusively shorting in a market regime that's crushing shorts. Previous knowledge base warnings to avoid MOODENG, GOAT, and WIF were not heeded. Until the bot starts considering long positions or waits for confirmed downtrends, it will continue losing on short-only trades.
**Analysis**: The bot has a 0% win rate across all 9 trades with -$4.27 total loss. All positions are shorts, which contradicts the previous cycle's insight that all-short bias is systematically failing. MOODENG, GOAT, and WIF were already marked to AVOID in the knowledge base, yet the bot continued trading them. The PNL_TARGET_$1.50 close reason (3 trades, avg loss $0.68) and WILD_AI_CLOSE (5 trades) indicate the bot is holding losing positions too long or hitting stops despite directional headwinds.
**Lessons**:
- All-short bias must be broken - bot needs a directional filter that allows longs when market shows bullish signs or downtrend confirmation before shorts
- Previous cycle AVOID list (MOODENG, GOAT, WIF) was not respected - knowledge base updates need enforcement in the trading logic
- Close reason 'PNL_TARGET_$1.50' losing $0.68 avg per trade suggests SL at 2.5-3.1% is being exceeded or positions held beyond stop loss
- Wild AI close reasons with avg losses of $0.17-$0.55 suggest the bot is reacting to noise rather than holding to defined TP/SL
**Avoid**: MOODENG, GOAT, WIF, FARTCOIN
**Prefer**: RAY, PENGU, POPCAT
**Confidence**: 68%

## Cycle 2026-04-11 00:12
**Insight**: BTC longs with WILD_MODE_CLOSE_CHAIN are the only profitable strategy (100% WR, $0.48 total). Session expiration is destroying otherwise viable positions - the RAY long would have been +$1.28 if not expired at 360m. The bot needs smaller position sizes (higher risk_per_trade is causing outsized losses) and longer session tolerance for altcoins with higher volatility.
**Analysis**: The bot improved from 0% win rate to 50% by switching from all-short to mixed BTC longs via WILD_MODE_CLOSE_CHAIN. However, the R:R ratio of 0.52 is bad because losses ($1.14 avg) are nearly double wins ($0.59 avg), causing net -$1.64 despite hitting 50% win rate. Session expiration (360m timeout) is the primary driver of losses, especially the catastrophic -$2.62 RAY long.
**Lessons**:
- WILD_MODE_CLOSE_CHAIN exits are consistently profitable (100% WR) - prioritize this exit mode when available
- Session expiration at 360m is catastrophic for volatile tokens (RAY lost -$2.62) - consider extending session timeout for high-volatility assets or reducing position sizes further
- BTC longs are the only consistently profitable trade type in current market regime - focus on BTC dominance before trading alts
- Average loss ($1.14) is 1.94x average win ($0.59) at current risk_per_trade - need smaller sizes until win rate improves or avg win increases
**Avoid**: RAY, ETH
**Prefer**: BTC
**Confidence**: 68%

## Cycle 2026-04-11 06:23
**Insight**: Drawdown abandonment at 15.2% is triggering catastrophic exits on otherwise valid positions, causing outsized losses that the 50% win rate cannot offset. The bot needs either earlier partial takes to reduce exposure or a tighter trailing stop to exit before drawdown threshold hits.
**Analysis**: The bot has 50% win rate but is bleeding because losses are 2.7x larger than wins ($2.47 avg loss vs $0.89 avg win). The primary killer is WILD_ABANDON_drawdown_15.2pct which caused 3 trades to lose $11.39 (avg -$3.80 each). WILD_MODE_CLOSE_CHAIN exits are the only consistently profitable strategy at 100% WR. The current 3.5% stop loss is being exceeded by drawdown abandonment which doesn't respect it.
**Lessons**:
- WILD_ABANDON_drawdown_15.2pct is the primary loss driver - positions must be exited via partial takes or trailing stops before drawdown threshold is hit
- Win rate of 50% is wasted if losses are 2.5x+ larger than wins - focus on loss size reduction before win rate improvement
- WILD_MODE_CLOSE_CHAIN exits (100% WR) should be the primary exit strategy - ensure trailing stop is positioned to trigger these exits
**Avoid**: WIF, PENGU, MOODENG, ETH
**Prefer**: RAY, BTC
**Confidence**: 72%

## Cycle 2026-04-13 13:23
**Insight**: Session expiration is the #1 loss driver - the 2 expired trades cost -$7.20 which exceeds ALL profits combined ($+4.75). The 2.0% trailing stop needs to trigger exits BEFORE session expiration, or session timeout must be reduced.
**Analysis**: Despite a solid 66.7% win rate, the bot is bleeding money because losses average 3x larger than wins ($3.60 vs $1.19). The primary damage comes from session expiration abandoning ETH (-$4.54) and SOL (-$2.66) at 361 minutes, while BTC and JUP longs are consistently profitable. The R:R ratio of 0.33 confirms the core problem: even with more winners, oversized losers destroy the portfolio.
**Lessons**:
- Session expiration at 361m is catastrophic for volatile alts - ETH and SOL combined lost $7.20 when abandoned, exceeding all profitable trades combined
- WILD_AI_CLOSE exits (FG<15 or PnL combo) are 100% WR - these signal momentum or fear shifts that should trigger trailing stop activation
- The 66.7% win rate is wasted if 2 losses ($7.20) exceed 4 wins ($4.75) - loss size reduction is more critical than win rate improvement
- All 6 trades are longs - consistent directional bias is fine if market is bullish, but ensure SL protects against reversal
**Avoid**: ETH, SOL
**Prefer**: BTC, JUP
**Confidence**: 72%

## Cycle 2026-04-13 21:46
**Insight**: Session expiration at 361 minutes is the #1 threat - these 2 trades lost $7.20 which exceeds ALL cumulative profits. The trailing stop was correctly increased from 0.5% to 2.5% to capture more gains before abandonment triggers, but session timeout needs to be reduced or positions must exit via trailing stop much earlier.
**Analysis**: The 81.8% win rate is excellent but the catastrophic session expiration losses ($7.20 on 2 trades) exceed all profits combined, while average loss ($3.60) is 2.6x the average win ($1.37), creating a destructive R:R ratio of 0.38. The WILD_MODE_CLOSE_CHAIN and WILD_AI_CLOSE exits are 100% WR and responsible for all profits, while session expiration at 361m is the primary destruction mechanism. BTC and JUP are the star performers while ETH and SOL consistently underperform.
**Lessons**:
- WILD_ABANDON_session_expired_361m caused -$7.20 in losses on just 2 trades - session timeout MUST be reduced to 240m or positions must have tighter trailing stop activation to exit before expiration
- BTC and JUP are the dominant performers with 100% WR each, contributing +$7.80 combined vs -$7.20 from expired ETH/SOL - concentrate allocation on these two tokens
- The 2.5% trailing stop is correctly calibrated - increasing from 0.5% allowed WILD_MODE_CLOSE_CHAIN exits to capture full profit on JUP ($2.91) and SOL ($2.11) trades
- With 81.8% WR, the problem is never about being wrong but about loss magnitude - always prioritize loss size reduction over win rate improvement
**Avoid**: ETH, SOL
**Prefer**: JUP, BTC
**Confidence**: 78%