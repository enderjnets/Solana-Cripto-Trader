# Auto-Learner Knowledge Base

No lessons learned yet.

## Cycle 2026-03-30 08:57
**Insight**: Three liquidation events at $23.59 each are the core problem - this happens when leverage is too high and stop loss is too tight, causing market movements to instantly exceed collateral. The strategy needs to survive normal market volatility, not get instantly liquidated on normal swings.
**Analysis**: This strategy is catastrophically broken with a 10.9% win rate and -$99.26 total loss. The three liquidations alone account for $70.76 (71% of total losses), each costing ~$23.59. The trailing stop change to 2.0% didn't help - 27 trailing stop trades all lost, suggesting positions are being stopped out before breaking even despite being in profit. Short positions in ADA, SOL, and ETH are the primary destroyers of capital.
**Lessons**:
- Never use 5x-10x leverage on crypto with tight stop losses - normal 1-2% swings liquidate positions instantly when leveraged
- Trailing stop of 2% is too aggressive for volatile tokens - 27 consecutive losses means the market keeps hitting the stop before continuing in your favor
- All three liquidations happened on short positions after rapid upside moves - avoid shorting during bullish momentum
- The last 10 trades show a clear pattern: timeouts hurt small, take profits work well, but liquidations are catastrophic and dwarf all wins
**Avoid**: ADA, SOL, ETH
**Prefer**: DOT, XRP
**Confidence**: 92%

## Cycle 2026-04-01 11:55
**Insight**: The R:R ratio is the core problem - with 72.7% WR you should be highly profitable, but large single losses on WIF (-$6.09) are wiping out multiple wins. The trailing stop change to 2% may have helped avoid liquidations but is now cutting winners short prematurely.
**Analysis**: The strategy has a solid 72.7% win rate but an abysmal 0.41 R:R ratio because avg losses ($4.34) more than double avg wins ($1.79). WIF caused $12.18 in losses on just 2 trades while RAY generated $11.22 profit on 4 trades. All 11 trades closed via EMERGENCY_CLOSE, suggesting the trailing stop of 2% is still too aggressive for volatile Solana tokens.
**Lessons**:
- WIF is extremely dangerous for shorts - single trade lost $6.09, likely due to rapid upside spikes during the session
- RAY consistently delivers wins on short positions (4/4, $11.22 total) - the clear preferred token for this strategy
- EMERGENCY_CLOSE dominating all 11 trades indicates the trailing stop is too tight relative to market volatility - positions can't reach their natural take profit levels
- With 72.7% WR and poor R:R, the problem is loss size, not win frequency - focus on letting winners run longer
**Avoid**: WIF, SOL
**Prefer**: RAY, BONK, GOAT
**Confidence**: 78%

## Cycle 2026-04-01 14:50
**Insight**: The 2.0% trailing stop is prematurely closing profitable positions. With 86% WR, letting winners run longer (higher TP and wider trailing stop) would dramatically improve R:R without significantly increasing risk, since the high win rate means most trades should hit take profit.
**Analysis**: The strategy has an excellent 86.2% win rate but is crippled by a terrible 0.34 R:R ratio - losses average 3x the size of wins. All 29 recent trades closed via EMERGENCY_CLOSE, indicating the 2.0% trailing stop is too tight, cutting winners short before they can fully develop. WIF alone destroyed $11.05 across 5 trades, while RAY generated $15.55 profit on 8 trades - the clear winner in this strategy.
**Lessons**:
- All 29 recent trades closed via EMERGENCY_CLOSE at 2.0% trailing stop - this confirms the trailing stop MUST be widened for Solana meme tokens which exhibit frequent 2-4% intraday swings
- WIF is catastrophic for shorts ($11.05 loss on 5 trades) - likely due to rapid upside liquidity spikes during the session, continue avoiding
- RAY is the star performer with 100% WR and $15.55 profit across 8 short trades - dominates total PnL
- With 86% WR and the trailing stop fix, the strategy should see immediate R:R improvement as winners are no longer cut short
**Avoid**: WIF, SOL, ETH
**Prefer**: RAY, JUP, GOAT, BONK, FARTCOIN
**Confidence**: 82%

## Cycle 2026-04-02 14:21
**Insight**: The core problem is that average wins ($0.62) nearly equal average losses ($0.60), so the strategy is essentially break-even on risk-adjusted basis despite a 65% win rate. The trailing stop at 2% combined with 8% take profit means positions are being stopped out before reaching their full profit potential, creating a ceiling on gains.
**Analysis**: The strategy has improved significantly from a catastrophic -$99.26 to +$35.90 total PnL, but the R:R ratio remains stuck at 1.03 - meaning wins and losses are nearly identical in dollar value. With a 65% win rate and 1.03 R:R, profitability is marginal. The 2.0% trailing stop is cutting winners short too early as evidenced by all recent trades exiting via EMERGENCY_CLOSE, preventing the strategy from capturing larger moves that 8% take profits should allow.
**Lessons**:
- FARTCOIN is a consistent loser on shorts with only 33% WR and -$0.97 total loss across 9 trades - avoid completely
- The 2% trailing stop was too tight - all 50 recent trades closed via EMERGENCY_CLOSE indicating positions can't reach natural take profit levels
- RAY remains the strongest performer by volume (22 trades) and should remain the primary trading token
- WIF has recovered to 60% WR after previous catastrophic performance - consider reintroducing with caution
**Avoid**: FARTCOIN, BTC, SOL
**Prefer**: RAY, PENGU, WIF, POPCAT
**Confidence**: 72%

## Cycle 2026-04-02 20:00
**Insight**: FARTCOIN transformed from an avoid token ($0.97 loss in April 2 cycle) to the top PnL contributor ($4.56 across 8 trades) - the strategy is adapting but knowledge base updates are lagging actual performance changes.
**Analysis**: The recent 50 trades show a concerning divergence: WR dropped to 50% from 62.9% overall, yet R:R improved to 1.79, indicating wins are larger but less frequent. FARTCOIN has emerged as a top performer (+$4.56) despite being on the avoid list previously, while JUP (-$2.34, 25% WR) and SOL (-$0.39, 43% WR) remain consistent losers. The EMERGENCY_CLOSE rate of 96% (48/50 trades) suggests the trailing stop at 3.75% is still too tight relative to market volatility.
**Lessons**:
- FARTCOIN reversed its fortunes completely - from -$0.97 loss to +$4.56 profit across 8 trades, with consistent $0.57 avg win. The previous avoid recommendation was premature; FARTCOIN now belongs on the prefer list.
- JUP is the worst performer by far (-$2.34, 25% WR) and should be permanently added to the avoid list - it's destroying capital on short positions.
- EMERGENCY_CLOSE at 96% confirms that even 3.75% trailing stop is too tight for Solana meme tokens which regularly exhibit 3-5% intraday swings - the 4.5% adjustment addresses this.
**Avoid**: JUP, SOL, BONK, MOODENG
**Prefer**: FARTCOIN, WIF, RAY, POPCAT, PENGU
**Confidence**: 74%