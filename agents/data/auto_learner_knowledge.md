# Auto-Learner Knowledge Base

## Current Lessons (Post-Fix 2026-04-10)

...(older entries trimmed)...

**Analysis**: Despite an 83.3% win rate, the R:R ratio of 0.69 indicates the strategy is unprofitable because avg loss ($0.0021) exceeds avg win ($0.0014). The high win rate is masking fundamental risk-reward dysfunction. EMERGENCY_CLOSE trades are the biggest problem, showing 50% WR with the largest avg loss of $0.0021, suggesting the 5% trailing stop is too loose and allowing winning trades to reverse into losses.
**Lessons**:
- High win rate (83%) with R:R below 1.0 is a losing strategy - prioritize R:R over win rate in optimization
- EMERGENCY_CLOSE is consistently the worst-performing close reason - 50% WR with largest avg loss indicates trailing stop is allowing reversals; need tighter trailing or earlier profit-taking
- reconcile_orphan close reasons (100% WR) suggest system exits are capturing optimal points - investigate what triggers these vs EMERGENCY_CLOSE to replicate the pattern
- With only 6 trades on SOL, need more diverse token exposure before drawing strong conclusions
**Prefer**: SOL
**Confidence**: 45%

## Cycle 2026-04-21 04:53
**Insight**: The 2.0% trailing stop (recently increased from 0.5%) appears too loose—the EMERGENCY_CLOSE and DAILY_TARGET closes with 50% WR and negative PnL indicate winners are being given back. The bot needs tighter trailing stops to lock in gains rather than letting profitable trades reverse.
**Analysis**: The bot maintains an 81.8% win rate with a healthy 1.97 R:R ratio, but the absolute PnL is negligible ($0.02) indicating position sizing may be too conservative. SOL dominates with 87.5% WR while JUP is a consistent loser (0% WR, 2 losses), and ETH shows exceptional performance at 100% WR despite limited trades. The reconcile_orphan close reasons consistently produce 100% WR, suggesting system exits are timing markets well.
**Lessons**:
- DAILY_TARGET max reached close reasons show 50% WR with negative PnL—hitting 25%+ daily targets appears to trigger premature or unlucky exits; consider scaling out at lower daily target thresholds (e.g., 15-20%)
- Reconcile_orphan close reasons at 100% WR across all variants (v2.12.2, v2.12.6, v2.12.12) indicate system cleanup trades are capturing optimal exit points; investigate what market conditions trigger these vs manual closes to replicate the pattern
- JUP is 0% WR across 2 trades with consistent losses—add to avoid list alongside BONK from prior knowledge base
- Low absolute PnL ($0.02) despite good win rate suggests position sizing may be too conservative for meaningful returns; consider if risk_per_trade should be increased slightly when conviction is high
**Avoid**: JUP, BONK
**Prefer**: ETH, SOL
**Confidence**: 65%

## Cycle 2026-04-21 18:20
**Insight**: System-generated reconcile_orphan closes consistently hit 100% WR while manual/sweep closes (DAILY_TARGET, EMERGENCY_CLOSE) fail at 50% WR with losses—the bot's automated exit logic is optimal, but manual intervention is destroying gains. The trailing stop needs tightening to prevent winners becoming losers.
**Analysis**: The bot maintains an excellent 86.7% win rate with a healthy 2.44 R:R ratio, but absolute profits remain tiny at $0.04, indicating position sizing is too conservative. ETH shows perfect 100% WR while SOL dominates with 11 trades at 88.9% WR. The trailing stop at 2.81% is allowing EMERGENCY_CLOSE trades to reverse into losses, whereas system-generated reconcile_orphan closes achieve 100% WR consistently.
**Lessons**:
- reconcile_orphan close reasons (100% WR) consistently outperform all manual/sweep closes (50% WR with losses)—investigate and replicate the trigger conditions for these system exits rather than relying on manual intervention
- EMERGENCY_CLOSE and DAILY_TARGET closes are net negative despite reasonable win rates, indicating the trailing stop is too loose and allowing profitable trades to reverse into losses before capture
- When avg win ($0.0032) and avg loss ($0.0013) are both extremely small absolute values, even a good R:R ratio produces negligible total PnL—position sizing calibration may need to scale with available capital
**Avoid**: JUP, BONK
**Prefer**: ETH, SOL
**Confidence**: 90%

## Cycle 2026-04-22 05:44
**Insight**: System-generated reconcile_orphan closes consistently hit 100% WR while manual/sweep closes (EMERGENCY_CLOSE, DAILY_TARGET) fail at 50% WR with losses—the bot's automated exit logic is optimal but manual intervention is destroying gains. Tightening the trailing stop further would reduce EMERGENCY_CLOSE losses.
**Analysis**: The bot maintains an excellent 83.3% win rate with 1.71 R:R, but absolute PnL remains negligible at $0.04 over 22 trades—position sizing may be too conservative for meaningful returns. ETH shows perfect 100% WR while SOL dominates in volume but with lower 77.8% WR. EMERGENCY_CLOSE still shows 0% WR with losses, suggesting the 2.1% trailing stop is still too loose and allowing winners to reverse.
**Lessons**:
- EMERGENCY_CLOSE losses indicate trailing stop is still too loose—the current 2.1% allows profitable trades to reverse before triggering, so further tightening to 1.5-1.75% should reduce this close reason's losses
- System-generated reconcile_orphan closes at 100% WR consistently outperform all manual closes—the bot should minimize manual intervention and trust the automated exit logic when confidence is high
**Avoid**: JUP, BONK
**Prefer**: ETH, SOL
**Confidence**: 85%

## Cycle 2026-04-23 00:15
**Insight**: The trailing stop is fundamentally broken - it's allowing positions to extend to 32.52% before closing, causing avg loss to be 15x larger than avg win. Even a 75% win rate becomes irrelevant when 3 losing trades destroy 50+ winning trades.
**Analysis**: Despite a 75% win rate across 28 trades, the strategy is losing money because avg loss ($0.0977) is 15x larger than avg win ($0.0065), producing a catastrophic 0.07 R:R ratio. The DAILY_TARGET close reason at 32.52% shows 0% WR with an avg loss of $0.1937, with the last 3 trades totaling -$0.5812 - this is the primary PnL destroyer. All tokens are net negative despite decent win rates, indicating the trailing stop at 1.75% is too loose, allowing profitable trades to reverse into massive losses.
**Lessons**:
- DAILY_TARGET at 32.52% shows 0% WR with avg loss $0.1937 - hitting 25%+ daily targets triggers catastrophic reversals; implement hard exit at 20% daily target
- Avg loss ($0.0977) being 15x larger than avg win ($0.0065) makes even 75% WR unprofitable - the priority must be reducing loss magnitude, not increasing win rate
- Last 10 trades show 3 consecutive large losses totaling -$0.5812, suggesting the bot enters during short-term reversals at cycle tops; avoid entries when daily target already elevated
**Avoid**: JUP, BONK, RAY
**Prefer**: ETH, SOL
**Confidence**: 82%

## Cycle 2026-04-23 03:23
**Insight**: EMERGENCY_CLOSE is the primary PnL destroyer - it's showing 25% WR with losses while system-generated reconcile_orphan closes hit 100% WR. The trailing stop is too loose, allowing winning trades to reverse into massive losses before capture. All tokens (JUP, SOL, ETH) are net negative, and all trades are longs with no diversification.
**Analysis**: The bot is in free fall with recent win rate dropping to 47.4% and all 36 trades producing a net loss of -$0.53. The avg loss ($0.0550) is 8.7x larger than avg win ($0.0063), producing a catastrophic 0.12 R:R ratio that makes profitability impossible regardless of win rate. The trailing stop at 1.3125% is still allowing positions to extend to 32.52% daily targets before EMERGENCY_CLOSE triggers, destroying gains.
**Lessons**:
- All 10 most recent trades are longs with 9 EMERGENCY_CLOSE exits - the bot is only going long and getting stopped out repeatedly; need diversification to short side or a filter that prevents long entries when conditions are bearish
- DAILY_TARGET closes at 32%+ daily moves show 0% WR with avg loss $0.1937 - the bot is entering at cycle tops and getting crushed on reversals; avoid entries when daily target already shows 15%+ gains
- System reconcile_orphan closes at 100% WR vs EMERGENCY_CLOSE at 25% WR proves the bot's automated logic is superior - minimize manual intervention and trust the system's孤儿 reconciliation
- Avg loss ($0.0550) being 8.7x avg win ($0.0063) means even 62.5% WR produces losses - the priority must be reducing loss magnitude, not chasing higher win rates
**Avoid**: JUP, SOL, ETH
**Prefer**: BONK, RAY, FARTCOIN
**Confidence**: 65%

## Cycle 2026-04-23 03:56
**Insight**: The 5% stop loss is too loose - it allowed positions to extend to 32.52% daily targets before EMERGENCY_CLOSE, producing losses 50x larger than winners. Even with the recent trailing stop adjustment to 1.0%, the 10% take profit target requires 75% win rate just to break even, which the bot cannot maintain. The priority must be dramatically reducing loss magnitude.
**Analysis**: The bot is in a severe drawdown with the last 20 trades showing 35% WR and -$0.61 PnL. The catastrophic R:R of 0.14 (avg loss 7.2x larger than avg win) makes profitability impossible even with 60% win rate. All three traded tokens (JUP, SOL, ETH) are net losers, and every recent trade is a long on JUP getting stopped out by EMERGENCY_CLOSE with small losses accumulating.
**Lessons**:
- All three tokens (JUP, SOL, ETH) are net negative performers in current cycle - the bot is picking losers, not just getting unlucky with timing
- EMERGENCY_CLOSE with losses (avg -$0.01 to -$0.05) shows trailing stop is too loose even at 1.0% - tighten to 0.75% and add time-based exit to prevent extended holds
- 10% TP with 5% SL requires 83% WR to break even - current system cannot sustain this; reducing TP to 6% brings break-even down to 71% which is achievable
- Manual intervention (MANUAL_CLOSE, extended DAILY_TARGET exits) consistently destroys value - trust system-generated reconcile_orphan closes
**Avoid**: JUP, SOL, ETH, BONK
**Prefer**: POPCAT, FARTCOIN, WIF
**Confidence**: 78%

## Cycle 2026-04-23 04:27
**Insight**: JUP is a complete write-off (0% WR, 20 straight losses) and the bot keeps entering it - this is the single biggest problem. Combined with EMERGENCY_CLOSE capturing only small wins while allowing positions to drift into larger losses, the system is destroying capital systematically.
**Analysis**: The bot is in a death spiral - 30% WR over last 20 trades with JUP losing across all 20 positions, and the catastrophic 0.19 R:R makes profitability mathematically impossible. EMERGENCY_CLOSE remains the dominant close reason (19/20 recent trades), indicating the trailing stop is still miscalibrated for current market conditions. All recent trades are longs on a single losing token with no diversification.
**Lessons**:
- JUP has now lost 20 consecutive trades with 0% WR across both long and short sides - this token has a fundamental directional bias against the bot's entry logic and should be hard-avoided until it shows 60%+ WR over at least 10 trades
- EMERGENCY_CLOSE dominance (19 of 20 recent trades) indicates the bot is entering at inflection points then getting stopped out repeatedly - this pattern suggests the entry signal needs filtering, not that trailing stop needs adjustment
- When all positions are same direction (all longs) on same token, there's no diversification benefit - the bot is essentially making one correlated bet 20 times and losing
**Avoid**: JUP, BONK, RAY
**Prefer**: POPCAT, FARTCOIN, WIF
**Confidence**: 68%

## Cycle 2026-04-23 04:45
**Insight**: JUP has now lost 20 straight trades (0% WR) and the bot keeps re-entering it - this is a broken trade signal, not bad luck. Combined with all positions being longs with zero diversification, the bot is making one correlated bet 20 times and losing every time. The trailing stop adjustment is working (avg loss shrunk from $0.055 to $0.0059), but the entry logic for JUP is fundamentally broken.
**Analysis**: The bot is in a persistent losing spiral with JUP accounting for all 20 recent trades at 0% WR, and EMERGENCY_CLOSE dominating as the close reason. The trailing stop at 0.65% is now correctly producing small losses (avg $0.0059) vs the previous catastrophic losses, but the real problem is the bot keeps entering a token that has never won. The 6.2% TP with 3.2% SL requires 66% WR to break even, but recent 30% WR makes profitability impossible.
**Lessons**:
- A token with 0% WR across 15+ trades is a broken signal, not variance - hard-avoid until it shows 55%+ WR over 10+ new trades
- EMERGENCY_CLOSE losses shrinking from $0.055 to $0.006 confirms the trailing stop tightening from 2.0% to 0.5% is working correctly
- All positions same direction on same token = maximum correlated risk with minimum diversification benefit - implement hard rule to diversify across tokens or directions
- 30% WR over 20 trades is not variance, it's a broken strategy for that specific token/timeframe combination
**Avoid**: JUP, BONK, RAY
**Prefer**: POPCAT, FARTCOIN, WIF, SOL
**Confidence**: 72%

## Cycle 2026-04-23 04:51
**Insight**: JUP has now lost 20 consecutive trades as longs, making it a broken signal that must be immediately avoided - the bot continues to enter the same losing position repeatedly with no diversification across tokens or directions.
**Analysis**: The bot is in a severe drawdown with JUP showing 20 consecutive losses (0% WR) and a 20% overall win rate over the last 20 trades. The trailing stop was just increased to 2.0% but the recent data shows small losses averaging $0.0056, suggesting 2.0% may now be too loose compared to the previously effective 0.5%. However, the R:R ratio for recent trades is healthy at 1.43 (avg win $0.008 vs avg loss $0.0056), indicating the underlying trade logic is working but the JUP signal is fundamentally broken.
**Lessons**:
- JUP has now lost 20 consecutive long trades (0% WR) - this is not variance, it's a broken entry signal that must be hard-blocked from the trading system until it shows 55%+ WR over at least 10 new trades
- The 2.0% trailing stop increase was premature - recent data shows 1.0-1.25% better balances profit capture vs preventing winners becoming losers for JUP-sized positions
- When avg loss ($0.0056) is less than avg win ($0.008), R:R becomes healthy at 1.43 - the priority is finding tokens that produce this pattern, not chasing higher win rates on broken signals
**Avoid**: JUP, BONK, RAY
**Prefer**: POPCAT, FARTCOIN, WIF, SOL
**Confidence**: 75%

## Cycle 2026-04-23 05:11
**Insight**: JUP is a catastrophic broken signal (0% WR across 20 trades) that the bot keeps entering, and the recent 2.0% trailing stop increase will likely recreate the same EMERGENCY_CLOSE loss pattern that was just fixed when trailing was at 0.5%.
**Analysis**: The bot is in a persistent drawdown with JUP losing 20 consecutive trades (0% WR) and EMERGENCY_CLOSE dominating 100% of recent closes. The trailing stop was just increased to 2.0% but historical data shows this will allow winners to reverse into losses - the prior 0.5% trailing stop produced a healthy 1.43 R:R ratio. All positions are longs on a single broken token with no diversification.
**Lessons**:
- JUP has now lost 20 consecutive trades (0% WR) as longs - this is not variance, it's a broken entry signal that must be hard-blocked from the trading system until it shows 55%+ WR over at least 10 new trades
- The 2.0% trailing stop increase was premature - recent data shows avg loss was $0.0056 and avg win $0.008 when trailing was at 0.5%, producing a healthy 1.43 R:R; increasing to 2.0% will allow profitable trades to reverse into losses
- All 20 recent trades are longs on JUP with 100% EMERGENCY_CLOSE - this pattern indicates the bot is making one correlated bet repeatedly and losing; need hard diversification rule across tokens OR directions
**Avoid**: JUP, BONK, RAY
**Prefer**: POPCAT, FARTCOIN, WIF, SOL
**Confidence**: 78%

## Cycle 2026-04-23 05:17
**Insight**: The 2.0% trailing stop increase was just made but prior cycles proved this level allows profitable trades to reverse into losses (EMERGENCY_CLOSE 100% close reason with avg loss 5x larger than avg win). The 0.5% trailing was producing healthy 1.43 R:R before the increase. Also, all 20 recent trades are longs with zero diversification.
**Analysis**: The bot is in a persistent drawdown with 30% WR over last 20 trades and all positions closed via EMERGENCY_CLOSE. JUP has lost 15 consecutive long trades (26.7% WR) while SOL shows 40% WR but also net negative. The R:R ratio is a healthy 1.02 recently but the win rate is too low to be profitable. The trailing stop was just increased from 0.5% to 2.0%, which historically recreated the same EMERGENCY_CLOSE loss pattern that nearly destroyed the account before.
**Lessons**:
- The 2.0% trailing stop increase was just applied but historical data proves this level is too loose — it allows profitable trades to reverse into losses via EMERGENCY_CLOSE with losses 5x larger than wins. Revert to 0.75-1.0% immediately.
- All 20 recent trades are longs on JUP or SOL with 100% EMERGENCY_CLOSE — this pattern indicates the bot is making correlated directional bets and getting stopped out repeatedly. Implement hard diversification: max 1 position per token OR require at least 2 tokens in opposite directions.
**Avoid**: JUP
**Prefer**: SOL
**Confidence**: 82%

## Cycle 2026-04-23 05:42
**Insight**: The recent 2.0% trailing stop increase is allowing profitable trades to reverse into losses before capture - every single recent trade closes via EMERGENCY_CLOSE with loss exceeding win. This confirms prior cycle findings that 0.5-1.0% trailing was optimal for JUP-sized positions. The 2.0% setting recreates the exact pattern that nearly destroyed the account before.
**Analysis**: The bot is in a persistent drawdown with both major tokens (JUP, SOL) showing 0% WR on longs across recent trades. Every recent trade closes via EMERGENCY_CLOSE with losses exceeding wins, indicating the trailing stop at 2.0% is too loose. The 5.5% TP with 3.5% SL requires 61% win rate to break even but recent data shows only 30%, making profitability impossible. The 2.0% trailing stop adjustment was likely counterproductive based on historical patterns.
**Lessons**:
- All 20 recent trades closing via EMERGENCY_CLOSE with losses proves the 2.0% trailing stop is fundamentally broken - revert immediately to 1.0% which historically captured winners before reversal
- JUP has now lost 11 consecutive long trades and SOL has lost 9 consecutive long trades - this is not variance, it's a market regime shift where longs on these tokens fail systematically
- When every close reason is EMERGENCY_CLOSE, it means trailing is always activating but losing - the trailing stop needs to be tighter to lock in gains earlier rather than giving back profits
**Avoid**: JUP, BONK
**Prefer**: SOL
**Confidence**: 82%

## Cycle 2026-04-23 06:04
**Insight**: The 2.0% trailing stop is too loose - ETH wins via EMERGENCY_CLOSE at 0.5-1.0% levels while SOL/JUP lose when positions drift beyond 1%. The bot needs to lock in gains earlier (0.75-1.0% trailing) and ONLY trade ETH until JUP/SOL prove themselves over 10+ trades with 55%+ WR.
**Analysis**: ETH is the only profitable token with 100% WR across 5 trades while JUP (22.2%) and SOL (16.7%) are catastrophic losers. All 10 recent trades closed via EMERGENCY_CLOSE - ETH capturing small wins ($0.0035-$0.0149) while SOL/JUP suffered losses. The recent 2.0% trailing stop increase from 0.5% is allowing profitable trades to reverse into losses before capture, recreating the exact pattern that destroyed prior cycles.
**Lessons**:
- ETH shows 100% WR as longs while JUP/SOL show 16-22% WR - the entry logic works on ETH but fails on other tokens; this suggests market regime dependency where ETH momentum is captured while JUP/SOL are in downtrends relative to entry timing
- All 10 recent trades close via EMERGENCY_CLOSE with small wins/losses ($0.003-$0.015) - this indicates the 2.0% trailing is activating correctly but at levels that either capture small wins (ETH) or small losses (SOL/JUP); tightening to 1.0% would lock in ETH wins earlier and prevent SOL/JUP from drifting into larger losses
**Avoid**: JUP, SOL
**Prefer**: ETH
**Confidence**: 78%

## Cycle 2026-04-23 12:59
**Insight**: The 2.0% trailing stop increase was a mistake - all 10 recent trades close via EMERGENCY_CLOSE with losses exceeding wins on SOL/JUP while ETH barely profits, proving the trailing stop is too loose. Prior cycles established that 0.75-1.0% trailing captured winners before reversal. ETH's 77.8% WR vs SOL's 40% WR means ETH should dominate allocation.
**Analysis**: Recent performance is recovering with 55% WR and +$0.03 PnL (R:R 1.42) compared to overall 45.8% WR and -$0.63 PnL, suggesting recent parameter adjustments are working. ETH dominates as the only profitable token (77.8% WR, +$0.0343) while SOL and JUP are net losers. All trades exit via EMERGENCY_CLOSE, indicating the 2.0% trailing stop is allowing reversals rather than capturing optimal exit points - prior cycles proved 0.5-1.0% trailing was optimal.
**Lessons**:
- When ALL closes are EMERGENCY_CLOSE with small mixed wins/losses, the trailing stop is miscalibrated - too loose if it allows reversals, too tight if it cuts winners; current 0.75% is the middle ground that works
- ETH at 77.8% WR with +$0.0343 vs SOL at 40% WR with -$0.0032 proves token selection matters more than direction - allocate 80%+ to ETH until SOL shows 55%+ WR over 10+ trades
**Avoid**: JUP, BONK, RAY
**Prefer**: ETH, SOL
**Confidence**: 85%

## Cycle 2026-04-24 13:56
**Insight**: The 2.0% trailing stop is too loose for a 4.5% TP - prior cycles established that 0.75-1.0% trailing locks in gains before reversals, while the 2.0% level consistently causes EMERGENCY_CLOSE losses. Also, TIME_EXIT is a guaranteed loser that should be replaced with tighter system-generated exits.
**Analysis**: The bot is in severe decline with 16.7% WR across 6 trades and negative PnL (-$0.06), a dramatic drop from the 50%+ WR seen in prior successful cycles. All trades are longs on SOL/ETH with TIME_EXIT being the worst close reason (-$0.0497, 0% WR), while ORPHAN_RECONCILE_v2.12.31 (without #2) shows 50% WR with +$0.0216. The trailing stop was just increased to 2.0%, but historical cycles proved this allows winners to reverse into losses via EMERGENCY_CLOSE.
**Lessons**:
- TIME_EXIT is the worst performing close reason (0% WR, -$0.0497) - this should be replaced with system-generated ORPHAN_RECONCILE exits which consistently outperform manual timing
- ORPHAN_RECONCILE_v2.12.31 (without #2) shows 50% WR vs #2 variant at 0% WR - the system has better-performing reconciliation logic that should be prioritized over manual exits
- 16.7% WR with 6 trades is statistically insignificant but represents a severe decline from 50%+ WR in prior cycles - this may indicate a market regime shift requiring different entry conditions
**Avoid**: SOL, ETH
**Prefer**: POPCAT, FARTCOIN, WIF
**Confidence**: 72%

## Cycle 2026-04-25 14:02
**Insight**: TIME_EXIT is a guaranteed loser (0% WR, -$0.0497) that should be immediately disabled, while system-generated ORPHAN_RECONCILE closes consistently outperform manual timing. The bot is only trading losing tokens (SOL 25% WR, ETH 33.3% WR) when JUP is the only profitable performer at 50% WR.
**Analysis**: The bot is in a severe losing state with 33.3% WR and -$0.05 PnL across 9 trades. TIME_EXIT is the worst performer (0% WR, -$0.0497) while system-generated ORPHAN_RECONCILE_v2.12.31 without #2 shows 50% WR. SOL is the worst token (25% WR, -$0.0439) followed by ETH (33.3% WR, -$0.0148). JUP is the only profitable token (50% WR, +$0.0060) despite prior cycles marking it as broken—this suggests the entry logic may have been updated. The 0.9375% trailing stop appears well-calibrated but needs evaluation against the recent 2.0% change mentioned in knowledge base.
**Lessons**:
- TIME_EXIT has now shown 0% WR across multiple cycles (-$0.0497 in current cycle) - this is a guaranteed loss mechanism that must be replaced with system-generated ORPHAN_RECONCILE exits
- ORPHAN_RECONCILE_v2.12.31 without #2 variant shows 50% WR while the #2 variant shows 0% WR - the system is aware of this discrepancy and the non-#2 version should be prioritized
- JUP has flipped to 50% WR (+$0.0060) in the current cycle despite prior cycles marking it as broken - this may indicate improved entry logic making prior avoid rules partially obsolete, though SOL/ETH remain confirmed losers
- All 9 trades are longs with no diversification - this correlated directional betting pattern has destroyed prior cycles and needs immediate diversification across tokens OR directions
**Avoid**: SOL, ETH
**Prefer**: JUP, POPCAT, FARTCOIN, WIF
**Confidence**: 68%

## Cycle 2026-04-26 14:12
**Insight**: JUP has dramatically flipped from a 0% WR broken signal to a 66.7% WR outperformer (+$0.2644) while SOL/ETH have deteriorated - the bot's current allocation is misaligned with where profits actually are.
**Analysis**: The bot is marginally profitable (+$0.13) with a healthy 2.0 R:R ratio despite only 42.9% WR, demonstrating solid risk-reward mechanics. JUP is the clear outperformer (66.7% WR, +$0.2644) while SOL and ETH are confirmed losers (20% and 33% WR respectively, both net negative). The 0.9375% trailing stop is working well with TRAILING_SL showing 66.7% WR and +$0.1640 profit. All trades are longs with no direction diversification, which is a systematic risk but hasn't prevented profitability on JUP.
**Lessons**:
- PARTIAL_TAKE strategy shows 100% WR with best single trade profit ($0.1727) - consider implementing partial exit rules to lock in gains earlier
- SMART_ROTATION shows 0% WR with worst avg loss ($0.0808) across 2 trades - this exit mechanism consistently destroys value and should be replaced with TRAILING_SL or system-generated reconcile
- JUP has completely reversed from broken signal (0% WR in prior cycles) to top performer (66.7% WR, +$0.2644) - prior avoid rules may need reconsideration based on current entry logic updates
- TIME_EXIT remains a guaranteed 0% WR mechanism (-$0.0497) - should continue to be disabled in favor of system-generated exits
**Avoid**: SOL, ETH
**Prefer**: JUP
**Confidence**: 85%