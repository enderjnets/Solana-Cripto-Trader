"""
sweep.py — Grid search sobre configuraciones de wild mode.
Encuentra la combinación con mejor sharpe ratio sujeto a constraints de seguridad.
"""
from __future__ import annotations
import json
import statistics
import itertools
import time
from pathlib import Path
from dataclasses import asdict
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from agents.wild_mode_backtest.simulator import (
    simulate_session, load_tokens_data, load_fg_data, POLICIES
)

OUTPUT = HERE.parent / 'data' / 'wild_mode_sweep_results.json'

PARAM_GRID = {
    'max_multiplier':       [1.3, 1.5, 1.8, 2.0],
    'max_levels':           [3, 4, 5],
    'max_chain_ratio':      [3.0, 4.0, 5.0],
    'max_global_margin_pct':[0.4, 0.6, 0.8],
    'max_drawdown_pct':     [0.10, 0.15, 0.20],
    'policy':               ['aggressive', 'balanced', 'conservative', 'ai_proxy'],
}

# Fixed params
FIXED = {
    'sl_pct': 0.025,
    'tp_pct': 0.05,
    'leverage': 5,
    'margin_per_pos': 20.0,
}

SESSIONS_PER_CONFIG = 50
TARGET_USD = 2.0


def grid_iterate():
    keys = list(PARAM_GRID.keys())
    for combo in itertools.product(*[PARAM_GRID[k] for k in keys]):
        yield dict(zip(keys, combo))


def run_config(tokens, fg, params):
    config = {**FIXED}
    for k, v in params.items():
        if k != 'policy':
            config[k] = v
    policy = params['policy']
    results = []
    for i in range(SESSIONS_PER_CONFIG):
        try:
            r = simulate_session(tokens, fg, config, policy, target_usd=TARGET_USD, seed=hash((tuple(sorted(params.items())), i)) % 2**31)
            results.append(r)
        except Exception as e:
            continue
    if not results:
        return None

    pnls = [r.realized_pnl for r in results]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(results)
    avg_pnl = statistics.mean(pnls)
    std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 0.001
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
    abandon_rate = sum(1 for r in results if r.outcome == 'abandoned') / len(results)
    target_hit_rate = sum(1 for r in results if r.outcome == 'target_hit') / len(results)
    avg_dd = statistics.mean([r.max_drawdown_pct for r in results])
    max_dd = max([r.max_drawdown_pct for r in results])
    avg_levels = statistics.mean([r.levels_opened for r in results])
    avg_duration = statistics.mean([r.duration_candles for r in results])

    return {
        'params': params,
        'n_sessions': len(results),
        'win_rate': round(win_rate, 4),
        'avg_pnl': round(avg_pnl, 4),
        'std_pnl': round(std_pnl, 4),
        'sharpe': round(sharpe, 4),
        'abandon_rate': round(abandon_rate, 4),
        'target_hit_rate': round(target_hit_rate, 4),
        'avg_drawdown': round(avg_dd, 4),
        'max_drawdown': round(max_dd, 4),
        'avg_levels': round(avg_levels, 2),
        'avg_duration_candles': round(avg_duration, 1),
    }


def main():
    tokens = load_tokens_data()
    fg = load_fg_data()
    print(f"Loaded {len(tokens)} tokens, {len(fg)} F&G entries")
    if not tokens:
        print("ERROR: no tokens — run fetch_history.py")
        return
    if not fg:
        print("WARNING: no F&G data, using neutral")

    combos = list(grid_iterate())
    print(f"Sweep: {len(combos)} configs × {SESSIONS_PER_CONFIG} sessions = {len(combos)*SESSIONS_PER_CONFIG} simulations")

    t0 = time.time()
    results = []
    for i, params in enumerate(combos):
        r = run_config(tokens, fg, params)
        if r:
            results.append(r)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            print(f"  [{i+1}/{len(combos)}] elapsed={elapsed:.0f}s eta={eta:.0f}s last_sharpe={r['sharpe'] if r else 'N/A'}")

    print(f"\nCompleted: {len(results)} valid configs in {time.time()-t0:.0f}s")

    # Filter by safety constraints
    safe = [r for r in results if r['abandon_rate'] < 0.20 and r['max_drawdown'] < 0.18 and r['win_rate'] > 0.50]
    print(f"After safety filter (abandon<20%, max_dd<18%, win>50%): {len(safe)} configs")

    # Sort by sharpe
    safe.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\nTop 10 configs by sharpe ratio:")
    print(f"{'#':<3} {'sharpe':<8} {'win%':<7} {'avgPnL':<9} {'abnd%':<7} {'maxDD':<7} {'lvls':<5} {'policy':<14} {'mult':<5} {'lvls':<5} {'ratio':<6} {'gMrg':<6}")
    for i, r in enumerate(safe[:10]):
        p = r['params']
        print(f"{i+1:<3} {r['sharpe']:<8.3f} {r['win_rate']*100:<7.1f} {r['avg_pnl']:<+9.3f} {r['abandon_rate']*100:<7.1f} {r['max_drawdown']*100:<7.1f} {r['avg_levels']:<5.1f} {p['policy']:<14} {p['max_multiplier']:<5.1f} {p['max_levels']:<5} {p['max_chain_ratio']:<6.1f} {p['max_global_margin_pct']:<6.2f}")

    OUTPUT.write_text(json.dumps({
        'sweep_summary': {
            'total_configs': len(combos),
            'valid_configs': len(results),
            'safe_configs': len(safe),
            'sessions_per_config': SESSIONS_PER_CONFIG,
            'target_usd': TARGET_USD,
        },
        'top_safe': safe[:10],
        'all_results': results,
    }, indent=2, default=str))
    print(f"\nResults saved to {OUTPUT}")


if __name__ == '__main__':
    main()
