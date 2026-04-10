"""
extract_insights.py — Bootstrap del wild_mode_knowledge.jsonl + insights.

Corre 2000 sesiones simuladas con la config top del sweep, las graba al
knowledge file con el formato que espera wild_mode_learner, computa insights
y actualiza martingale_engine con los defaults ganadores.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import asdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from agents.wild_mode_backtest.simulator import (
    simulate_session, load_tokens_data, load_fg_data
)
from agents import wild_mode_learner

SWEEP_FILE = HERE.parent / 'data' / 'wild_mode_sweep_results.json'
KNOWLEDGE_FILE = HERE.parent / 'data' / 'wild_mode_knowledge.jsonl'
INSIGHTS_FILE = HERE.parent / 'data' / 'wild_mode_insights.json'

N_BOOTSTRAP_SESSIONS = 2000
SESSIONS_PER_CONFIG = 100  # bootstrap with several top configs


def session_to_record(r, config_name: str, idx: int) -> dict:
    """Convert SessionResult to wild_mode_knowledge.jsonl record schema."""
    now = datetime.now(timezone.utc)
    started = now - timedelta(minutes=r.duration_candles * 60)
    return {
        'session_id': f'sim_{config_name}_{idx}',
        'started_at': started.isoformat(),
        'ended_at': now.isoformat(),
        'duration_minutes': float(r.duration_candles * 60),
        'outcome': r.outcome,
        'starting_equity': r.starting_equity,
        'ending_equity': r.ending_equity,
        'realized_pnl': r.realized_pnl,
        'target_usd': 2.0,
        'starting_fg': r.starting_fg,
        'ending_fg': r.ending_fg,
        'chains_created': r.chains_count,
        'total_levels_opened': r.levels_opened,
        'max_multiplier_used': r.max_multiplier_used,
        'avg_multiplier': r.avg_multiplier,
        'max_drawdown_pct': r.max_drawdown_pct,
        'guardrails_hit_count': 0,
        'ai_decisions_count': 0,
        'config_name': config_name,
        'policy': r.policy,
        'source': 'simulation_bootstrap',
    }


def main():
    tokens = load_tokens_data()
    fg = load_fg_data()
    print(f"Loaded {len(tokens)} tokens, {len(fg)} F&G entries")

    # Load top configs from sweep
    if not SWEEP_FILE.exists():
        print("ERROR: run sweep.py first")
        return
    sweep = json.loads(SWEEP_FILE.read_text())
    top_configs = sweep.get('top_safe', [])[:5]
    if not top_configs:
        print("ERROR: no safe configs in sweep")
        return

    print(f"\nBootstrapping {N_BOOTSTRAP_SESSIONS} sessions across top {len(top_configs)} configs")
    print(f"Knowledge file: {KNOWLEDGE_FILE}")

    # Backup existing knowledge if any
    if KNOWLEDGE_FILE.exists():
        backup = KNOWLEDGE_FILE.with_suffix('.jsonl.bak')
        backup.write_bytes(KNOWLEDGE_FILE.read_bytes())
        print(f"Backed up existing knowledge to {backup.name}")
        KNOWLEDGE_FILE.unlink()

    fixed = {'sl_pct': 0.025, 'tp_pct': 0.05, 'leverage': 5, 'margin_per_pos': 20.0}
    total_written = 0
    sessions_per_cfg = N_BOOTSTRAP_SESSIONS // len(top_configs)

    t0 = time.time()
    for cfg_idx, c in enumerate(top_configs):
        params = c['params']
        config = {**fixed,
                  'max_multiplier': params['max_multiplier'],
                  'max_levels': params['max_levels'],
                  'max_chain_ratio': params['max_chain_ratio'],
                  'max_global_margin_pct': params['max_global_margin_pct'],
                  'max_drawdown_pct': params['max_drawdown_pct']}
        cfg_name = f"top{cfg_idx+1}_{params['policy']}"
        print(f"\n  [{cfg_idx+1}/{len(top_configs)}] {cfg_name} sharpe={c['sharpe']:.3f} win={c['win_rate']*100:.0f}%")

        # Mix of target_usd values to capture both modes
        for i in range(sessions_per_cfg):
            target = 2.0 if i % 3 != 0 else 0.0  # 2/3 with target $2, 1/3 free mode
            try:
                r = simulate_session(tokens, fg, config, params['policy'],
                                     target_usd=target, seed=hash((cfg_idx, i)) % 2**31)
                rec = session_to_record(r, cfg_name, i)
                rec['target_usd'] = target
                wild_mode_learner.append_session(rec)
                total_written += 1
            except Exception as e:
                continue

    elapsed = time.time() - t0
    print(f"\nDone. Wrote {total_written} sessions in {elapsed:.0f}s")

    # Compute insights
    print(f"\nComputing insights...")
    insights = wild_mode_learner.compute_insights()
    print(json.dumps(insights, indent=2, default=str))

    print(f"\n--- LLM CONTEXT (preview) ---")
    print(wild_mode_learner.generate_llm_context())

    # Print winning config recommendation
    best = top_configs[0]
    print(f"\n{'='*60}")
    print(f"WINNING CONFIG (apply to martingale_engine.py):")
    print(f"{'='*60}")
    p = best['params']
    print(f"  MAX_LEVEL_MULTIPLIER     = {p['max_multiplier']}")
    print(f"  MAX_LEVELS_PER_SYMBOL    = {p['max_levels']}")
    print(f"  MAX_CHAIN_TOTAL_RATIO    = {p['max_chain_ratio']}")
    print(f"  MAX_TOTAL_MARGIN_PCT     = {p['max_global_margin_pct']}")
    print(f"  MAX_DRAWDOWN_FROM_START_PCT = {p['max_drawdown_pct']}")
    print(f"  Recommended policy:       {p['policy']} (heuristic prior)")
    print(f"  Expected sharpe:          {best['sharpe']:.3f}")
    print(f"  Expected win rate:        {best['win_rate']*100:.1f}%")
    print(f"  Expected avg PnL:         ${best['avg_pnl']:+.2f}")
    print(f"  Expected max drawdown:    {best['max_drawdown']*100:.1f}%")


if __name__ == '__main__':
    main()
