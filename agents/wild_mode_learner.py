"""
wild_mode_learner.py — Auto-aprendizaje específico del Modo Salvaje.

Recolecta el resultado de cada sesión de wild mode en wild_mode_knowledge.jsonl
(append-only) y produce insights condensados que se inyectan al prompt del LLM
en futuras sesiones para que el sistema mejore automáticamente con cada uso.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict

log = logging.getLogger('wild_mode_learner')

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
KNOWLEDGE_FILE = DATA / 'wild_mode_knowledge.jsonl'
INSIGHTS_FILE = DATA / 'wild_mode_insights.json'

# ── Append session ───────────────────────────────────────────────────────────
def append_session(record: dict) -> None:
    """Append una sesión completa al knowledge file (jsonl)."""
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_FILE, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    except Exception as e:
        log.error(f'append_session error: {e}')

# ── Load sessions ────────────────────────────────────────────────────────────
def load_recent_sessions(n: int = 50) -> list[dict]:
    if not KNOWLEDGE_FILE.exists():
        return []
    try:
        with open(KNOWLEDGE_FILE) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        sessions = []
        for ln in lines[-n:]:
            try:
                sessions.append(json.loads(ln))
            except Exception:
                continue
        return sessions
    except Exception as e:
        log.error(f'load_recent_sessions error: {e}')
        return []

def load_all_sessions() -> list[dict]:
    return load_recent_sessions(n=10**9)

# ── Compute insights ─────────────────────────────────────────────────────────
def compute_insights() -> dict:
    sessions = load_all_sessions()
    if not sessions:
        return {'total_sessions': 0, 'win_rate': 0, 'updated_at': datetime.now(timezone.utc).isoformat()}

    total = len(sessions)
    wins = sum(1 for s in sessions if float(s.get('realized_pnl', 0)) > 0)
    losses = sum(1 for s in sessions if float(s.get('realized_pnl', 0)) < 0)
    flat = total - wins - losses
    win_rate = wins / total if total > 0 else 0

    avg_pnl = sum(float(s.get('realized_pnl', 0)) for s in sessions) / total
    avg_duration = sum(float(s.get('duration_minutes', 0)) for s in sessions) / total

    # Outcomes breakdown
    outcomes = defaultdict(int)
    for s in sessions:
        outcomes[s.get('outcome', 'unknown')] += 1

    # Multiplier analysis (winning vs losing sessions)
    win_mults = [float(s.get('avg_multiplier', 1)) for s in sessions if float(s.get('realized_pnl', 0)) > 0]
    loss_mults = [float(s.get('avg_multiplier', 1)) for s in sessions if float(s.get('realized_pnl', 0)) < 0]
    avg_win_mult = sum(win_mults) / len(win_mults) if win_mults else 0
    avg_loss_mult = sum(loss_mults) / len(loss_mults) if loss_mults else 0

    # Levels analysis
    win_levels = [int(s.get('total_levels_opened', 0)) for s in sessions if float(s.get('realized_pnl', 0)) > 0]
    loss_levels = [int(s.get('total_levels_opened', 0)) for s in sessions if float(s.get('realized_pnl', 0)) < 0]
    avg_win_levels = sum(win_levels) / len(win_levels) if win_levels else 0
    avg_loss_levels = sum(loss_levels) / len(loss_levels) if loss_levels else 0

    # F&G analysis: win rate por bucket de F&G
    fg_buckets = {'extreme_fear': [], 'fear': [], 'neutral': [], 'greed': [], 'extreme_greed': []}
    for s in sessions:
        fg = s.get('starting_fg')
        if fg is None:
            continue
        try:
            fg = int(fg)
        except Exception:
            continue
        bucket = 'neutral'
        if fg <= 20:
            bucket = 'extreme_fear'
        elif fg <= 40:
            bucket = 'fear'
        elif fg <= 60:
            bucket = 'neutral'
        elif fg <= 80:
            bucket = 'greed'
        else:
            bucket = 'extreme_greed'
        fg_buckets[bucket].append(float(s.get('realized_pnl', 0)))

    fg_stats = {}
    for k, vals in fg_buckets.items():
        if vals:
            fg_stats[k] = {
                'n': len(vals),
                'win_rate': sum(1 for v in vals if v > 0) / len(vals),
                'avg_pnl': sum(vals) / len(vals),
            }

    insights = {
        'total_sessions': total,
        'wins': wins,
        'losses': losses,
        'flat': flat,
        'win_rate': round(win_rate, 4),
        'avg_pnl_usd': round(avg_pnl, 4),
        'avg_duration_minutes': round(avg_duration, 1),
        'outcomes': dict(outcomes),
        'avg_winning_multiplier': round(avg_win_mult, 2),
        'avg_losing_multiplier': round(avg_loss_mult, 2),
        'avg_winning_levels': round(avg_win_levels, 1),
        'avg_losing_levels': round(avg_loss_levels, 1),
        'fg_buckets': fg_stats,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }

    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(INSIGHTS_FILE, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
    except Exception as e:
        log.warning(f'save insights error: {e}')

    return insights

# ── Generate prompt context for LLM ──────────────────────────────────────────
def generate_llm_context() -> str:
    """Bloque de texto para inyectar en build_decision_prompt.
       Lee insights cacheados, o los computa si no existen."""
    insights = {}
    if INSIGHTS_FILE.exists():
        try:
            with open(INSIGHTS_FILE) as f:
                insights = json.load(f)
        except Exception:
            insights = {}

    if not insights or insights.get('total_sessions', 0) < 3:
        return ''  # not enough data

    lines = ['APRENDIZAJES DE SESIONES PASADAS:']
    lines.append(f"- Total: {insights['total_sessions']} sesiones, win rate {insights['win_rate']*100:.0f}% (avg PnL ${insights['avg_pnl_usd']:+.2f})")

    if insights.get('avg_winning_multiplier', 0) > 0 and insights.get('avg_losing_multiplier', 0) > 0:
        lines.append(f"- Multipliers ganadores ~{insights['avg_winning_multiplier']:.2f}x vs perdedores ~{insights['avg_losing_multiplier']:.2f}x")

    if insights.get('avg_winning_levels', 0) > 0:
        lines.append(f"- Niveles promedio en sesiones ganadoras: {insights['avg_winning_levels']:.1f} vs perdedoras: {insights['avg_losing_levels']:.1f}")

    fg = insights.get('fg_buckets', {})
    for bucket, stats in fg.items():
        if stats.get('n', 0) >= 3:
            lines.append(f"- F&G {bucket}: win rate {stats['win_rate']*100:.0f}% en {stats['n']} sesiones (PnL avg ${stats['avg_pnl']:+.2f})")

    outcomes = insights.get('outcomes', {})
    if outcomes:
        oc = ', '.join(f'{k}={v}' for k, v in outcomes.items())
        lines.append(f"- Outcomes: {oc}")

    return '\n'.join(lines)

# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    insights = compute_insights()
    print(json.dumps(insights, indent=2, default=str))
    print()
    print('--- LLM CONTEXT ---')
    print(generate_llm_context() or '(insuficientes datos)')
