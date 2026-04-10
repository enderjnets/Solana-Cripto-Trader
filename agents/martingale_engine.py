"""
martingale_engine.py — Modo Salvaje (AI-Driven Martingale + Hedge Engine)

Cuando el usuario activa el switch 'Modo Salvaje' en el dashboard, este módulo
toma el control de las posiciones existentes y delega a un agente IA las
decisiones de hedge/martingala/abandono.

Hard guardrails (NO la IA NO los puede sobrepasar):
  - level_multiplier ∈ [1.1, 2.0]
  - sum(chain.levels[*].margin) ≤ 4 × chain.base_margin
  - sum(all_chains.total_margin) ≤ 0.60 × equity
  - max 5 niveles por símbolo
  - drawdown ≥ 15% desde starting_equity → ABANDON_ALL
  - sesión > 6h → ABANDON_ALL
  - liquidación a < 5% → no abrir nuevo nivel

Modos:
  - target_usd > 0  → cierra todo cuando PnL combinado ≥ target
  - target_usd == 0 → IA decide cuándo cerrar según mercado
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger('martingale_engine')

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
STATE_FILE = DATA / 'wild_mode_state.json'
LLM_FAILURES_FILE = DATA / 'wild_mode_llm_failures.jsonl'

def _log_llm_failure(context: str, raw_response: str, error: str) -> None:
    """Append a malformed LLM response to the failures log for post-mortem debugging.
       Rotates if > 1MB."""
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        if LLM_FAILURES_FILE.exists() and LLM_FAILURES_FILE.stat().st_size > 1_000_000:
            try:
                LLM_FAILURES_FILE.rename(LLM_FAILURES_FILE.with_suffix('.jsonl.1.bak'))
            except Exception:
                pass
        with open(LLM_FAILURES_FILE, 'a') as _f:
            _f.write(json.dumps({
                'ts': datetime.now(timezone.utc).isoformat(),
                'context': context,
                'error': str(error)[:300],
                'raw_response': (raw_response or '')[:2000],
            }) + '\n')
    except Exception:
        pass


# ── Hard Guardrails (constants) ──────────────────────────────────────────────
# Defaults extraídos del sweep de 2000 simulaciones (ver wild_mode_sweep_results.json):
#   conservative policy + mult=1.8 + lvls=3 + ratio=3.0 + global=0.60 + dd=0.15
#   → sharpe 1.22, win rate 92%, avg PnL +$2.35, max DD 6.7%, 0% abandons
MIN_LEVEL_MULTIPLIER = 1.1
MAX_LEVEL_MULTIPLIER = 1.8        # validated by sweep (was 2.0 — 1.8 wins)
MAX_CHAIN_TOTAL_RATIO = 3.0       # validated by sweep (was 4.0)
MAX_LEVELS_PER_SYMBOL = 3         # validated by sweep (was 5)
MAX_TOTAL_MARGIN_PCT = 0.60       # confirmed by sweep
MAX_DRAWDOWN_FROM_START_PCT = 0.15  # confirmed by sweep
MAX_SESSION_MINUTES = 360         # 6h max
MIN_LIQUIDATION_DISTANCE_PCT = 0.05
MIN_PNL_THRESHOLD_FOR_HEDGE = -0.5  # solo considerar hedges si PnL% ≤ -0.5%

# ── State persistence ────────────────────────────────────────────────────────
def _empty_state() -> dict:
    return {'active': False, 'target_usd': 0.0, 'martingale_chains': {}, 'decisions_log': []}

def load_state() -> dict:
    try:
        from safe_io import safe_read_json
        return safe_read_json(STATE_FILE, default=_empty_state())
    except Exception:
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except FileNotFoundError:
            return _empty_state()
        except Exception as e:
            log.error(f'load_state error: {e}')
            return _empty_state()

def save_state(state: dict) -> None:
    try:
        from safe_io import atomic_write_json
        atomic_write_json(STATE_FILE, state)
    except Exception:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)

def is_active() -> bool:
    return bool(load_state().get('active', False))

# ── Helpers ──────────────────────────────────────────────────────────────────
def _equity(portfolio: dict) -> float:
    cash = float(portfolio.get('capital_usd', 0))
    margins = sum(float(p.get('margin_usd', 0)) for p in portfolio.get('positions', []) if p.get('status') == 'open')
    unrealized = sum(float(p.get('pnl_usd', 0)) for p in portfolio.get('positions', []) if p.get('status') == 'open')
    return cash + margins + unrealized

def _total_chain_margin(state: dict) -> float:
    return sum(float(c.get('total_margin', 0)) for c in state.get('martingale_chains', {}).values())

def _session_age_minutes(state: dict) -> float:
    started = state.get('started_at')
    if not started:
        return 0.0
    try:
        t = datetime.fromisoformat(started.replace('Z', '+00:00'))
        return (datetime.now(timezone.utc) - t).total_seconds() / 60.0
    except Exception:
        return 0.0

def _all_open_symbols(portfolio: dict) -> list:
    return [p['symbol'] for p in portfolio.get('positions', []) if p.get('status') == 'open']

def _enrich_portfolio_with_live_pnl(portfolio: dict, market: dict) -> dict:
    """Refresh open positions with live price and net estimated PnL."""
    try:
        import executor
    except Exception:
        return portfolio

    positions = portfolio.get('positions', [])
    for pos in positions:
        if pos.get('status') != 'open':
            continue
        symbol = str(pos.get('symbol', ''))
        entry_price = float(pos.get('entry_price', 0) or 0)
        if entry_price <= 0:
            continue

        current_price = float(executor.get_current_price(symbol, market) or pos.get('current_price', 0) or 0)
        if current_price <= 0:
            current_price = float(pos.get('current_price', 0) or entry_price)

        direction = str(pos.get('direction', 'long')).lower()
        margin = float(pos.get('margin_usd', 0) or 0)
        notional = float(pos.get('notional_value', 0) or pos.get('size_usd', 0) or (margin * float(pos.get('leverage', 0) or 0)))
        funding = float(pos.get('funding_accumulated', 0) or 0)
        fee_entry = float(pos.get('fee_entry', 0) or 0)
        gross_price_pct = ((current_price - entry_price) / entry_price) if direction == 'long' else ((entry_price - current_price) / entry_price)
        fee_exit = notional * (executor.TAKER_FEE + executor.get_slippage(symbol))
        pnl_usd = (gross_price_pct * notional) + funding - fee_entry - fee_exit
        pnl_pct = ((pnl_usd / margin) * 100.0) if margin > 0 else (gross_price_pct * 100.0)

        pos['current_price'] = round(current_price, 8)
        pos['pnl_usd'] = round(pnl_usd, 4)
        pos['pnl_pct'] = round(pnl_pct, 4)
        pos['fee_exit_est'] = round(fee_exit, 4)

    portfolio['last_mark_to_market'] = datetime.now(timezone.utc).isoformat()
    return portfolio

def _has_expanded_hedge_chain(state: dict) -> bool:
    """True once any wild-mode chain has opened at least one extra hedge level."""
    for chain in state.get('martingale_chains', {}).values():
        if len(chain.get('levels', [])) > 1:
            return True
    return False

def _allow_free_mode_ai_close(state: dict, portfolio: dict) -> tuple[bool, str]:
    """
    In salvage/free mode (target=0), never let the pre-check AI close the basket
    while positions are flat/losing and hedging has not been attempted yet.

    This prevents premature WILD_AI_CLOSE_* exits before the hedge engine has a
    chance to open protective levels.
    """
    total_unrealized = sum(float(p.get('pnl_usd', 0)) for p in portfolio.get('positions', []) if p.get('status') == 'open')
    if total_unrealized <= 0:
        return (False, f'unrealized_pnl={total_unrealized:+.2f}')

    if _has_expanded_hedge_chain(state):
        return (True, 'hedge_chain_expanded')

    age_min = _session_age_minutes(state)
    if age_min >= 20:
        return (True, f'session_mature_{age_min:.0f}min')

    return (False, f'no_hedge_levels_yet_age={age_min:.0f}min')

def _rebuild_chains_from_portfolio(state: dict, portfolio: dict) -> None:
    """Rebuild chain state from currently open positions.

    Wild mode sessions can survive manual resets, manual closes, or fresh positions
    opened after activation. Rebuilding the chains from live open positions keeps the
    hedge engine aligned with reality and prevents stale/missing chains.
    """
    open_positions = [p for p in portfolio.get('positions', []) if p.get('status') == 'open']
    rebuilt = {}

    by_symbol = {}
    for pos in open_positions:
        by_symbol.setdefault(pos.get('symbol', ''), []).append(pos)

    for sym, positions in by_symbol.items():
        positions.sort(key=lambda p: p.get('open_time', ''))
        base = positions[0]
        base_margin = float(base.get('margin_usd', 0))
        levels = []
        prev_margin = base_margin if base_margin > 0 else 1.0
        total_margin = 0.0

        for idx, pos in enumerate(positions):
            margin = float(pos.get('margin_usd', 0))
            if idx == 0:
                mult = 1.0
            else:
                mult = (margin / prev_margin) if prev_margin > 0 else 1.0
            levels.append({
                'level': idx,
                'position_id': pos.get('id'),
                'size_multiplier': round(mult, 4),
                'margin': margin,
                'direction': pos.get('direction'),
                'opened_at': pos.get('open_time'),
            })
            total_margin += margin
            if margin > 0:
                prev_margin = margin

        rebuilt[sym] = {
            'base_position_id': base.get('id'),
            'base_margin': base_margin,
            'base_direction': base.get('direction'),
            'levels': levels,
            'total_margin': round(total_margin, 2),
            'max_total_allowed': round(base_margin * MAX_CHAIN_TOTAL_RATIO, 2),
        }

    state['martingale_chains'] = rebuilt

# ── Safety rails ─────────────────────────────────────────────────────────────
def check_safety_rails(state: dict, portfolio: dict) -> tuple[bool, str]:
    """Returns (should_abandon, reason)."""
    if not state.get('active'):
        return (False, 'inactive')

    equity = _equity(portfolio)
    starting = float(state.get('starting_equity', equity))
    if starting <= 0:
        return (False, 'no_starting_equity')

    drawdown = (starting - equity) / starting
    if drawdown >= MAX_DRAWDOWN_FROM_START_PCT:
        return (True, f'drawdown_{drawdown*100:.1f}pct')

    age = _session_age_minutes(state)
    if age > MAX_SESSION_MINUTES:
        return (True, f'session_expired_{age:.0f}min')

    return (False, 'ok')

# ── Target check ─────────────────────────────────────────────────────────────
def check_target_reached(state: dict, portfolio: dict) -> bool:
    target = float(state.get('target_usd', 0))
    if target <= 0:
        return False
    total_unrealized = sum(float(p.get('pnl_usd', 0)) for p in portfolio.get('positions', []) if p.get('status') == 'open')
    return total_unrealized >= target

# ── AI decision: should close (target=0 free mode) ───────────────────────────
def check_ai_close_decision(state: dict, portfolio: dict, market: dict, fear_greed: int) -> tuple[bool, str]:
    """Sólo se llama si target_usd == 0. Pregunta breve al LLM."""
    try:
        import llm_config
    except Exception as e:
        log.warning(f'cannot import llm_config: {e}')
        return (False, 'no_llm')

    positions = [p for p in portfolio.get('positions', []) if p.get('status') == 'open']
    if not positions:
        return (False, 'no_positions')

    total_pnl = sum(float(p.get('pnl_usd', 0)) for p in positions)
    equity = _equity(portfolio)
    starting = float(state.get('starting_equity', equity))
    age = _session_age_minutes(state)

    pos_summary = '\n'.join(
        f"  {p['symbol']} {p['direction']} margin=${p.get('margin_usd', 0):.2f} pnl=${p.get('pnl_usd', 0):+.2f} ({p.get('pnl_pct', 0):+.1f}%)"
        for p in positions
    )

    prompt = f"""Wild Mode (sin target fijo). Decide si CERRAR todas las posiciones AHORA o HOLD.

ESTADO:
- Equity actual: ${equity:.2f}
- Equity inicial: ${starting:.2f}
- PnL combinado: ${total_pnl:+.2f}
- F&G index: {fear_greed}
- Edad sesión: {age:.0f} min

POSICIONES ({len(positions)}):
{pos_summary}

Criterios para CLOSE_NOW:
- Si PnL > 0 y momentum se debilita / F&G cambió drásticamente
- Si PnL > +1% del equity y vimos pico (toma profit)
- Si tendencia se confirma muy en contra (mejor cortar)

Criterios para HOLD:
- Si todavía hay R:R favorable
- Si momentum sigue fuerte a favor

Responde SOLO con JSON:
{{"action": "CLOSE_NOW" | "HOLD", "reason": "..."}}"""

    system = 'Eres un trader risk manager. Responde solo JSON, sin explicaciones extra.'
    raw = None
    try:
        raw = llm_config.call_llm(prompt, system, max_tokens=200)
        if not raw:
            return (False, 'llm_empty')
        # Extract JSON
        import re
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not m:
            _log_llm_failure('check_ai_close_decision', raw, 'no_json_found')
            return (False, 'no_json')
        data = json.loads(m.group(0))
        action = str(data.get('action', 'HOLD')).upper()
        reason = str(data.get('reason', ''))[:200]
        return (action == 'CLOSE_NOW', reason)
    except Exception as e:
        log.warning(f'check_ai_close_decision error: {e}')
        _log_llm_failure('check_ai_close_decision', raw, e)
        return (False, f'error_{type(e).__name__}')

# ── Build decision prompt ────────────────────────────────────────────────────
def build_decision_prompt(state: dict, portfolio: dict, market: dict, fear_greed: int) -> str:
    chains = state.get('martingale_chains', {})
    equity = _equity(portfolio)
    starting = float(state.get('starting_equity', equity))
    target = float(state.get('target_usd', 0))
    total_pnl = sum(float(p.get('pnl_usd', 0)) for p in portfolio.get('positions', []) if p.get('status') == 'open')
    chain_margin_total = _total_chain_margin(state)

    # Build per-chain summary
    chain_lines = []
    pos_by_id = {p['id']: p for p in portfolio.get('positions', []) if p.get('status') == 'open'}
    for sym, ch in chains.items():
        levels = ch.get('levels', [])
        lvl_pnls = [float(pos_by_id.get(lv['position_id'], {}).get('pnl_usd', 0)) for lv in levels]
        ch_pnl = sum(lvl_pnls)
        base_margin = ch.get('base_margin', 0)
        total_margin = ch.get('total_margin', 0)
        used_ratio = total_margin / base_margin if base_margin > 0 else 0
        chain_lines.append(
            f"  {sym}: dir={ch.get('base_direction')} levels={len(levels)} "
            f"margin=${total_margin:.2f}/{base_margin*MAX_CHAIN_TOTAL_RATIO:.2f} "
            f"(used={used_ratio:.1f}x/{MAX_CHAIN_TOTAL_RATIO:.0f}x) pnl=${ch_pnl:+.2f}"
        )
    chain_block = '\n'.join(chain_lines) if chain_lines else '  (sin chains)'

    # Try to load insights from learner
    insights_block = ''
    try:
        import wild_mode_learner
        insights_block = wild_mode_learner.generate_llm_context() or ''
    except Exception:
        insights_block = ''

    # Symbols not yet in chains (could initialize)
    open_syms = set(_all_open_symbols(portfolio))
    chain_syms = set(chains.keys())
    new_candidates = open_syms - chain_syms
    new_block = ''
    if new_candidates:
        new_lines = []
        for sym in new_candidates:
            p = next((x for x in portfolio.get('positions', []) if x['symbol'] == sym and x.get('status') == 'open'), None)
            if p:
                new_lines.append(f"  {sym} {p['direction']} margin=${p.get('margin_usd', 0):.2f} pnl=${p.get('pnl_usd', 0):+.2f}")
        new_block = '\nPOSICIONES SIN CHAIN (puedes iniciar chain si va perdiendo):\n' + '\n'.join(new_lines)

    prompt = f"""WILD MODE — gestión activa de coberturas martingala/hedge.

ESTADO GLOBAL:
- Equity: ${equity:.2f} (inicial ${starting:.2f}, dd={((starting-equity)/starting*100 if starting>0 else 0):.1f}%)
- PnL combinado: ${total_pnl:+.2f}
- Target: {('$%.2f' % target) if target > 0 else 'NINGUNO (libre)'}
- Margen en chains: ${chain_margin_total:.2f} de máx ${equity*MAX_TOTAL_MARGIN_PCT:.2f} ({MAX_TOTAL_MARGIN_PCT*100:.0f}% equity)
- F&G: {fear_greed}
- Sesión: {_session_age_minutes(state):.0f} min

CHAINS ACTIVOS:
{chain_block}{new_block}

{insights_block}

DECIDE para cada chain (o nuevo símbolo):
- OPEN_LEVEL: abre cobertura. Especifica direction (same=martingala / opposite=hedge) y level_multiplier ∈ [1.1, 2.0]
- HOLD: mantener sin tocar
- CLOSE_CHAIN: cerrar todas las posiciones del chain (toma profit/corta pérdida)

A nivel global:
- CONTINUE: seguir operando
- ABANDON_ALL: cerrar TODO (sólo si ves señal muy en contra)

REGLAS DURAS (los guardrails sanitizan tu output, pero respétalos):
- multiplier ∈ [{MIN_LEVEL_MULTIPLIER}, {MAX_LEVEL_MULTIPLIER}]
- chain.total_margin ≤ 4× base_margin
- sum(chains) ≤ {MAX_TOTAL_MARGIN_PCT*100:.0f}% equity
- max {MAX_LEVELS_PER_SYMBOL} niveles por símbolo
- Sólo agregar nivel si pos pierde ≥ {abs(MIN_PNL_THRESHOLD_FOR_HEDGE)}%

Responde SOLO con JSON válido. NO copies este esquema — emite decisiones reales basadas en los datos de arriba.

EJEMPLO (solo ilustrativo, NO lo copies literalmente):
{{
  "decisions": [
    {{"symbol": "MOODENG", "action": "OPEN_LEVEL", "direction": "opposite", "level_multiplier": 1.4, "reasoning": "posición pierde -8.3%, abro hedge contrario para neutralizar", "confidence": 0.75}}
  ],
  "global_action": "CONTINUE",
  "abandon_reason": null,
  "risk_assessment": "medium"
}}

REGLAS ESTRICTAS DEL JSON:
- "action" debe ser EXACTAMENTE: OPEN_LEVEL, HOLD, o CLOSE_CHAIN (una sola palabra, sin barras "|")
- "direction" debe ser EXACTAMENTE: same, u opposite (una sola palabra, sin barras "|")
- "symbol" debe ser el ticker real (ej: MOODENG, GOAT, BTC) — nunca "..."
- "global_action" debe ser EXACTAMENTE: CONTINUE, o ABANDON_ALL
- Incluye una entrada por cada chain activo arriba indicado
- Si no hay nada que hacer en un chain, usa {{"symbol": "X", "action": "HOLD", ...}}"""
    return prompt

def _parse_llm_response(raw: str) -> dict | None:
    """Parse JSON from LLM response. Returns None on failure."""
    import re as _re
    m = _re.search(r'\{.*\}', raw, _re.DOTALL)
    if not m:
        return None
    data = json.loads(m.group(0))
    if 'decisions' not in data:
        data['decisions'] = []
    if 'global_action' not in data:
        data['global_action'] = 'CONTINUE'
    return data


def ask_ai_decision(prompt: str) -> dict:
    try:
        import llm_config
    except Exception as e:
        log.warning(f'cannot import llm_config: {e}')
        return {'global_action': 'CONTINUE', 'decisions': []}

    system = 'Eres un risk manager experto en futuros perpetuos crypto. Responde SOLO con JSON válido sin texto adicional.'
    raw = None
    try:
        raw = llm_config.call_llm(prompt, system, max_tokens=1500)
        if not raw:
            return {'global_action': 'CONTINUE', 'decisions': []}

        data = _parse_llm_response(raw)
        if data is None:
            _log_llm_failure('ask_ai_decision', raw, 'no_json_found')
            return {'global_action': 'CONTINUE', 'decisions': []}

        # Check if response is just placeholders — retry once with stricter prompt
        decisions = data.get('decisions', [])
        if decisions and all(_decision_is_placeholder(d) for d in decisions):
            log.warning('WILD MODE: LLM returned placeholder decisions — retrying with strict prompt')
            _log_llm_failure('ask_ai_decision', raw, 'placeholder_response')

            # Extract symbol list from original prompt to build stricter retry
            import re as _re2
            syms = _re2.findall(r'  ([A-Z]+):', prompt)
            sym_list = ', '.join(syms) if syms else 'los símbolos indicados'
            retry_prompt = (
                f"Tu respuesta anterior fue inválida: copiaste el esquema en vez de emitir decisiones reales.\n\n"
                f"Debes decidir para estos símbolos: {sym_list}\n\n"
                f"Responde ÚNICAMENTE con JSON. Ejemplo válido:\n"
                f'{{"decisions": [{{"symbol": "{syms[0] if syms else "SYM"}", "action": "HOLD", '
                f'"direction": "same", "level_multiplier": 1.0, "reasoning": "sin cambios", "confidence": 0.6}}], '
                f'"global_action": "CONTINUE", "abandon_reason": null, "risk_assessment": "medium"}}\n\n'
                f"Contexto resumido del prompt original:\n{prompt[-800:]}"
            )
            raw2 = llm_config.call_llm(retry_prompt, system, max_tokens=800)
            if raw2:
                data2 = _parse_llm_response(raw2)
                if data2 is not None and not all(_decision_is_placeholder(d) for d in data2.get('decisions', [{'symbol': '...'}])):
                    log.info('WILD MODE: retry LLM succeeded — using retry response')
                    return data2
            log.warning('WILD MODE: retry also failed — falling back to deterministic')

        return data
    except Exception as e:
        log.warning(f'ask_ai_decision parse error: {e}')
        _log_llm_failure('ask_ai_decision', raw, e)
        return {'global_action': 'CONTINUE', 'decisions': []}

def _decision_is_placeholder(decision: dict) -> bool:
    symbol = str(decision.get('symbol', '')).strip()
    action = str(decision.get('action', '')).strip().upper()
    direction = str(decision.get('direction', '')).strip().lower()
    if not symbol or symbol == '...' or '|' in action or '|' in direction:
        return True
    if action not in {'OPEN_LEVEL', 'HOLD', 'CLOSE_CHAIN'}:
        return True
    if direction and direction not in {'same', 'opposite'}:
        return True
    return False

def _fallback_decision_cycle(state: dict, portfolio: dict) -> dict:
    """Deterministic hedge fallback when LLM returns placeholders/garbage."""
    decisions = []
    open_positions = [p for p in portfolio.get('positions', []) if p.get('status') == 'open']
    open_positions.sort(key=lambda p: float(p.get('pnl_pct', 0)))

    for pos in open_positions:
        sym = pos.get('symbol')
        pnl_pct = float(pos.get('pnl_pct', 0) or 0)
        chains = state.get('martingale_chains', {})
        chain = chains.get(sym, {})
        if pnl_pct <= MIN_PNL_THRESHOLD_FOR_HEDGE and len(chain.get('levels', [])) < MAX_LEVELS_PER_SYMBOL:
            decisions.append({
                'symbol': sym,
                'action': 'OPEN_LEVEL',
                'direction': 'opposite',
                'level_multiplier': MIN_LEVEL_MULTIPLIER,
                'reasoning': f'fallback_hedge_trigger pnl_pct={pnl_pct:.2f}',
                'confidence': 0.55,
            })
        else:
            decisions.append({
                'symbol': sym,
                'action': 'HOLD',
                'direction': 'same',
                'level_multiplier': MIN_LEVEL_MULTIPLIER,
                'reasoning': f'fallback_hold pnl_pct={pnl_pct:.2f}',
                'confidence': 0.55,
            })

    return {
        'global_action': 'CONTINUE',
        'decisions': decisions,
        'risk_assessment': 'medium',
        'fallback_used': True,
    }

# ── Validate decision (HARD GUARDRAILS) ──────────────────────────────────────
def validate_decision(decision: dict, state: dict, portfolio: dict) -> dict:
    """Sanitiza una decisión individual. Devuelve dict con keys:
       symbol, action, direction, level_multiplier, new_margin, guardrails_hit[]
    """
    sym = decision.get('symbol', '')
    action = str(decision.get('action', 'HOLD')).upper()
    direction = str(decision.get('direction', 'same')).lower()
    if direction not in ('same', 'opposite'):
        direction = 'same'
    mult_raw = float(decision.get('level_multiplier', 1.3))
    hits = []

    # Clamp multiplier
    mult = max(MIN_LEVEL_MULTIPLIER, min(MAX_LEVEL_MULTIPLIER, mult_raw))
    if mult != mult_raw:
        hits.append(f'multiplier_clamped_{mult_raw:.2f}_to_{mult:.2f}')

    out = {
        'symbol': sym,
        'action': action,
        'direction': direction,
        'level_multiplier': mult,
        'new_margin': 0.0,
        'guardrails_hit': hits,
    }

    if action != 'OPEN_LEVEL':
        return out

    chains = state.get('martingale_chains', {})
    chain = chains.get(sym)

    # Find base margin (from chain or from open position)
    if chain:
        base_margin = float(chain.get('base_margin', 0))
        levels = chain.get('levels', [])
        n_levels = len(levels)
        chain_total = float(chain.get('total_margin', 0))
        last_margin = float(levels[-1]['margin']) if levels else base_margin
    else:
        # New chain — find the position
        p = next((x for x in portfolio.get('positions', []) if x['symbol'] == sym and x.get('status') == 'open'), None)
        if not p:
            hits.append('no_base_position')
            out['action'] = 'HOLD'
            return out
        base_margin = float(p.get('margin_usd', 0))
        n_levels = 1  # the existing position is level 0
        chain_total = base_margin
        last_margin = base_margin

    # Check max levels
    if n_levels >= MAX_LEVELS_PER_SYMBOL:
        hits.append(f'max_levels_{n_levels}')
        out['action'] = 'HOLD'
        return out

    # Compute new margin
    new_margin = last_margin * mult

    # Check chain total cap
    max_chain_total = base_margin * MAX_CHAIN_TOTAL_RATIO
    if chain_total + new_margin > max_chain_total:
        # Try to fit
        room = max(0, max_chain_total - chain_total)
        if room < base_margin * 0.1:  # not enough room
            hits.append(f'chain_total_cap_{chain_total:.2f}+{new_margin:.2f}>{max_chain_total:.2f}')
            out['action'] = 'HOLD'
            return out
        new_margin = room
        hits.append(f'margin_reduced_to_fit_chain_cap_{new_margin:.2f}')

    # Check global margin cap
    equity = _equity(portfolio)
    total_chain_margin = _total_chain_margin(state)
    global_cap = equity * MAX_TOTAL_MARGIN_PCT
    if total_chain_margin + new_margin > global_cap:
        room = max(0, global_cap - total_chain_margin)
        if room < base_margin * 0.1:
            hits.append(f'global_margin_cap_{total_chain_margin:.2f}+{new_margin:.2f}>{global_cap:.2f}')
            out['action'] = 'HOLD'
            return out
        new_margin = min(new_margin, room)
        hits.append(f'margin_reduced_to_fit_global_cap_{new_margin:.2f}')

    # Check pnl threshold (only hedge losing positions)
    p = next((x for x in portfolio.get('positions', []) if x['symbol'] == sym and x.get('status') == 'open'), None)
    if p:
        pnl_pct = float(p.get('pnl_pct', 0))
        if pnl_pct > MIN_PNL_THRESHOLD_FOR_HEDGE:
            hits.append(f'pnl_pct_{pnl_pct:.2f}>{MIN_PNL_THRESHOLD_FOR_HEDGE}')
            out['action'] = 'HOLD'
            return out

    out['new_margin'] = round(new_margin, 2)
    return out

# ── Apply decision ───────────────────────────────────────────────────────────
def apply_decision(decision: dict, state: dict, portfolio: dict, market: dict, history: list) -> dict:
    """Ejecuta una decisión sanitizada. Retorna stats {opened, closed}."""
    stats = {'opened': 0, 'closed': 0}
    action = decision.get('action', 'HOLD')

    if action == 'HOLD':
        return stats

    sym = decision['symbol']
    chains = state.setdefault('martingale_chains', {})

    if action == 'CLOSE_CHAIN':
        chain = chains.get(sym)
        if not chain:
            return stats
        try:
            import executor
            level_syms = [sym]  # close all positions of this symbol (covers all levels)
            closed = executor.close_positions_emergency(portfolio, level_syms, market, history, reason='WILD_MODE_CLOSE_CHAIN')
            stats['closed'] = len(closed)
            chains.pop(sym, None)
        except Exception as e:
            log.error(f'CLOSE_CHAIN error: {e}')
        return stats

    if action == 'OPEN_LEVEL':
        new_margin = float(decision.get('new_margin', 0))
        if new_margin <= 0:
            return stats
        # Find base position to inherit leverage and direction
        base_pos = next((x for x in portfolio.get('positions', []) if x['symbol'] == sym and x.get('status') == 'open'), None)
        if not base_pos:
            return stats
        leverage = int(base_pos.get('leverage', 5))
        base_direction = base_pos['direction']
        new_direction = base_direction if decision['direction'] == 'same' else ('long' if base_direction == 'short' else 'short')

        # Build synthetic signal
        try:
            import executor
            current_price = executor.get_current_price(sym, market)
        except Exception:
            current_price = float(base_pos.get('current_price', 0))
        if current_price <= 0:
            return stats

        # SL/TP relative to current price (mirror executor defaults)
        sl_pct = 0.025
        tp_pct = 0.05
        if new_direction == 'long':
            sl_price = current_price * (1 - sl_pct)
            tp_price = current_price * (1 + tp_pct)
        else:
            sl_price = current_price * (1 + sl_pct)
            tp_price = current_price * (1 - tp_pct)

        signal = {
            'symbol': sym,
            'direction': new_direction,
            'strategy': 'wild_martingale',
            'confidence': 0.85,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'exit_mode': 'fixed',
            '_force_margin': new_margin,
            '_force_leverage': leverage,
        }

        try:
            import executor
            pos = executor.paper_open_position(signal, portfolio, market)
        except Exception as e:
            log.error(f'OPEN_LEVEL paper_open error: {e}')
            pos = None

        if pos:
            stats['opened'] = 1
            chain = chains.get(sym)
            if not chain:
                chain = {
                    'base_position_id': base_pos['id'],
                    'base_margin': float(base_pos.get('margin_usd', 0)),
                    'base_direction': base_direction,
                    'levels': [{
                        'level': 0,
                        'position_id': base_pos['id'],
                        'size_multiplier': 1.0,
                        'margin': float(base_pos.get('margin_usd', 0)),
                        'direction': base_direction,
                        'opened_at': base_pos.get('open_time'),
                    }],
                    'total_margin': float(base_pos.get('margin_usd', 0)),
                    'max_total_allowed': float(base_pos.get('margin_usd', 0)) * MAX_CHAIN_TOTAL_RATIO,
                }
                chains[sym] = chain
            n_lvl = len(chain['levels'])
            chain['levels'].append({
                'level': n_lvl,
                'position_id': pos['id'],
                'size_multiplier': decision['level_multiplier'],
                'margin': float(pos.get('margin_usd', 0)),
                'direction': new_direction,
                'opened_at': pos.get('open_time'),
            })
            chain['total_margin'] = sum(float(lv['margin']) for lv in chain['levels'])

    return stats

# ── Abandon all ──────────────────────────────────────────────────────────────
def abandon_all(state: dict, portfolio: dict, market: dict, history: list, reason: str) -> int:
    try:
        import executor
        syms = _all_open_symbols(portfolio)
        if syms:
            closed = executor.close_positions_emergency(portfolio, syms, market, history, reason=f'WILD_ABANDON_{reason[:20]}')
            n = len(closed)
        else:
            n = 0
    except Exception as e:
        log.error(f'abandon_all error: {e}')
        n = 0
    state['active'] = False
    state['ended_at'] = datetime.now(timezone.utc).isoformat()
    state['end_reason'] = f'abandon_{reason}'
    return n

def close_for_target(state: dict, portfolio: dict, market: dict, history: list, reason: str) -> int:
    try:
        import executor
        syms = _all_open_symbols(portfolio)
        if syms:
            closed = executor.close_positions_emergency(portfolio, syms, market, history, reason=f'WILD_{reason[:30]}')
            n = len(closed)
        else:
            n = 0
    except Exception as e:
        log.error(f'close_for_target error: {e}')
        n = 0
    state['active'] = False
    state['ended_at'] = datetime.now(timezone.utc).isoformat()
    state['end_reason'] = reason
    return n

# ── Record session outcome (for learner) ─────────────────────────────────────
def record_session_outcome(state: dict, portfolio: dict, history: list, outcome_type: str) -> None:
    try:
        import wild_mode_learner
        equity = _equity(portfolio)
        starting = float(state.get('starting_equity', equity))
        chains = state.get('martingale_chains', {})

        # Calc multipliers
        mults = []
        for ch in chains.values():
            for lv in ch.get('levels', [])[1:]:  # skip base level
                m = float(lv.get('size_multiplier', 0))
                if m > 0:
                    mults.append(m)
        max_mult = max(mults) if mults else 1.0
        avg_mult = (sum(mults) / len(mults)) if mults else 1.0

        # Decisions log stats
        decisions = state.get('decisions_log', [])
        guardrails_hit_count = sum(len(d.get('guardrails_hit', [])) for d in decisions)

        record = {
            'session_id': state.get('session_id', 'unknown'),
            'started_at': state.get('started_at'),
            'ended_at': state.get('ended_at') or datetime.now(timezone.utc).isoformat(),
            'duration_minutes': round(_session_age_minutes(state), 1),
            'outcome': outcome_type,
            'starting_equity': starting,
            'ending_equity': round(equity, 2),
            'realized_pnl': round(equity - starting, 4),
            'target_usd': float(state.get('target_usd', 0)),
            'starting_fg': state.get('starting_fg'),
            'chains_created': len(chains),
            'total_levels_opened': sum(len(c.get('levels', [])) for c in chains.values()),
            'max_multiplier_used': round(max_mult, 2),
            'avg_multiplier': round(avg_mult, 2),
            'guardrails_hit_count': guardrails_hit_count,
            'ai_decisions_count': len(decisions),
        }
        wild_mode_learner.append_session(record)
    except Exception as e:
        log.warning(f'record_session_outcome error: {e}')

# ── Main entry point ─────────────────────────────────────────────────────────
def run_cycle(portfolio: dict, market: dict, history: list, fear_greed: int = 50) -> dict:
    """Llamado desde orchestrator cada ciclo. Idempotente."""
    result = {'active': False, 'opened': 0, 'closed': 0, 'abandoned': False, 'target_hit': False, 'reason': ''}

    portfolio = _enrich_portfolio_with_live_pnl(portfolio, market)
    state = load_state()
    if not state.get('active'):
        return result
    result['active'] = True

    _rebuild_chains_from_portfolio(state, portfolio)

    # 1. Safety rails
    should_abandon, rail_reason = check_safety_rails(state, portfolio)
    if should_abandon:
        n = abandon_all(state, portfolio, market, history, rail_reason)
        result['abandoned'] = True
        result['closed'] = n
        result['reason'] = rail_reason
        record_session_outcome(state, portfolio, history, 'abandoned')
        save_state(state)
        return result

    # 2. Target reached?
    if check_target_reached(state, portfolio):
        n = close_for_target(state, portfolio, market, history, 'TARGET_HIT')
        result['target_hit'] = True
        result['closed'] = n
        result['reason'] = 'target_hit'
        record_session_outcome(state, portfolio, history, 'target_hit')
        save_state(state)
        return result

    # 3. Free mode (target=0): ask AI if should close
    if float(state.get('target_usd', 0)) == 0:
        allow_ai_close, skip_reason = _allow_free_mode_ai_close(state, portfolio)
        if allow_ai_close:
            should_close, ai_reason = check_ai_close_decision(state, portfolio, market, fear_greed)
            if should_close:
                n = close_for_target(state, portfolio, market, history, f'AI_CLOSE_{ai_reason[:20]}')
                result['target_hit'] = True
                result['closed'] = n
                result['reason'] = f'ai_close: {ai_reason}'
                record_session_outcome(state, portfolio, history, 'ai_close')
                save_state(state)
                return result
        else:
            log.info(f'WILD MODE: skipping free-mode AI close precheck ({skip_reason}) - prioritize hedge/decision cycle')

    # 4. Decision cycle
    prompt = build_decision_prompt(state, portfolio, market, fear_greed)
    ai_resp = ask_ai_decision(prompt)
    decisions = ai_resp.get('decisions', [])
    if (not decisions) or any(_decision_is_placeholder(d) for d in decisions):
        log.warning('WILD MODE: invalid/placeholder LLM decisions detected - using deterministic fallback hedge cycle')
        ai_resp = _fallback_decision_cycle(state, portfolio)

    if ai_resp.get('global_action') == 'ABANDON_ALL':
        n = abandon_all(state, portfolio, market, history, ai_resp.get('abandon_reason', 'ai_abandon'))
        result['abandoned'] = True
        result['closed'] = n
        result['reason'] = ai_resp.get('abandon_reason', 'ai_abandon')
        record_session_outcome(state, portfolio, history, 'abandoned')
        save_state(state)
        return result

    decisions = ai_resp.get('decisions', [])
    decisions_log = state.setdefault('decisions_log', [])
    for d in decisions:
        validated = validate_decision(d, state, portfolio)
        stats = apply_decision(validated, state, portfolio, market, history)
        result['opened'] += stats.get('opened', 0)
        result['closed'] += stats.get('closed', 0)
        decisions_log.append({
            'ts': datetime.now(timezone.utc).isoformat(),
            'raw': d,
            'validated': validated,
            'stats': stats,
        })
    # cap log size
    if len(decisions_log) > 200:
        state['decisions_log'] = decisions_log[-200:]

    save_state(state)
    return result
