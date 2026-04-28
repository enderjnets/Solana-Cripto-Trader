"""
simulator.py — Replay determinístico de sesiones de wild mode sobre histórico.

NO llama LLM. Usa policies heurísticas para tomar decisiones de hedge/abandon.
Aplica EXACTAMENTE los mismos guardrails que martingale_engine.validate_decision.

Cada sesión: arranca con N posiciones base aleatorias, avanza candle a candle,
abre/cierra hedges según policy, termina cuando target_hit / abandoned / expired.
"""
from __future__ import annotations
import json
import random
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'data' / 'history'

# ── Hard guardrails (same as martingale_engine) ──────────────────────────────
MIN_LEVEL_MULT = 1.1
MAX_LEVEL_MULT = 2.0
MAX_CHAIN_RATIO = 4.0
MAX_LEVELS = 5
MAX_GLOBAL_MARGIN_PCT = 0.60
MAX_DRAWDOWN_PCT = 0.15
MIN_PNL_FOR_HEDGE = -0.005  # need ≥0.5% loss to consider hedge


# ── Position model ───────────────────────────────────────────────────────────
@dataclass
class SimPosition:
    symbol: str
    direction: str           # 'long' or 'short'
    entry_price: float
    margin: float
    leverage: int
    sl_price: float
    tp_price: float
    open_idx: int            # candle index when opened
    chain_id: str
    level: int
    closed: bool = False
    close_idx: int = -1
    close_price: float = 0.0
    pnl_usd: float = 0.0
    close_reason: str = ''

    @property
    def notional(self) -> float:
        return self.margin * self.leverage

    def update_pnl(self, current_price: float) -> float:
        if self.closed:
            return self.pnl_usd
        if self.direction == 'long':
            pct = (current_price - self.entry_price) / self.entry_price
        else:
            pct = (self.entry_price - current_price) / self.entry_price
        # Approx fees: 0.04% taker round-trip
        fee = self.notional * 0.0008
        self.pnl_usd = self.notional * pct - fee
        return self.pnl_usd

    def check_sl_tp(self, high: float, low: float, idx: int) -> bool:
        """Returns True if hit SL or TP."""
        if self.closed:
            return False
        if self.direction == 'long':
            if low <= self.sl_price:
                self.closed = True
                self.close_idx = idx
                self.close_price = self.sl_price
                self.update_pnl(self.sl_price)
                self.close_reason = 'SL'
                return True
            if high >= self.tp_price:
                self.closed = True
                self.close_idx = idx
                self.close_price = self.tp_price
                self.update_pnl(self.tp_price)
                self.close_reason = 'TP'
                return True
        else:  # short
            if high >= self.sl_price:
                self.closed = True
                self.close_idx = idx
                self.close_price = self.sl_price
                self.update_pnl(self.sl_price)
                self.close_reason = 'SL'
                return True
            if low <= self.tp_price:
                self.closed = True
                self.close_idx = idx
                self.close_price = self.tp_price
                self.update_pnl(self.tp_price)
                self.close_reason = 'TP'
                return True
        return False


@dataclass
class SimChain:
    symbol: str
    base_direction: str
    base_margin: float
    levels: list = field(default_factory=list)  # list[SimPosition]

    @property
    def total_margin(self) -> float:
        return sum(lv.margin for lv in self.levels if not lv.closed)

    @property
    def total_pnl(self) -> float:
        return sum(lv.pnl_usd for lv in self.levels)


@dataclass
class SessionResult:
    outcome: str             # 'target_hit' | 'ai_close' | 'abandoned' | 'expired' | 'all_closed'
    starting_equity: float
    ending_equity: float
    realized_pnl: float
    duration_candles: int
    chains_count: int
    levels_opened: int
    max_multiplier_used: float
    avg_multiplier: float
    starting_fg: int
    ending_fg: int
    max_drawdown_pct: float
    config: dict = field(default_factory=dict)
    policy: str = ''


# ── F&G lookup helper ────────────────────────────────────────────────────────
def fg_at_timestamp(fg_data: list, ts_ms: int) -> int:
    """Return F&G value closest to the given timestamp (ms)."""
    if not fg_data:
        return 50
    ts_sec = ts_ms / 1000
    best = min(fg_data, key=lambda d: abs(d['timestamp'] - ts_sec))
    return int(best['value'])


# ── Heuristic decision policies ──────────────────────────────────────────────
def policy_aggressive(chain: SimChain, current_price: float, fg: int, equity: float, total_chain_margin: float, config: dict) -> dict | None:
    last = chain.levels[-1]
    last.update_pnl(current_price)
    if last.pnl_usd >= 0:
        return None
    pnl_pct = last.pnl_usd / last.notional
    if pnl_pct > -0.01:  # need ≥1% loss
        return None
    if len(chain.levels) >= config.get('max_levels', MAX_LEVELS):
        return None
    return {'symbol': chain.symbol, 'action': 'OPEN_LEVEL', 'multiplier': config.get('max_multiplier', 1.8), 'direction': 'same'}


def policy_balanced(chain: SimChain, current_price: float, fg: int, equity: float, total_chain_margin: float, config: dict) -> dict | None:
    last = chain.levels[-1]
    last.update_pnl(current_price)
    if last.pnl_usd >= 0:
        return None
    pnl_pct = last.pnl_usd / last.notional
    if pnl_pct > -0.015:  # need ≥1.5% loss
        return None
    if len(chain.levels) >= config.get('max_levels', MAX_LEVELS):
        return None
    # In extreme F&G use opposite direction (real hedge)
    direction = 'opposite' if (fg <= 20 or fg >= 80) else 'same'
    mult = min(config.get('max_multiplier', 1.5), 1.4)
    return {'symbol': chain.symbol, 'action': 'OPEN_LEVEL', 'multiplier': mult, 'direction': direction}


def policy_conservative(chain: SimChain, current_price: float, fg: int, equity: float, total_chain_margin: float, config: dict) -> dict | None:
    last = chain.levels[-1]
    last.update_pnl(current_price)
    if last.pnl_usd >= 0:
        return None
    pnl_pct = last.pnl_usd / last.notional
    if pnl_pct > -0.02:  # need ≥2% loss
        return None
    if len(chain.levels) >= config.get('max_levels', MAX_LEVELS) - 1:  # one less than max
        return None
    return {'symbol': chain.symbol, 'action': 'OPEN_LEVEL', 'multiplier': 1.2, 'direction': 'same'}


def policy_ai_proxy(chain: SimChain, current_price: float, fg: int, equity: float, total_chain_margin: float, config: dict) -> dict | None:
    """Simulates what an LLM might do — adapts to context."""
    last = chain.levels[-1]
    last.update_pnl(current_price)
    if last.pnl_usd >= 0:
        return None
    pnl_pct = last.pnl_usd / last.notional
    if pnl_pct > -0.01:
        return None
    if len(chain.levels) >= config.get('max_levels', MAX_LEVELS):
        return None
    # If extreme F&G, more confident in mean reversion → larger martingale
    # If neutral, smaller hedge
    if fg <= 25 and chain.base_direction == 'short':
        mult = 1.6  # high confidence in continuation of fear
        direction = 'same'
    elif fg >= 75 and chain.base_direction == 'long':
        mult = 1.6
        direction = 'same'
    elif pnl_pct < -0.025:
        # Bigger loss → switch to hedge
        mult = 1.4
        direction = 'opposite'
    else:
        mult = 1.3
        direction = 'same'
    mult = min(mult, config.get('max_multiplier', MAX_LEVEL_MULT))
    return {'symbol': chain.symbol, 'action': 'OPEN_LEVEL', 'multiplier': mult, 'direction': direction}


POLICIES = {
    'aggressive': policy_aggressive,
    'balanced': policy_balanced,
    'conservative': policy_conservative,
    'ai_proxy': policy_ai_proxy,
}


# ── Validate decision (mirror of martingale_engine.validate_decision) ────────
def validate_open_level(decision: dict, chain: SimChain, equity: float, total_chain_margin: float, config: dict) -> tuple[float, list]:
    """Returns (new_margin, guardrails_hit). new_margin=0 means rejected."""
    hits = []
    mult = decision['multiplier']
    mult = max(MIN_LEVEL_MULT, min(config.get('max_multiplier', MAX_LEVEL_MULT), mult))
    if mult != decision['multiplier']:
        hits.append('mult_clamped')

    if not chain.levels:
        return (0, hits + ['no_base'])

    base_margin = chain.base_margin
    last_margin = chain.levels[-1].margin
    new_margin = last_margin * mult

    chain_total = chain.total_margin
    max_chain = base_margin * config.get('max_chain_ratio', MAX_CHAIN_RATIO)
    if chain_total + new_margin > max_chain:
        room = max(0, max_chain - chain_total)
        if room < base_margin * 0.1:
            return (0, hits + ['chain_cap'])
        new_margin = room
        hits.append('chain_cap_reduced')

    global_cap = equity * config.get('max_global_margin_pct', MAX_GLOBAL_MARGIN_PCT)
    if total_chain_margin + new_margin > global_cap:
        room = max(0, global_cap - total_chain_margin)
        if room < base_margin * 0.1:
            return (0, hits + ['global_cap'])
        new_margin = min(new_margin, room)
        hits.append('global_cap_reduced')

    return (round(new_margin, 2), hits)


# ── Main simulator ───────────────────────────────────────────────────────────
def simulate_session(
    tokens_data: dict,         # {token: list[candles]}
    fg_data: list,
    config: dict,
    policy_name: str,
    starting_capital: float = 100.0,
    target_usd: float = 0.0,
    max_session_candles: int = 96,  # 4 days
    n_initial_positions: int = 4,
    seed: Optional[int] = None,
) -> SessionResult:
    if seed is not None:
        random.seed(seed)
    policy_fn = POLICIES.get(policy_name, policy_balanced)

    sl_pct = config.get('sl_pct', 0.025)
    tp_pct = config.get('tp_pct', 0.05)
    leverage = config.get('leverage', 5)
    margin_per_pos = config.get('margin_per_pos', 20.0)

    available_tokens = list(tokens_data.keys())
    if not available_tokens:
        return SessionResult('error', 0, 0, 0, 0, 0, 0, 1.0, 1.0, 50, 50, 0, config, policy_name)

    # Pick a random window (60% of length is usable as start)
    min_len = min(len(tokens_data[t]) for t in available_tokens)
    if min_len < max_session_candles + 10:
        return SessionResult('error_short', 0, 0, 0, 0, 0, 0, 1.0, 1.0, 50, 50, 0, config, policy_name)

    start_idx = random.randint(0, min_len - max_session_candles - 1)

    # Initial F&G
    first_ts = tokens_data[available_tokens[0]][start_idx][0]
    starting_fg = fg_at_timestamp(fg_data, first_ts)

    # Open n_initial positions on random tokens, direction biased by F&G
    chains: dict[str, SimChain] = {}
    pick_tokens = random.sample(available_tokens, min(n_initial_positions, len(available_tokens)))
    for tok in pick_tokens:
        candle = tokens_data[tok][start_idx]
        entry_price = candle[4]  # close
        # Direction: F&G < 30 → bias short, > 70 → bias long, else random
        if starting_fg < 30:
            direction = 'short' if random.random() < 0.7 else 'long'
        elif starting_fg > 70:
            direction = 'long' if random.random() < 0.7 else 'short'
        else:
            direction = random.choice(['long', 'short'])
        if direction == 'long':
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)
        pos = SimPosition(
            symbol=tok, direction=direction, entry_price=entry_price,
            margin=margin_per_pos, leverage=leverage,
            sl_price=sl_price, tp_price=tp_price,
            open_idx=start_idx, chain_id=f'chain_{tok}', level=0
        )
        chains[tok] = SimChain(symbol=tok, base_direction=direction, base_margin=margin_per_pos, levels=[pos])

    cash = starting_capital - sum(c.base_margin for c in chains.values())
    if cash < 0:
        return SessionResult('error_capital', starting_capital, starting_capital, 0, 0, len(chains), len(chains), 1.0, 1.0, starting_fg, starting_fg, 0, config, policy_name)

    starting_equity = starting_capital
    max_drawdown = 0.0
    outcome = 'expired'

    # Replay candles
    for i in range(max_session_candles):
        candle_idx = start_idx + i
        # Update prices for all chains
        all_pnl = 0.0
        active_margin = 0.0
        for sym, ch in chains.items():
            if candle_idx >= len(tokens_data[sym]):
                continue
            cd = tokens_data[sym][candle_idx]
            high, low, close = cd[2], cd[3], cd[4]
            for lv in ch.levels:
                if not lv.closed:
                    lv.check_sl_tp(high, low, candle_idx)
                    lv.update_pnl(close)
                    if not lv.closed:
                        active_margin += lv.margin
            all_pnl += ch.total_pnl

        # Equity calc
        # cash already deducted margins; closed positions return margin+pnl to cash
        # Easier: rebuild equity
        realized = sum(lv.pnl_usd + lv.margin for ch in chains.values() for lv in ch.levels if lv.closed)
        unrealized = sum(lv.pnl_usd for ch in chains.values() for lv in ch.levels if not lv.closed)
        in_use_margin = sum(lv.margin for ch in chains.values() for lv in ch.levels if not lv.closed)
        equity = (starting_capital - sum(lv.margin for ch in chains.values() for lv in ch.levels)) + realized + in_use_margin + unrealized
        # Simplify: equity = starting + realized_pnl + unrealized_pnl
        equity = starting_capital + sum(lv.pnl_usd for ch in chains.values() for lv in ch.levels)

        dd = (starting_equity - equity) / starting_equity if starting_equity > 0 else 0
        max_drawdown = max(max_drawdown, dd)

        # Check abandon (drawdown)
        if dd >= config.get('max_drawdown_pct', MAX_DRAWDOWN_PCT):
            outcome = 'abandoned'
            # Close all open positions at current price
            for ch in chains.values():
                for lv in ch.levels:
                    if not lv.closed:
                        cd = tokens_data[lv.symbol][candle_idx] if candle_idx < len(tokens_data[lv.symbol]) else None
                        price = cd[4] if cd else lv.entry_price
                        lv.closed = True
                        lv.close_idx = candle_idx
                        lv.close_price = price
                        lv.update_pnl(price)
                        lv.close_reason = 'ABANDON'
            break

        # Check target
        if target_usd > 0 and unrealized >= target_usd:
            outcome = 'target_hit'
            for ch in chains.values():
                for lv in ch.levels:
                    if not lv.closed:
                        cd = tokens_data[lv.symbol][candle_idx] if candle_idx < len(tokens_data[lv.symbol]) else None
                        price = cd[4] if cd else lv.entry_price
                        lv.closed = True
                        lv.close_idx = candle_idx
                        lv.close_price = price
                        lv.update_pnl(price)
                        lv.close_reason = 'TARGET'
            break

        # Free mode: simulate AI close (heuristic — close if pnl > 1% equity & momentum weakening)
        if target_usd == 0 and unrealized >= starting_equity * 0.01 and i > 5:
            # Simple heuristic: if PnL has been positive last 3 candles but not growing, close
            outcome = 'ai_close'
            for ch in chains.values():
                for lv in ch.levels:
                    if not lv.closed:
                        cd = tokens_data[lv.symbol][candle_idx] if candle_idx < len(tokens_data[lv.symbol]) else None
                        price = cd[4] if cd else lv.entry_price
                        lv.closed = True
                        lv.close_idx = candle_idx
                        lv.close_price = price
                        lv.update_pnl(price)
                        lv.close_reason = 'AI_CLOSE'
            break

        # Apply hedge policy on each chain (only on chains with at least one open level)
        cur_fg = fg_at_timestamp(fg_data, tokens_data[available_tokens[0]][candle_idx][0])
        total_chain_margin = sum(c.total_margin for c in chains.values())
        for ch in chains.values():
            open_levels = [lv for lv in ch.levels if not lv.closed]
            if not open_levels:
                continue
            current_price = tokens_data[ch.symbol][candle_idx][4] if candle_idx < len(tokens_data[ch.symbol]) else open_levels[-1].entry_price
            decision = policy_fn(ch, current_price, cur_fg, equity, total_chain_margin, config)
            if not decision or decision['action'] != 'OPEN_LEVEL':
                continue
            new_margin, hits = validate_open_level(decision, ch, equity, total_chain_margin, config)
            if new_margin <= 0:
                continue
            # Open new level
            base_dir = ch.base_direction
            new_dir = base_dir if decision['direction'] == 'same' else ('long' if base_dir == 'short' else 'short')
            if new_dir == 'long':
                sl_p = current_price * (1 - sl_pct)
                tp_p = current_price * (1 + tp_pct)
            else:
                sl_p = current_price * (1 + sl_pct)
                tp_p = current_price * (1 - tp_pct)
            new_pos = SimPosition(
                symbol=ch.symbol, direction=new_dir, entry_price=current_price,
                margin=new_margin, leverage=leverage,
                sl_price=sl_p, tp_price=tp_p,
                open_idx=candle_idx, chain_id=ch.symbol, level=len(ch.levels)
            )
            new_pos.size_multiplier = decision['multiplier']
            ch.levels.append(new_pos)
            total_chain_margin += new_margin

        # Check if all positions closed naturally
        any_open = any(not lv.closed for ch in chains.values() for lv in ch.levels)
        if not any_open:
            outcome = 'all_closed'
            break

    # Finalize
    final_equity = starting_capital + sum(lv.pnl_usd for ch in chains.values() for lv in ch.levels)
    realized_pnl = final_equity - starting_capital

    # Multipliers used (skipping level 0)
    mults = []
    for ch in chains.values():
        for lv in ch.levels[1:]:
            m = getattr(lv, 'size_multiplier', None)
            if m:
                mults.append(m)
    max_mult = max(mults) if mults else 1.0
    avg_mult = sum(mults) / len(mults) if mults else 1.0

    end_fg = fg_at_timestamp(fg_data, tokens_data[available_tokens[0]][min(start_idx + i, len(tokens_data[available_tokens[0]]) - 1)][0])

    return SessionResult(
        outcome=outcome,
        starting_equity=starting_capital,
        ending_equity=round(final_equity, 4),
        realized_pnl=round(realized_pnl, 4),
        duration_candles=i + 1,
        chains_count=len(chains),
        levels_opened=sum(len(c.levels) for c in chains.values()),
        max_multiplier_used=round(max_mult, 2),
        avg_multiplier=round(avg_mult, 2),
        starting_fg=starting_fg,
        ending_fg=end_fg,
        max_drawdown_pct=round(max_drawdown, 4),
        config=config,
        policy=policy_name,
    )


# ── Loaders ──────────────────────────────────────────────────────────────────
def load_tokens_data() -> dict:
    out = {}
    for f in DATA_DIR.glob('*_1h.json'):
        if 'fear_greed' in f.name:
            continue
        token = f.stem.replace('_1h', '')
        try:
            out[token] = json.loads(f.read_text())
        except Exception:
            continue
    return out


def load_fg_data() -> list:
    f = DATA_DIR / 'fear_greed_120d.json'
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except Exception:
        return []


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tokens = load_tokens_data()
    fg = load_fg_data()
    print(f"Loaded {len(tokens)} tokens, {len(fg)} F&G entries")
    if not tokens:
        print("No data — run fetch_history.py first")
        exit(1)

    config = {
        'sl_pct': 0.025, 'tp_pct': 0.05, 'leverage': 5,
        'max_multiplier': 1.5, 'max_levels': 4,
        'max_chain_ratio': 4.0, 'max_global_margin_pct': 0.6,
        'max_drawdown_pct': 0.15, 'margin_per_pos': 20.0,
    }
    print("\n--- Single session test ---")
    r = simulate_session(tokens, fg, config, 'balanced', target_usd=2.0, seed=42)
    print(json.dumps(asdict(r), indent=2, default=str))

    print("\n--- 50 sessions, balanced policy ---")
    results = [simulate_session(tokens, fg, config, 'balanced', target_usd=2.0, seed=i) for i in range(50)]
    wins = sum(1 for r in results if r.realized_pnl > 0)
    avg_pnl = sum(r.realized_pnl for r in results) / len(results)
    abandons = sum(1 for r in results if r.outcome == 'abandoned')
    print(f"Wins: {wins}/50 ({wins*2}%) | Avg PnL: ${avg_pnl:+.2f} | Abandons: {abandons}")
