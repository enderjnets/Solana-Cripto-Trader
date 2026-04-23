import logging
log = logging.getLogger("dashboard")
#!/usr/bin/env python3
"""
Solana Cripto Trader - Professional Trading Dashboard
Flask backend serving real-time data from bot data files.
Port: 8001
"""

import json
import os
import math
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request, Response, send_from_directory
import uuid

app = Flask(__name__)

# ── Data paths ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA = BASE / "agents" / "data"
RESET_ATTACHMENTS_DIR = DATA / "reset_attachments"
RESET_ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_ATTACH_EXT = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf', 'txt', 'csv'}
WATCHDOG_LOG = Path("/home/enderj/.config/solana-jupiter-bot/modular.live.log")

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        log.error(f'JSON decode error in {path}: {e}')
        return {}
    except Exception as e:
        log.error(f'Error reading {path}: {e}')
        return {}

def load_portfolio():
    raw = load_json(DATA / "portfolio.json")
    if not raw:
        return {"positions": [], "capital_usd": 500, "initial_capital": 500,
                "wins": 0, "losses": 0, "total_trades": 0}
    return raw

def load_trade_history():
    """Load all trades from trade_history.json (authoritative full history).
    
    BUG FIX (2026-03-23): Previously used portfolio.json positions as primary source,
    which only contained the last ~10 trades. trade_history.json has 2200+ trades.
    
    AUDIT FIX (2026-04-10): REMOVED portfolio.json fallback. The fallback was
    leaking OPEN positions into trade_history when the file was empty (post-reset),
    causing duplicate entries and false accounting discrepancies. If trade_history.json
    is missing/empty/invalid, we return [] — never mix open positions with history.
    """
    path = DATA / "trade_history.json"
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("trades", [])
    except Exception:
        pass
    return []

# ── Agent Notes (Chat) ────────────────────────────────────────────────────────
AGENT_NOTES_FILE = DATA / "agent_notes.json"

def load_notes():
    try:
        with open(AGENT_NOTES_FILE) as f:
            return json.load(f)
    except Exception:
        return {"messages": [], "last_updated": None}

def save_notes(data):
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(AGENT_NOTES_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default

def safe_parse_dt(s):
    """Parse ISO datetime string to naive datetime (strips timezone).
    Handles both timezone-aware and naive strings consistently."""
    from datetime import datetime as _dt
    if not s:
        return None
    try:
        d = _dt.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is not None:
            d = d.replace(tzinfo=None)
        return d
    except (ValueError, TypeError):
        return None

def _load_reset_baseline():
    """Resolve the current reset baseline from reset_history.json[-1].

    v2.12.29: single source of truth for all post-reset filtering.
    The legacy reset_log_20260326.json (paper-era, 2026-04-05) is NO LONGER
    consulted — it misleadingly pulled in 20 pre-scale trades.

    Returns (reset_dt, reset_capital, reset_date_str) or (None, None, None).
    """
    rh = load_json(DATA / "reset_history.json")
    if not (isinstance(rh, list) and rh):
        return None, None, None
    latest = rh[-1]
    reset_date_str = latest.get("reset_date")
    if not reset_date_str:
        return None, None, None
    reset_dt = safe_parse_dt(reset_date_str)
    if reset_dt is None:
        return None, None, None
    # Prefer new_capital (post-scale amount), fall back to capital_after
    reset_capital = safe_float(latest.get("new_capital") or latest.get("capital_after") or 0.0)
    return reset_dt, reset_capital, reset_date_str


def estimate_open_position_pnl(pos: dict, current_price: float | None = None) -> dict:
    """Estimate net PnL if the open position were closed now."""
    symbol = str(pos.get('symbol', ''))
    direction = str(pos.get('direction', 'long')).lower()
    entry_price = safe_float(pos.get('entry_price', 0))
    margin = safe_float(pos.get('margin_usd', 0))
    leverage = safe_float(pos.get('leverage', 0))
    notional = safe_float(pos.get('notional_value', 0)) or safe_float(pos.get('size_usd', 0)) or (margin * leverage)
    funding = safe_float(pos.get('funding_accumulated', 0))
    fee_entry = safe_float(pos.get('fee_entry', 0))
    current_price = safe_float(current_price if current_price is not None else pos.get('current_price', 0))

    if entry_price <= 0 or current_price <= 0:
        return {
            'pnl_usd': safe_float(pos.get('pnl_usd', 0)),
            'pnl_pct': safe_float(pos.get('pnl_pct', 0)),
            'current_price': current_price,
            'notional_value': notional,
        }

    gross_price_pct = ((current_price - entry_price) / entry_price) if direction == 'long' else ((entry_price - current_price) / entry_price)
    gross_pnl_usd = (gross_price_pct * notional) + funding
    fee_exit_est = 0.0
    try:
        import sys
        sys.path.insert(0, str(DATA.parent))
        import executor as ex
        fee_exit_est = notional * (ex.TAKER_FEE + ex.get_slippage(symbol))
    except Exception:
        fee_exit_est = notional * 0.001

    pnl_usd = gross_pnl_usd - fee_entry - fee_exit_est
    pnl_pct = ((pnl_usd / margin) * 100) if margin > 0 else (gross_price_pct * 100)
    return {
        'pnl_usd': round(pnl_usd, 4),
        'pnl_pct': round(pnl_pct, 4),
        'current_price': current_price,
        'notional_value': notional,
    }

# ── Version & Changelog ──────────────────────────────────────────────────────
VERSION = "2.12.29-live"
CHANGELOG = [
    {
        "version": "2.12.29-live",
        "date": "2026-04-23",
        "title": "Unificar dashboard reset baseline a reset_history.json[-1]",
        "changes": [
            "BUG: api_stats leia reset_log_20260326.json (archivo de era paper, reset_date 2026-04-05) directamente. Mezclaba 20 trades pre-scale (era $8 capital) con 11 post-scale (era $100). User veia PF 0.181 vs realidad 0.118, WR 58% vs real 45%.",
            "FIX: nuevo helper _load_reset_baseline() como fuente unica — consume reset_history.json[-1] (scale 2026-04-22 $8->$100).",
            "FIX: api_stats, api_stats_post_reset, api_equity usan helper. Legacy reset_log_20260326.json NO se consulta mas.",
            "FIX: api_trades filtra por reset_dt por default (antes retornaba 31 mezclados). Query ?all=1 para historia completa (audit/debug).",
            "RESULTADO: dashboard muestra 11 trades / 5W-6L-0F / Net -$0.56 / PF 0.118 / WR 45.5% (realidad post-scale, consistente con gráficas que ya estaban correctas).",
        ]
    },
    {
        "version": "2.12.28-live",
        "date": "2026-04-23",
        "title": "Wallet-total equity + initial_capital inmutable (dashboard muestra realidad on-chain)",
        "changes": [
            "PROBLEMA: user reporto dashboard Equity \$91.48 vs user expected \$102. Root causes: (1) initial_capital fue resetado a \$90.10 por tools/reconcile_all_orphans.py (bug: cada reconcile sobrescribia baseline), (2) bot equity excluye SOL fuel + dust residues que el user SI tiene on-chain.",
            "FIX A: tools/reconcile_all_orphans.py ya NO sobrescribe initial_capital. Solo lo establece si falta (nuevo portfolio). Preserva baseline permanente.",
            "FIX B: dashboard/app.py nuevo helper _fetch_wallet_equity() (cache 60s) query on-chain USDC + SOL + JUP + ETH x Jupiter prices (fallback a market_latest.json si Jupiter API down). Nuevo endpoint /api/wallet-equity.",
            "INTEGRATION: api_stats usa wallet_total como equity cuando disponible, fallback a bot_equity si RPC fails. Resto de metricas (return_pct, drawdown) flowean naturalmente.",
            "STATE FIX: portfolio.initial_capital \$90.10 -> \$102.90 (real deposit value = USDC \$100.10 + SOL fuel ~\$2.80).",
            "VERIFICACION: /api/wallet-equity reporta wallet_total=\$105.62, /api/stats return_pct=+2.64% (ganancia real). Dashboard ahora muestra realidad on-chain inclusive SOL fuel + dust.",
            "Nota: endpoint original cachea 60s para no spammear RPC. Paper bot falla silencioso y cae a bot_equity, compatible.",
        ]
    },
    {
        "version": "2.12.27-live",
        "date": "2026-04-23",
        "title": "Fix dashboard equity curve baseline (reset_history stale desde paper era)",
        "changes": [
            "BUG: dashboard /api/equity usaba reset_history.json[-1] como baseline de equity curve. Latest entry era 2026-04-13 con new_capital=500 (era paper bot). Dashboard mostraba equity en rango $500 cuando live bot esta en $100.",
            "EVIDENCIA: user screenshot mostro equity curve desde $500 bajando a $500 (extraño zoom) con discrepancias notorias vs trades reales (que son $10 positions con $0.05-0.30 pnl cada uno).",
            "FIX: agregado entry a reset_history.json reflejando v2.12.20 scale ($8 -> $100). Baseline ahora $100.10 correcto, equity curve $99-100 range matchea reality.",
            "VERIFICADO via /api/equity: first=$100.10, last=$99.59, max=$100.17, min=$99.59. Refleja pnl real del día (-$0.51 net).",
            "Otras charts afectadas positivamente: wins/losses donut ahora solo cuenta trades post-reset (3W/3L = 50/50), symbol_pnl refleja reality (ETH -$0.027, JUP -$0.32, SOL -$0.16), distribution histogram muestra 6 trades post-scale real range.",
            "NO code change — solo state fix en agents/data/reset_history.json (archivo gitignored). Versión bumped para documentar el fix.",
        ]
    },
    {
        "version": "2.12.26-live",
        "date": "2026-04-23",
        "title": "FIX daily_target reset usa equity (no solo capital_usd) + safety >20% skip",
        "changes": [
            "ROOT CAUSE: orchestrator.py:273 day-reset guardaba starting_capital = capital_usd (solo USDC libre). Cuando hay positions abiertas al day-rollover 00:00 UTC, capital_usd es artificialmente bajo (invested locked). El siguiente calculo daily_pnl_pct aparenta +30% ganancia falsa - dispara DAILY_TARGET_MAX_REACHED - force-close positions que suelen ser en losses.",
            "EVIDENCIA: 2026-04-23 00:05 UTC, 3 positions cerraron con loss total -$0.58 reportando DAILY_TARGET_MAX_REACHED: 32.52% >= 5.0%. Pero 32.52% era falso (starting_capital=62.17 vs equity real 92.17).",
            "FIX orchestrator.py: day-reset ahora calcula equity = capital_usd + sum(position.margin_usd) + sum(position.pnl_usd). Matchea la logica de daily_target.py:69-78 que ya tenia correcto pero no corria por timing.",
            "FIX daily_target.py: safety check defensivo - si daily_pnl_pct > 20% (claramente bug), log warning y skip trigger. Defensa en profundidad por si otra fuente escribe starting_capital malo.",
            "STATE fix manual: daily_target_state.json starting_capital 62.17 -> 91.59 (equity real), target_reached true -> false, closed_at null.",
            "NEW tools/test_daily_target_reset.py - 4 unit tests (no positions, positions+unrealized, legit 5%, absurd 30% safety). ALL PASSED.",
            "Impacto: bug causaba perdidas directamente atribuibles de ~$0.50-1/dia cuando positions abiertas al rollover. Backtester 14d net P&L -$0.50 incluye estas forced-close losses.",
        ]
    },
    {
        "version": "2.12.25-live",
        "date": "2026-04-22",
        "title": "Config bump: LIVE_MAX_POSITIONS 1 -> 3 (aliga con observed behavior)",
        "changes": [
            "OBSERVED: durante monitoreo autonomo (iter 7, 15:34-15:39 UTC) bot abrio 3 posiciones simultaneas (SOL+ETH+JUP) pese a LIVE_MAX_POSITIONS=1 en .env. Probable: wild_mode o daily_target logic bypassea el cap, o el var es suggestion no hard cap.",
            "USER DECISION: 3 concurrent positions es aceptable (max exposure $30 con $10 position size y $90 capital = 33% exposure max).",
            ".env + .env.example: LIVE_MAX_POSITIONS 1 -> 3 para alinear config con intent/reality.",
            "NO restart orch necesario — bot ya operando con 3 pos. Cambio se activara oficialmente en proximo watchdog respawn.",
            "Impacto: mas diversificacion (SOL + ETH + JUP simultaneos), menos USDC libre para switching, pero matchea behavior del LLM cuando detecta multiples senales.",
        ]
    },
    {
        "version": "2.12.24-live",
        "date": "2026-04-22",
        "title": "Phase 1 prep cleanup — .env.example sync + backtester fix & enrich",
        "changes": [
            "FIX agents/backtester.py load_trade_history — manejaba solo formato list, ahora soporta dict con trades key (actual formato post-v2.12.16).",
            "FIX agents/backtester.py by_token — usaba campo token pero trade_history guarda symbol. Ahora por-token breakdown funciona correctamente.",
            "ENRICH backtester: argparse con --mode (live/paper/paper_drift) + --days N window. Formato $ 4 decimales (vs 2 que round a zero valores pequenos).",
            ".env.example sync v2.12.24: documentados 3 advanced config vars faltantes (ORCH_LOCK_FILE, ORCH_PGREP_PATTERN, WATCHDOG_PREFIX). PAPERCLIP_API_KEY sigue hardcoded como fallback en paperclip_client.py pero documentado como override opcional.",
            "Live trading STATS baseline (backtester run): 23 trades / WR 69.6% (16W-7L) / PF 8.94 / payoff 3.91x / max DD $0.0023. Por token: SOL WR 71% (14t), ETH WR 75% (4t), JUP WR 60% (5t). Edge matematicamente validado pre-Phase-1-gate.",
        ]
    },
    {
        "version": "2.12.23-live",
        "date": "2026-04-22",
        "title": "FIX reconcile_critical false-positives — asymmetric check (root cause del patron de crisis 2-3/dia)",
        "changes": [
            "ROOT CAUSE: agents/reconcile.py usaba abs() para comparar wallet balance vs position.tokens, flagging excess (wallet > position) como critical, no solo shortage. User fuel real ~0.048 SOL vs hardcoded 0.010 FUEL_RESERVE causaba false-positives 25-33% cada vez que se abria posicion SOL.",
            "EVIDENCIA: v2.12.16 SOL(25.4%) = false positive (wallet 0.039 SOL, fuel 0.029 excess post-0.01-reserve). v2.12.22 05:08 SOL(33.4%) = mismo bug (wallet 0.163 SOL, 0.048 fuel excess). JUP(100.9%) del v2.12.16 era real orphan (wallet=0, correcto).",
            "FIX agents/reconcile.py: asymmetric check. Solo flag SHORTAGE (actual < expected) como warning/critical. Excess (actual > expected) = info severity, benign (user fuel, residue, airdrops, other positions).",
            "Discrepancy dataclass: nuevo campo is_shortage=True para log differentiation (SHORTAGE vs excess).",
            "NEW tools/test_reconcile_asymmetric.py — 6 unit tests cubren: SOL excess false-positive, JUP orphan real, ETH partial close, ETH exact match, JUP tiny excess benign, JUP 1% shortage warning. ALL PASSED.",
            "Impacto esperado: crisis reconcile_critical 2-3/dia → <0.5/dia. Downtime bot 4h/dia → <1h/dia. Cero riesgo aditivo — solo reduce falsos positivos.",
        ]
    },
    {
        "version": "2.12.22-live",
        "date": "2026-04-22",
        "title": "Phase 1 prep — Telegram signals module + health badge + runbook + auto-learner reset",
        "changes": [
            "NEW agents/telegram_signals.py — signal broadcaster pattern paperclip_client.py. Hooks on_trade_opened + on_trade_closed + on_daily_report. DISABLED por defecto (TELEGRAM_ENABLED=false). Activacion Phase 2: crear bot via BotFather + canal + set tokens + flip flag.",
            "WIRE executor.py — telegram hooks en 3 sitios (line 1054 open + 1493 + 1558 close) en paralelo a paperclip. try/except wrapper: si telegram falla, trading sigue normal.",
            ".env / .env.example — added TELEGRAM_ENABLED + TELEGRAM_SIGNALS_BOT_TOKEN + TELEGRAM_SIGNALS_CHANNEL.",
            "NEW dashboard health badge — verde/rojo/gris dot + live info (HB age, positions, capital) en header. Poll /api/health cada 30s. CSS + HTML + JS additive, no altera layout existente.",
            "NEW docs/RUNBOOK.md (250 lines) — incident response procedures: kill switch, orphan, close fail 6024, HB stale, watchdog+orch dead, dust on-chain, capital scale. Includes quick-status links, rollback procedure, Paperclip company UUID map.",
            "NEW tools/auto_learner_soft_reset.py + APPLIED — reseteo counters de auto_learner post capital-scale v2.12.20 ($8 → $100). Total_trades_learned 19→0, adaptation_count 5→0, last_trade_count frozen a 22 como baseline. Params (SL/TP/leverage/risk_per_trade) preservados — son % scale-invariant. Tokens_to_avoid/prefer preservados.",
            "CERO cambios en trading logic — todos los items son capa observability + comunicacion + higiene.",
        ]
    },
    {
        "version": "2.12.21-live",
        "date": "2026-04-22",
        "title": "Drift devnet smoke test — infra validada (Phase 1.5 parallel)",
        "changes": [
            "FIX agents/drift_client.py:_load_keypair — soporta el formato custom del Jupiter bot (dict con private_key como JSON-array-string) ademas del formato standard de Solana CLI (list of 64 ints). Backward-compatible.",
            "NEW tools/drift_devnet_mint_usdc.py — portado de origin/drift-integration. Mintea devnet USDC via Drift token_faucet program (V4v1mQiAdLz4qwckEb45WqHYceYizoib39cDBHSWfaB). Idempotente: crea ATA si falta + mint. Usa DRIFT_WALLET_PATH del .env.",
            "Devnet smoke EXECUTED: Drift user PDA creada, USDC ATA creada, 100 USDC devnet minted, $0.5 depositado como collateral, 2 market orders (long+short) broadcast + aceptadas por Drift.",
            "GAP identificado: Pyth oracle de SOL-PERP en devnet esta STALE (~52 dias lag), AMM rechaza fills con oracle not valid. NO es bug del codigo — es issue del ambiente devnet. Mainnet oracles actualizan ~400ms, no deberia repetirse.",
            "Conclusion: infra Drift validada end-to-end excepto fill (bloqueado por oracle stale). Siguiente paso NO es mainnet inmediato — esperar Phase 2 gate (5-7 dias estabilidad Jupiter spot a $100) antes de $20 walk-forward en Drift mainnet.",
        ]
    },
    {
        "version": "2.12.20-live",
        "date": "2026-04-22",
        "title": "Capital scale $8 -> $100 + position size $2 -> $10 (Phase 1 acelerada)",
        "changes": [
            "USER deposit: +$92 USDC (wallet $8.10 -> $100.10) + 0.033 SOL fuel (0.015 -> 0.049).",
            "ENV: LIVE_MAX_POSITION_USD $2 -> $10 (5x scaling proporcional al capital). LIVE_MAX_POSITIONS mantenido en 1 hasta Phase 2 gate.",
            "STATE: portfolio.json capital_usd $8.09 -> $100.10, initial_capital reset para fresh baseline Phase 1. daily_target_state + wild_mode_state reset a $100 baseline.",
            "Ratio fee/position mejorado: $0.12 fee sobre $10 posicion = 1.2% (antes era 6% con $2). Break-even alcanzable con edge actual (expectancy +$0.0017/trade NET con $2, escalara ~5x a +$0.008/trade con $10).",
            "Backups .bak_<epoch> creados para portfolio, trade_history, daily_target, wild_mode, .env (rollback si necesario).",
            "Phase 1 acelerada: 5-7 dias observacion con cron health_check monitoreando. Gate a Phase 2: 0 crises auto-detectadas + edge positivo sobre 30+ trades a capital escalado.",
        ]
    },
    {
        "version": "2.12.19-live",
        "date": "2026-04-22",
        "title": "Phase 1 observability — weekly_report + health_check + /api/health",
        "changes": [
            "NEW: tools/weekly_report.py — compila métricas paper vs live side-by-side (7/14/30 days). Modos table, json, gate-check. Usado para validar edge + Phase 1 gate criteria.",
            "NEW: tools/health_check.py — cron-friendly health probe. Checks: heartbeat, orch alive (CWD match), watchdog alive, kill_switch, close_failures counter. Flag --alert crea Paperclip issue urgent dedupado 15min.",
            "NEW: /api/health endpoint en dashboard/app.py — expone {healthy, orch, watchdog, heartbeat, kill_switch, close_failures, portfolio} JSON. Usado por health_check cron + external monitors.",
            "CRONTAB: */5 * * * * python3 tools/health_check.py --alert → detecta crisis <5min, alerta SOLAAA automáticamente.",
            "Phase 1 observability lista. Live edge confirmado: WR 83.3% sobre 22 trades últimos 7d, expectancy +$0.0017/trade NET (muy small — scale ayudaría).",
        ]
    },
    {
        "version": "2.12.18-live",
        "date": "2026-04-21",
        "title": "Fix Paperclip routing — SOLAA (paper) -> SOLAAA (Solana Cripto Trading Live)",
        "changes": [
            "FIX: agents/paperclip_client.py COMPANY_ID hardcoded apuntaba a SOLAA (paper, UUID 782b926b...) en lugar de SOLAAA (live, UUID 2b0a3d7d-4929-42ef-9602-86324c0a202a). Issues v2.12.16 y v2.12.17 fueron posteados a la company equivocada.",
            "FIX: .env PAPERCLIP_API_KEY cambiado de agent key de SOLAA (scoped) a board key (acceso a todas las companies).",
            "MIGRATION: SOLAA-115 (v2.12.16) -> SOLAAA-19, SOLAA-116 (v2.12.17) -> SOLAAA-18. SOLAA originals marcadas cancelled con redirect note.",
            "NOTA: memoria de 6 dias decia SOLAAA=live, SOLAA=paper — confirmado correcto. Mi audit anterior infirio mal que solo existia una company. Listado de 5 companies verificado: SOL/SOLA (archived), SOLAA (paper), SOLAAA (live), BIT (BitTrader YouTube).",
        ]
    },
    {
        "version": "2.12.17-live",
        "date": "2026-04-21",
        "title": "Per-symbol slippage config + escalating retry ladder — fix ETH close-loop",
        "changes": [
            "ROOT CAUSE v2.12.16 post-audit: ETH Wormhole (bridged token, thin liquidity) fallaba Jupiter close con error 6024 (slippage exceeded) usando MAX_SLIPPAGE_BPS=100 global. Fresh-quote retry hardcoded 500bps tampoco pasaba. 3 fails consecutivos habrian disparado kill switch.",
            "FIX sistemico: agents/executor.py nuevo helper get_max_slippage_bps(symbol) lee SLIPPAGE_BPS_<SYMBOL> env var (ej SLIPPAGE_BPS_ETH=800) con fallback a MAX_SLIPPAGE_BPS global. Aplica a entry y close swaps.",
            "FIX retry ladder: agents/jupiter_swap.py fresh-quote retry ahora escala progresivamente max(500, slippage_bps*2) -> min(max(1500, slippage_bps*8), 2000bps). Cap absoluto 2000bps por seguridad.",
            ".env: SLIPPAGE_BPS_ETH=800 agregado. ETH close ahora usa ladder 800 -> 1600 -> 2000bps. SOL/JUP mantienen ladder 100 -> 500 -> 1500bps.",
            "CLEANUP: sweep JUP residue 0.107 tokens ($0.018) on-chain via tools/sweep_token.py. Wallet libre de dust.",
            "Audit confirmado: bot estable post-deploy, LLM decide HOLD ETH con R/R 2.82x, close_failures=1 (lejos del kill switch threshold=3), HB fresh, no error spam. Close de ETH usara 800bps al proximo trigger natural (LLM sell o SL/TP).",
        ]
    },
    {
        "version": "2.12.16-live",
        "date": "2026-04-21",
        "title": "Multi-orphan reconcile tool + recovery de SOL+JUP del kill switch loop",
        "changes": [
            "Bot se detuvo 08:45 MDT por kill switch auto-disparado tras 3 posiciones abiertas (SOL+JUP+ETH) con discrepancias vs wallet. Causa raiz: DNS flap de token.jup.ag impidio persistir los close-tx al portfolio.json tras trailing SL.",
            "NEW: tools/reconcile_all_orphans.py (v2.12.16) — generaliza reconcile_both_orphans.py para manejar N orphans. Flags --dry-run (audit sin escribir) y --limit N (default 100, antes 30) para ventana de getSignaturesForAddress. Token-mint verification por posicion evita matchear el mismo close tx a multiples orphans.",
            "FIX: bug de total_trades counter que no incrementaba tras reconcile.",
            "FIX: auto-backup de portfolio.json y trade_history.json a .bak_<epoch> antes de escribir en modo live.",
            "RECOVERY: 2 orphans reconciliadas (SOL +$0.0072, JUP +$0.0048); ETH correctamente identificada como no-orphan (0.00086215 on-chain match portfolio). Bot relanzado, nueva posicion SOL abierta en primer ciclo post-recovery.",
        ]
    },
    {
        "version": "2.12.15-live",
        "date": "2026-04-21",
        "title": "UX: surface LLM eval timestamp + pnl at eval — elimina confusion dashboard vs LLM",
        "changes": [
            "User reporto: dashboard JUP header -0.34% (losing) vs LLM text 'ganadora +0.49%'. Investigacion: ambos NET correctos, pero calculados con precio de momentos distintos (live vs cycle-cached, ~60s gap).",
            "Fix UX: /api/ai-thinking ahora incluye llm_evaluated_at + llm_pnl_pct_at_eval + llm_pnl_usd_at_eval por posicion (leidos de position_decisions.json, ya poblados por v2.12.12 fix).",
            "Frontend AI panel: nueva linea meta 'Evaluado hace Xm · LLM vio pnl Y%' debajo del RAZONAMIENTO DEL LLM. Si delta live vs eval > 0.2%: warning 'precio movio Z% desde analisis'. Si eval > 5min: warning 'eval antigua'.",
            "Sin cambios backend/trading. LLM prompt sigue recibiendo pnl correct del ciclo (v2.12.12 NET). Dashboard header sigue fresh con live price (user sabe valor actual).",
            "User ahora entiende por que los dos numeros pueden diferir: tiempo transcurrido + movimiento de precio intercycle.",
        ]
    },
    {
        "version": "2.12.14-live",
        "date": "2026-04-21",
        "title": "Maintenance: sweep residue 0.000863 ETH → USDC, resync portfolio state",
        "changes": [
            "Accounting audit revelo que los ~$2 aparentemente faltantes (wallet 8.06 vs 10.0 esperado) estaban en realidad como 0.000863 ETH SPL (~$1.99) on-chain — residue del manual_reconcile v2.12.12 que solo actualizo state sin ejecutar swap reverso.",
            "Sweep ejecutado via tools/sweep_token.py --symbol ETH --amount 0.000863 --slippage-bps 500 --priority veryHigh. Tx 67ni6n36Zn... → +$1.998 USDC.",
            "State sync: portfolio.capital_usd 8.063 → 10.0612 (wallet real), daily_target_state reset a nuevo baseline, trade_history +1 entry maintenance_sweep (audit trail).",
            "Wallet final: $10.06 USDC + 0.016 SOL = baseline ~idéntico al inicial de Yonathan ($10 + 0.022 SOL).",
            "Real net loss acumulado del día: -$0.43 (fees Solana por retries y priority veryHigh × multiple cycles). Dentro del rango esperado para un día con tantos deploys + manual interventions.",
        ]
    },
    {
        "version": "2.12.13-live",
        "date": "2026-04-21",
        "title": "FIX ROOT CAUSE: emergency_close.py now persists portfolio after on-chain close",
        "changes": [
            "Root cause orphan pattern identificado: tools/emergency_close.py (cron every min) cerraba positions via Jupiter OK on-chain pero NO actualizaba portfolio.json. Next orchestrator cycle veia position todavia 'open', reconcile detectaba mismatch wallet vs position.tokens → 100% discrepancy → kill switch.",
            "Fix: emergency_close.py ahora on Jupiter swap success: (1) marca pos closed en portfolio.json, (2) credita usdc_received a capital_usd, (3) mueve pos a trade_history con tx_signature_close, (4) incrementa total_trades/wins/losses.",
            "Manual recovery: 2 orphan positions (JUP + SOL del cycle 00:22 UTC) movidas a history con pnl $0 estimado (tx close irrecoverable). capital_usd resynced con wallet $8.06.",
            "Bot ahora capaz de cerrar positions JUP/ETH via emergency cron sin crear orphans. Combined con v2.12.10 retry y v2.12.12 net pnl = close reliability completa.",
        ]
    },
    {
        "version": "2.12.12-live",
        "date": "2026-04-21",
        "title": "FIX: unificar pos.pnl_usd a NET (post-fees) — elimina discrepancia dashboard vs LLM",
        "changes": [
            "User reporto: dashboard JUP header muestra -1.14% (losing) pero LLM razonamiento dice +0.39% (ganadora). Mismo fenomeno con ETH.",
            "Root cause: paper_update_positions guardaba GROSS pnl_usd en portfolio (solo price delta). Dashboard /api/positions calculaba NET aparte (gross - fee_entry - fee_exit_est). LLM leia pos.pnl_usd directo = GROSS.",
            "Fix executor.py paper_update_positions: deducir fee_entry + fee_exit_est al escribir pos.pnl_usd. Mantiene pos.gross_pnl_usd para audit.",
            "Impact: LLM, quant_score, reporter, dashboard ahora leen el mismo NET pnl — single source of truth. Decisiones HOLD/CLOSE se basan en P&L realizable (lo que el user recuperaría si cierra ahora).",
            "Ejemplo con JUP: entry 0.1696, current 0.1702, notional $2. Gross +$0.0077 (+0.39%). Fee_entry $0.002, fee_exit_est $0.004 → Net pnl = -$0.001 (-0.05% sobre margen).",
            "Los SL/TP thresholds (pnl_pct > 12%, < -8%) siguen funcionando — solo el numero es mas preciso ahora.",
        ]
    },
    {
        "version": "2.12.11-live",
        "date": "2026-04-21",
        "title": "Tune fresh-quote retry slippage 200→500bps (Wormhole wETH needs more)",
        "changes": [
            "First v2.12.10 live test: ETH close fresh-quote retry at 200bps STILL failed Jupiter 6024. Manual sweep at 500bps succeeded immediately (pnl +$0.008).",
            "Raise retry slippage 200→500bps en jupiter_swap.execute_swap fresh-quote path. Wormhole wETH spread puede ser >2% en volatilidad.",
            "Infra v2.12.10 works as designed — this is just parameter tuning.",
            "Recovery aplicado: ETH position manual-swept (tx 2MdfvghndaTE...) + marked closed in trade_history.",
        ]
    },
    {
        "version": "2.12.10-live",
        "date": "2026-04-20",
        "title": "Reliability: broadcast retry + fresh-quote + priority veryHigh + CoinGecko cache + log rotation",
        "changes": [
            "A1 jupiter_swap broadcast_and_confirm: 3 retries exponential backoff. Slippage errors (Jupiter 6024) NO retry mismo tx (needs fresh quote).",
            "A2 execute_swap fresh-quote retry on 6024: una vez, slippage 200bps + mismo priority level. Evita close fallidos por movimiento intra-4s precio.",
            "A3 real_close_position priority_fee_level: medium → veryHigh. Closes son time-critical (position on-chain), priority para ejecución rápida (<2s).",
            "A4 stale quote detection: si quote >3s old antes de build, auto-refresh. Previene blockhash expired / slippage acumulado mientras build+sign.",
            "Bonus A: priority_level passthrough a Jupiter API (antes hardcoded 'high' en build_swap_transaction, ahora usa param del caller).",
            "B1 market_data.py cache 10min TTL para fetch_coingecko_24h + fetch_hourly_trends. Reduce 195 warnings/sesion → ~20.",
            "B2 run_watchdog.sh log truncation: si /tmp/solana_live_*.log >100MB, keep last 20MB en startup. Previene disk pressure.",
            "C1 NEW tools/sweep_token.py — generaliza emergency_sol_to_usdc para cualquier simbolo (JUP, ETH, etc). --priority veryHigh default.",
            "C2 llm_config per-provider cooldown: Claude 429 → skip Claude 300s, salta directo a MiniMax. Evita hammer de requests garantizados a fallar.",
            "Root cause JUP/ETH close 6/6 fails: Jupiter error 6024 (slippage exceeded) en preflight. Investigacion en plan v2.12.10 con evidencia log + codepath.",
            "Expected post-deploy: close success JUP/ETH 0% → 85%+, warnings CoinGecko -80%, log size bounded.",
        ]
    },
    {
        "version": "2.12.9-live",
        "date": "2026-04-20",
        "title": "FIX: hours_open=0 bug + missing trade_history append on live normal close",
        "changes": [
            "FIX 1 (risk_manager.py:611): _quant_score buscaba pos.get(opened_at) pero portfolio usa open_time. Consecuencia: hours_open siempre 0 → LLM prompt decia Horas abierta: 0.0h → LLM decidia HOLD creyendo posicion recien abierta aunque llevaba 10h. Ahora fallback a open_time.",
            "FIX 2 (executor.py B2c): mi patch v2.12.1 live normal close (SL/TP/trailing/time) hacia continue ANTES del history.append paper. Consecuencia: trades cerrados por SL/TP no aparecian en trade_history.json ni en el dashboard. Capital_usd portfolio diverge de wallet real. Ahora append({**pos}) + closed.append(pos) + on_trade_closed Paperclip.",
            "Impact del bug de horas: quant score factor MUY_ANTIGUA/ANTIGUA nunca disparaba. SL_PELIGRO logic (SL cerca + pos vieja) nunca activaba. LLM subestimaba decay de posicion -> HOLD excesivo.",
            "Impact del history bug: usuario veia posiciones cerradas pero dashboard y total_trades desync. Reconcile state sincronizado (capital_usd $4.03 -> $8.04 post-fix).",
            "Defense-in-depth cerrada v2.12.3 + v2.12.6 + v2.12.8 + v2.12.9 cubren ahora: open tracking, pre-flight, orphan detection, reconcile nativo, persist inmediato, history append, hours_open fix.",
        ]
    },
    {
        "version": "2.12.8-live",
        "date": "2026-04-19",
        "title": "HOTFIX: reconcile false 100% + immediate portfolio persist post-swap",
        "changes": [
            "FIX reconcile.py: para positions SOL ahora compara native SOL balance (- fuel reserve 0.01) vs position.tokens, no el SPL wSOL balance. Jupiter swap entrega SOL NATIVO al unwrap, no wrapped SOL — comparar SPL siempre daba 0 → 100% discrepancy FALSA → kill switch on every legit SOL position.",
            "FIX real_open_position + real_close_position: persist inmediato de portfolio.json tras swap on-chain exitoso (atomic write via .tmp + replace). Previene state loss si orchestrator muere entre swap y reporter stage 6 save.",
            "Root cause de 3 orphans detectados hoy: close on-chain ejecutaba OK pero portfolio no persistía hasta stage 6 (reporter). Si el proceso moría entre stage 4 (executor close) y stage 6, state se perdía — resultando en position ghost que reconcile interpretaba como orphan.",
            "Recovery aplicado: 3rd orphan del día (SOL_live_1776654770, pnl +$0.0021) movido a history via tools/reconcile_both_orphans.py. Capital $10.03 USDC sincronizado con wallet on-chain.",
            "Fixes compuestos v2.12.3 + v2.12.6 + v2.12.8: defense-in-depth completa — pre-flight check evita wasted broadcasts + orphan detection gracefully quarantine + persist inmediato post-swap + reconcile correcto.",
        ]
    },
    {
        "version": "2.12.7-live",
        "date": "2026-04-19",
        "title": "Phase A: whitelist expandida a JUP + ETH (3x superficie de oportunidad)",
        "changes": [
            "MINT_MAP expandido: JUP (JUPyiwrYJFsk...) + ETH Wormhole (7vfCXTUXx5WJ...). DECIMALS_MAP ya tenia entries (JUP=6, ETH=8).",
            "TRADE_WHITELIST env: SOL → SOL,JUP,ETH. Ahora el bot puede ejecutar las 3 majors que el LLM recomienda con alta conviction.",
            "Pre-deploy verify_mints.py: SOL 0.009% impact, JUP 0.000%, ETH 0.003% — todos con reverse routes OK. Promovido a tools/ como script permanente.",
            "LIVE_MAX_POSITIONS=1 y LIVE_MAX_POSITION_USD=2 sin cambios — solo 1 posicion activa a la vez, bot escoge best signal entre SOL/JUP/ETH.",
            "Motivacion: audit mostro que LLM generaba 4 long signals oversold_bounce (JUP 0.83/RSI 15.2, ETH 0.82/RSI 19.6, SOL 0.81/RSI 20.1, FARTCOIN 0.80/RSI 16.8) pero solo SOL ejecutaba. JUP tenia mayor conviction que SOL.",
            "Phase B (Drift perps) deferido — se activa post-validacion 1 semana Phase A con criterios: 5+ trades completos, 0 orphans, 0 kill switches, P&L neutral/positivo.",
        ]
    },
    {
        "version": "2.12.6-live",
        "date": "2026-04-19",
        "title": "HOTFIX: emergency_close.py pre-flight check (orphan detection gap)",
        "changes": [
            "FIX: tools/emergency_close.py ahora hace pre-flight balance check antes del Jupiter swap. Antes llamaba Jupiter directo sin usar real_close_position, por lo que mi P3a fix (v2.12.3) no aplicaba — orphans disparaban 3 fails → kill switch.",
            "Ahora: si wallet_sol - 0.005 reserve < position.tokens, marca entry como orphan y continue (NO cuenta como broadcast fail, NO activa kill switch).",
            "Incident recovery: 2 positions reconciled (pnl +$0.0008 + +$0.0007 = +$0.0015 total).",
            "Post-fix ambos flows (orchestrator executor + cron emergency_close) protegidos uniformemente.",
        ]
    },
    {
        "version": "2.12.5-live",
        "date": "2026-04-19",
        "title": "LLM optimization: reorder chain + strategy cache + position throttle",
        "changes": [
            "REMOVED GPT-5.4 from call_llm chain (port 18793 muerto, generaba 25 errors/30min). Sigue disponible via LLM_ALLOW_GPT5=true env var si lo reviven.",
            "REORDERED chain: Claude Sonnet 4.6 (primary, OAuth Max) → MiniMax M2.7 (secondary) → Qwen 2.5 14B local (tertiary).",
            "NEW ai_strategy cache: 3min + invalidacion por movimiento precio >=0.3%. Reuse signals_latest.json. Esperado: ~50% reduccion de strategy LLM calls en mercado rangebound.",
            "NEW risk_manager throttle: skip LLM position_decisions si pnl_pct <0.5% AND distSL>3% AND distTP>3% AND lastEval<3min. Reuse cached decision. Esperado: ~60-70% reduccion cuando position calm.",
            "Logs limpios: 0 GPT-5.4 errors por ciclo (vs 25+ antes). Latencia ~200ms menos por ciclo.",
            "Estado actual al deploy: Claude 429 (rate limit) y MiniMax 529 (peak surge) intermitentes — Qwen local queda como respaldo. Chain recuperara primary cuando rate limits pasen.",
        ]
    },
    {
        "version": "2.12.4-live",
        "date": "2026-04-19",
        "title": "UI: display 4 decimales en pnl_usd < $1 (trades pequeños ya no muestran $0.00)",
        "changes": [
            "fmt$ y _fmtUsd ahora usan 4 decimales cuando |pnl| < $1 (antes siempre 2 → pnls subcentavo como $0.0016 mostraban $0.00).",
            "Ejemplo: trade de $2 margen con +0.08% movimiento ahora muestra +$0.0016 en vez de +$0.00, consistente con el % mostrado.",
            "Sin cambios en storage ni cálculos, solo formato de display.",
        ]
    },
    {
        "version": "2.12.3-live",
        "date": "2026-04-19",
        "title": "Hardening post-incident: pre-flight balance check + WILD_MODE flag + emergency tolerance + 5 fixes",
        "changes": [
            "INCIDENT RECAP: entre 02:26-02:40 UTC position legitima abrio y cerro on-chain correctamente (pnl +0.0016 USD), pero el portfolio no persistio el close. Orchestrator murio, emergency_close cron trato de cerrar la posicion fantasma y activo kill switch permanente. Bot 15h offline.",
            "FIX P3a: real_close_position pre-flight check — si wallet SOL < position tokens, retorna orphan sin hacer broadcast. Evita gas wasted en swaps que iban a fallar con 'insufficient lamports'.",
            "FIX P3b: close paths (emergency + normal) manejan orphan gracefully — marcan needs_manual_reconcile y notifican Paperclip, NO activan kill switch automatico.",
            "FIX P3c: emergency_close.py tolerance — 3 consecutive fails antes de activar kill switch (antes: 1 fail = bloqueo permanente). Transient RPC issues ya no bloquean bot.",
            "FIX P3d: LIQUIDATED close path tambien detecta mode=live y rutea a real_close_position (era solo path unpatched despues de B2).",
            "FIX P3e: Wild Mode feature flag WILD_MODE_ENABLED (default false). Previene auto-activacion que causaba false daily_target_hit al abrir positions.",
            "FIX P3f: equity_history.json format migration (list legacy → dict) — elimina los 68 warnings/sesion 'list indices must be integers or slices'.",
            "NEW tools/reconcile_orphan.py — one-shot recovery para sincronizar portfolio con on-chain truth. Uso: python3 tools/reconcile_orphan.py cuando hay position huerfana.",
            "State recovery aplicado: moved SOL_live_1776565594 to trade_history (pnl +0.0016, tx_close 3etBpV4J...), capital_usd $8.02 → $10.02, daily_target + wild_mode reset, kill switch cleared.",
            "Defense in depth resultante: 3 safeguards escalados (pre-flight check al abrir/cerrar + fail tolerance + manual reconcile quarantine) que detectan desync antes de bloquear bot.",
        ]
    },
    {
        "version": "2.12.2-live",
        "date": "2026-04-18",
        "title": "HOTFIX: datetime shadow en real_open_position (Drift branch causaba UnboundLocalError)",
        "changes": [
            "FIX: removido from datetime import dentro del branch DRIFT_ENABLED en real_open_position. El import local hacia datetime variable local de toda la funcion, disparando UnboundLocalError cuando DRIFT_ENABLED=false y el flow llegaba a datetime.now() del Jupiter path normal.",
            "Impacto del bug: entre v2.12.0 y v2.12.2, cada intento de abrir posicion Jupiter ejecutaba el swap on-chain pero crashaba al construir position dict. Resultado: 4 trades fantasma entre 19:07-19:35 que dejaron ~0.09 SOL huerfano en wallet.",
            "Recovery adicional: swap 0.093 SOL -> $7.97 USDC para consolidar baseline, resync portfolio.capital_usd y daily_target_state a balance real on-chain.",
            "Estado final: 0.047 SOL wallet (0.023 fuel + 0.023 ghost accepted), $7.88 USDC. Equity total equivalente $9.91 vs inicial $11.89 (perdida real ~$0.03 solo en fees de todos los swaps).",
        ]
    },
    {
        "version": "2.12.1-live",
        "date": "2026-04-18",
        "title": "FIX: 3 bugs live mode — capital_usd, real_close_position, emergency close swap",
        "changes": [
            "FIX B1: real_open_position ahora decrementa portfolio.capital_usd post-swap (matches paper_open_position:971 pattern). Sin esto, _equity doble-contaba cash+margin → disparo falso de daily_target_hit 20% al instante de abrir cualquier live trade.",
            "FIX B2: NEW agents/executor.py::real_close_position() — ejecuta swap reverso TOKEN→USDC via Jupiter. Ambos paths close ahora detectan mode=live: close_positions_emergency (emergency_close, daily_target, smart_rotation, portfolio_tp) y normal close (SL/TP/TRAILING_SL/TIME_EXIT).",
            "FIX B3: live close path ya no llega al capital_usd += (margin + pnl) que inflaba fantasma; el nuevo path usa USDC real recibido del swap. Paper path intacto sin cambios de comportamiento.",
            "Incident recovery: portfolio.capital_usd 11.99 → 8.00 (USDC real en wallet tras el trade fantasma), daily_target_state reset (starting_capital=8, target_reached=false), wild_mode_state clear. Los 0.023 SOL huérfanos quedan como fuel extra (~400 tx runway adicional).",
            "Root cause: primer live trade SOL $2 @ $86.12 (tx miLhJRHj...QWW) disparó el chain: open sin decrement → 20% falso daily target → wild mode abandon → emergency close paper-only → SOL quedó en wallet sin registrar. Daño real: $0.01 en fees.",
        ]
    },
    {
        "version": "2.12.0-live",
        "date": "2026-04-18",
        "title": "Drift Protocol perps — longs+shorts con leverage 1-2x (dormant hasta validación)",
        "changes": [
            "PORT agents/drift_client.py desde origin/drift-integration — driftpy wrapper SOL-PERP",
            "NEW agents/drift_adapter.py — 8 safety gates: DRIFT_ENABLED + LIVE + !kill_switch + daily_loss + whitelist + confidence + leverage_cap + collateral",
            "Hook en executor.real_open_position — si DRIFT_ENABLED=true, ruta a drift_adapter; Jupiter queda como fallback y para sol_topup",
            "SHORTS HABILITADOS — bypass del Jupiter direction-must-be-long guard cuando Drift activo",
            "safety.validate_drift_startup() — chequea driftpy, env vars, drift_state.json",
            "reconcile.py extendido — compara SOL-PERP base on-chain vs portfolio.json positions mode=drift cada 10 ciclos",
            "NEW endpoint GET /api/drift/status — collateral, leverage, funding, posición actual",
            "NEW tarjeta 🎯 Drift en header del dashboard",
            "NEW tools/drift_setup.py — CLI init subaccount + deposit USDC",
            "NEW tools/drift_devnet_smoke.py — validación long+short round-trip",
            "Defaults conservadores: DRIFT_ENABLED=false (dormant), MAX_LEVERAGE=2, DEFAULT=1, MAX_COLLATERAL=$3, MIN_USDC_RESERVE=$5",
            "Rollout: devnet smoke 1 día → mainnet $3 colateral → primer trade → escalar a leverage 2x",
        ]
    },
    {
        "version": "2.11.0-live",
        "date": "2026-04-18",
        "title": "SOL auto-replenish: bot se compra su propio combustible",
        "changes": [
            "NEW agents/sol_topup.py — swap USDC→SOL automático cuando balance < SOL_MIN_BALANCE",
            "Hook en orchestrator.run_cycle cada 10 ciclos (piggyback con reconcile) — solo si LIVE_TRADING_ENABLED=true",
            "Defense-in-depth 6 gates: SOL_AUTO_TOPUP_ENABLED, LIVE, !kill_switch, balance<min, usdc_reserve_floor, daily_cap",
            "Estado diario persistido en agents/data/sol_topup_state.json (spent_usd_today + events)",
            "NEW endpoint GET /api/sol/fuel → balance actual, días/trades estimados, histórico del día",
            "NEW tarjeta 🔋 SOL Fuel en header del dashboard — verde/amarillo/rojo según ratio vs SOL_MIN_BALANCE",
            "Paperclip notification automática por cada top-up exitoso (priority=low, status=done)",
            "Defaults conservadores: SOL_AUTO_TOPUP_ENABLED=false, $1/evento, max $3/día, reserva USDC $2",
            "Feature dormant hasta activación manual — rollout en 5 pasos (shadow → dry-run → $0.50 manual → live → 72h validación)",
        ]
    },
    {
        "version": "2.10.1-live",
        "date": "2026-04-18",
        "title": "UI: Botón ON/OFF del bot en header (kill switch toggle)",
        "changes": [
            "NEW botón en header junto a Reset: 🟢 ON / 🔴 OFF según estado kill switch",
            "Click con confirm() dialog → POST /api/safety/kill-switch (activate/deactivate)",
            "Auto-sync vía polling cada 10s (hook en refreshAll) — si kill switch se activa por otro canal (cron emergency_close, DAILY_LOSS trigger, reconcile critical), el botón se pinta rojo solo",
            "Reconciliación visual: si falla request, refreshBotStatus restaura UI al estado real del server",
            "Reusa endpoints safety.py del Sprint 1 — zero cambio al motor de trading",
            "SL/TP de posiciones abiertas NO se tocan al apagar — solo pausa aperturas nuevas",
            "Ambos dashboards (paper 8081 + live 8082) ganan el botón",
        ]
    },
    {
        "version": "2.10.0-live",
        "date": "2026-04-18",
        "title": "Sync master v2.10.0 recovery + Sprint 1 Safety Nets en live (4-layer defense)",
        "changes": [
            "MERGE: todos los 12 commits del recovery paper (v2.9.0 → v2.10.0) en branch live",
            "Defense-in-depth 4 capas para aperturas: TRADE_WHITELIST_STRICT (hardcoded) + safety.is_whitelisted (env) + MIN_SL 2% + MAX_NOTIONAL 50%",
            "Capa 1: TRADE_WHITELIST_STRICT bloquea memecoins sí o sí (hardcoded en executor.py)",
            "Capa 2: safety.is_whitelisted permite restricción adicional vía TRADE_WHITELIST env var (opcional)",
            "Claude OAuth Bearer fix heredado — el live puede usar Claude Max sin 401",
            "Accounting A-E fixes + near-target guard + cache LLM selectivo + session-expired guards",
            "Live-specific preservado: agents/safety.py + run_watchdog_live.sh + endpoints /api/safety/*",
            "Snapshot preservado: branch snapshot-live-pre-merge-2026-04-18",
        ]
    },
    {
        "version": "2.10.0",
        "date": "2026-04-18",
        "title": "Recovery post-desastre -$57: whitelist estricta + SL mínimo + Claude OAuth fix",
        "changes": [
            "CRITICAL TRADE_WHITELIST_STRICT={SOL,BTC,ETH,XRP,JUP}: memecoins excluidos (causaron -$57.27 hoy)",
            "CRITICAL MIN_SL_DISTANCE_PCT=2.0: rechaza SL a <2%% del entry (MOODENG tenia 0.38%% -> -33%%)",
            "SAFETY MAX_NOTIONAL_PCT_EQUITY=0.50: cap notional a 50%% del equity (MOODENG era 160%%)",
            "FIX Claude OAuth: lee ~/.claude/.credentials.json con auto-refresh (resuelve 401 cascada)",
            "FIX Claude auth header: OAuth tokens (sk-ant-oat01-) ahora usan Bearer (no x-api-key)",
            "INCLUYE v2.9.1 filter LLM errors + v2.9.5 cache LLM selectivo + v2.9.6 B2 near-target guard",
            "INCLUYE accounting fixes A-E (fee apportionment, slippage, emergency close audit fields)",
            "EXCLUYE v2.9.2+3 (Claude Sonnet primary via x-api-key) -- reemplazado por Fix OAuth Bearer",
            "Snapshot pre-recovery: branch snapshot-disaster-2026-04-18 (commit 45009a4) preservado",
            "Replay desde v2.9.0 (commit 4969ebf); tests verificados: 28 passed, 10 pre-existing fails",
        ]
    },
    {
        "version": "2.9.6",
        "date": "2026-04-18",
        "title": "B2: no abrir posiciones nuevas cerca del daily target (>=80%)",
        "changes": [
            "NEW near-target guard en executor.py: skip nuevas aperturas si daily_pnl >= 80%% del TARGET_MAX_PCT",
            "Previene el patron 2026-04-18: bot abrio 3 posiciones con daily_pnl ya ~4% -> force-close 1.6h despues en perdida",
            "Lee current_pnl_pct + target_pct de daily_target_state.json (v2.9.0 auto-reset activo)",
            "Fallback: si daily_target_state.json falta o esta corrupto, comportamiento v2.9.5 (abrir normalmente)",
        ]
    },
    {
        "version": "2.9.5",
        "date": "2026-04-18",
        "title": "Fix regresion v2.6.0 cache LLM + equity history extendido",
        "changes": [
            "FIX _should_call_llm_wild: ahora SIEMPRE llama al LLM si alguna chain esta en loss (anti session_expired)",
            "Cache v2.6.0 solo aplica a chains ganadoras/neutrales -- previene las perdidas de $-75.91 del 2026-04-17",
            "FIX equity_history.json: cap 500 puntos (~12.5h) -> 5000 con downsample automatico (~30 dias de historia)",
            "Contexto: auditoria revelo que cache agresivo de v2.6.0 dejaba chains perdedoras sin supervision por 10 min, acumulando hasta WILD_ABANDON_session_expired_181m",
            "Backup en capas: v2.9.0 CHAIN_MAX_LOSS_PCT 15% y pre-expiry loss guard siguen activos",
        ]
    },
    {
        "version": "2.9.0-live",
        "date": "2026-04-17",
        "title": "Sprint 1 Safety Nets en branch live: kill switch + daily loss + whitelist + startup validation",
        "changes": [
            "NEW agents/safety.py: modulo centralizado fail-safe (todos los hooks con try/except)",
            "NEW kill switch: /tmp/solana_live_killswitch -- orchestrator skipea ciclo si existe",
            "NEW MAX_DAILY_LOSS_USD: si se alcanza, activa kill switch automatico",
            "NEW TRADE_WHITELIST env var: filtro de tokens en signal loop (vacio = allow all = paper behavior)",
            "NEW validate_startup(): chequea kill switch inactivo + si LIVE_TRADING_ENABLED=true exige wallet/RPC/limites configurados",
            "NEW MAX_SLIPPAGE_BPS: cutoff para swaps reales (aun no usado; se activa en Sprint 2 con real_open_position)",
            "NUEVOS endpoints: GET /api/safety/status, POST /api/safety/kill-switch",
            "Retrocompatible: sin env vars, comportamiento identico al paper. Safe merge back to master.",
        ]
    },
    {
        "version": "2.9.0",
        "date": "2026-04-17",
        "title": "WILD session-expired guards + daily_target auto-reset",
        "changes": [
            "NEW CHAIN_MAX_LOSS_PCT=15%: auto-close chain cuando pérdida ≥15% del base margin (runaway cutoff)",
            "NEW pre-expiry loss guard: cierra chains en pérdida al 75%% del timeout (135min) vs esperar al abandono a 180min",
            "FIX daily_target_state.json auto-reset cuando la fecha cambia (antes podia quedar stale)",
            "Contexto: hoy el bot perdio -$38.20 por 5 WILD_ABANDON_session_expired; con estas guardas habria ahorrado ~$20-25",
            "Ambas guardas reusan validate_decision + apply_decision del profit-target loop (mismo path probado)",
        ]
    },
    {
        "version": "2.8.1",
        "date": "2026-04-17",
        "title": "Hotfix: cambio de idioma fuerza regeneracion LLM en WILD mode",
        "changes": [
            "FIX: al cambiar idioma, se invalida _last_llm_ts en wild_mode_state.json",
            "El proximo ciclo WILD (~10s a 2min) regenera el reasoning en el nuevo idioma",
            "Antes: el tag (EN)/(ES) permanecia hasta 10min por el cache v2.6.0 sin movimiento de precio",
            "NUEVO endpoint: POST /api/force-reasoning-refresh -- invocado por toggleLang() automaticamente",
        ]
    },
    {
        "version": "2.8.0",
        "date": "2026-04-17",
        "title": "LLM razona en ingles cuando idioma = EN (Fase 2 i18n)",
        "changes": [
            "NEW: agents/lang_utils.py con get_user_language() + lang_directive()",
            "NEW: martingale_engine y risk_manager apendizan directiva EN al prompt cuando user_profile.json language=en -- el LLM devuelve reasoning traducido",
            "NEW: reasoning_lang persistido en wild_mode_state.json y position_decisions.json para saber en que idioma fue generado cada texto",
            "FIX: dashboard muestra tag (ES)/(EN) SOLO si hay desajuste real entre idioma del texto y idioma seleccionado -- tag desaparece cuando el ciclo regenera en idioma correcto",
            "FAIL-SAFE: si lang_utils falla o user_profile.json esta ausente, bot opera normal en espanol (cero regresion)",
            "NOTA: razonamientos persistidos antes del cambio de idioma muestran su tag original hasta que el ciclo WILD regenera (~2min)",
        ]
    },
    {
        "version": "2.7.0",
        "date": "2026-04-16",
        "title": "i18n completo: traducción ES/EN de modal AI Thinking + UI dinámica",
        "changes": [
            "FIX: modal 'Que esta pensando la IA' ahora traduce 100% al ingles (titulo, KPIs, badges, labels, fallbacks)",
            "NEW: 25 keys nuevas en TRANSLATIONS (ES+EN) para modal AI Thinking y strings dinamicos",
            "NEW: helper T(key) en JS -- reemplaza TRANSLATIONS[currentLang].key con fallback seguro",
            "FIX: _actionBadge() traduce HOLD/OPEN_LEVEL/CLOSE_CHAIN/CLOSE/ABANDON segun idioma",
            "FIX: renderAIThinking() usa templates con {n}/{ts}/{c} para interpolar datos en strings traducidos",
            "NEW: applyLang() re-renderiza modal AI Thinking si esta abierto -- cambio de idioma en vivo",
            "FIX: botones 'Cerrar' en tabla posiciones, 'Sin trades que mostrar', 'Sin log disponible' ahora se traducen",
            "NOTA: el razonamiento del LLM (p.llm_reasoning) queda en ES -- marcado con '(ES)' en modo EN como Fase 2",
        ]
    },
    {
        "version": "2.6.0",
        "date": "2026-04-15",
        "title": "Eficiencia LLM: cache inteligente -94% llamadas",
        "changes": [
            "NEW: _should_call_llm_wild() en martingale_engine -- skip LLM si precio no movio >0.3% y <10min desde ultima llamada",
            "NEW: _strategy_needs_refresh() en orchestrator -- reemplaza 'if True:' con cache basado en F&G + precio + vela 30m",
            "AI Strategy: de ~1440 llamadas/dia a ~96 (cache 15min, refresh en vela 30m o movimiento >1%)",
            "WILD mode: de ~1440 llamadas/dia a ~72 (cache 10min, override si precio movio o cadena escalada)",
            "Reduccion total estimada: 94% (de ~2880 a ~168 llamadas/dia) sin perdida de calidad",
            "Env vars de tuning: WILD_LLM_CACHE_SEC, WILD_LLM_PRICE_THRESHOLD, STRATEGY_CACHE_SEC",
        ]
    },
    {
        "version": "2.5.0",
        "date": "2026-04-16",
        "title": "Resiliencia LLM: Qwen local + guardia anti-abandono",
        "changes": [
            "NEW: call_qwen_local() -- Qwen 2.5 14B via Ollama como 4to fallback local (sin internet, sin coste)",
            "Cadena LLM: GPT-5.4 -> MiniMax M2.7 -> Claude Sonnet 4.6 -> Qwen 2.5 14B local",
            "NEW: _llm_is_down() en martingale_engine -- detecta circuit breaker activo",
            "NEW: LLM_DOWN_GUARD en run_cycle() -- cierre preventivo a 75% del timeout si LLM caido",
            "Elimina causa raiz de WILD_ABANDON_session_expired cuando LLM no responde",
            "QWEN_URL / QWEN_MODEL configurables via .env para cualquier modelo Ollama",
        ]
    },
    {
        "version": "2.4.0",
        "date": "2026-04-15",
        "title": "Portabilidad multi-instancia",
        "changes": [
            "FIX: llm_config.py — MINIMAX_API_KEY ahora lee de env var primero (portable), fallback a archivo legacy (ROG sin cambios)",
            "FIX: chat_agent.py — minimax key lee env var antes que archivo; LOG_FILE usa Path.home() (funciona en Mac/Linux)",
            "FIX: reporter.py — ruta minimax.json para TTS usa Path.home() en lugar de /home/enderj/",
            "FIX: run_watchdog.sh — cd usa dirname BASH_SOURCE (portable); logs usan $HOME en vez de /home/enderj/",
            "NEW: .env.example completo con MINIMAX_API_KEY, GPT5_URL, PAPERCLIP, OPENCLAW_WEBHOOK documentados",
            "NUEVO colaborador puede arrancar con: cp .env.example .env + pip install -r requirements.txt + bash run_watchdog.sh",
        ]
    },
    {
        "version": "2.3.0",
        "date": "2026-04-15",
        "title": "Auditoría post-deploy: 5 fixes críticos",
        "changes": [
            "FIX: MAX_SESSION_MINUTES 360 → 180 (martingale_engine.py) — reduce pérdidas por session_expired de -$35.23 histórico",
            "FIX: equity_history.json usaba capital_usd ($299) en vez de total_value ($490) — gráfico mostraba falsa caída",
            "FIX: orchestrator.py — eliminado _dt = __import__('datetime').datetime (línea no usada causaba WARNING cada ciclo)",
            "FIX: LLM chain ampliada: GPT-5.4 → MiniMax M2.7 → Claude Sonnet 4.6 (tertiary) — cero cierres sin LLM",
            "FIX: Proceso zombie dashboard PID 178985 consumía 92.8% CPU desde Apr 14 — eliminado",
        ]
    },
    {
        "version": "2.2.0",
        "date": "2026-04-15",
        "title": "OpenClaw webhooks + plugins Fase 1 & 2",
        "changes": [
            "Nuevo: agents/openclaw_webhook.py — notificaciones Telegram en tiempo real via OpenClaw",
            "Integración: executor.py notifica trade abierto/cerrado via webhook",
            "Integración: martingale_engine.py notifica cadena WILD cerrada y nuevo nivel",
            "OpenClaw 2026.4.5 → 2026.4.14 actualizado en PC ROG",
            "Plugins Fase 1 activados: active-memory, diffs, llm-task",
            "Plugins Fase 2 activados: webhooks con rutas solana-alerts y github-solana",
            "nginx en ROG puerto 18790 como proxy externo para webhooks",
            "Fix: conflicto system/user systemd en openclaw-gateway (--force + KillMode=control-group)",
        ]
    },
    {
        "version": "2.1.0",
        "date": "2026-04-14",
        "title": "Auditoría de seguridad & fixes críticos",
        "changes": [
            "FIX CRÍTICO: executor.py — 4 rutas de early-return (kill switch, daily target, max loss) perdían trades en trade_history.json al actualizar solo portfolio.json",
            "FIX: Escrituras atómicas en save_portfolio() y save_history() usando atomic_write_json() — previene corrupción de archivos en crash",
            "FIX: reporter.py + performance_tracker.py — filtro 'status==closed' retornaba lista vacía; avg_loss_usd mostraba 0 con pérdidas reales",
            "FIX: equity_history.json sin actualizar desde Apr 3 — ahora se actualiza en cada ciclo del reporter",
            "FIX: token_scanner.py — Jupiter DNS falla sin fallback; ahora usa caché local válido hasta 6h",
            "Dashboard: solana-dashboard.service tenía 2 procesos huérfanos usando 41% CPU en puerto 8081 — limpiados",
        ]
    },
    {
        "version": "2.0.5",
        "date": "2026-04-12",
        "title": "Wild Mode & template LLM fix",
        "changes": [
            "FIX: Bot sin operaciones — template LLM bloqueaba señales técnicas",
            "FIX: Wild Mode margin starvation + filtro Extreme Fear",
            "Ajuste: dashboard estilos ait-wild-inline y tarjeta Motor Martingala",
        ]
    },
    {
        "version": "2.0.4",
        "date": "2026-04-08",
        "title": "Hardening runtime & watchdog",
        "changes": [
            "FIX: Invertir lógica shorts cuando RSI < 30 en tendencia bajista",
            "FIX: Filtro anti-reversión SHORT (RSI<30 + tendencia)",
            "FIX: Watchdog ownership split entre systemd y run_watchdog manual",
            "FIX: Bot no ejecuta trades después del restart",
            "Hardening: single orchestrator lock + canonical state path",
        ]
    },
    {
        "version": "2.0.3",
        "date": "2026-04-07",
        "title": "Pérdidas constantes — fix shorts",
        "changes": [
            "Investigación de pérdidas sistemáticas en posiciones short",
            "Ajuste de parámetros de entrada y salida para shorts",
        ]
    },
    {
        "version": "2.0.0",
        "date": "2026-04-03",
        "title": "Wild Mode + Martingale Engine",
        "changes": [
            "Nuevo: Wild Mode con engine Martingala para maximizar ganancias en tendencia",
            "Nuevo: Análisis LLM combinado con señales técnicas (stoch_rsi, rsi_bb, golden_cross, etc.)",
            "Nuevo: Auto-Learner v2.0 — parámetros adaptativos basados en historial",
            "Nuevo: 6 estrategias activas en modo COMBO",
            "Dashboard: modal AI Thinking en tiempo real",
            "Dashboard: barra P&L reactiva en historial de trades",
        ]
    },
    {
        "version": "1.0.0",
        "date": "2026-03-20",
        "title": "v1.0-rentable — primera versión rentable confirmada",
        "changes": [
            "Primera versión con profit factor > 1 confirmado en paper trading",
            "Pipeline completo: Market Data → Risk → Strategy → Executor → Reporter",
            "Integración con Paperclip AI (CEO, CTO, Engineer, Analyst)",
            "Dashboard web en puerto 8081",
            "Modo paper_drift (Drift Protocol simulado con 5x leverage)",
        ]
    },
]

# ── HTML template (single-file SPA) ──────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Solana Cripto Trader — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #1c2128;
    --border: #30363d;
    --text: #e6edf3;
    --text2: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --orange: #ff7b72;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  a { color: var(--blue); text-decoration: none; }

  /* ── Header ── */
  .header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  .header-left { display: flex; align-items: center; gap: 16px; }
  .logo { font-size: 18px; font-weight: 700; color: var(--blue); letter-spacing: -0.5px; }
  .logo span { color: var(--purple); }
  .status-badge {
    display: flex; align-items: center; gap: 6px;
    background: var(--bg3); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
  .dot-red { background: var(--red); }
  .dot-yellow { background: var(--yellow); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

  .header-right { display: flex; align-items: center; gap: 12px; }
  .capital-header { font-size: 13px; color: var(--text2); }
  .capital-header .val { color: var(--text); font-weight: 600; font-size: 15px; }
  .capital-header .chg { font-size: 12px; margin-left: 4px; }
  .last-update { font-size: 11px; color: var(--text2); }
  .btn-refresh {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--blue); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 12px; transition: all .2s;
  }
  .btn-refresh:hover { background: var(--blue); color: var(--bg); }
  .btn-reset {
    background: var(--bg3); border: 1px solid var(--orange);
    color: var(--orange); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 12px; transition: all .2s; margin-left: 8px;
  }
  .btn-reset:hover { background: var(--orange); color: var(--bg); }
  /* v2.10.0-live: bot ON/OFF toggle (kill switch) */
  .btn-bot {
    border: 1px solid var(--border);
    padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 12px;
    transition: all .2s; margin-left: 8px;
    font-weight: 600;
    min-width: 72px;
    background: var(--bg3);
    color: var(--text2);
  }
  .btn-bot.on  { border-color: #3fb950; color: #3fb950; }
  .btn-bot.on:hover  { background: #3fb950; color: var(--bg); }
  .btn-bot.off { border-color: #f85149; color: #f85149; }
  .btn-bot.off:hover { background: #f85149; color: #fff; }
  .btn-bot.loading { opacity: 0.5; cursor: wait; }
  .btn-bot:disabled { opacity: 0.4; cursor: not-allowed; }
  /* v2.11.0-live: SOL fuel badge */
  .fuel-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 10px; margin-left: 8px;
    border-radius: 6px; font-size: 11px; font-weight: 600;
    border: 1px solid var(--border); background: var(--bg3);
    color: var(--text2); font-family: ui-monospace, monospace;
    cursor: default; user-select: none;
  }
  .fuel-badge.ok     { border-color: #3fb950; color: #3fb950; }
  .fuel-badge.warn   { border-color: #d29922; color: #d29922; }
  .fuel-badge.low    { border-color: #f85149; color: #f85149; }
  .fuel-badge.off    { opacity: 0.5; }
  /* v2.12.0-live: Drift perps badge */
  .drift-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 10px; margin-left: 8px;
    border-radius: 6px; font-size: 11px; font-weight: 600;
    border: 1px solid var(--border); background: var(--bg3);
    color: var(--text2); font-family: ui-monospace, monospace;
    cursor: default; user-select: none;
  }
  .drift-badge.active  { border-color: #8957e5; color: #d2a8ff; }
  .drift-badge.long    { border-color: #3fb950; color: #3fb950; }
  .drift-badge.short   { border-color: #f85149; color: #f85149; }
  .drift-badge.dormant { opacity: 0.5; }
  .drift-badge.err     { border-color: #d29922; color: #d29922; }

  /* v2.12.22 health badge */
  .health-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 12px; font-size: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border); font-weight: 500;
  }
  .health-dot { font-size: 14px; line-height: 1; transition: color 0.3s; }
  .health-dot.healthy { color: #10b981; }
  .health-dot.unhealthy { color: #ef4444; animation: health-pulse 1s infinite; }
  .health-dot.unknown { color: #6b7280; }
  @keyframes health-pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }
  .mode-badge {
    font-size: 11px; padding: 2px 8px; border-radius: 10px;
    background: rgba(188,140,255,0.15); color: var(--purple); border: 1px solid var(--purple);
  }

  /* ── Layout ── */
  .container { max-width: 1600px; margin: 0 auto; padding: 20px 24px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .grid-7 { display: grid; grid-template-columns: repeat(7, 1fr); gap: 12px; }
  .card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }
  .card-title {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text2); margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
  }
  .card-title .icon { font-size: 14px; }
  section { margin-bottom: 20px; }
  .section-title {
    font-size: 13px; font-weight: 600; color: var(--text2);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
  }

  /* ── KPI Cards ── */
  .kpi-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .kpi-label { font-size: 11px; color: var(--text2); font-weight: 500; letter-spacing: 0.3px; }
  .kpi-value { font-size: 22px; font-weight: 700; line-height: 1.2; }
  .kpi-sub { font-size: 11px; color: var(--text2); }
  .kpi-icon { font-size: 20px; align-self: flex-end; opacity: 0.6; margin-top: -8px; }
  .kpi-card.positive { border-left: 3px solid var(--green); }
  .kpi-card.negative { border-left: 3px solid var(--red); }
  .kpi-card.neutral { border-left: 3px solid var(--blue); }
  .kpi-card.warning { border-left: 3px solid var(--yellow); }

  /* Colors */
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .neu { color: var(--blue); }
  .warn { color: var(--yellow); }

  /* Real-time update animations */
  @keyframes flashGreen {
    0% { background-color: rgba(63, 185, 80, 0.3); }
    100% { background-color: transparent; }
  }
  @keyframes flashRed {
    0% { background-color: rgba(248, 81, 73, 0.3); }
    100% { background-color: transparent; }
  }
  .flash-update { animation: flashGreen 0.5s ease-out; }
  .price-cell { 
    font-family: 'Monaco', 'Consolas', monospace; 
    font-weight: 600;
    transition: color 0.3s ease;
  }
  .price-up { color: var(--green) !important; }
  .price-down { color: var(--red) !important; }

  /* ── Charts ── */
  .chart-container { position: relative; width: 100%; }
  .chart-sm { height: 200px; }
  .chart-md { height: 260px; }
  .chart-lg { height: 340px; }
  canvas { max-width: 100%; }

  /* ── Tables ── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead th {
    text-align: left; padding: 8px 10px; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2);
    border-bottom: 1px solid var(--border); white-space: nowrap;
  }
  tbody tr { border-bottom: 1px solid rgba(48,54,61,0.5); transition: background .15s; }
  tbody tr:hover { background: rgba(88,166,255,0.04); }
  tbody td { padding: 8px 10px; white-space: nowrap; }
  .chain-sub-row { background: rgba(188,140,255,0.05); border-left: 3px solid var(--purple); }
  .chain-sub-row td:first-child { padding-left: 24px; }
  .chain-sub-row:hover { background: rgba(188,140,255,0.1); }
  .chain-toggle { background: none; border: 1px solid var(--purple); color: var(--purple); border-radius: 4px; padding: 1px 5px; font-size: 10px; cursor: pointer; margin-right: 4px; transition: all .15s; }
  .chain-toggle:hover { background: var(--purple); color: var(--bg); }
  /* ── Reason badge (clickable) ── */
  .reason-btn { background: none; border: 1px solid var(--border); color: var(--text2); border-radius: 4px; padding: 2px 7px; font-size: 10px; cursor: pointer; transition: all .15s; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; }
  .reason-btn:hover { border-color: var(--blue); color: var(--blue); }
  /* ── AI Reasoning Modal ── */
  #reasonModal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 9999; align-items: center; justify-content: center; }
  #reasonModal.open { display: flex; }
  .reason-modal-box { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; max-width: 580px; width: 90%; max-height: 80vh; overflow-y: auto; position: relative; }
  .reason-modal-title { font-size: 13px; font-weight: 700; color: var(--blue); margin-bottom: 6px; }
  .reason-modal-sym { font-size: 11px; color: var(--text2); margin-bottom: 14px; }
  .reason-modal-type { display: inline-block; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 11px; color: var(--yellow); margin-bottom: 14px; }
  .reason-modal-body { font-size: 13px; line-height: 1.7; color: var(--text); white-space: pre-wrap; border-left: 3px solid var(--blue); padding-left: 12px; }
  .reason-modal-close { position: absolute; top: 14px; right: 16px; background: none; border: none; color: var(--text2); font-size: 18px; cursor: pointer; }
  .reason-modal-close:hover { color: var(--text); }
  /* ── Reset Modal ── */
  #resetModal{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:9999;align-items:center;justify-content:center;}
  #resetModal.open{display:flex;}
  .reset-modal-box{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:28px;max-width:520px;width:92%;max-height:90vh;overflow-y:auto;position:relative;}
  .reset-modal-title{font-size:14px;font-weight:700;color:var(--orange);margin-bottom:20px;}
  .reset-modal-field{margin-bottom:16px;}
  .reset-modal-field label{display:block;font-size:11px;color:var(--text2);margin-bottom:5px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}
  .reset-modal-field input[type=number],.reset-modal-field textarea{width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:8px 10px;font-size:13px;box-sizing:border-box;}
  .reset-modal-field textarea{resize:vertical;min-height:80px;font-family:inherit;}
  .reset-drop-zone{border:2px dashed var(--border);border-radius:8px;padding:20px;text-align:center;cursor:pointer;color:var(--text2);font-size:12px;transition:border-color 0.2s;}
  .reset-drop-zone.drag-over{border-color:var(--orange);color:var(--orange);}
  .reset-drop-zone input[type=file]{display:none;}
  .reset-file-list{margin-top:10px;display:flex;flex-direction:column;gap:4px;}
  .reset-file-item{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text2);background:var(--bg3);border-radius:4px;padding:4px 8px;}
  .reset-file-item button{background:none;border:none;color:var(--red);cursor:pointer;font-size:14px;line-height:1;padding:0 2px;}
  .reset-modal-actions{display:flex;gap:10px;justify-content:flex-end;margin-top:20px;}
  .reset-modal-actions button{padding:8px 20px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;}
  .reset-btn-cancel{background:var(--bg3);color:var(--text2);}
  .reset-btn-confirm{background:var(--orange);color:#000;}
  /* ── Version Badge ── */
  /* v2.10.0-live: banner visual para distinguir del paper */
  .live-mode-banner {
    position: fixed; top: 0; left: 0; right: 0;
    background: linear-gradient(90deg, #b91c1c, #dc2626, #b91c1c);
    color: white; text-align: center; padding: 8px 16px;
    font-weight: 700; font-size: 13px; letter-spacing: 2px;
    z-index: 9999; box-shadow: 0 2px 8px rgba(0,0,0,.4);
    display: none;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  .live-mode-banner.active { display: block; }
  body.live-mode { padding-top: 34px; }
  .version-badge.is-live {
    background: #dc2626 !important;
    color: white !important;
    border-color: #ef4444 !important;
    font-weight: 700;
  }

  .version-badge {
    display: inline-block; font-size: 10px; font-weight: 700;
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--purple); border-radius: 20px; padding: 2px 9px;
    cursor: pointer; letter-spacing: 0.5px; transition: border-color 0.2s, color 0.2s;
    vertical-align: middle; margin-left: 8px;
  }
  .version-badge:hover { border-color: var(--purple); color: #d4b8ff; }
  /* ── Changelog Modal ── */
  #changelogModal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75); z-index: 9999; align-items: center; justify-content: center; }
  #changelogModal.open { display: flex; }
  .changelog-box { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 24px; max-width: 640px; width: 94%; max-height: 82vh; overflow-y: auto; position: relative; }
  .changelog-title { font-size: 15px; font-weight: 700; color: var(--purple); margin-bottom: 18px; }
  .changelog-close { position: absolute; top: 14px; right: 16px; background: none; border: none; color: var(--text2); font-size: 18px; cursor: pointer; }
  .changelog-close:hover { color: var(--text); }
  .changelog-version { margin-bottom: 20px; }
  .changelog-ver-header { display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; }
  .changelog-ver-tag { font-size: 12px; font-weight: 700; color: var(--purple); background: rgba(188,140,255,0.12); border: 1px solid rgba(188,140,255,0.3); border-radius: 20px; padding: 2px 10px; }
  .changelog-ver-date { font-size: 11px; color: var(--text2); }
  .changelog-ver-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 6px; }
  .changelog-ver-list { list-style: none; padding: 0; }
  .changelog-ver-list li { font-size: 12px; color: var(--text2); padding: 3px 0 3px 16px; position: relative; line-height: 1.5; }
  .changelog-ver-list li::before { content: "▸"; position: absolute; left: 0; color: var(--purple); opacity: 0.6; }
  .changelog-ver-divider { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
  .reset-btn-confirm:disabled{opacity:0.4;cursor:not-allowed;}

  /* ── AI Thinking Modal ── */
  #aiThinkingModal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.85); z-index:10000; align-items:flex-start; justify-content:center; overflow-y:auto; padding:20px 0; }
  #aiThinkingModal.open { display:flex; }
  .ait-box { background:var(--bg); border:1px solid #30363d; border-radius:14px; width:min(900px,96vw); padding:0; overflow:hidden; position:relative; }
  /* header */
  .ait-header { background:linear-gradient(135deg,rgba(188,140,255,0.15),rgba(88,166,255,0.1)); border-bottom:1px solid #30363d; padding:18px 24px; display:flex; align-items:center; justify-content:space-between; }
  .ait-header-left { display:flex; align-items:center; gap:12px; }
  .ait-brain { font-size:28px; animation:ait-pulse 2s infinite; }
  @keyframes ait-pulse { 0%,100%{transform:scale(1);filter:brightness(1)} 50%{transform:scale(1.1);filter:brightness(1.3)} }
  .ait-title { font-size:16px; font-weight:700; background:linear-gradient(90deg,#bc8cff,#58a6ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .ait-subtitle { font-size:11px; color:var(--text2); margin-top:2px; }
  .ait-live { display:flex; align-items:center; gap:6px; font-size:11px; font-weight:700; color:#3fb950; }
  .ait-live-dot { width:8px; height:8px; border-radius:50%; background:#3fb950; box-shadow:0 0 8px #3fb950; animation:pulse 1.5s infinite; }
  .ait-close-btn { background:none; border:1px solid #30363d; color:var(--text2); border-radius:6px; padding:4px 10px; cursor:pointer; font-size:14px; }
  .ait-close-btn:hover { color:var(--text); border-color:var(--text2); }
  /* refresh bar */
  .ait-refresh-bar { height:2px; background:var(--bg3); position:relative; overflow:hidden; }
  .ait-refresh-fill { height:100%; background:linear-gradient(90deg,#bc8cff,#58a6ff); transition:width 1s linear; }
  /* global context */
  .ait-global { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:#30363d; border-bottom:1px solid #30363d; }
  .ait-kpi { background:var(--bg2); padding:12px 16px; text-align:center; }
  .ait-kpi-val { font-size:18px; font-weight:700; }
  .ait-kpi-lbl { font-size:10px; color:var(--text2); margin-top:2px; text-transform:uppercase; letter-spacing:0.5px; }
  /* body */
  .ait-body { padding:20px 24px; display:flex; flex-direction:column; gap:16px; }
  /* position card */
  .ait-pos-card { background:var(--bg2); border:1px solid #30363d; border-radius:10px; overflow:hidden; transition:border-color .2s; }
  .ait-pos-card:hover { border-color:#bc8cff44; }
  .ait-pos-header { display:flex; align-items:center; gap:10px; padding:12px 16px; border-bottom:1px solid #30363d; }
  .ait-sym { font-size:16px; font-weight:700; }
  .ait-pos-pnl { margin-left:auto; font-size:15px; font-weight:700; }
  .ait-decision-badge { font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px; letter-spacing:0.5px; }
  .ait-badge-hold { background:rgba(210,153,34,0.2); color:#d29922; border:1px solid #d2992244; }
  .ait-badge-open { background:rgba(63,185,80,0.2); color:#3fb950; border:1px solid #3fb95044; }
  .ait-badge-close { background:rgba(88,166,255,0.2); color:#58a6ff; border:1px solid #58a6ff44; }
  .ait-badge-abandon { background:rgba(248,81,73,0.2); color:#f85149; border:1px solid #f8514944; }
  .ait-pos-body { padding:12px 16px; display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  /* alignment */
  .ait-align { font-size:11px; padding:6px 10px; border-radius:6px; background:var(--bg3); }
  .ait-align-split { border-left:3px solid #d29922; }
  .ait-align-agree { border-left:3px solid #3fb950; }
  /* confidence bar */
  .ait-conf-wrap { display:flex; align-items:center; gap:8px; }
  .ait-conf-track { flex:1; height:6px; background:var(--bg3); border-radius:3px; overflow:hidden; }
  .ait-conf-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,#bc8cff,#58a6ff); transition:width .5s ease; }
  .ait-conf-lbl { font-size:10px; color:var(--text2); width:30px; text-align:right; }
  /* quant reasons */
  .ait-reasons { display:flex; flex-wrap:wrap; gap:4px; }
  .ait-reason-chip { font-size:10px; padding:2px 7px; border-radius:10px; background:var(--bg3); color:var(--text2); border:1px solid #30363d; }
  .ait-reason-chip.good { color:#3fb950; border-color:#3fb95033; background:rgba(63,185,80,0.08); }
  .ait-reason-chip.bad  { color:#f85149; border-color:#f8514933; background:rgba(248,81,73,0.08); }
  /* LLM reasoning block */
  .ait-llm-block { grid-column:1/-1; background:rgba(88,166,255,0.04); border:1px solid rgba(88,166,255,0.15); border-radius:8px; padding:12px 14px; }
  .ait-llm-title { font-size:10px; font-weight:700; color:var(--blue); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; display:flex; align-items:center; gap:6px; }
  /* v2.12.15: LLM eval metadata */
  .ait-eval-meta { font-size:11px; color:var(--text2); margin-bottom:8px; padding:6px 10px; background:var(--bg3); border-radius:6px; border-left:2px solid #8957e5; }
  .ait-eval-meta .ait-stale { color:#d29922; font-weight:600; }
  .ait-eval-meta .ait-drift { color:#f0883e; font-weight:600; }

  .ait-llm-text { font-size:12px; color:var(--text); line-height:1.7; white-space:pre-wrap; }
  /* guardrails */
  .ait-guardrail { font-size:11px; color:#f85149; background:rgba(248,81,73,0.08); border:1px solid rgba(248,81,73,0.2); border-radius:6px; padding:4px 10px; display:flex; align-items:center; gap:6px; }
  /* metrics row */
  .ait-metrics { display:flex; gap:16px; flex-wrap:wrap; }
  .ait-metric { font-size:11px; color:var(--text2); }
  .ait-metric span { color:var(--text); font-weight:600; }
  /* wild mode section */
  .ait-wild-header { display:flex; align-items:center; gap:8px; font-size:12px; font-weight:700; color:var(--orange); margin-bottom:10px; }
  .ait-wild-chain { background:rgba(255,123,114,0.06); border:1px solid rgba(255,123,114,0.2); border-radius:8px; padding:10px 14px; margin-bottom:8px; }
  /* loading state */
  .ait-loading { padding:60px 24px; text-align:center; color:var(--text2); }
  .ait-loading-spinner { font-size:32px; animation:ait-pulse 1s infinite; margin-bottom:12px; }
  /* distance bars */
  .ait-dist-row { display:flex; align-items:center; gap:6px; font-size:10px; color:var(--text2); }
  .ait-dist-bar { flex:1; height:4px; background:var(--bg3); border-radius:2px; overflow:hidden; position:relative; }
  .ait-dist-sl { height:100%; background:#f85149; border-radius:2px; }
  .ait-dist-tp { height:100%; background:#3fb950; border-radius:2px; position:absolute; right:0; top:0; }
  .badge {
    display: inline-block; padding: 2px 7px; border-radius: 4px;
    font-size: 10px; font-weight: 600; text-transform: uppercase;
  }
  .badge-long { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge-short { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge-win { background: rgba(63,185,80,0.12); color: var(--green); }
  .badge-loss { background: rgba(248,81,73,0.12); color: var(--red); }
  .badge-hold { background: rgba(88,166,255,0.12); color: var(--blue); }

  /* ── Pagination ── */
  .pagination { display: flex; align-items: center; gap: 8px; margin-top: 12px; justify-content: flex-end; }
  .page-btn {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text2); padding: 4px 10px; border-radius: 4px;
    cursor: pointer; font-size: 11px;
  }
  .page-btn:hover, .page-btn.active { background: var(--blue); color: var(--bg); border-color: var(--blue); }
  .page-info { font-size: 11px; color: var(--text2); }

  /* ── Filters ── */
  .filters { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
  select, input {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text); padding: 5px 10px; border-radius: 6px;
    font-size: 12px; outline: none;
  }
  select:focus, input:focus { border-color: var(--blue); }

  /* ── Log ── */
  .log-container {
    background: #0a0e14; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; font-family: 'Consolas', monospace;
    font-size: 11px; max-height: 280px; overflow-y: auto;
  }
  .log-line { padding: 2px 0; line-height: 1.6; }
  .log-ok { color: var(--green); }
  .log-err { color: var(--red); }
  .log-warn { color: var(--yellow); }
  .log-info { color: var(--text2); }
  .log-auto { font-size: 10px; color: var(--text2); margin-top: 8px; }

  /* ── Advanced Metrics ── */
  .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(48,54,61,0.5); }
  .metric-row:last-child { border-bottom: none; }
  .metric-label { color: var(--text2); font-size: 12px; }
  .metric-value { font-weight: 600; font-size: 13px; }

  /* ── Equity toolbar ── */
  .chart-toolbar { display: flex; gap: 6px; margin-bottom: 8px; align-items: center; }
  .tb-btn {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text2); padding: 3px 9px; border-radius: 4px;
    cursor: pointer; font-size: 11px;
  }
  .tb-btn:hover, .tb-btn.active { background: var(--blue); color: var(--bg); border-color: var(--blue); }
  .tb-sep { flex: 1; }

  /* ── Discrepancy Warning ── */
  .discrepancy-banner {
    background: rgba(210,153,34,0.08); border: 1px solid rgba(210,153,34,0.4);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;
  }
  .discrepancy-banner .disc-title {
    font-size: 12px; font-weight: 700; color: var(--yellow); margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
  }
  .discrepancy-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  }
  .disc-item { display: flex; flex-direction: column; gap: 2px; }
  .disc-label { font-size: 10px; color: var(--text2); }
  .disc-value { font-size: 14px; font-weight: 600; }
  .disc-note { font-size: 10px; color: var(--text2); margin-top: 8px; line-height: 1.5; }
  @media (max-width: 600px) {
    .discrepancy-grid { grid-template-columns: repeat(2, 1fr); }
  }

  /* ── Empty state ── */
  .empty { text-align: center; padding: 30px; color: var(--text2); font-size: 13px; }

  /* ── Responsive ── */
  @media (max-width: 1200px) {
    .grid-7 { grid-template-columns: repeat(4, 1fr); }
  }
  @media (max-width: 900px) {
    .grid-3, .grid-4 { grid-template-columns: 1fr 1fr; }
    .grid-7 { grid-template-columns: repeat(3, 1fr); }
    .header { flex-wrap: wrap; gap: 8px; }
  }
  @media (max-width: 600px) {
    .grid-2 { grid-template-columns: 1fr; }
    .grid-3, .grid-4, .grid-7 { grid-template-columns: 1fr 1fr; }
    .container { padding: 12px; }
    .header-right { flex-wrap: wrap; }
    .chart-lg { height: 220px; }
    .chart-md { height: 180px; }
    .chart-sm { height: 160px; }
    table { font-size: 11px; }
    thead th, tbody td { padding: 5px 6px; }
  }
  /* ── Agent Chat ── */
  .chat-section{margin-top:20px}
  .chat-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;overflow:hidden;border-left:3px solid var(--purple)}
  .chat-header{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--bg3);border-bottom:1px solid var(--border);font-size:12px;color:var(--text2)}
  .chat-badge{background:rgba(63,185,80,.15);color:var(--green);padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
  .chat-messages{max-height:400px;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
  .cmsg{display:flex;flex-direction:column;gap:2px}
  .cmsg-agent{align-items:flex-start}
  .cmsg-user{align-items:flex-end}
  .cmsg-meta{font-size:10px;margin-bottom:2px}
  .cmsg-agent .cmsg-meta{color:var(--purple)}
  .cmsg-user .cmsg-meta{color:var(--green)}
  .cmsg-bubble{padding:8px 12px;font-size:12px;line-height:1.5;max-width:85%;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
  .cmsg-agent .cmsg-bubble{background:#1f1f3d;border:1px solid #3d3d6e;border-radius:12px 12px 12px 2px}
  .cmsg-user .cmsg-bubble{background:#1a3d1a;border:1px solid #2d6e2d;border-radius:12px 12px 2px 12px}
  .chat-input-row{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border);background:var(--bg3)}
  .chat-input-row input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px 12px;border-radius:8px;font-size:12px}
  .chat-input-row input:focus{border-color:var(--purple);outline:none}
  .chat-input-row button{background:var(--purple);color:#fff;border:none;padding:8px 16px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600}
  .chat-input-row button:hover{opacity:.85}
  .chat-input-row button:disabled{opacity:.4;cursor:not-allowed}
  .chat-hint{font-size:10px;color:var(--text2);padding:4px 12px 8px;text-align:center}
  .typing-wrap{display:none;padding:4px 14px}
  .typing-wrap.active{display:flex;align-items:flex-start;flex-direction:column}
  .typing-label{color:var(--purple);font-size:10px;margin-bottom:3px}
  .typing-row{background:#1f1f3d;border:1px solid #3d3d6e;border-radius:12px 12px 12px 2px;padding:8px 14px;display:flex;align-items:center;gap:8px}
  .typing-dots{display:flex;gap:4px}
  .typing-dots .dot{width:7px;height:7px;border-radius:50%;background:var(--purple);animation:dotBounce 1.4s infinite ease-in-out both}
  .typing-dots .dot:nth-child(2){animation-delay:.2s}
  .typing-dots .dot:nth-child(3){animation-delay:.4s}
  @keyframes dotBounce{0%,80%,100%{transform:scale(.4);opacity:.4}40%{transform:scale(1);opacity:1}}
  .typing-text{color:var(--text2);font-size:11px;animation:tPulse 2s infinite}
  @keyframes tPulse{0%,100%{opacity:.5}50%{opacity:1}}

/* ── Trades Summary Bar ─────────────────────────────────────────── */
.trades-summary {
  display: flex;
  gap: 1.5rem;
  align-items: center;
  padding: 0.55rem 1rem;
  margin-bottom: 0.75rem;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  font-size: 0.82rem;
  flex-wrap: wrap;
}
.trades-summary .ts-label { color: var(--text-muted, #aaa); margin-right: 0.25rem; }
.trades-summary .ts-val   { font-weight: 700; }
.trades-summary .ts-sep   { color: rgba(255,255,255,0.15); }

/* ── AI Thinking: Wild Chain inline block ───────────────────────── */
.ait-wild-inline { grid-column:1/-1;
  border-left: 3px solid #fb923c;
  background: transparent;
  padding: 3px 0 3px 10px;
  margin-bottom: 10px;
}
.ait-wild-inline-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  font-weight: 700;
  color: #fb923c;
  margin-bottom: 3px;
  text-transform: uppercase;
  letter-spacing: .5px;
}
.ait-wild-inline-header .ait-chain-meta {
  color: var(--text2);
  font-weight: 400;
  text-transform: none;
  letter-spacing: 0;
}
.ait-wild-inline-header .ait-chain-pnl {
  margin-left: auto;
  font-size: 11px;
}
.ait-wild-inline-text {
  font-size: 12px;
  color: var(--text);
  line-height: 1.7;
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <div class="live-mode-banner" id="liveModeBanner">🔴 LIVE TRADING MODE — REAL MONEY AT RISK (v2.10.0-live, mainnet Solana)</div>
    <div class="logo">◎ Solana <span>Cripto</span> Trader<span class="version-badge" id="versionBadge" onclick="openChangelog()" title="Ver historial de versiones">v—</span></div>
    <div class="status-badge" id="statusBadge">
      <div class="status-dot dot-green" id="statusDot"></div>
      <span id="statusText">ACTIVE</span>
    </div>
    <span class="mode-badge" id="modeBadge">PAPER</span>
  </div>
  <div class="header-right">
    <div class="capital-header">
      Equity: <span class="val" id="headerCapital">$0.00</span>
      <span class="chg" id="headerChg">+0.00%</span>
    </div>
    <div class="last-update" id="lastUpdate">Actualizado: —</div>
    <button class="btn-refresh" onclick="refreshAll()">⟳ Refresh</button>
    <button id="langToggle" onclick="toggleLang()" style="font-size:12px;padding:4px 10px;background:var(--bg3);border:1px solid var(--border);color:var(--text);border-radius:6px;cursor:pointer;font-weight:600;">🇬🇧 EN</button>
    <button class="btn-reset" onclick="resetBot()">🔄 Reset</button>
    <button class="btn-bot" id="botToggleBtn" onclick="toggleBot()" title="Encender/Apagar bot (kill switch)">⏳</button>
    <span class="fuel-badge" id="solFuelBadge" title="SOL fuel (Jupiter swap fees)">🔋 —</span>
    <span class="drift-badge dormant" id="driftBadge" title="Drift perps status">🎯 Drift: —</span>
    <span class="health-badge" id="healthBadge" title="Bot infra health (/api/health)">
      <span class="health-dot unknown" id="healthDot">●</span>
      <span id="healthLabel">health: —</span>
    </span>
  </div>
</div>

<div class="container">

  <!-- KPI ROW -->
  <section>
    <div class="grid-7" id="kpiRow">
      <div class="kpi-card neutral">
        <div class="kpi-label">Equity</div>
        <div class="kpi-value neu" id="kpiCapital">$—</div>
        <div class="kpi-sub" id="kpiCapitalSub">—</div>
      </div>
      <div class="kpi-card" id="kpiPnlCard">
        <div class="kpi-label">P&amp;L Realizado</div>
        <div class="kpi-value" id="kpiPnl">$—</div>
        <div class="kpi-sub" id="kpiPnlSub">—</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Win Rate <span style="font-size:9px;color:var(--text2);">(excl. flat)</span></div>
        <div class="kpi-value" id="kpiWinRate">—%</div>
        <div class="kpi-sub" id="kpiWinRateSub">W/L: —/—</div>
        <div class="kpi-sub" id="kpiWinRateExcl" style="margin-top:2px;font-size:10px;color:var(--text2);">incl. flat: —%</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Profit Factor</div>
        <div class="kpi-value" id="kpiPF">—</div>
        <div class="kpi-sub">ratio ganancia/pérdida</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Total Trades</div>
        <div class="kpi-value" id="kpiTrades">—</div>
        <div class="kpi-sub" id="kpiTradesSub">—</div>
        <div class="kpi-sub" id="kpiFlatSub" style="margin-top:2px;font-size:10px;color:var(--yellow);">flat: —</div>
      </div>
      <div class="kpi-card" id="kpiDdCard">
        <div class="kpi-label">Drawdown</div>
        <div class="kpi-value" id="kpiDD">—%</div>
        <div class="kpi-sub">desde el pico</div>
      </div>
      <div class="kpi-card neutral">
        <div class="kpi-label">Posiciones</div>
        <div class="kpi-value" id="kpiPos">—</div>
        <div class="kpi-sub" id="kpiPosSub">abiertas</div>
      </div>
    </div>
  </section>

  <!-- POST-RESET section removed: main KPIs now show post-reset data -->

  <!-- ACCOUNTING DISCREPANCY BANNER -->
  <div class="discrepancy-banner" id="discBanner" style="display:none;">
    <div class="disc-title">⚠️ Discrepancia Contable Detectada</div>
    <div class="discrepancy-grid">
      <div class="disc-item">
        <span class="disc-label">Cambio Real de Capital</span>
        <span class="disc-value warn" id="discRealChange">—</span>
      </div>
      <div class="disc-item">
        <span class="disc-label">PnL Registrado en Trades</span>
        <span class="disc-value" id="discRecordedPnl">—</span>
      </div>
      <div class="disc-item">
        <span class="disc-label">Gap No Contabilizado</span>
        <span class="disc-value neg" id="discGap">—</span>
      </div>
      <div class="disc-item">
        <span class="disc-label">EMERGENCY_CLOSEs (pnl=0)</span>
        <span class="disc-value warn" id="discEmergency">—</span>
      </div>
    </div>
    <div class="disc-note" id="discNote">
      ℹ️ Diferencia entre capital total y suma de PnL registrados. Puede incluir fees acumuladas no contabilizadas en ciclos anteriores. En EMERGENCY_CLOSE el margen se devuelve correctamente; la discrepancia proviene de fees de salida históricas. Capital actual ($<span id="discCapNote">—</span>) refleja contabilidad post-fix. Gap: <span id="discGapNote">—</span>
    </div>
  </div>

  <!-- RESET MODAL -->
  <div id="resetModal" onclick="if(event.target===this)closeResetModal()">
    <div class="reset-modal-box">
      <button class="reason-modal-close" onclick="closeResetModal()">&#x2715;</button>
      <div class="reset-modal-title">&#x1F504; Reset del Bot</div>
      <div class="reset-modal-field">
        <label>Capital inicial (USD)</label>
        <input type="number" id="resetCapitalInput" value="200" min="100" step="50">
      </div>
      <div class="reset-modal-field">
        <label>Motivo del reset</label>
        <textarea id="resetNotesInput" placeholder="Describe por que estas reseteando: que salio mal, que vas a cambiar, que aprendiste..."></textarea>
      </div>
      <div class="reset-modal-field">
        <label>Archivos adjuntos (capturas, PDFs, imagenes)</label>
        <div class="reset-drop-zone" id="resetDropZone" onclick="document.getElementById('resetFileInput').click()">
          <input type="file" id="resetFileInput" multiple accept="image/*,.pdf,.txt,.csv" onchange="handleResetFiles(this.files)">
          &#x1F4CE; Arrastra archivos aqui o haz clic para seleccionar<br>
          <span style="font-size:10px;opacity:0.6;">PNG, JPG, PDF, etc. &middot; Max 10 archivos &middot; 20MB c/u</span>
        </div>
        <div class="reset-file-list" id="resetFileList"></div>
      </div>
      <div class="reset-modal-actions">
        <button class="reset-btn-cancel" onclick="closeResetModal()">Cancelar</button>
        <button class="reset-btn-confirm" id="resetConfirmBtn" onclick="confirmReset()">Confirmar Reset</button>
      </div>
    </div>
  </div>

  <!-- OPEN POSITIONS (moved up) -->
  <section>
    <div class="section-title" style="display:flex;align-items:center;justify-content:space-between;">
      <span data-i18n="openPositions">⚡ Posiciones Abiertas</span>
      <div style="display:flex;align-items:center;gap:10px;">
        <span id="totalPnlBadge" style="font-size:16px;font-weight:700;padding:4px 12px;border-radius:8px;background:var(--bg3);"></span>
        <div style="display:flex;align-items:center;gap:6px;">
          <label style="font-size:11px;color:var(--text2);">Target $:</label>
          <input type="number" id="pnlTargetInput" placeholder="ej: 5" style="width:60px;font-size:12px;" step="0.5" min="0">
          <button onclick="setPnlTarget()" style="font-size:11px;padding:3px 8px;background:var(--green);color:#000;border:none;border-radius:4px;cursor:pointer;">Set</button>
          <button onclick="clearPnlTarget()" style="font-size:11px;padding:3px 8px;background:var(--bg3);color:var(--text2);border:1px solid var(--bg3);border-radius:4px;cursor:pointer;">Clear</button>
          <label style="display:flex;align-items:center;gap:4px;font-size:11px;cursor:pointer;margin-left:8px;" title="Modo Salvaje: IA gestiona coberturas martingala automaticamente">
            <input type="checkbox" id="wildModeSwitch" onchange="toggleWildMode()" style="cursor:pointer;">
            <span style="color:var(--orange);font-weight:600;">🔥 Salvaje</span>
          </label>
          <span id="wildModeBadge" style="display:none;font-size:10px;padding:3px 8px;border-radius:6px;background:var(--orange);color:#000;font-weight:700;"></span>
          <span id="pnlTargetStatus" style="font-size:10px;color:var(--text2);"></span>
        </div>
        <button onclick="openAIThinking()" id="aiThinkingBtn" style="font-size:11px;padding:4px 12px;background:linear-gradient(135deg,#bc8cff,#58a6ff);color:#000;border:none;border-radius:6px;cursor:pointer;font-weight:700;letter-spacing:0.3px;" data-i18n="aiThinking">🧠 Qué está pensando la IA</button>
        <button onclick="closeAllPositions()" style="font-size:11px;padding:4px 12px;background:var(--red);color:#fff;border:none;border-radius:6px;cursor:pointer;font-weight:600;" data-i18n="closeAll">Cerrar Todas</button>
      </div>
    </div>
    <div class="card">
      <div class="table-wrap" id="positionsTable">
        <div class="empty" data-i18n="loadingPositions">Cargando posiciones...</div>
      </div>
    </div>
  </section>

  <!-- TRADE HISTORY -->
  <section>
    <div class="section-title"><span data-i18n="tradeHistory">📋 Historial de Trades</span></div>
    <div class="card">
      <div class="filters">
        <select id="filterSymbol" onchange="filterTrades()">
          <option value="" data-i18n="allSymbols">Todos los símbolos</option>
        </select>
        <select id="filterResult" onchange="filterTrades()">
          <option value="" data-i18n="allResults">Todos los resultados</option>
          <option value="win" data-i18n="winners">Ganadores</option>
          <option value="loss" data-i18n="losers">Perdedores</option>
          <option value="flat" data-i18n="flat">Sin cambio (FLAT)</option>
          <option value="nonflat" data-i18n="nonFlat">Solo W/L (excl. FLAT)</option>
        </select>
        <select id="filterDir" onchange="filterTrades()">
          <option value="" data-i18n="longShort">Long &amp; Short</option>
          <option value="long">Long</option>
          <option value="short">Short</option>
        </select>
      </div>
      <div id="tradesSummary" class="trades-summary" style="display:none"></div>
      <div class="table-wrap" id="tradesTable">
        <div class="empty" data-i18n="loadingTrades">Cargando trades...</div>
      </div>
      <div class="pagination" id="tradesPagination"></div>
    </div>
  </section>

  <!-- EQUITY + DISTRIBUTIONS -->
  <section>
    <div class="grid-2">
      <!-- Equity Curve -->
      <div class="card" style="grid-column: span 1;">
        <div class="card-title"><span class="icon">📈</span> <span data-i18n="equityCurve">Equity Curve</span></div>
        <div class="chart-toolbar">
          <button class="tb-btn active" onclick="setEquityRange('all', this)">All</button>
          <button class="tb-btn" onclick="setEquityRange(50, this)">50T</button>
          <button class="tb-btn" onclick="setEquityRange(100, this)">100T</button>
          <button class="tb-btn" onclick="setEquityRange(200, this)">200T</button>
          <div class="tb-sep"></div>
          <button class="tb-btn" onclick="resetZoom()">↺ Reset</button>
        </div>
        <div class="chart-container chart-lg">
          <canvas id="equityChart"></canvas>
        </div>
      </div>

      <!-- Right column: pie + bar -->
      <div style="display:flex; flex-direction:column; gap:16px;">
        <div class="card">
          <div class="card-title"><span class="icon">🥧</span> <span data-i18n="winsVsLosses">Wins vs Losses</span></div>
          <div class="chart-container chart-sm">
            <canvas id="pieChart"></canvas>
          </div>
        </div>
        <div class="card">
          <div class="card-title"><span class="icon">📊</span> <span data-i18n="pnlBySymbol">P&amp;L por Símbolo</span></div>
          <div class="chart-container chart-sm">
            <canvas id="symbolChart"></canvas>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- P&L HISTOGRAM -->
  <section>
    <div class="card">
      <div class="card-title"><span class="icon">📉</span> <span data-i18n="pnlDistribution">Distribución de P&amp;L por Trade</span></div>
      <div class="chart-container chart-md">
        <canvas id="histChart"></canvas>
      </div>
    </div>
  </section>


  <!-- RESET HISTORY -->
  <section>
    <div class="section-title" data-i18n="resetHistory">📊 Historial de Resets</div>
    <div class="card">
      <div class="table-wrap" id="resetHistoryTable">
        <div class="empty" data-i18n="noResets">Sin resets registrados</div>
      </div>
    </div>
  </section>

  <!-- AGENT CHAT -->
  <section class="chat-section">
    <div class="section-title" data-i18n="chatTitle">💬 Agente de Trading — Chat en Vivo</div>
    <div class="chat-card">
      <div class="chat-header">
        <span>🤖 <strong data-i18n="agentName">Solana Trading Agent</strong></span>
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="chat-badge" id="chatBadge">0 notas</span>
          <button onclick="clearChat()" title="Limpiar chat — el historial queda guardado para el agente"
            style="font-size:10px;padding:2px 8px;background:var(--bg);border:1px solid var(--border);color:var(--text2);border-radius:4px;cursor:pointer;" data-i18n="clearChat">🗑 Limpiar</button>
        </div>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="empty" data-i18n="connectingAgent">Conectando con el agente...</div>
      </div>
      <div class="typing-wrap" id="chatTyping">
        <div class="typing-label">🤖 Agente</div>
        <div class="typing-row">
          <div class="typing-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
          <span class="typing-text" id="chatTypingText" data-i18n="thinking">Pensando...</span>
        </div>
      </div>
      <div class="chat-input-row">
        <input type="text" id="chatInput" placeholder="Escribe una nota o pregunta al agente..." data-i18n-placeholder="chatPlaceholder" maxlength="500"
               onkeydown="if(event.key==='Enter')sendChatNote()">
        <button onclick="sendChatNote()" id="chatSendBtn" data-i18n="send">Enviar ▸</button>
      </div>
      <div class="chat-hint" data-i18n="chatHint">🚀 El agente responde automáticamente. Presiona Enter para enviar.</div>
    </div>
  </section>

  <!-- BOTTOM ROW: LOG + ADVANCED METRICS -->
  <section>
    <div class="grid-2">
      <!-- Watchdog Log -->
      <div class="card">
        <div class="card-title"><span class="icon">🔍</span> <span data-i18n="watchdogLog">Watchdog Log</span></div>
        <div class="log-container" id="logContainer">
          <div class="log-info" data-i18n="logLoading">Cargando log...</div>
        </div>
        <div class="log-auto" data-i18n="autoRefresh">🔄 Auto-refresh cada 30s</div>
      </div>

      <!-- Advanced Metrics -->
      <div class="card">
        <div class="card-title"><span class="icon">🧮</span> <span data-i18n="advancedMetrics">Métricas Avanzadas</span></div>
        <div id="advancedMetrics">
          <div class="metric-row">
            <span class="metric-label">Sharpe Ratio</span>
            <span class="metric-value" id="mSharpe">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Avg Win</span>
            <span class="metric-value pos" id="mAvgWin">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Avg Loss</span>
            <span class="metric-value neg" id="mAvgLoss">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Mejor Trade</span>
            <span class="metric-value pos" id="mBestTrade">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Peor Trade</span>
            <span class="metric-value neg" id="mWorstTrade">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Max Racha Ganadora</span>
            <span class="metric-value pos" id="mMaxWinStreak">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Max Racha Perdedora</span>
            <span class="metric-value neg" id="mMaxLossStreak">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Profit Factor</span>
            <span class="metric-value" id="mPF2">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Win Rate (excl. flat)</span>
            <span class="metric-value" id="mWRExcl">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Trades FLAT/EMERGENCY</span>
            <span class="metric-value warn" id="mFlatTrades">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Capital Inicial</span>
            <span class="metric-value" id="mInitCap">—</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Retorno Total</span>
            <span class="metric-value" id="mReturn">—</span>
          </div>
        </div>
      </div>
    </div>
  </section>

</div><!-- /container -->

<script>
// ── State ──────────────────────────────────────────────────────────────────
let allTrades = [];
let filteredTrades = [];
let currentPage = 1;
const PAGE_SIZE = 50;
let equityChart = null, pieChart = null, symbolChart = null, histChart = null;
let equityAllData = { labels: [], datasets: [] };

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  refreshAll();
  loadVersionBadge();
  // Refresh completo cada 30 segundos
  setInterval(refreshAll, 10000);  // 10s refresh (was 30s)
  // Refresh de posiciones cada 3 segundos (tiempo real)
  setInterval(loadPositionsRealtime, 3000);
});

async function loadVersionBadge() {
  try {
    const res = await fetch('/api/version');
    const data = await res.json();
    const _verEl = document.getElementById('versionBadge');
    _verEl.textContent = 'v' + data.version;
    // v2.10.0-live: detectar live mode y mostrar banner
    if (data.version && data.version.endsWith('-live')) {
      _verEl.classList.add('is-live');
      const _banner = document.getElementById('liveModeBanner');
      if (_banner) _banner.classList.add('active');
      document.body.classList.add('live-mode');
      document.title = '🔴 LIVE — ' + document.title;
    }
  } catch(e) {}
}

async function updateHealthBadge() {
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    const dot = document.getElementById('healthDot');
    const label = document.getElementById('healthLabel');
    if (!dot || !label) return;
    if (h.healthy) {
      dot.className = 'health-dot healthy';
      const age = h.heartbeat && h.heartbeat.age_s != null ? Math.round(h.heartbeat.age_s) + 's' : '—';
      const pos = h.portfolio ? h.portfolio.open_positions : 0;
      const cap = h.portfolio ? h.portfolio.capital_usd : 0;
      label.textContent = `HB ${age} · ${pos}pos · $${(cap||0).toFixed(2)}`;
    } else {
      dot.className = 'health-dot unhealthy';
      const r = [];
      if (h.orch && !h.orch.alive) r.push('orch dead');
      if (h.watchdog && !h.watchdog.alive) r.push('watchdog dead');
      if (h.kill_switch && h.kill_switch.active) r.push('kill switch');
      if (h.close_failures && h.close_failures.count >= h.close_failures.threshold) r.push('close fails');
      if (!h.heartbeat || h.heartbeat.age_s == null || h.heartbeat.age_s > 300) r.push('HB stale');
      label.textContent = 'UNHEALTHY: ' + (r.join(', ') || 'check /api/health');
    }
  } catch (e) {
    const dot = document.getElementById('healthDot');
    const label = document.getElementById('healthLabel');
    if (dot) dot.className = 'health-dot unknown';
    if (label) label.textContent = 'health: unreachable';
  }
}
// v2.12.22: poll /api/health every 30s
updateHealthBadge();
setInterval(updateHealthBadge, 30000);

async function refreshAll() {
  try {
    await Promise.all([
      loadStats(),
      loadEquity(),
      loadPositions(),
      loadTrades(),
      loadLog(),
      loadResetHistory(),
      refreshBotStatus(),
      refreshFuel(),
      refreshDrift()
    ]);
    document.getElementById('lastUpdate').textContent =
      'Actualizado: ' + new Date().toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch(e) {
    console.error('Refresh error:', e);
  }
}

let _resetFiles = [];

function openResetModal() {
  _resetFiles = [];
  document.getElementById('resetFileList').innerHTML = '';
  document.getElementById('resetNotesInput').value = '';
  document.getElementById('resetCapitalInput').value = '200';
  document.getElementById('resetModal').classList.add('open');
  setTimeout(() => document.getElementById('resetCapitalInput').focus(), 100);
}
function closeResetModal() {
  document.getElementById('resetModal').classList.remove('open');
}
/* ── Bot ON/OFF toggle (v2.10.0-live) ── */
/* ── Drift Perps badge (v2.12.0-live) ── */
async function refreshDrift() {
  const el = document.getElementById('driftBadge');
  if (!el) return;
  try {
    const r = await fetch('/api/drift/status');
    const d = await r.json();
    if (!d.ok) { el.className = 'drift-badge err'; el.textContent = '🎯 Drift: err'; return; }
    const s = d.status;
    el.classList.remove('active','long','short','dormant','err');
    if (!s.enabled) {
      el.classList.add('dormant');
      el.textContent = '🎯 Drift: off';
      el.title = 'DRIFT_ENABLED=false — feature dormant';
      return;
    }
    if (s.error) {
      el.classList.add('err');
      el.textContent = '🎯 Drift: err';
      el.title = 'Drift error: ' + s.error;
      return;
    }
    if (!s.live) {
      el.classList.add('dormant');
      el.textContent = '🎯 Drift: paper';
      return;
    }
    const base = s.sol_perp_base || 0;
    const hasPos = Math.abs(base) > 0.001;
    if (!hasPos) {
      el.classList.add('active');
      el.textContent = `🎯 Drift: flat · $${(s.collateral_free_usd||0).toFixed(2)}`;
    } else if (base > 0) {
      el.classList.add('long');
      el.textContent = `🎯 LONG ${base.toFixed(3)} @ $${(s.sol_perp_mark||0).toFixed(2)} · ${(s.current_leverage||0).toFixed(1)}x`;
    } else {
      el.classList.add('short');
      el.textContent = `🎯 SHORT ${Math.abs(base).toFixed(3)} @ $${(s.sol_perp_mark||0).toFixed(2)} · ${(s.current_leverage||0).toFixed(1)}x`;
    }
    el.title = [
      `collateral total: $${(s.collateral_total_usd||0).toFixed(2)} (cap $${s.max_collateral_cap})`,
      `collateral free:  $${(s.collateral_free_usd||0).toFixed(2)}`,
      `leverage:         ${(s.current_leverage||0).toFixed(2)}x (cap ${s.max_leverage_cap}x)`,
      `SOL-PERP base:    ${base.toFixed(4)}`,
      `mark price:       $${(s.sol_perp_mark||0).toFixed(2)}`,
      `funding/hr:       ${((s.funding_hourly||0)*100).toFixed(4)}%`,
      `whitelist:        ${(s.whitelist||[]).join(', ')}`,
    ].join('\n');
  } catch(e) {
    el.className = 'drift-badge err';
    el.textContent = '🎯 Drift: err';
    console.warn('refreshDrift', e);
  }
}

/* ── SOL Fuel badge (v2.11.0-live) ── */
async function refreshFuel() {
  const el = document.getElementById('solFuelBadge');
  if (!el) return;
  try {
    const r = await fetch('/api/sol/fuel');
    const d = await r.json();
    if (!d.ok) { el.className = 'fuel-badge off'; el.textContent = '🔋 —'; return; }
    const f = d.fuel;
    el.classList.remove('ok', 'warn', 'low', 'off');
    if (!f.live) {
      el.classList.add('off');
      el.textContent = '🔋 paper';
      el.title = 'Paper mode — SOL fuel no aplica';
      return;
    }
    const bal = f.sol_balance != null ? f.sol_balance.toFixed(4) : '?';
    const trades = f.trades_remaining_estimate != null ? f.trades_remaining_estimate : '?';
    const ratio = f.sol_balance != null && f.sol_min_balance > 0 ? f.sol_balance / f.sol_min_balance : 0;
    let cls = 'ok';
    if (ratio < 1) cls = 'low';
    else if (ratio < 3) cls = 'warn';
    el.classList.add(cls);
    el.textContent = `🔋 ${bal} SOL · ~${trades} tx`;
    const autoTxt = f.enabled ? 'ON' : 'OFF';
    const lastTxt = f.last_event ? ` | último: ${f.last_event.ts.substring(0,10)} +${f.last_event.out_sol.toFixed(4)} SOL` : '';
    el.title = `SOL balance: ${bal}\nmin threshold: ${f.sol_min_balance}\nUSDC: ${f.usdc_balance != null ? f.usdc_balance.toFixed(2) : '?'} (reserve: ${f.usdc_reserve_floor})\nauto-topup: ${autoTxt} ($${f.topup_amount_usd}/evt, max $${f.max_daily_usd}/día)\nhoy: $${f.spent_today_usd.toFixed(2)} en ${f.events_today} top-up(s)${lastTxt}`;
  } catch(e) {
    el.className = 'fuel-badge off';
    el.textContent = '🔋 err';
    console.warn('refreshFuel', e);
  }
}

async function refreshBotStatus() {
  try {
    const r = await fetch('/api/safety/status');
    const d = await r.json();
    if (!d.ok) return;
    const btn = document.getElementById('botToggleBtn');
    if (!btn) return;
    btn.classList.remove('on', 'off', 'loading');
    // kill_switch_active=true → bot APAGADO (🔴)
    // kill_switch_active=false → bot ENCENDIDO (🟢)
    if (d.kill_switch_active) {
      btn.classList.add('off');
      btn.textContent = '🔴 OFF';
      btn.title = 'Bot APAGADO — click para encender. Razón: ' + (d.kill_switch_reason || '—');
    } else {
      btn.classList.add('on');
      btn.textContent = '🟢 ON';
      btn.title = 'Bot ENCENDIDO — click para apagar (pausa aperturas nuevas, no toca posiciones abiertas)';
    }
  } catch (e) {
    console.warn('refreshBotStatus error', e);
  }
}

async function toggleBot() {
  const btn = document.getElementById('botToggleBtn');
  const isOn = btn.classList.contains('on');
  const action = isOn ? 'activate' : 'deactivate';
  const actionLabel = isOn ? 'APAGAR' : 'ENCENDER';
  const msg = isOn
    ? '¿Apagar el bot?\n\nSe pausarán las aperturas nuevas.\nLas posiciones abiertas siguen gestionadas (SL/TP activos).'
    : '¿Encender el bot?\n\nVolverá a abrir posiciones cuando las señales pasen los filtros.';
  if (!confirm(msg)) return;
  btn.classList.add('loading');
  btn.textContent = '⏳';
  try {
    const r = await fetch('/api/safety/kill-switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: action, reason: 'dashboard_button:' + actionLabel.toLowerCase()}),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'unknown');
    await refreshBotStatus();
  } catch (e) {
    alert('Error toggling bot: ' + e.message);
    await refreshBotStatus();
  }
}

async function resetBot() { openResetModal(); }

/* ── Changelog ── */
async function openChangelog() {
  document.getElementById('changelogModal').classList.add('open');
  const content = document.getElementById('changelogContent');
  try {
    const res = await fetch('/api/version');
    const data = await res.json();
    let html = '';
    data.changelog.forEach((entry, idx) => {
      if (idx > 0) html += '<hr class="changelog-ver-divider">';
      html += `<div class="changelog-version">
        <div class="changelog-ver-header">
          <span class="changelog-ver-tag">v${entry.version}</span>
          <span class="changelog-ver-date">${entry.date}</span>
        </div>
        <div class="changelog-ver-title">${entry.title}</div>
        <ul class="changelog-ver-list">
          ${entry.changes.map(c => `<li>${c}</li>`).join('')}
        </ul>
      </div>`;
    });
    content.innerHTML = html;
  } catch(e) {
    content.innerHTML = '<span style="color:var(--red)">Error al cargar el changelog.</span>';
  }
}
function closeChangelog() {
  document.getElementById('changelogModal').classList.remove('open');
}

function handleResetFiles(files) {
  for (const f of Array.from(files)) {
    if (_resetFiles.length >= 10) break;
    if (f.size > 20 * 1024 * 1024) { alert(f.name + ': demasiado grande (max 20MB)'); continue; }
    if (!_resetFiles.find(x => x.name === f.name)) _resetFiles.push(f);
  }
  renderResetFileList();
}
function renderResetFileList() {
  const list = document.getElementById('resetFileList');
  list.innerHTML = _resetFiles.map((f, i) =>
    `<div class="reset-file-item">
      <span>${_fileIcon(f.name)}</span>
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(f.name)}</span>
      <span style="color:var(--text2);flex-shrink:0">${(f.size/1024).toFixed(0)}KB</span>
      <button onclick="_resetFiles.splice(${i},1);renderResetFileList()" title="Quitar">&#x2715;</button>
    </div>`).join('');
}
function _fileIcon(name) {
  const ext = (name.split('.').pop()||'').toLowerCase();
  if (['png','jpg','jpeg','gif','webp'].includes(ext)) return '🖼';
  if (ext === 'pdf') return '📄';
  return '📎';
}

// Drag & drop setup — called once on DOMContentLoaded (see below)
function _initResetDrop() {
  const zone = document.getElementById('resetDropZone');
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    handleResetFiles(e.dataTransfer.files);
  });
}

async function confirmReset() {
  const capitalNum = parseFloat(document.getElementById('resetCapitalInput').value);
  const notes = document.getElementById('resetNotesInput').value.trim();
  if (isNaN(capitalNum) || capitalNum < 100) { alert(TRANSLATIONS[currentLang].invalidCapital); return; }
  const btn = document.getElementById('resetConfirmBtn');
  btn.disabled = true; btn.textContent = 'Procesando...';
  try {
    let attachments = [];
    if (_resetFiles.length > 0) {
      const fd = new FormData();
      _resetFiles.forEach(f => fd.append('files', f));
      const up = await fetch('/api/reset-upload', { method: 'POST', body: fd });
      const upd = await up.json();
      attachments = upd.files || [];
    }
    const r = await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ capital: capitalNum, notes, attachments })
    });
    const d = await r.json();
    if (d.success) {
      closeResetModal();
      alert(TRANSLATIONS[currentLang].resetSuccess.replace('${capital}', '$' + capitalNum));
      location.reload();
    } else {
      alert('❌ Error: ' + (d.error || 'Unknown'));
    }
  } catch(e) {
    alert('❌ Error de conexión: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Confirmar Reset';
  }
}

// ── API Calls ──────────────────────────────────────────────────────────────
async function loadStats() {
  const r = await fetch('/api/stats');
  const d = await r.json();

  // Header
  document.getElementById('headerCapital').textContent = fmt$(d.capital_usd);
  const chgEl = document.getElementById('headerChg');
  chgEl.textContent = fmtPct(d.return_pct, true);
  chgEl.className = 'chg ' + (d.return_pct >= 0 ? 'pos' : 'neg');
  document.getElementById('modeBadge').textContent = (d.mode || 'PAPER').toUpperCase().replace('_', ' ');

  // Status
  const status = d.portfolio_status || 'ACTIVE';
  const dot = document.getElementById('statusDot');
  document.getElementById('statusText').textContent = status;
  dot.className = 'status-dot ' + (status === 'ACTIVE' ? 'dot-green' : status === 'PAUSED' ? 'dot-yellow' : 'dot-red');

  // KPIs
  document.getElementById('kpiCapital').textContent = fmt$(d.capital_usd);
  document.getElementById('kpiCapitalSub').textContent = 'Inicial: ' + fmt$(d.initial_capital);

  const pnl = d.realized_pnl;
  const pnlEl = document.getElementById('kpiPnl');
  const pnlSubEl = document.getElementById('kpiPnlSub');
  const pnlCard = document.getElementById('kpiPnlCard');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + fmt$(pnl);
  pnlEl.className = 'kpi-value ' + (pnl >= 0 ? 'pos' : 'neg');
  const unrealized_val = d.unrealized_pnl || 0; pnlSubEl.textContent = TRANSLATIONS[currentLang].unrealized + ": " + (unrealized_val >= 0 ? "+" : "") + fmt$(unrealized_val);
  pnlSubEl.className = 'kpi-sub ' + (pnl >= 0 ? 'pos' : 'neg');
  pnlCard.className = 'kpi-card ' + (pnl >= 0 ? 'positive' : 'negative');

  // Primary: excl. flat win rate (signal quality), secondary: gross incl. flat
  const wr = d.win_rate_excl_flat || 0;  // excl. flat is the meaningful metric
  const wrGross = d.win_rate || 0;        // gross incl. flat shown as secondary
  const wrEl = document.getElementById('kpiWinRate');
  wrEl.textContent = fmtNum(wr, 1) + '%';
  wrEl.className = 'kpi-value ' + (wr >= 50 ? 'pos' : wr >= 40 ? 'warn' : 'neg');
  document.getElementById('kpiWinRateSub').textContent = `W/L: ${d.wins}/${d.losses}`;
  document.getElementById('kpiWinRateExcl').textContent = `incl. flat: ${fmtNum(wrGross, 1)}%`;

  const pf = d.profit_factor;
  const pfEl = document.getElementById('kpiPF');
  pfEl.textContent = fmtNum(pf, 2);
  pfEl.className = 'kpi-value ' + (pf >= 1.5 ? 'pos' : pf >= 1 ? 'warn' : 'neg');

  document.getElementById('kpiTrades').textContent = d.total_trades;
  document.getElementById('kpiTradesSub').textContent =
    `${d.wins} wins · ${d.losses} losses`;
  const flatCount = d.flat_trades || 0;
  const flatEl = document.getElementById('kpiFlatSub');
  flatEl.textContent = `flat: ${flatCount} (${fmtNum(flatCount / d.total_trades * 100, 0)}%)`;
  flatEl.style.color = flatCount > 100 ? 'var(--yellow)' : 'var(--text2)';

  const dd = d.drawdown_pct;
  const ddEl = document.getElementById('kpiDD');
  ddEl.textContent = fmtNum(dd, 2) + '%';
  ddEl.className = 'kpi-value ' + (dd <= 2 ? 'pos' : dd <= 5 ? 'warn' : 'neg');
  document.getElementById('kpiDdCard').className = 'kpi-card ' + (dd <= 2 ? 'positive' : dd <= 5 ? 'warning' : 'negative');

  const op = d.open_positions;
  document.getElementById('kpiPos').textContent = op;
  document.getElementById('kpiPosSub').textContent = op === 0 ? TRANSLATIONS[currentLang].noExposure : `$${fmtNum(d.total_exposure_usd, 0)} exp.`;

  // Advanced metrics
  document.getElementById('mSharpe').textContent = fmtNum(d.sharpe_ratio, 3);
  document.getElementById('mSharpe').className = 'metric-value ' + (d.sharpe_ratio >= 1 ? 'pos' : d.sharpe_ratio >= 0 ? 'warn' : 'neg');
  document.getElementById('mAvgWin').textContent = '+' + fmt$(d.avg_win_usd);
  document.getElementById('mAvgLoss').textContent = '-' + fmt$(d.avg_loss_usd);
  document.getElementById('mBestTrade').textContent = '+' + fmt$(d.best_trade);
  document.getElementById('mWorstTrade').textContent = '-' + fmt$(Math.abs(d.worst_trade));
  document.getElementById('mMaxWinStreak').textContent = d.max_win_streak + ' trades';
  document.getElementById('mMaxLossStreak').textContent = d.max_loss_streak + ' trades';
  document.getElementById('mPF2').textContent = fmtNum(d.profit_factor, 2);
  document.getElementById('mPF2').className = 'metric-value ' + (pf >= 1.5 ? 'pos' : pf >= 1 ? 'warn' : 'neg');
  const wrExclEl = document.getElementById('mWRExcl');
  const wre = d.win_rate_excl_flat || 0;
  wrExclEl.textContent = fmtNum(wre, 1) + '%';
  wrExclEl.className = 'metric-value ' + (wre >= 50 ? 'pos' : wre >= 40 ? 'warn' : 'neg');
  document.getElementById('mFlatTrades').textContent =
    `${d.flat_trades || 0} (${d.emergency_zero_pnl || 0} EMRG)`;
  document.getElementById('mInitCap').textContent = fmt$(d.initial_capital);
  const retEl = document.getElementById('mReturn');
  retEl.textContent = fmtPct(d.return_pct, true);
  retEl.className = 'metric-value ' + (d.return_pct >= 0 ? 'pos' : 'neg');

  // Accounting discrepancy banner
  const gap = d.accounting_gap || 0;
  const realChange = d.real_capital_change || 0;
  const recPnl = d.total_pnl || 0;
  const emergencyZero = d.emergency_zero_pnl || 0;
  const banner = document.getElementById('discBanner');
  // Show banner only if gap > 5% of capital OR > $25 (whichever is larger)
  const capitalForThreshold = d.capital_usd || 500;
  const thresholdPct = capitalForThreshold * 0.05;
  const threshold = Math.max(thresholdPct, 25);
  if (Math.abs(gap) > threshold) {  // show if gap > 5% of capital or $25 abs
    banner.style.display = '';
    const rcEl = document.getElementById('discRealChange');
    rcEl.textContent = (realChange >= 0 ? '+' : '') + fmt$(realChange);
    rcEl.className = 'disc-value ' + (realChange >= 0 ? 'pos' : 'neg');
    const rpEl = document.getElementById('discRecordedPnl');
    rpEl.textContent = (recPnl >= 0 ? '+' : '') + fmt$(recPnl);
    rpEl.className = 'disc-value ' + (recPnl >= 0 ? 'pos' : 'neg');
    const gapEl = document.getElementById('discGap');
    gapEl.textContent = (gap >= 0 ? '+' : '') + fmt$(gap);
    gapEl.className = 'disc-value ' + (gap >= 0 ? 'pos' : 'neg');
    document.getElementById('discEmergency').textContent = emergencyZero + ' trades';
    document.getElementById('discCapNote').textContent = fmt$(d.capital_usd);
    const gapNoteEl = document.getElementById('discGapNote');
    if (gapNoteEl) {
      gapNoteEl.textContent = (gap >= 0 ? '+' : '') + fmt$(gap) + ' (' + fmtNum(Math.abs(gap) / capitalForThreshold * 100, 1) + '% del capital)';
    }
  } else {
    banner.style.display = 'none';
  }
}



async function loadEquity() {
  const r = await fetch('/api/equity');
  const d = await r.json();

  equityAllData = {
    labels: d.labels,
    trade_pnls: d.trade_pnls,
    equity: d.equity,
    winners: d.winners,
    losers: d.losers
  };
  updateEquityChart('all');

  // Pie chart
  const wins = d.total_wins || 0, losses = d.total_losses || 0;
  pieChart.data.datasets[0].data = [wins, losses];
  pieChart.update();

  // Symbol P&L bar
  const syms = Object.keys(d.symbol_pnl || {}).sort((a,b) => d.symbol_pnl[b] - d.symbol_pnl[a]);
  const symVals = syms.map(s => d.symbol_pnl[s]);
  symbolChart.data.labels = syms;
  symbolChart.data.datasets[0].data = symVals;
  symbolChart.data.datasets[0].backgroundColor = symVals.map(v => v >= 0 ? 'rgba(63,185,80,0.7)' : 'rgba(248,81,73,0.7)');
  symbolChart.data.datasets[0].borderColor = symVals.map(v => v >= 0 ? '#3fb950' : '#f85149');
  symbolChart.update();

  // Histogram
  const pnls = d.trade_pnls || [];
  if (pnls.length > 0) {
    const min = Math.min(...pnls), max = Math.max(...pnls);
    const bins = 20;
    const step = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    const binLabels = [];
    for (let i = 0; i < bins; i++) binLabels.push(fmtNum(min + i * step, 2));
    pnls.forEach(p => {
      const idx = Math.min(Math.floor((p - min) / step), bins - 1);
      counts[Math.max(0, idx)]++;
    });
    histChart.data.labels = binLabels;
    histChart.data.datasets[0].data = counts;
    histChart.data.datasets[0].backgroundColor = binLabels.map(l => parseFloat(l) >= 0 ? 'rgba(63,185,80,0.6)' : 'rgba(248,81,73,0.6)');
    histChart.update();
  }
}

async function loadPositions() {
  const r = await fetch('/api/positions');
  const d = await r.json();
  renderPositions(d);
}

// Función de actualización en tiempo real (cada 3 seg)


// Total PnL badge + auto-close target
let _pnlTarget = 0;
function updateTotalPnl(positions) {
  const total = positions.reduce((s, p) => s + (p.pnl_usd || 0), 0);
  const el = document.getElementById('totalPnlBadge');
  const sign = total >= 0 ? '+' : '';
  el.textContent = 'Total: ' + sign + '$' + total.toFixed(2);
  el.style.color = total >= 0 ? 'var(--green)' : 'var(--red)';
  // Check auto-close target
  const wildSw = document.getElementById('wildModeSwitch');
  const wildActive = wildSw && wildSw.checked;
  if (!wildActive && _pnlTarget > 0 && total >= _pnlTarget) {
    autoCloseAllPositions();
    _pnlTarget = 0;
    document.getElementById('pnlTargetStatus').textContent = '✅ Target hit!';
    document.getElementById('pnlTargetInput').value = '';
  }
}
function setPnlTarget() {
  const val = parseFloat(document.getElementById('pnlTargetInput').value);
  if (val > 0) {
    _pnlTarget = val;
    document.getElementById('pnlTargetStatus').textContent = 'Target: $' + val.toFixed(2);
  } else {
    clearPnlTarget();
  }
}
function clearPnlTarget() {
  _pnlTarget = 0;
  document.getElementById('pnlTargetInput').value = '';
  document.getElementById('pnlTargetStatus').textContent = 'Sin target';
  setTimeout(() => document.getElementById('pnlTargetStatus').textContent = '', 3000);
}

// ── Wild Mode (Modo Salvaje) ─────────────────────────────────────────────
async function toggleWildMode() {
  const sw = document.getElementById('wildModeSwitch');
  const badge = document.getElementById('wildModeBadge');
  if (sw.checked) {
    const targetMode = _pnlTarget > 0
      ? `con target $${_pnlTarget.toFixed(2)} (cierra al alcanzar)`
      : `SIN target — la IA decidirá cuándo cerrar según mercado`;
    if (!confirm(TRANSLATIONS[currentLang].confirmWildOn)) {
      sw.checked = false;
      return;
    }
    try {
      const r = await fetch('/api/wild-mode/activate', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({target: _pnlTarget})
      });
      const j = await r.json();
      if (j.ok) {
        badge.style.display = 'inline-block';
        badge.textContent = _pnlTarget > 0 ? `🔥 $${_pnlTarget.toFixed(2)}` : '🔥 IA-LIBRE';
      } else {
        alert('Error: ' + (j.error || 'unknown'));
        sw.checked = false;
      }
    } catch(e) {
      alert('Error de red: ' + e);
      sw.checked = false;
    }
  } else {
    if (!confirm(TRANSLATIONS[currentLang].confirmWildOff)) {
      sw.checked = true;
      return;
    }
    try {
      await fetch('/api/wild-mode/deactivate', {method: 'POST'});
      badge.style.display = 'none';
    } catch(e) { alert('Error: ' + e); }
  }
}

async function loadWildModeState() {
  try {
    const r = await fetch('/api/wild-mode/state');
    const s = await r.json();
    if (s && s.active) {
      document.getElementById('wildModeSwitch').checked = true;
      const badge = document.getElementById('wildModeBadge');
      badge.style.display = 'inline-block';
      const tgt = parseFloat(s.target_usd || 0);
      if (tgt > 0) {
        _pnlTarget = tgt;
        document.getElementById('pnlTargetInput').value = tgt;
        document.getElementById('pnlTargetStatus').textContent = `Target: $${tgt.toFixed(2)} 🔥`;
        badge.textContent = `🔥 $${tgt.toFixed(2)}`;
      } else {
        badge.textContent = '🔥 IA-LIBRE';
      }
    }
  } catch(e) { /* silencioso */ }
}
// Carga estado al iniciar
window.addEventListener('DOMContentLoaded', loadWildModeState);
async function autoCloseAllPositions() {
  try {
    await fetch('/api/close-all-target', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({target: _pnlTarget}) });
    _pnlTarget = 0;
    document.getElementById('pnlTargetInput').value = '';
    document.getElementById('pnlTargetStatus').textContent = '\u2705 Target reached!';
    setTimeout(refreshAll, 1000);
  } catch(e) { console.error(e); }
}
async function closeAllPositions() {
  if (!confirm(TRANSLATIONS[currentLang].confirmCloseAll)) return;
  try {
    await fetch('/api/close-all', { method: 'POST' });
    refreshAll();
  } catch(e) { alert('Error: ' + e); }
}
async function closePosition(symbol) {
  if (!confirm(TRANSLATIONS[currentLang].confirmCloseOne.replace('${symbol}', symbol))) return;
  try {
    await fetch('/api/close-position', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({symbol: symbol}) });
    refreshAll();
  } catch(e) { alert('Error: ' + e); }
}

async function loadPositionsRealtime() {
  try {
    const r = await fetch('/api/positions');
    const d = await r.json();
    renderPositions(d);
    updateTotalPnl(d.positions || []);
    // Actualizar timestamp
    document.getElementById('lastUpdate').textContent =
      'Actualizado: ' + new Date().toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch(e) {
    // Silencioso en errores de realtime
  }
}

function renderPositions(d) {
  const wrap = document.getElementById('positionsTable');
  if (!d.positions || d.positions.length === 0) {
    wrap.innerHTML = `<div class="empty">${TRANSLATIONS[currentLang].noOpenPositions}</div>`;
    return;
  }
  let html = `<table><thead><tr>
    <th>Símbolo</th><th>Dirección</th><th>Entrada</th><th>Precio Act.</th>
    <th>Ganancia real estimada</th><th>% neto est.</th><th>Margin</th><th>Tamaño</th>
    <th>SL</th><th>TP</th><th>Tiempo</th><th>Estrategia</th>
    <th>Acción</th>
  </tr></thead><tbody>`;
  for (const p of d.positions) {
    const pnlCls = p.pnl_usd >= 0 ? 'pos' : 'neg';
    const flashCls = p.price_changed ? 'flash-update' : '';
    const hasHedges = p.chain_levels && p.chain_levels.length > 0;
    const symId = p.symbol.replace(/[^a-zA-Z0-9]/g, '_');
    const toggleBtn = hasHedges
      ? `<button class="chain-toggle" id="ctbtn_${symId}" onclick="toggleChainRows('${symId}')" title="${p.chain_levels.length} cobertura(s)">▶ ${p.chain_levels.length}</button>`
      : '';
    // Chain combined PnL badge (if in chain)
    const chainBadge = hasHedges
      ? (() => {
          const totalPnl = p.pnl_usd + p.chain_levels.reduce((s,lv) => s + (lv.pnl_usd||0), 0);
          const cls = totalPnl >= 0 ? 'pos' : 'neg';
          return ` <span class="${cls}" style="font-size:10px;" title="PnL combinado de la cadena">[${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}]</span>`;
        })()
      : '';
    html += `<tr class="${flashCls}">
      <td>${toggleBtn}<strong>${p.symbol}</strong>${chainBadge}</td>
      <td><span class="badge badge-${p.direction}">${p.direction.toUpperCase()}</span></td>
      <td>${fmtPrice(p.entry_price)}</td>
      <td class="price-cell">${fmtPrice(p.current_price)}</td>
      <td class="${pnlCls}">${p.pnl_usd >= 0 ? '+' : ''}${fmt$(p.pnl_usd)}</td>
      <td class="${pnlCls}">${fmtPct(p.pnl_pct, true)}</td>
      <td>${fmt$(p.margin_usd)}</td>
      <td>${fmt$(p.size_usd)}</td>
      <td class="neg">${fmtPrice(p.sl_price)}</td>
      <td class="pos">${fmtPrice(p.tp_price)}</td>
      <td>${p.time_open || '—'}</td>
      <td><button onclick="closePosition('${p.symbol}')" style="font-size:10px;padding:2px 8px;background:var(--orange);color:#fff;border:none;border-radius:4px;cursor:pointer;">${T('closeBtnCell')}</button></td>
      <td>${p.strategy || '—'}</td>
    </tr>`;
    // Sub-rows for hedge levels (hidden by default)
    if (hasHedges) {
      for (const lv of p.chain_levels) {
        const lvPnlCls = (lv.pnl_usd||0) >= 0 ? 'pos' : 'neg';
        const lvDir = lv.direction || '—';
        const lvBadge = `<span class="badge badge-${lvDir}" style="font-size:9px;">${lvDir.toUpperCase()}</span>`;
        html += `<tr class="chain-sub-row" id="csub_${symId}_${lv.level}" style="display:none;">
          <td><span style="color:var(--purple);font-size:10px;">Nivel ${lv.level}</span> ${lvBadge} <span style="font-size:9px;color:var(--text2);">x${lv.size_multiplier||1}</span></td>
          <td>${lvBadge}</td>
          <td>${fmtPrice(lv.entry_price)}</td>
          <td class="price-cell">${fmtPrice(lv.current_price)}</td>
          <td class="${lvPnlCls}">${(lv.pnl_usd||0) >= 0 ? '+' : ''}${fmt$(lv.pnl_usd||0)}</td>
          <td class="${lvPnlCls}">${fmtPct(lv.pnl_pct||0, true)}</td>
          <td>${fmt$(lv.margin||0)}</td>
          <td>—</td><td>—</td><td>—</td>
          <td style="font-size:10px;color:var(--text2);">${lv.opened_at ? new Date(lv.opened_at).toLocaleTimeString() : '—'}</td>
          <td>—</td><td style="font-size:10px;color:var(--purple);">Cobertura</td>
        </tr>`;
      }
    }
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function toggleChainRows(symId) {
  const btn = document.getElementById('ctbtn_' + symId);
  const rows = document.querySelectorAll('[id^="csub_' + symId + '_"]');
  const isOpen = btn && btn.textContent.startsWith('▼');
  rows.forEach(r => r.style.display = isOpen ? 'none' : '');
  if (btn) btn.textContent = btn.textContent.replace(isOpen ? '▼' : '▶', isOpen ? '▶' : '▼');
}

// ── AI Reasoning Modal ──────────────────────────────────────────────
const _REASON_LABELS = {
  'SL':                    { label: 'Stop Loss alcanzado', text: 'El precio tocó el nivel de Stop Loss predefinido. La posición se cerró automáticamente para limitar la pérdida máxima según los parámetros de riesgo configurados.' },
  'TP':                    { label: 'Take Profit alcanzado', text: 'El precio llegó al objetivo de ganancia (Take Profit). La posición se cerró automáticamente al alcanzar el retorno objetivo.' },
  'TRAILING_SL':           { label: 'Trailing Stop activado', text: 'El stop loss dinámico (trailing) se activó. El precio retrocedió desde su máximo y alcanzó el nivel de stop móvil, asegurando parte de las ganancias acumuladas.' },
  'TIME_EXIT':             { label: 'Tiempo máximo alcanzado', text: 'La posición superó el tiempo máximo permitido de permanencia. El sistema la cerró automáticamente para liberar capital y evitar exposición prolongada.' },
  'WILD_MODE_CLOSE_CHAIN': { label: 'Modo Salvaje — Cadena cerrada', text: 'El motor de modo salvaje cerró la cadena completa porque el PnL combinado de todos los niveles llegó a positivo (o condiciones de mercado óptimas). La IA evaluó el contexto (RSI, MACD, tendencia) y determinó el momento ideal para realizar la ganancia.' },
  'MANUAL_CLOSE':          { label: 'Cierre manual', text: 'El usuario cerró esta posición manualmente desde el dashboard.' },
  'MANUAL_CLOSE_ALL':      { label: 'Cierre manual de todas las posiciones', text: 'El usuario cerró todas las posiciones abiertas manualmente desde el dashboard.' },
  'EMERGENCY_CLOSE':       { label: 'Cierre de emergencia', text: 'El Risk Manager detectó condiciones de riesgo extremo (drawdown excesivo, exposición peligrosa) y cerró la posición automáticamente como medida de protección del capital.' },
  'ABANDON_ALL':           { label: 'Modo Salvaje — Abandono total', text: 'El modo salvaje alcanzó el límite máximo de pérdida (15% de drawdown desde el inicio de la sesión) y activó el protocolo de abandono. Todas las posiciones de la cadena se cerraron para preservar el capital restante.' },
  'LIQUIDATED':            { label: 'Liquidación', text: 'La posición fue liquidada por el sistema al alcanzar el precio de liquidación (margen insuficiente para mantener la posición abierta con el leverage configurado).' },
};

function _reasonFallback(closeReason, pnl) {
  if (!closeReason) return 'No hay información disponible sobre el motivo de cierre de esta posición.';
  const r = closeReason.toUpperCase();
  // PNL_TARGET pattern
  if (r.startsWith('PNL_TARGET')) {
    const amt = closeReason.replace(/[^0-9.]/g, '');
    return amt
      ? 'Se alcanzó el target de ganancia de $' + amt + ' configurado por el usuario. El sistema cerró automáticamente todas las posiciones para asegurar el objetivo diario.'
      : 'Se alcanzó el target de ganancia configurado. El sistema cerró las posiciones automáticamente.';
  }
  // WILD_* dynamic patterns (generated by martingale_engine.py)
  if (r.startsWith('WILD_AI_CLOSE_F') || r.includes('F&G') || r.includes('MIEDO') || r.includes('FEAR')) {
    const fg = closeReason.match(/\d+/)?.[0] || '?';
    return 'La IA del Modo Salvaje evaluó el contexto de mercado y decidió cerrar la sesión. Fear & Greed Index = ' + fg + ' (miedo extremo), que históricamente precede recuperaciones en crypto. La IA realizó la ganancia/pérdida en el momento óptimo para proteger capital antes de mayor volatilidad.';
  }
  if (r.startsWith('WILD_AI_CLOSE_PNL') || r.includes('PNL COMBINADO') || r.includes('PNL_COMB')) {
    return 'La IA del Modo Salvaje cerró todas las posiciones de la cadena porque el PnL combinado de todos los niveles alcanzó un valor positivo. La IA evaluó RSI, MACD y tendencia para confirmar el momento óptimo de salida y asegurar la ganancia acumulada.';
  }
  if (r.startsWith('WILD_AI_CLOSE')) {
    const detail = closeReason.replace(/^WILD_AI_CLOSE_?/i, '').trim();
    return 'La IA del Modo Salvaje evaluó las condiciones de mercado y decidió cerrar la sesión proactivamente.' + (detail ? ' Razón: ' + detail + '.' : '') + ' El sistema libera capital para la siguiente oportunidad.';
  }
  if (r.startsWith('WILD_ABANDON')) {
    const detail = closeReason.replace(/^WILD_ABANDON_?/i, '').trim();
    const mins = detail.match(/(\d+)m/)?.[1];
    return mins
      ? 'El Modo Salvaje cerró todas las posiciones porque la sesión superó ' + mins + ' minutos sin alcanzar el objetivo. Cierre preventivo para proteger capital antes del siguiente ciclo.'
      : 'El Modo Salvaje activó el protocolo de abandono. ' + (detail || 'La sesión superó los límites de tiempo o pérdida configurados.') + ' Todas las posiciones de la cadena fueron cerradas.';
  }
  if (r.startsWith('WILD_TARGET_HIT') || r.startsWith('WILD_TARGET')) {
    return 'El Modo Salvaje alcanzó el objetivo de ganancia de la sesión. Todas las posiciones de la cadena se cerraron para asegurar el profit objetivo antes de que el mercado revierta.';
  }
  if (r === 'WILD_MODE_CLOSE_CHAIN' || r.startsWith('WILD_MODE_CLOSE_CHAIN')) {
    return 'El motor de Martingala cerró toda la cadena de posiciones porque el PnL combinado de todos los niveles alcanzó el objetivo de la sesión. Las posiciones se cerraron juntas para maximizar la ganancia acumulada antes de que el mercado revierta.';
  }
  if (r.startsWith('WILD_')) {
    const detail = closeReason.replace(/^WILD_/i, '').trim();
    return 'Cierre por motor Wild Mode. ' + (detail || 'El sistema de martingala ejecutó un cierre automático según sus reglas internas.');
  }
  const entry = _REASON_LABELS[r];
  if (entry) return entry.text;
  return 'Razón de cierre: ' + closeReason + '. No hay descripción detallada disponible para este tipo de cierre.';
}

function showReasonByIdx(idx) {
  const d = (window._tradeReasonData || {})[idx];
  if (!d) return;
  showReason(d.symbol, d.closeReason, d.aiReasoning, d.pnl, d.closeTime);
}

function showReason(symbol, closeReason, aiReasoning, pnl, closeTime) {
  const modal = document.getElementById('reasonModal');
  const r = (closeReason || '').toUpperCase();
  const entry = _REASON_LABELS[r] || {};
  let typeLabel = entry.label || closeReason || '—';
  // Dynamic labels for WILD_* close reasons
  if (!entry.label) {
    if (r.startsWith('WILD_AI_CLOSE'))           typeLabel = 'Modo Salvaje — Cierre por IA';
    else if (r.startsWith('WILD_ABANDON'))        typeLabel = 'Modo Salvaje — Sesión abandonada';
    else if (r.startsWith('WILD_TARGET'))         typeLabel = 'Modo Salvaje — Target alcanzado';
    else if (r.startsWith('WILD_'))               typeLabel = 'Modo Salvaje — Cierre automático';
  }
  const ts = closeTime ? new Date(closeTime).toLocaleString() : '';
  document.getElementById('rmSym').textContent = symbol + (ts ? '  ·  ' + ts : '');
  document.getElementById('rmType').textContent = typeLabel;
  // Priority: real AI reasoning > fallback description
  const body = (aiReasoning && aiReasoning.trim().length > 10)
    ? aiReasoning.trim()
    : _reasonFallback(closeReason, pnl);
  document.getElementById('rmBody').textContent = body;
  modal.classList.add('open');
}

function closeReasonModal() {
  document.getElementById('reasonModal').classList.remove('open');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeReasonModal(); closeAIThinking(); closeResetModal(); }
});

// ── AI Thinking Modal ────────────────────────────────────────────────
let _aitTimer = null;
let _aitCountdownVal = 20;
let _aitCountdownTimer = null;

async function openAIThinking() {
  document.getElementById('aiThinkingModal').classList.add('open');
  await fetchAIThinking();
  _startAitCountdown();
}

function closeAIThinking() {
  document.getElementById('aiThinkingModal').classList.remove('open');
  clearTimeout(_aitTimer);
  clearInterval(_aitCountdownTimer);
}

function _startAitCountdown() {
  clearInterval(_aitCountdownTimer);
  _aitCountdownVal = 20;
  _updateAitBar();
  _aitCountdownTimer = setInterval(() => {
    _aitCountdownVal--;
    _updateAitBar();
    if (_aitCountdownVal <= 0) {
      clearInterval(_aitCountdownTimer);
      fetchAIThinking().then(_startAitCountdown);
    }
  }, 1000);
}

function _updateAitBar() {
  const pct = (_aitCountdownVal / 20) * 100;
  const bar = document.getElementById('aitRefillBar');
  const lbl = document.getElementById('aitCountdown');
  if (bar) bar.style.width = pct + '%';
  if (lbl) lbl.textContent = T('aitCountdown').replace('{n}', _aitCountdownVal);
}

async function fetchAIThinking() {
  try {
    const r = await fetch('/api/ai-thinking');
    const d = await r.json();
    renderAIThinking(d);
  } catch(e) {
    document.getElementById('aitContent').innerHTML =
      '<div class="ait-loading"><div>⚠️ ' + T('aitError') + ' ' + e.message + '</div></div>';
  }
}

function _actionBadge(action) {
  if (!action) return '';
  const a = action.toUpperCase();
  if (a === 'HOLD')        return `<span class="ait-decision-badge ait-badge-hold">${T('badgeHold')}</span>`;
  if (a === 'OPEN_LEVEL')  return `<span class="ait-decision-badge ait-badge-open">${T('badgeOpen')}</span>`;
  if (a === 'CLOSE_CHAIN') return `<span class="ait-decision-badge ait-badge-close">${T('badgeCloseChain')}</span>`;
  if (a === 'CLOSE')       return `<span class="ait-decision-badge ait-badge-close">${T('badgeClose')}</span>`;
  if (a.includes('ABANDON'))return `<span class="ait-decision-badge ait-badge-abandon">${T('badgeAbandon')}</span>`;
  return `<span class="ait-decision-badge ait-badge-hold">${action}</span>`;
}

function _pnlColor(v) { return v > 0 ? '#3fb950' : v < 0 ? '#f85149' : 'var(--text2)'; }
function _fmtUsd(v) {
  const s = v >= 0 ? '+' : '';
  const abs = Math.abs(v);
  const dec = (abs > 0 && abs < 1) ? 4 : 2;
  return s + '$' + abs.toFixed(dec);
}
function _fmtPct(v) { const s = v >= 0 ? '+' : ''; return s + v.toFixed(2) + '%'; }

function renderAIThinking(d) {
  const fg = d.fear_greed || {};
  const fgColor = fg.value <= 25 ? '#f85149' : fg.value >= 75 ? '#3fb950' : '#d29922';
  const ts = d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : '—';
  document.getElementById('aitSubtitle').textContent = T('aitLastAnalysis').replace('{ts}', ts).replace('{c}', d.cycle || '—');

  // Global KPIs
  const ddColor = (d.drawdown_pct||0) > 10 ? '#f85149' : (d.drawdown_pct||0) > 5 ? '#d29922' : '#3fb950';
  let html = `<div class="ait-global">
    <div class="ait-kpi">
      <div class="ait-kpi-val" style="color:${fgColor}">${fg.value || '—'}</div>
      <div class="ait-kpi-lbl">Fear & Greed</div>
      <div style="font-size:10px;color:${fgColor};margin-top:2px;">${fg.label || ''}</div>
    </div>
    <div class="ait-kpi">
      <div class="ait-kpi-val">$${(d.equity||0).toFixed(2)}</div>
      <div class="ait-kpi-lbl">Equity</div>
    </div>
    <div class="ait-kpi">
      <div class="ait-kpi-val" style="color:${ddColor}">-${(d.drawdown_pct||0).toFixed(1)}%</div>
      <div class="ait-kpi-lbl">Drawdown</div>
    </div>
    <div class="ait-kpi">
      <div class="ait-kpi-val">${d.open_positions || 0}</div>
      <div class="ait-kpi-lbl">${T('aitKpiOpenPos')}</div>
    </div>
  </div>
  <div class="ait-body">`;

  // Per-position cards — wild chain info integrated inline (no duplicate cards)
  const chains = d.wild_chains || {};
  const positions = d.positions || [];
  if (positions.length === 0) {
    html += `<div style="text-align:center;padding:30px;color:var(--text2);">${T('aitEmpty')}</div>`;
  }
  for (const p of positions) {
    const pnlColor = _pnlColor(p.pnl_usd || 0);
    const alignClass = (p.alignment||'').includes('SPLIT') ? 'ait-align-split' : 'ait-align-agree';
    const confPct = Math.round((p.confidence || 0) * 100);
    const action = p.action || p.llm_action || 'HOLD';
    const badgeHtml = _actionBadge(action);
    // Build quant reason chips
    const reasons = (p.quant_reasons || []).map(r => {
      const isGood = /TP_|RR_|PROFIT|GAIN|BULL/i.test(r);
      const isBad  = /SL_|LOSS|BEAR|LIQUIDAT|PANIC|FEAR/i.test(r);
      const cls = isGood ? 'good' : isBad ? 'bad' : '';
      return `<span class="ait-reason-chip ${cls}">${r}</span>`;
    }).join('');

    // Distance bars
    const sl = p.dist_sl_pct || 0;
    const tp = p.dist_tp_pct || 0;
    const total = sl + tp || 1;
    const slW = Math.min(100, (sl / total) * 100).toFixed(0);
    const tpW = Math.min(100, (tp / total) * 100).toFixed(0);

    html += `<div class="ait-pos-card">
      <div class="ait-pos-header">
        <span class="ait-sym">${p.symbol}</span>
        <span class="badge badge-${p.direction}">${(p.direction||'').toUpperCase()}</span>
        ${badgeHtml}
        <div class="ait-conf-wrap" style="width:120px;">
          <div class="ait-conf-track"><div class="ait-conf-fill" style="width:${confPct}%"></div></div>
          <span class="ait-conf-lbl">${confPct}%</span>
        </div>
        <span class="ait-pos-pnl" style="color:${pnlColor}">${_fmtUsd(p.pnl_usd||0)} (${_fmtPct(p.pnl_pct||0)})</span>
      </div>
      <div class="ait-pos-body">
        <div>
          <div style="font-size:10px;color:var(--text2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px;">${T('aitQuantAnalysis')}</div>
          <div class="ait-align ${alignClass}" style="margin-bottom:8px;">${escHtml(p.alignment || '—')}</div>
          <div class="ait-reasons">${reasons || '<span style="font-size:11px;color:var(--text2);">' + T('aitNoQuant') + '</span>'}</div>
        </div>
        <div>
          <div style="font-size:10px;color:var(--text2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px;">${T('aitDistances')}</div>
          <div class="ait-metrics" style="margin-bottom:8px;">
            <div class="ait-metric">SL <span style="color:#f85149">${sl.toFixed(2)}%</span></div>
            <div class="ait-metric">TP <span style="color:#3fb950">${tp.toFixed(2)}%</span></div>
            <div class="ait-metric">R/R <span>${(p.rr_remaining||0).toFixed(2)}x</span></div>
            <div class="ait-metric">${T('aitOpenFor')} <span>${p.hours_open||0}h</span></div>
          </div>
          <div class="ait-dist-row">
            <span style="color:#f85149;min-width:18px;">SL</span>
            <div class="ait-dist-bar">
              <div class="ait-dist-sl" style="width:${slW}%"></div>
              <div class="ait-dist-tp" style="width:${tpW}%"></div>
            </div>
            <span style="color:#3fb950;min-width:18px;">TP</span>
          </div>
        </div>
        ${(()=>{
          const chain = chains[p.symbol];
          if (!chain) return '';
          const craw = (chain.last_decision || {}).raw || {};
          const cval = (chain.last_decision || {}).validated || {};
          const cguards = cval.guardrails_hit || [];
          const cpnlColor = _pnlColor(chain.chain_pnl || 0);
          let chtml = '<div class=\"ait-wild-inline\">';
          const cRlang = (chain.last_decision || {}).reasoning_lang || 'es';
          const cLangTag = cRlang !== currentLang ? ' (' + cRlang.toUpperCase() + ')' : '';
          chtml += '<div class=\"ait-wild-inline-header\">' + T('aitMartingale') + cLangTag;
          chtml += '<span class=\"ait-chain-meta\">' + T('aitChainMeta').replace('{n}', chain.n_levels||1).replace('{m}', (chain.total_margin||0).toFixed(2)) + '</span>';
          chtml += '<span class=\"ait-chain-pnl\" style=\"color:' + cpnlColor + ';font-weight:700;\">' + _fmtUsd(chain.chain_pnl||0) + '</span>';
          chtml += '</div>';
          if (craw.reasoning && !craw.reasoning.startsWith('fallback')) {
            chtml += '<div class=\"ait-wild-inline-text\">' + escHtml(craw.reasoning) + '</div>';
          }
          cguards.forEach(g => { chtml += '<div class=\"ait-guardrail\">🛡️ ' + escHtml(g) + '</div>'; });
          chtml += '</div>';
          return chtml;
        })()}
        <div class="ait-llm-block">
          <div class="ait-llm-title">${T('aitLlmTitle')}${(p.llm_reasoning_lang||'es')!==currentLang?' ('+(p.llm_reasoning_lang||'es').toUpperCase()+')':''} <span style="color:var(--text2);font-weight:400;text-transform:none;">${p.llm_source ? '· ' + p.llm_source : ''}</span></div>
          ${(function(){
              // v2.12.15: LLM eval metadata — explica delta entre header (live) y LLM text (cycle-stale)
              if (!p.llm_evaluated_at) return '';
              const evalTs = Date.parse(p.llm_evaluated_at);
              if (isNaN(evalTs)) return '';
              const ageSec = Math.max(0, Math.floor((Date.now() - evalTs) / 1000));
              const ageStr = ageSec < 60 ? ageSec+'s' : (ageSec<3600 ? Math.floor(ageSec/60)+'m'+(ageSec%60)+'s' : Math.floor(ageSec/3600)+'h');
              const llmPnl = (p.llm_pnl_pct_at_eval || 0);
              const livePnl = (p.pnl_pct || 0);
              const delta = Math.abs(livePnl - llmPnl);
              const staleWarn = ageSec > 300 ? ' <span class="ait-stale">· ⚠ eval antigua</span>' : '';
              const driftWarn = delta > 0.2 ? ' <span class="ait-drift">· precio movió '+(livePnl - llmPnl >= 0 ? '+' : '')+(livePnl - llmPnl).toFixed(2)+'% desde análisis</span>' : '';
              return '<div class="ait-eval-meta">Evaluado hace '+ageStr+' · LLM vio pnl '+(llmPnl>=0?'+':'')+llmPnl.toFixed(2)+'%'+staleWarn+driftWarn+'</div>';
          })()}
          <div class="ait-llm-text">${p.llm_reasoning && p.llm_reasoning.length > 15 && !p.llm_reasoning.includes('workdir:') ? escHtml(p.llm_reasoning) : '<span style="color:var(--text2);font-style:italic;">' + T('aitLlmAnalyzing') + '</span>'}</div>
        </div>
      </div>
    </div>`;
  }

  html += `</div>`; // end ait-body
  document.getElementById('aitContent').innerHTML = html;
}

async function loadTrades() {
  const r = await fetch('/api/trades?limit=500');
  const d = await r.json();
  allTrades = d.trades || [];
  populateFilters();
  filterTrades();
}

async function loadLog() {
  const r = await fetch('/api/log');
  const d = await r.json();
  const el = document.getElementById('logContainer');
  if (!d.lines || d.lines.length === 0) {
    el.innerHTML = '<div class="log-info">' + T('logEmpty') + '</div>';
    return;
  }
  el.innerHTML = d.lines.map(line => {
    let cls = 'log-info';
    if (/✅|exitosamente|código: 0/i.test(line)) cls = 'log-ok';
    else if (/⚠️|error|fail|código: 1|exception/i.test(line)) cls = 'log-err';
    else if (/⚙️|🔄|ejecutando/i.test(line)) cls = 'log-warn';
    return `<div class="log-line ${cls}">${escHtml(line)}</div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}

let _resetHistoryData = [];

function showResetDetail(idx) {
  const h = _resetHistoryData[idx];
  if (!h) return;
  const dateStr = new Date(h.reset_date).toLocaleString();
  document.getElementById('rmSym').textContent = dateStr + '  ·  $' + (h.initial_capital||0).toFixed(0) + ' → $' + (h.new_capital||0).toFixed(0);
  document.getElementById('rmType').textContent = 'Reset — ' + (h.reason || 'Manual');
  let body = (h.notes && h.notes.trim()) ? h.notes : '(sin notas)';
  if (h.attachments && h.attachments.length) {
    body += '\n\n📎 Archivos adjuntos (' + h.attachments.length + '):\n';
    h.attachments.forEach(f => {
      const name = f.split('/').pop();
      const ext = (name.split('.').pop()||'').toLowerCase();
      const isImg = ['png','jpg','jpeg','gif','webp'].includes(ext);
      body += (isImg ? '🖼 ' : '📄 ') + name + '  →  /api/reset-file/' + f + '\n';
    });
  }
  document.getElementById('rmBody').textContent = body;
  document.getElementById('reasonModal').classList.add('open');
}

async function loadResetHistory() {
  const r = await fetch('/api/reset-history');
  const d = await r.json();
  const el = document.getElementById('resetHistoryTable');
  
  if (!d.history || d.history.length === 0) {
    el.innerHTML = `<div class="empty">${TRANSLATIONS[currentLang].noResets}</div>`;
    return;
  }
  
  // Ordenar por fecha (más reciente primero)
  const history = d.history.sort((a, b) => new Date(b.reset_date) - new Date(a.reset_date));
  _resetHistoryData = history;
  
  let html = `<table>
    <thead>
      <tr>
        <th>#</th>
        <th>Fecha Reset</th>
        <th>Capital Inicial</th>
        <th>Capital Final</th>
        <th>Retorno</th>
        <th>Trades</th>
        <th>W/L/F</th>
        <th>Win Rate</th>
        <th>Mejor</th>
        <th>Peor</th>
      </tr>
    </thead>
    <tbody>`;
  
  history.forEach((h, i) => {
    const num = history.length - i;
    const date = new Date(h.reset_date).toLocaleDateString('es-MX', { day: '2-digit', month: 'short', year: '2-digit', hour: '2-digit', minute: '2-digit' });
    const ret = h.return_pct || 0;
    const returnCls = ret >= 0 ? 'positive' : 'negative';
    const returnSign = ret >= 0 ? '+' : '';
    const wr = h.win_rate || 0;
    const best = h.best_trade_usd || 0;
    const worst = h.worst_trade_usd || 0;
    const flat = h.flat || 0;

    const hasNotes = h.notes && h.notes.trim();
    const hasFiles = h.attachments && h.attachments.length > 0;
    const notesBadge = hasNotes ? `<span onclick="showResetDetail(${i})" title="Ver notas" style="cursor:pointer;color:var(--yellow);margin-left:4px;">💬</span>` : '';
    const filesBadge = hasFiles ? `<span onclick="showResetDetail(${i})" style="cursor:pointer;color:var(--blue);margin-left:4px;" title="Ver adjuntos">📎${h.attachments.length}</span>` : '';
    html += `<tr>
      <td><strong>${num}</strong></td>
      <td>${date}${notesBadge}${filesBadge}</td>
      <td>$${(h.initial_capital||0).toFixed(2)}</td>
      <td>$${(h.final_capital||0).toFixed(2)}</td>
      <td class="${returnCls}">${returnSign}${ret.toFixed(2)}%</td>
      <td>${h.total_trades||0}</td>
      <td><span class="positive">${h.wins||0}</span>/<span class="negative">${h.losses||0}</span>/${flat}</td>
      <td>${wr.toFixed(1)}%</td>
      <td class="positive">$${best.toFixed(2)}</td>
      <td class="negative">$${worst.toFixed(2)}</td>
    </tr>`;
  });
  
  html += '</tbody></table>';
  
  // Agregar resumen de tendencia
  if (history.length >= 2) {
    const avgReturn = history.reduce((sum, h) => sum + (h.return_pct || 0), 0) / history.length;
    const avgWinRate = history.reduce((sum, h) => sum + (h.win_rate || 0), 0) / history.length;
    const improving = (history[0].return_pct || 0) > (history[1].return_pct || 0);
    const trend = improving ? '📈 Mejorando' : '📉 Empeorando';
    const trendCls = improving ? 'positive' : 'negative';
    
    html += `<div style="margin-top: 12px; padding: 10px; background: var(--bg3); border-radius: 6px; font-size: 12px;">
      <strong>Tendencia:</strong> <span class="${trendCls}">${trend}</span> |
      <strong>Retorno Promedio:</strong> ${avgReturn >= 0 ? '+' : ''}${avgReturn.toFixed(2)}% |
      <strong>Win Rate Promedio:</strong> ${avgWinRate.toFixed(1)}%
    </div>`;
  }
  
  el.innerHTML = html;
}

// ── Filtering / Pagination ─────────────────────────────────────────────────
function populateFilters() {
  const symbols = [...new Set(allTrades.map(t => t.symbol))].sort();
  const sel = document.getElementById('filterSymbol');
  const current = sel.value;
  sel.innerHTML = `<option value="">${TRANSLATIONS[currentLang].allSymbols}</option>`;
  symbols.forEach(s => sel.innerHTML += `<option value="${s}" ${s === current ? 'selected' : ''}>${s}</option>`);
}

function filterTrades() {
  const sym = document.getElementById('filterSymbol').value;
  const res = document.getElementById('filterResult').value;
  const dir = document.getElementById('filterDir').value;
  filteredTrades = allTrades.filter(t => {
    if (sym && t.symbol !== sym) return false;
    if (dir && t.direction !== dir) return false;
    if (res) {
      if (res === 'win' && t.pnl_usd <= 0) return false;
      if (res === 'loss' && t.pnl_usd >= 0) return false;
      if (res === 'flat' && t.pnl_usd !== 0) return false;
      if (res === 'nonflat' && t.pnl_usd === 0) return false;
    }
    return true;
  });
  currentPage = 1;
  renderTradesPage();
  updateTradesSummary();
}

function updateTradesSummary() {
  const el = document.getElementById('tradesSummary');
  if (!el) return;
  if (!filteredTrades.length) { el.style.display = 'none'; return; }

  const wins   = filteredTrades.filter(t => (t.pnl_usd || 0) > 0);
  const losses = filteredTrades.filter(t => (t.pnl_usd || 0) < 0);
  const net    = filteredTrades.reduce((s, t) => s + (t.pnl_usd || 0), 0);
  const winSum = wins.reduce((s, t) => s + (t.pnl_usd || 0), 0);
  const lssSum = losses.reduce((s, t) => s + (t.pnl_usd || 0), 0);
  const wr     = filteredTrades.length
    ? (wins.length / filteredTrades.length * 100).toFixed(1)
    : '0.0';

  const netCls = net >= 0 ? 'pos' : 'neg';
  el.innerHTML =
    '<span><span class="ts-label">NET P&L</span>' +
    '<span class="ts-val ' + netCls + '">' + (net >= 0 ? '+' : '') + fmt$(net) + '</span></span>' +
    '<span class="ts-sep">·</span>' +
    '<span><span class="ts-label">' + wins.length + ' WIN</span>' +
    '<span class="ts-val pos">+' + fmt$(winSum) + '</span></span>' +
    '<span class="ts-sep">·</span>' +
    '<span><span class="ts-label">' + losses.length + ' LOSS</span>' +
    '<span class="ts-val neg">' + fmt$(lssSum) + '</span></span>' +
    '<span class="ts-sep">·</span>' +
    '<span><span class="ts-label">Win Rate</span>' +
    '<span class="ts-val">' + wr + '%</span></span>';
  el.style.display = 'flex';
}

function renderTradesPage() {
  const start = (currentPage - 1) * PAGE_SIZE;
  const page = filteredTrades.slice(start, start + PAGE_SIZE);
  const wrap = document.getElementById('tradesTable');

  if (page.length === 0) {
    wrap.innerHTML = '<div class="empty">' + T('noTrades') + '</div>';
    document.getElementById('tradesPagination').innerHTML = '';
    return;
  }

  let html = `<table><thead><tr>
    <th>#</th><th>Fecha</th><th>Símbolo</th><th>Dirección</th>
    <th>Estrategia</th><th>Entrada</th><th>Salida</th>
    <th>P&L $</th><th>P&L %</th><th>Resultado</th><th>Razón cierre</th>
  </tr></thead><tbody>`;

  // Store trade data for modal lookups — avoids HTML attribute quote-escaping issues
  window._tradeReasonData = {};
  page.forEach((t, i) => {
    const idx = start + i;
    window._tradeReasonData[idx] = {
      symbol: t.symbol, closeReason: t.close_reason || '',
      aiReasoning: t.ai_reasoning || '', pnl: t.pnl_usd, closeTime: t.close_time || ''
    };
  });

  page.forEach((t, i) => {
    const idx = start + i;
    const pnlCls = t.pnl_usd > 0 ? 'pos' : t.pnl_usd < 0 ? 'neg' : '';
    const resultBadge = t.pnl_usd > 0 ? 'badge-win' : t.pnl_usd < 0 ? 'badge-loss' : 'badge-hold';
    const resultTxt = t.pnl_usd > 0 ? 'WIN' : t.pnl_usd < 0 ? 'LOSS' : 'FLAT';
    const reasonLabel = t.close_reason || '—';
    const reasonCell = `<button class="reason-btn" onclick="showReasonByIdx(${idx})" title="Ver razonamiento de la IA">${reasonLabel}</button>`;
    html += `<tr>
      <td class="text2">${idx + 1}</td>
      <td>${t.open_time ? t.open_time.replace('T',' ').slice(0,16) : '—'}</td>
      <td><strong>${t.symbol}</strong></td>
      <td><span class="badge badge-${t.direction}">${t.direction.toUpperCase()}</span></td>
      <td>${t.strategy || '—'}</td>
      <td>${fmtPrice(t.entry_price)}</td>
      <td>${fmtPrice(t.close_price)}</td>
      <td class="${pnlCls}">${t.pnl_usd >= 0 ? '+' : ''}${fmt$(t.pnl_usd)}</td>
      <td class="${pnlCls}">${fmtPct(t.pnl_pct, true)}</td>
      <td><span class="badge ${resultBadge}">${resultTxt}</span></td>
      <td>${reasonCell}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / PAGE_SIZE);
  let pg = `<span class="page-info">${filteredTrades.length} trades · Pág ${currentPage}/${totalPages}</span>`;
  if (currentPage > 1) pg += `<button class="page-btn" onclick="goPage(${currentPage-1})">← ${TRANSLATIONS[currentLang].prev}</button>`;
  // page numbers
  const from = Math.max(1, currentPage - 2), to = Math.min(totalPages, currentPage + 2);
  for (let p = from; p <= to; p++) {
    pg += `<button class="page-btn ${p===currentPage?'active':''}" onclick="goPage(${p})">${p}</button>`;
  }
  if (currentPage < totalPages) pg += `<button class="page-btn" onclick="goPage(${currentPage+1})">${TRANSLATIONS[currentLang].next} →</button>`;
  document.getElementById('tradesPagination').innerHTML = pg;
}

function goPage(p) { currentPage = p; renderTradesPage(); }

// ── Charts Init ────────────────────────────────────────────────────────────
function initCharts() {
  Chart.defaults.color = '#8b949e';
  Chart.defaults.borderColor = '#30363d';

  // Equity
  const equityCtx = document.getElementById('equityChart').getContext('2d');
  equityChart = new Chart(equityCtx, {
    type: 'line',
    data: { labels: [], datasets: [{
      label: 'Equity',
      data: [], fill: true,
      borderColor: '#58a6ff', borderWidth: 2,
      backgroundColor: 'rgba(88,166,255,0.08)',
      tension: 0.3, pointRadius: 0, pointHoverRadius: 4
    },{
      label: TRANSLATIONS[currentLang].winners,
      data: [], type: 'scatter',
      backgroundColor: 'rgba(63,185,80,0.8)',
      pointStyle: 'triangle', pointRadius: 5, showLine: false
    },{
      label: TRANSLATIONS[currentLang].losers,
      data: [], type: 'scatter',
      backgroundColor: 'rgba(248,81,73,0.8)',
      pointStyle: 'rectRot', pointRadius: 4, showLine: false
    }]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex === 0) return ' Equity: $' + ctx.parsed.y.toFixed(2);
              return ' P&L: $' + (ctx.raw.pnl || 0).toFixed(2);
            }
          }
        },
        zoom: {
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
          pan: { enabled: true, mode: 'x' }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(48,54,61,0.5)' }, ticks: { maxTicksLimit: 8, font: { size: 10 } } },
        y: {
          grid: { color: 'rgba(48,54,61,0.5)' },
          ticks: { callback: v => '$' + v.toFixed(0), font: { size: 10 } }
        }
      }
    }
  });

  // Pie
  const pieCtx = document.getElementById('pieChart').getContext('2d');
  pieChart = new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: [TRANSLATIONS[currentLang].winners, TRANSLATIONS[currentLang].losers],
      datasets: [{ data: [0,0], backgroundColor: ['rgba(63,185,80,0.8)','rgba(248,81,73,0.8)'], borderWidth: 2, borderColor: '#161b22' }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: { position: 'right', labels: { font: { size: 11 }, boxWidth: 10 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}` } }
      }
    }
  });

  // Symbol bar
  const symCtx = document.getElementById('symbolChart').getContext('2d');
  symbolChart = new Chart(symCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], borderWidth: 1, borderRadius: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ' P&L: $' + ctx.parsed.y.toFixed(2) } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        y: { grid: { color: 'rgba(48,54,61,0.5)' }, ticks: { callback: v => '$' + v.toFixed(1), font: { size: 10 } } }
      }
    }
  });

  // Histogram
  const histCtx = document.getElementById('histChart').getContext('2d');
  histChart = new Chart(histCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], borderWidth: 1, borderRadius: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` Trades: ${ctx.parsed.y}` } } },
      scales: {
        x: { grid: { display: false }, title: { display: true, text: 'P&L por Trade ($)', font: { size: 10 } }, ticks: { maxTicksLimit: 10, font: { size: 10 } } },
        y: { grid: { color: 'rgba(48,54,61,0.5)' }, title: { display: true, text: 'Frecuencia', font: { size: 10 } }, ticks: { font: { size: 10 } } }
      }
    }
  });
}

function updateEquityChart(range) {
  if (!equityAllData.equity) return;
  let eq = equityAllData.equity;
  let lb = equityAllData.labels;
  let tr = equityAllData.trade_pnls;
  let wi = equityAllData.winners;
  let lo = equityAllData.losers;

  if (range !== 'all' && typeof range === 'number') {
    const n = eq.length;
    const from = Math.max(0, n - range);
    eq = eq.slice(from);
    lb = lb.slice(from);
    tr = tr.slice(from);
    // filter scatter points within range
    wi = (wi || []).filter(p => p.x >= from).map(p => ({...p, x: p.x - from}));
    lo = (lo || []).filter(p => p.x >= from).map(p => ({...p, x: p.x - from}));
  }

  equityChart.data.labels = lb;
  equityChart.data.datasets[0].data = eq;
  equityChart.data.datasets[1].data = (wi||[]).map(p => ({ x: p.x, y: p.y, pnl: p.pnl }));
  equityChart.data.datasets[2].data = (lo||[]).map(p => ({ x: p.x, y: p.y, pnl: p.pnl }));
  equityChart.update();
}

function setEquityRange(range, btn) {
  document.querySelectorAll('.chart-toolbar .tb-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  updateEquityChart(range);
}

function resetZoom() { equityChart.resetZoom(); }

// ── Helpers ────────────────────────────────────────────────────────────────
function fmt$(v) {
  // v2.12.4-live fmt$: show higher precision for small pnls (< $1 abs uses 4 decimals)
  const n = v || 0;
  const abs = Math.abs(n);
  if (abs > 0 && abs < 1) return '$' + n.toFixed(4);
  return '$' + n.toFixed(2);
}
function fmtNum(v, d=2) { return (v || 0).toFixed(d); }
function fmtPct(v, sign=false) {
  const n = (v || 0).toFixed(2);
  return (sign && v > 0 ? '+' : '') + n + '%';
}
function fmtPrice(v) {
  if (!v || v === 0) return '—';
  if (v < 0.0001) return v.toExponential(4);
  if (v < 1) return '$' + v.toFixed(8);
  if (v < 10) return '$' + v.toFixed(4);
  if (v < 1000) return '$' + v.toFixed(2);
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Agent Chat (SSE real-time) ────────────────────────────────────────────
const TRANSLATIONS = {
  es: {
    tradeHistory:"📋 Historial de Trades", openPositions:"⚡ Posiciones Abiertas",
    equityCurve:"📈 Equity Curve", winsVsLosses:"🥧 Wins vs Losses",
    pnlBySymbol:"📊 P&L por Símbolo", pnlDistribution:"📉 Distribución de P&L por Trade",
    resetHistory:"📊 Historial de Resets", chatTitle:"💬 Agente de Trading — Chat en Vivo",
    advancedMetrics:"🧮 Métricas Avanzadas", watchdogLog:"🔍 Watchdog Log",
    agentName:"Solana Trading Agent",
    closeAll:"Cerrar Todas", close:"Cerrar", send:"Enviar ▸", clearChat:"🗑 Limpiar",
    aiThinking:"🧠 Qué está pensando la IA",
    allSymbols:"Todos los símbolos", allResults:"Todos los resultados",
    winners:"Ganadores", losers:"Perdedores",
    flat:"Sin cambio (FLAT)", nonFlat:"Solo W/L (excl. FLAT)", longShort:"Long & Short",
    loadingPositions:"Cargando posiciones...", loadingTrades:"Cargando trades...",
    connectingAgent:"Conectando con el agente...", noResets:"Sin resets registrados",
    noNotes:"Sin notas aún. El agente dejará actualizaciones aquí.",
    noTrades:"Sin trades que mostrar", noOpenPositions:"⚡ Sin posiciones abiertas en este momento",
    thinking:"Pensando...", analyzingPos:"Analizando posiciones...",
    reviewingMarket:"Revisando mercado...", preparingResp:"Preparando respuesta...", writing:"Escribiendo...",
    chatHint:"🚀 El agente responde automáticamente. Presiona Enter para enviar.",
    chatPlaceholder:"Escribe una nota o pregunta al agente...",
    notes:"notas", note:"nota", autoRefresh:"🔄 Auto-refresh cada 30s",
    invalidCapital:"Capital inválido (mínimo $100)",
    confirmReset:"⚠️ ¿Resetear el bot a ${capital}?\n\nEsto borrará TODO el historial de trades.",
    resetSuccess:"✅ Bot reseteado a ${capital}\n\nLa página se recargará.",
    confirmCloseAll:"¿Cerrar TODAS las posiciones abiertas?",
    confirmCloseOne:"¿Cerrar posición de ${symbol}?",
    confirmClearChat:"¿Limpiar el chat?\nEl historial queda guardado para el agente.",
    confirmWildOn:"Activar MODO SALVAJE?\n\nLa IA abrirá coberturas martingala automáticamente.",
    confirmWildOff:"¿Desactivar Modo Salvaje? (las posiciones quedan abiertas)",
    unrealized:"Unrealized", noExposure:"sin exposición", prev:"Anterior", next:"Siguiente",
    // ── AI Thinking modal (v2.7.0) ──
    aitTitle:"Qué está pensando la IA",
    aitSubtitleDefault:"Análisis en tiempo real de cada posición abierta",
    aitLive:"EN VIVO", aitClose:"✕ Cerrar",
    aitLoading:"Consultando a la IA...",
    aitCountdown:"Actualiza en {n}s",
    aitError:"Error consultando la IA:",
    aitLastAnalysis:"Último análisis: {ts} · Ciclo #{c}",
    aitKpiOpenPos:"Posiciones abiertas",
    aitEmpty:"Sin posiciones abiertas para analizar",
    aitQuantAnalysis:"Análisis cuantitativo",
    aitNoQuant:"Sin señales cuantitativas",
    aitDistances:"Distancias", aitOpenFor:"Abierta",
    aitMartingale:"🔥 Motor Martingala",
    aitChainMeta:"{n} nivel(es) · ${m} margen",
    aitLlmTitle:"🤖 Razonamiento del LLM",
    aitLlmAnalyzing:"Analizando condiciones del mercado...",
    badgeHold:"⏸ HOLD — Esperar", badgeOpen:"➕ ABRIR COBERTURA",
    badgeCloseChain:"✅ CERRAR CADENA", badgeClose:"✅ CERRAR",
    badgeAbandon:"🚨 ABANDONAR",
    logEmpty:"Sin log disponible", logLoading:"Cargando log...",
    closeBtnCell:"Cerrar", changelogLoading:"Cargando...",
  },
  en: {
    tradeHistory:"📋 Trade History", openPositions:"⚡ Open Positions",
    equityCurve:"📈 Equity Curve", winsVsLosses:"🥧 Wins vs Losses",
    pnlBySymbol:"📊 P&L by Symbol", pnlDistribution:"📉 P&L Distribution per Trade",
    resetHistory:"📊 Reset History", chatTitle:"💬 Trading Agent — Live Chat",
    advancedMetrics:"🧮 Advanced Metrics", watchdogLog:"🔍 Watchdog Log",
    agentName:"Solana Trading Agent",
    closeAll:"Close All", close:"Close", send:"Send ▸", clearChat:"🗑 Clear",
    aiThinking:"🧠 What is the AI thinking",
    allSymbols:"All symbols", allResults:"All results",
    winners:"Winners", losers:"Losers",
    flat:"No change (FLAT)", nonFlat:"W/L only (excl. FLAT)", longShort:"Long & Short",
    loadingPositions:"Loading positions...", loadingTrades:"Loading trades...",
    connectingAgent:"Connecting to agent...", noResets:"No resets recorded",
    noNotes:"No notes yet. The agent will leave updates here.",
    noTrades:"No trades to show", noOpenPositions:"⚡ No open positions at this time",
    thinking:"Thinking...", analyzingPos:"Analyzing positions...",
    reviewingMarket:"Reviewing market...", preparingResp:"Preparing response...", writing:"Writing...",
    chatHint:"🚀 The agent responds automatically. Press Enter to send.",
    chatPlaceholder:"Write a note or question to the agent...",
    notes:"notes", note:"note", autoRefresh:"🔄 Auto-refresh every 30s",
    invalidCapital:"Invalid capital (minimum $100)",
    confirmReset:"⚠️ Reset bot to ${capital}?\n\nThis will erase ALL trade history.",
    resetSuccess:"✅ Bot reset to ${capital}\n\nPage will reload.",
    confirmCloseAll:"Close ALL open positions?",
    confirmCloseOne:"Close position for ${symbol}?",
    confirmClearChat:"Clear chat?\nHistory is saved for the agent.",
    confirmWildOn:"Activate WILD MODE?\n\nThe AI will open martingale hedges automatically.",
    confirmWildOff:"Deactivate Wild Mode? (positions remain open)",
    unrealized:"Unrealized", noExposure:"no exposure", prev:"Previous", next:"Next",
    // ── AI Thinking modal (v2.7.0) ──
    aitTitle:"What is the AI thinking",
    aitSubtitleDefault:"Real-time analysis of each open position",
    aitLive:"LIVE", aitClose:"✕ Close",
    aitLoading:"Querying the AI...",
    aitCountdown:"Refresh in {n}s",
    aitError:"Error querying the AI:",
    aitLastAnalysis:"Last analysis: {ts} · Cycle #{c}",
    aitKpiOpenPos:"Open positions",
    aitEmpty:"No open positions to analyze",
    aitQuantAnalysis:"Quantitative analysis",
    aitNoQuant:"No quantitative signals",
    aitDistances:"Distances", aitOpenFor:"Open for",
    aitMartingale:"🔥 Martingale Engine",
    aitChainMeta:"{n} level(s) · ${m} margin",
    aitLlmTitle:"🤖 LLM Reasoning",
    aitLlmAnalyzing:"Analyzing market conditions...",
    badgeHold:"⏸ HOLD — Wait", badgeOpen:"➕ OPEN HEDGE",
    badgeCloseChain:"✅ CLOSE CHAIN", badgeClose:"✅ CLOSE",
    badgeAbandon:"🚨 ABANDON",
    logEmpty:"No log available", logLoading:"Loading log...",
    closeBtnCell:"Close", changelogLoading:"Loading...",
  }
};

let currentLang = localStorage.getItem('lang') || 'es';

// Helper: resuelve clave de traducción con fallback al propio key
const T = (key) => (TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][key]) || key;

function applyLang() {
  const t = TRANSLATIONS[currentLang];
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (t[key] !== undefined) el.textContent = t[key];
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (t[key] !== undefined) el.placeholder = t[key];
  });
  const btn = document.getElementById('langToggle');
  if (btn) btn.textContent = currentLang === 'es' ? '🇬🇧 EN' : '🇪🇸 ES';
  fetch('/api/set-language', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({language: currentLang})}).catch(()=>{});
  // Re-render modal AI Thinking si está abierto — fuerza traducción de contenido dinámico
  const aitModal = document.getElementById('aiThinkingModal');
  if (aitModal && aitModal.style.display === 'flex') {
    fetchAIThinking();
  }
}

function toggleLang() {
  currentLang = currentLang === 'es' ? 'en' : 'es';
  localStorage.setItem('lang', currentLang);
  applyLang();
  // v2.8.1: invalida cache WILD para regenerar reasoning en nuevo idioma
  fetch('/api/force-reasoning-refresh', {method:'POST'}).catch(()=>{});
}

document.addEventListener('DOMContentLoaded', () => { applyLang(); _initResetDrop(); });

let chatSSE = null;
let chatLastCount = 0;
let chatTypingInterval = null;

function renderChat(notes){
  const msgs = notes.messages || [];
  const count = msgs.length;
  const t_b=TRANSLATIONS[currentLang]; document.getElementById('chatBadge').textContent = count + ' ' + (count===1?t_b.note:t_b.notes);
  if(count===0){
    document.getElementById('chatMessages').innerHTML=`<div class="empty">${TRANSLATIONS[currentLang].noNotes}</div>`;
    return;
  }
  let h='';
  for(const m of msgs){
    const cls = m.sender==='agent' ? 'cmsg-agent' : 'cmsg-user';
    const label = m.sender==='agent' ? '🤖 Agente' : '👤 Ender';
    const time = new Date(m.ts).toLocaleString('es-MX',{timeZone:'America/Denver',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    h += '<div class="cmsg '+cls+'"><div class="cmsg-meta">'+label+' · '+time+'</div><div class="cmsg-bubble">'+escHtml(m.text)+'</div></div>';
  }
  document.getElementById('chatMessages').innerHTML = h;
  const box = document.getElementById('chatMessages');
  box.scrollTop = box.scrollHeight;
  // New agent message arrived → hide typing
  if(count > chatLastCount && chatLastCount > 0){
    const latest = msgs[msgs.length-1];
    if(latest && latest.sender==='agent'){
      showChatTyping(false);
      if(window._chatTimeout){clearTimeout(window._chatTimeout);window._chatTimeout=null;}
    }
  }
  chatLastCount = count;
}

function showChatTyping(show){
  const el = document.getElementById('chatTyping');
  if(show){
    el.className = 'typing-wrap active';
    const box = document.getElementById('chatMessages');
    box.scrollTop = box.scrollHeight;
    if(!chatTypingInterval){
      const t_s=TRANSLATIONS[currentLang]; const statuses=[t_s.thinking, t_s.analyzingPos, t_s.reviewingMarket, t_s.preparingResp, t_s.writing];
      let idx=0;
      chatTypingInterval = setInterval(()=>{
        idx=(idx+1)%statuses.length;
        document.getElementById('chatTypingText').textContent=statuses[idx];
      },3000);
    }
  } else {
    el.className = 'typing-wrap';
    if(chatTypingInterval){clearInterval(chatTypingInterval);chatTypingInterval=null;}
  }
}

async function clearChat(){
  if(!confirm(TRANSLATIONS[currentLang].confirmClearChat)) return;
  await fetch('/api/chat/clear',{method:'POST'});
  // SSE detecta el cambio y llama renderChat() automaticamente
}

async function sendChatNote(){
  const inp = document.getElementById('chatInput');
  const text = inp.value.trim();
  if(!text) return;
  inp.value = '';
  inp.disabled = true;
  document.getElementById('chatSendBtn').disabled = true;
  showChatTyping(true);
  // Typing se oculta cuando llega mensaje del agente via SSE (no polling)
  // Safety timeout: ocultar despues de 60s si no llega respuesta
  if(window._chatTimeout) clearTimeout(window._chatTimeout);
  window._chatTimeout = setTimeout(() => { showChatTyping(false); }, 60000);
  try{
    await fetch('/api/agent-notes',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,sender:'user'})
    });
  }catch(e){console.error('Chat send error:',e);showChatTyping(false);}
  inp.disabled = false;
  document.getElementById('chatSendBtn').disabled = false;
  inp.focus();
}

let chatPollInterval = null;
async function pollChatStatus(){
  if(chatPollInterval) return; // ya está corriendo
  chatPollInterval = setInterval(async () => {
    try {
      const r = await fetch('/api/chat-status');
      const d = await r.json();
      if (!d.thinking) {
        // Don't hide typing here - wait for actual agent message via SSE
        // Only stop polling
        clearInterval(chatPollInterval);
        chatPollInterval = null;
        // Give SSE 2 seconds to deliver the message before hiding
        setTimeout(() => {
          const el = document.getElementById('chatTyping');
          if(el.className.includes('active')) showChatTyping(false);
        }, 2000);
      }
    } catch(e) {
      clearInterval(chatPollInterval);
      chatPollInterval = null;
    }
  }, 2000);
}

function connectChatSSE(){
  if(chatSSE) chatSSE.close();
  chatSSE = new EventSource('/api/agent-notes/stream');
  chatSSE.addEventListener('init', function(e){
    try{renderChat(JSON.parse(e.data).notes);}catch(x){}
  });
  chatSSE.addEventListener('update', function(e){
    try{renderChat(JSON.parse(e.data).notes);}catch(x){}
  });
  chatSSE.onmessage = function(e){
    try{const d=JSON.parse(e.data);if(d.notes)renderChat(d.notes);}catch(x){}
  };
  chatSSE.onerror = function(){
    chatSSE.close();
    setTimeout(connectChatSSE, 3000);
  };
}

// Load chat on page load + connect SSE
(async function(){
  try{
    const d = await(await fetch('/api/agent-notes')).json();
    renderChat(d);
  }catch(e){}
  connectChatSSE();
})();
</script>

<!-- ── AI Thinking Modal ── -->
<div id="aiThinkingModal" onclick="if(event.target===this)closeAIThinking()">
  <div class="ait-box">
    <div class="ait-header">
      <div class="ait-header-left">
        <span class="ait-brain">🧠</span>
        <div>
          <div class="ait-title" data-i18n="aitTitle">Qué está pensando la IA</div>
          <div class="ait-subtitle" id="aitSubtitle" data-i18n="aitSubtitleDefault">Análisis en tiempo real de cada posición abierta</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;">
        <div class="ait-live"><div class="ait-live-dot"></div><span data-i18n="aitLive">EN VIVO</span></div>
        <span id="aitCountdown" style="font-size:10px;color:var(--text2);"></span>
        <button class="ait-close-btn" onclick="closeAIThinking()" data-i18n="aitClose">✕ Cerrar</button>
      </div>
    </div>
    <div class="ait-refresh-bar"><div class="ait-refresh-fill" id="aitRefillBar" style="width:100%"></div></div>
    <div id="aitContent"><div class="ait-loading"><div class="ait-loading-spinner">🧠</div><div data-i18n="aitLoading">Consultando a la IA...</div></div></div>
  </div>
</div>

<!-- ── AI Reasoning Modal ── -->
<div id="reasonModal" onclick="if(event.target===this)closeReasonModal()">
  <div class="reason-modal-box">
    <button class="reason-modal-close" onclick="closeReasonModal()">✕</button>
    <div class="reason-modal-title">🤖 Razonamiento de la IA</div>
    <div class="reason-modal-sym" id="rmSym"></div>
    <div class="reason-modal-type" id="rmType"></div>
    <div class="reason-modal-body" id="rmBody"></div>
  </div>
</div>

<!-- CHANGELOG MODAL -->
<div id="changelogModal" onclick="if(event.target===this)closeChangelog()">
  <div class="changelog-box">
    <button class="changelog-close" onclick="closeChangelog()">✕</button>
    <div class="changelog-title">📋 Historial de versiones</div>
    <div id="changelogContent" data-i18n="changelogLoading">Cargando...</div>
  </div>
</div>

</body>
</html>
"""

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/version')
def api_version():
    return jsonify({"version": VERSION, "changelog": CHANGELOG})


@app.route('/api/health')
def api_health():
    """v2.12.19 Phase 1: real-time live bot health probe.

    Returns JSON with orch liveness, heartbeat age, kill-switch state,
    close_failures counter, positions count, capital_usd.
    Used by cron health_check.py and external monitors.
    """
    import os as _os, subprocess as _sp, time as _time
    from pathlib import Path as _P

    bot_dir = _P("/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live")
    hb_file = _P("/tmp/solana_live_heartbeat")
    ks_file = _P("/tmp/solana_live_killswitch")
    cf_file = _P("/tmp/solana_live_close_failures")

    # Heartbeat
    hb_age = None
    if hb_file.exists():
        try:
            hb_age = _time.time() - float(hb_file.read_text().strip())
        except (ValueError, OSError):
            pass

    # Orch + watchdog liveness (CWD match)
    orch_alive = False
    orch_pid = None
    watchdog_alive = False
    watchdog_pid = None
    try:
        pids = _sp.check_output(["pgrep", "-f", "orchestrator.py"], text=True).strip().splitlines()
        for pid in pids:
            pid = pid.strip()
            try:
                if _os.readlink(f"/proc/{pid}/cwd") == str(bot_dir):
                    orch_alive = True
                    orch_pid = int(pid)
                    break
            except (OSError, FileNotFoundError):
                pass
    except _sp.CalledProcessError:
        pass

    try:
        lines = _sp.check_output(["pgrep", "-af", "run_watchdog.sh"], text=True).splitlines()
        for line in lines:
            pid = line.split()[0]
            try:
                if _os.readlink(f"/proc/{pid}/cwd") == str(bot_dir):
                    watchdog_alive = True
                    watchdog_pid = int(pid)
                    break
            except (OSError, FileNotFoundError):
                pass
    except _sp.CalledProcessError:
        pass

    # Kill switch
    kill_switch_active = ks_file.exists()
    kill_switch_reason = ks_file.read_text(errors="ignore")[:200].strip() if kill_switch_active else None

    # Close failures
    close_failures = 0
    if cf_file.exists():
        try:
            close_failures = int(cf_file.read_text().strip() or 0)
        except ValueError:
            close_failures = -1

    # Portfolio
    port = load_portfolio()
    open_positions = [p for p in port.get("positions", []) if p.get("status") == "open"]
    capital_usd = port.get("capital_usd", 0)
    total_trades = port.get("total_trades", 0)

    # Overall health
    healthy = (
        orch_alive and watchdog_alive and not kill_switch_active
        and (hb_age is not None and hb_age < 300)
        and close_failures < 3
    )

    return jsonify({
        "healthy": healthy,
        "version": VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "orch": {"alive": orch_alive, "pid": orch_pid},
        "watchdog": {"alive": watchdog_alive, "pid": watchdog_pid},
        "heartbeat": {
            "age_s": round(hb_age, 1) if hb_age is not None else None,
            "fresh": hb_age is not None and hb_age < 120,
        },
        "kill_switch": {"active": kill_switch_active, "reason": kill_switch_reason},
        "close_failures": {"count": close_failures, "threshold": 3},
        "portfolio": {
            "capital_usd": capital_usd,
            "open_positions": len(open_positions),
            "total_trades": total_trades,
            "symbols_open": [p.get("symbol") for p in open_positions],
        },
    })


# v2.12.28: wallet-total equity helper (on-chain USDC + SOL + tokens × prices).
# Caches for 60s to avoid spamming RPC on every dashboard refresh.
_WALLET_EQUITY_CACHE = {"ts": 0, "data": None}

def _fetch_wallet_equity():
    """Query on-chain wallet balances × prices. Returns dict with wallet_total,
    balances, prices, or None if unavailable (paper bot, RPC fail, etc.)."""
    import time as _time
    import os as _os
    try:
        import requests as _rq
    except ImportError:
        return None

    now = _time.time()
    if _WALLET_EQUITY_CACHE["data"] and (now - _WALLET_EQUITY_CACHE["ts"] < 60):
        return _WALLET_EQUITY_CACHE["data"]

    wallet = _os.environ.get("HOT_WALLET_ADDRESS", "").strip()
    if not wallet or wallet == "MOCK_NO_PUBKEY":
        return None

    rpc = _os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    mints = {
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "JUP":  "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "ETH":  "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
    }
    sol_mint = "So11111111111111111111111111111111111111112"

    try:
        balances = {}
        # Native SOL
        r = _rq.post(rpc, json={"jsonrpc": "2.0", "id": 1, "method": "getBalance",
                                "params": [wallet]}, timeout=5).json()
        balances["SOL"] = r["result"]["value"] / 1e9

        # SPL tokens
        for sym, mint in mints.items():
            r = _rq.post(rpc, json={
                "jsonrpc": "2.0", "id": 2, "method": "getTokenAccountsByOwner",
                "params": [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
            }, timeout=5).json()
            total = sum(float(a["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"] or 0)
                        for a in r.get("result", {}).get("value", []))
            balances[sym] = total

        # Prices from Jupiter (primary) with market_latest.json fallback
        sol_px = jup_px = eth_px = 0.0
        try:
            ids = f"{sol_mint},{mints['JUP']},{mints['ETH']}"
            px_r = _rq.get(f"https://price.jup.ag/v6/price?ids={ids}", timeout=5).json()
            px_data = px_r.get("data", {})
            sol_px = float(px_data.get(sol_mint, {}).get("price", 0) or 0)
            jup_px = float(px_data.get(mints["JUP"], {}).get("price", 0) or 0)
            eth_px = float(px_data.get(mints["ETH"], {}).get("price", 0) or 0)
        except Exception:
            pass
        # v2.12.28 fallback: if any price is 0 (Jupiter API down, stale, etc.),
        # use market_latest.json which bot updates every cycle.
        if sol_px == 0 or jup_px == 0 or eth_px == 0:
            try:
                _ml = load_json(DATA / "market_latest.json") or {}
                _tokens = _ml.get("tokens", {}) if isinstance(_ml, dict) else {}
                if sol_px == 0 and "SOL" in _tokens:
                    sol_px = float(_tokens["SOL"].get("price", 0) or 0)
                if jup_px == 0 and "JUP" in _tokens:
                    jup_px = float(_tokens["JUP"].get("price", 0) or 0)
                if eth_px == 0 and "ETH" in _tokens:
                    eth_px = float(_tokens["ETH"].get("price", 0) or 0)
            except Exception:
                pass

        prices = {"SOL": sol_px, "JUP": jup_px, "ETH": eth_px}

        wallet_total = (
            balances["USDC"]
            + balances["SOL"] * sol_px
            + balances["JUP"] * jup_px
            + balances["ETH"] * eth_px
        )
        result = {
            "wallet_total": round(wallet_total, 4),
            "balances": {k: round(v, 6) for k, v in balances.items()},
            "prices": {k: round(v, 6) for k, v in prices.items()},
            "ts": now,
        }
        _WALLET_EQUITY_CACHE["ts"] = now
        _WALLET_EQUITY_CACHE["data"] = result
        return result
    except Exception as _e:
        # Silent fail — fallback to bot equity
        return None


@app.route('/api/wallet-equity')
def api_wallet_equity():
    """Expose wallet_total equity (on-chain USDC + SOL + tokens × prices)."""
    data = _fetch_wallet_equity()
    if data is None:
        return jsonify({"error": "wallet equity unavailable", "wallet_total": None}), 200
    return jsonify(data)


@app.route('/api/stats')
def api_stats():
    """
    Compute stats from POST-RESET trades only.
    Capital = free_cash + invested_margin + unrealized_pnl (true equity).
    Pre-reset corrupted data (959 EMERGENCY_CLOSEs etc.) is excluded.
    """
    from datetime import datetime as dt

    risk = load_json(DATA / "risk_report.json")
    port = load_portfolio()
    comp = load_json(DATA / "compound_state.json")

    # --- Determine reset baseline (v2.12.29: reset_history.json[-1]) ---
    reset_dt, reset_capital, _reset_date_str = _load_reset_baseline()
    has_reset = reset_dt is not None
    if not has_reset:
        reset_capital = 500.0
    # Read initial_capital directly from portfolio (set correctly by reset endpoint)
    initial_capital = safe_float(port.get("initial_capital", 500.0))

    # --- Load trades, filter to post-reset if applicable ---
    trades_raw = load_trade_history()
    closed_all = [t for t in trades_raw if t.get("status") in ("closed", None, "")]

    if has_reset and reset_dt:
        closed = []
        for t in closed_all:
            close_time = t.get("close_time", "")
            if not close_time:
                continue
            try:
                ct = safe_parse_dt(close_time)
                if ct is not None and reset_dt is not None and ct >= reset_dt:
                    closed.append(t)
            except (ValueError, TypeError):
                continue
    else:
        closed = closed_all

    # Open positions: from portfolio.json (more reliable for current state)
    port_positions = port.get("positions", [])
    open_pos = [t for t in port_positions if t.get("status") == "open"]
    if not open_pos:
        open_pos = [t for t in trades_raw if t.get("status") == "open"]
    total_exp = sum(safe_float(t.get("size_usd", 0)) for t in open_pos)
    market_live = load_json(DATA / "market_latest.json") or {}
    market_tokens = market_live.get("tokens", {}) if isinstance(market_live, dict) else {}

    # --- Compute real metrics from closed trades ---
    win_trades  = [t for t in closed if safe_float(t.get("pnl_usd", 0)) > 0]
    loss_trades = [t for t in closed if safe_float(t.get("pnl_usd", 0)) < 0]
    flat_trades = [t for t in closed if safe_float(t.get("pnl_usd", 0)) == 0]
    wins   = len(win_trades)
    losses = len(loss_trades)
    flats  = len(flat_trades)
    total  = len(closed)

    # Win rate including flat trades — industry standard
    win_rate    = (wins / total * 100) if total > 0 else 0.0
    # Win rate excluding flat — shows signal quality without noise
    active_total = wins + losses
    win_rate_excl_flat = (wins / active_total * 100) if active_total > 0 else 0.0

    gross_win   = sum(safe_float(t.get("pnl_usd", 0)) for t in win_trades)
    gross_loss  = abs(sum(safe_float(t.get("pnl_usd", 0)) for t in loss_trades))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0.0
    avg_win_usd   = (gross_win  / wins)   if wins   > 0 else 0.0
    avg_loss_usd  = (gross_loss / losses) if losses > 0 else 0.0
    total_pnl     = gross_win - gross_loss

    # Accounting discrepancy: EMERGENCY_CLOSEs with pnl=0 (post-reset only)
    emergency_zero = [t for t in flat_trades if "EMERGENCY" in str(t.get("close_reason", ""))]
    emergency_zero_count = len(emergency_zero)

    # Streaks + best/worst
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    best = worst = 0.0
    closed_sorted = sorted(closed, key=lambda t: (t.get("open_time") or ""))
    for t in closed_sorted:
        p = safe_float(t.get("pnl_usd", 0))
        best = max(best, p)
        worst = min(worst, p)
        if p > 0:
            cur_win += 1; cur_loss = 0
        elif p < 0:
            cur_loss += 1; cur_win = 0
        else:
            cur_win = cur_loss = 0
        max_win_streak  = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Capital: paper_capital from master_state IS the equity
    # The executor DOES deduct margin_usd from capital when opening positions.
    # So capital_usd = free capital only. True equity = capital + locked margin + unrealized PnL.
    capital_usd = safe_float(port.get("capital_usd", initial_capital))
    unrealized = 0.0
    for p in open_pos:
        sym = str(p.get("symbol", "")).upper()
        mkt = market_tokens.get(sym, {}) if isinstance(market_tokens, dict) else {}
        current_price = safe_float(mkt.get("price", p.get("current_price", 0)))
        unrealized += estimate_open_position_pnl(p, current_price).get("pnl_usd", 0.0)
    margin_locked = sum(safe_float(p.get("margin_usd", 0)) for p in open_pos)
    invested = sum(safe_float(p.get("margin_usd", p.get("size_usd", 0))) for p in open_pos)
    # True equity includes free capital + margin locked in positions + unrealized PnL
    bot_equity = capital_usd + margin_locked + unrealized

    # v2.12.28: prefer wallet-total equity (on-chain) when available.
    # Includes SOL fuel + dust residues that bot equity ignores.
    # Falls back to bot_equity if RPC fails (paper bot, network issue).
    _wallet_data = _fetch_wallet_equity()
    if _wallet_data and _wallet_data.get("wallet_total") is not None:
        equity = _wallet_data["wallet_total"]
    else:
        equity = bot_equity

    return_pct = ((equity - initial_capital) / initial_capital * 100) \
                 if initial_capital > 0 else 0.0

    # Drawdown: compute from post-reset closed trade equity curve
    cap  = initial_capital
    peak = initial_capital
    max_dd = 0.0
    for t in closed_sorted:
        cap += safe_float(t.get("pnl_usd", 0))
        if cap > peak:
            peak = cap
        dd = ((peak - cap) / peak * 100) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Sharpe: daily returns approximation
    pnls = [safe_float(t.get("pnl_usd", 0)) for t in closed_sorted if safe_float(t.get("pnl_usd", 0)) != 0]
    if len(pnls) >= 2:
        import statistics
        mean_r = statistics.mean(pnls)
        std_r  = statistics.stdev(pnls)
        sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Accounting discrepancy: use equity (not free capital) vs recorded PnL
    # Suppress warning if gap ≈ initial_capital - equity_with_no_trades (manual reset scenario)
    real_capital_change = equity - initial_capital
    # Subtract unrealized PnL — it is already accounted for by open positions
    accounting_gap = real_capital_change - total_pnl - unrealized
    # If no closed trades and gap ≈ initial_capital, it's a manual reset — suppress
    if abs(total_pnl) < 0.01 and abs(accounting_gap - initial_capital) < 1.0:
        accounting_gap = 0
        real_capital_change = 0

    return jsonify({
        "capital_usd":        round(equity, 2),
        "initial_capital":    round(initial_capital, 2),
        "total_pnl":          round(total_pnl + unrealized, 2),
        "realized_pnl":       round(total_pnl, 2),
        "unrealized_pnl":     round(unrealized, 2),
        "return_pct":         round(return_pct, 2),
        "total_trades":       total,
        "wins":               wins,
        "losses":             losses,
        "flat_trades":        flats,
        "emergency_zero_pnl": emergency_zero_count,
        "win_rate":           round(win_rate, 2),
        "win_rate_excl_flat": round(win_rate_excl_flat, 2),
        "avg_win_usd":        round(avg_win_usd, 2),
        "avg_loss_usd":       round(avg_loss_usd, 2),
        "profit_factor":      round(profit_factor, 3),
        "sharpe_ratio":       round(safe_float(sharpe), 3),
        "drawdown_pct":       round(max_dd, 2),
        "open_positions":     len(open_pos),
        "total_exposure_usd": round(total_exp, 2),
        "best_trade":         round(best, 4),
        "worst_trade":        round(worst, 4),
        "max_win_streak":     max_win_streak,
        "max_loss_streak":    max_loss_streak,
        "portfolio_status":   risk.get("portfolio_status", "ACTIVE"),
        "mode":               port.get("mode", "paper"),
        "accounting_gap":     round(accounting_gap, 2),
        "real_capital_change":round(real_capital_change, 2),
    })


@app.route('/api/stats_post_reset')
def api_stats_post_reset():
    """Metrics computed only from trades AFTER the last capital reset.
    Used for visualization only — auto_learner always uses full history."""
    from datetime import datetime as dt

    # v2.12.29: single source via helper, no legacy fallback
    reset_dt, reset_capital, reset_date_str = _load_reset_baseline()
    if reset_dt is None:
        return jsonify({"has_reset": False})

    # Filter closed trades after reset
    trades_raw = load_trade_history()
    post_reset = []
    for t in trades_raw:
        if t.get("status") not in ("closed", None, ""):
            continue
        close_time = t.get("close_time", "")
        if not close_time:
            continue
        try:
            ct = safe_parse_dt(close_time)
            if ct is not None and reset_dt is not None and ct >= reset_dt:
                post_reset.append(t)
        except (ValueError, TypeError):
            continue

    post_reset.sort(key=lambda t: t.get("close_time", ""))

    total = len(post_reset)
    wins = sum(1 for t in post_reset if safe_float(t.get("pnl_usd", 0)) > 0)
    losses = sum(1 for t in post_reset if safe_float(t.get("pnl_usd", 0)) < 0)
    win_rate = (wins / total * 100) if total > 0 else 0.0
    total_pnl = sum(safe_float(t.get("pnl_usd", 0)) for t in post_reset)

    # Current capital: paper_capital from master_state IS the equity
    # The orchestrator does NOT deduct margin when opening positions
    # So capital_usd already represents total equity
    port = load_portfolio()
    current_capital = safe_float(port.get("capital_usd", reset_capital))
    open_pos = [p for p in port.get("positions", []) if p.get("status") == "open"]

    # Drawdown since reset
    cap = reset_capital
    peak = reset_capital
    max_dd = 0.0
    for t in post_reset:
        cap += safe_float(t.get("pnl_usd", 0))
        if cap > peak:
            peak = cap
        dd = ((peak - cap) / peak * 100) if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return_pct = ((current_capital - reset_capital) / reset_capital * 100) if reset_capital > 0 else 0.0

    return jsonify({
        "has_reset": True,
        "reset_date": reset_date_str,
        "reset_capital": round(reset_capital, 2),
        "post_reset_capital": round(current_capital, 2),
        "post_reset_pnl": round(total_pnl, 2),
        "post_reset_trades": total,
        "post_reset_wins": wins,
        "post_reset_losses": losses,
        "post_reset_win_rate": round(win_rate, 1),
        "post_reset_drawdown": round(max_dd, 2),
        "post_reset_return_pct": round(return_pct, 2),
    })


@app.route('/api/equity')
def api_equity():
    from datetime import datetime as dt

    port = load_portfolio()
    trades_raw = load_trade_history()
    closed_all = [t for t in trades_raw if t.get("status") in ("closed", None, "")]

    # Filter to post-reset trades if reset exists
    # v2.12.29: single source via helper, no legacy fallback
    _reset_dt_helper, _reset_cap_helper, _reset_date_helper = _load_reset_baseline()
    reset_log = {"reset_date": _reset_date_helper, "capital_after": _reset_cap_helper, "new_capital": _reset_cap_helper} if _reset_dt_helper else {}
    has_reset = bool(reset_log and reset_log.get("reset_date"))
    reset_dt = None
    if has_reset:
        try:
            reset_dt = safe_parse_dt(reset_log["reset_date"])
        except (ValueError, TypeError):
            has_reset = False

    if has_reset and reset_dt:
        closed = []
        for t in closed_all:
            close_time = t.get("close_time", "")
            if not close_time:
                continue
            try:
                ct = safe_parse_dt(close_time)
                if ct is not None and reset_dt is not None and ct >= reset_dt:
                    closed.append(t)
            except (ValueError, TypeError):
                continue
    else:
        closed = closed_all

    closed.sort(key=lambda t: t.get("close_time", ""))

    # Build equity curve from reset baseline
    # reset_history uses new_capital; legacy file uses capital_after
    _reset_cap = reset_log.get("capital_after") or reset_log.get("new_capital")
    _fallback_cap = safe_float(port.get("initial_capital", 500))
    init_cap = safe_float(_reset_cap) if (has_reset and _reset_cap) else _fallback_cap
    capital = init_cap
    equity_full = [init_cap]
    labels_full = ["Start"]
    trade_pnls = []
    winners = []
    losers  = []
    symbol_pnl: dict = {}
    wins = losses = 0

    for i, t in enumerate(closed):
        p = safe_float(t.get("pnl_usd", 0))
        capital += p
        equity_full.append(round(capital, 4))
        ts = t.get("close_time", "")
        labels_full.append(ts[:16].replace("T", " ") if ts else str(i+1))
        trade_pnls.append(p)

        sym = t.get("symbol", "UNK")
        symbol_pnl[sym] = symbol_pnl.get(sym, 0) + p

        idx = i + 1  # offset by 1 (Start point is index 0)
        if p > 0:
            wins += 1
            winners.append({"x": idx, "y": round(capital, 2), "pnl": p})
        elif p < 0:
            losses += 1
            losers.append({"x": idx, "y": round(capital, 2), "pnl": p})

    # Downsample equity curve for chart performance (max 500 points)
    MAX_CHART_PTS = 500
    n = len(equity_full)
    if n > MAX_CHART_PTS:
        step = n / MAX_CHART_PTS
        indices = [0] + [int(i * step) for i in range(1, MAX_CHART_PTS)] + [n - 1]
        indices = sorted(set(indices))
        equity = [equity_full[i] for i in indices]
        labels = [labels_full[i] for i in indices]
        # Re-map winners/losers x to downsampled index
        idx_map = {orig: new for new, orig in enumerate(indices)}
        def nearest(orig_x):
            best = min(indices, key=lambda v: abs(v - orig_x))
            return idx_map[best]
        winners = [{"x": nearest(w["x"]), "y": w["y"], "pnl": w["pnl"]} for w in winners]
        losers  = [{"x": nearest(l["x"]), "y": l["y"], "pnl": l["pnl"]} for l in losers]
    else:
        equity = equity_full
        labels = labels_full

    return jsonify({
        "equity": equity,
        "labels": labels,
        "trade_pnls": trade_pnls,
        "winners": winners,
        "losers":  losers,
        "total_wins": wins,
        "total_losses": losses,
        "symbol_pnl": {k: round(v, 4) for k, v in symbol_pnl.items()},
    })


@app.route('/api/trades')
def api_trades():
    limit = int(request.args.get("limit", 50))
    show_all = request.args.get("all", "").lower() in ("1", "true", "yes")
    trades_raw = load_trade_history()
    closed = [t for t in trades_raw if t.get("status") in ("closed", None, "")]

    # v2.12.29: filter to post-reset by default (consistency with /api/stats).
    # Use ?all=1 for full history (audit/debug).
    if not show_all:
        _reset_dt, _, _ = _load_reset_baseline()
        if _reset_dt is not None:
            _filtered = []
            for _t in closed:
                _ct = safe_parse_dt(_t.get("close_time", ""))
                if _ct is not None and _ct >= _reset_dt:
                    _filtered.append(_t)
            closed = _filtered

    # sort newest first
    closed.sort(key=lambda t: t.get("close_time", ""), reverse=True)
    result = []
    for t in closed[:limit]:
        result.append({
            "id":           t.get("id", ""),
            "symbol":       t.get("symbol", ""),
            "direction":    t.get("direction", "long"),
            "strategy":     t.get("strategy", ""),
            "entry_price":  safe_float(t.get("entry_price", 0)),
            "close_price":  safe_float(t.get("close_price", 0)),
            "pnl_usd":      round(safe_float(t.get("pnl_usd", 0)), 4),
            "pnl_pct":      round(safe_float(t.get("pnl_pct", 0)), 4),
            "open_time":    t.get("open_time", ""),
            "close_time":   t.get("close_time", ""),
            "close_reason": t.get("close_reason", ""),
            "ai_reasoning": t.get("ai_reasoning", ""),
            "margin_usd":   safe_float(t.get("margin_usd", 0)),
            "leverage":     t.get("leverage", 1),
            "confidence":   safe_float(t.get("confidence", 0)),
        })
    return jsonify({"trades": result, "total": len(closed)})


# Cache de precios para no spamear la API
_price_cache = {"prices": {}, "last_fetch": 0}
PRICE_CACHE_TTL = 2  # segundos


def _jupiter_fetch(mints: dict) -> dict:
    """Fetch live prices from Jupiter API v3 for real-time dashboard updates.
    v3 returns data directly at root level (not wrapped in 'data' key).
    """
    import requests as req
    try:
        mint_ids = ",".join(mints.values())
        r = req.get(f"https://lite-api.jup.ag/price/v3?ids={mint_ids}", timeout=5)
        if r.status_code == 200:
            raw = r.json()
            prices = {}
            for symbol, mint in mints.items():
                price_data = raw.get(mint, {})
                if price_data:
                    prices[symbol] = float(price_data.get("usdPrice", 0))
            return prices
    except Exception:
        pass
    return {}


def get_live_prices(symbols: list) -> dict:
    """Get live prices with 2-second cache."""
    import time as _time
    now = _time.time()
    
    if now - _price_cache["last_fetch"] < PRICE_CACHE_TTL and _price_cache["prices"]:
        return _price_cache["prices"]
    
    # Cargar mints de market_data
    market = load_json(DATA / "market_latest.json")
    tokens = market.get("tokens", {})
    mints = {s: tokens[s]["mint"] for s in symbols if s in tokens and "mint" in tokens[s]}
    
    if mints:
        live = _jupiter_fetch(mints)
        if live:
            _price_cache["prices"] = live
            _price_cache["last_fetch"] = now
            return live
    
    return {}





@app.route('/api/close-all-target', methods=['POST'])
def api_close_all_target():
    """Close all positions automatically when PnL target reached. No confirmation."""
    try:
        data = request.get_json(force=True) or {}
        target = data.get('target', 0)
        import sys
        sys.path.insert(0, str(DATA.parent))
        import executor as ex
        port = load_portfolio()
        market = load_json(DATA / "market_latest.json") or {}
        history = load_trade_history()
        symbols = [p["symbol"] for p in port.get("positions", []) if p.get("status") == "open"]
        if symbols:
            reason = f"PNL_TARGET_${target:.2f}" if target else "PNL_TARGET"
            closed = ex.close_positions_emergency(port, symbols, market, history, reason=reason)
            ex.save_portfolio(port)
            ex.save_history(history)
            _sync_wild_mode_after_close([p["symbol"] for p in closed])
            return jsonify({"ok": True, "closed": len(closed), "target": target})
        return jsonify({"ok": True, "closed": 0})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/close-all', methods=['POST'])
def api_close_all():
    """Close all open positions via emergency close."""
    try:
        import sys
        sys.path.insert(0, str(DATA.parent))
        import executor as ex
        port = load_portfolio()
        market = load_json(DATA / "market_latest.json") or {}
        history = load_trade_history()
        symbols = [p["symbol"] for p in port.get("positions", []) if p.get("status") == "open"]
        if symbols:
            closed = ex.close_positions_emergency(port, symbols, market, history, reason="MANUAL_CLOSE_ALL")
            ex.save_portfolio(port)
            ex.save_history(history)
            _sync_wild_mode_after_close([p["symbol"] for p in closed])
            return jsonify({"ok": True, "closed": len(closed)})
        return jsonify({"ok": True, "closed": 0})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/close-position', methods=['POST'])
def api_close_position():
    """Close a specific position by symbol."""
    try:
        data = request.get_json(force=True)
        symbol = data.get("symbol", "")
        if not symbol:
            return jsonify({"error": "symbol required"}), 400
        import sys
        sys.path.insert(0, str(DATA.parent))
        import executor as ex
        port = load_portfolio()
        market = load_json(DATA / "market_latest.json") or {}
        history = load_trade_history()
        closed = ex.close_positions_emergency(port, [symbol], market, history, reason="MANUAL_CLOSE")
        ex.save_portfolio(port)
        _sync_wild_mode_after_close([symbol])
        ex.save_history(history)
        return jsonify({"ok": True, "closed": len(closed)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _sync_wild_mode_after_close(closed_symbols: list):
    """After manual close, sync wild_mode chains to remove orphaned position_ids.
       If all chains gone, deactivate wild mode and record outcome."""
    try:
        import sys
        sys.path.insert(0, str(DATA.parent))
        import martingale_engine as me
        state = me.load_state()
        if not state.get('active'):
            return
        chains = state.setdefault('martingale_chains', {})
        for sym in closed_symbols:
            chains.pop(sym, None)
        if not chains:
            from datetime import datetime, timezone
            state['active'] = False
            state['ended_at'] = datetime.now(timezone.utc).isoformat()
            state['end_reason'] = 'manual_close_all_positions'
            try:
                _port_now = load_portfolio()
                _hist_now = load_trade_history()
                me.record_session_outcome(state, _port_now, _hist_now, 'manual_off')
            except Exception:
                pass
        me.save_state(state)
    except Exception as _e:
        print(f"wild_mode sync error (non-fatal): {_e}")


@app.route('/api/wild-mode/activate', methods=['POST'])
def api_wild_mode_activate():
    try:
        data = request.get_json(force=True)
        target = float(data.get('target', 0))
        if target < 0:
            return jsonify({"ok": False, "error": "target must be >= 0"}), 400
        port = load_portfolio()
        cash = float(port.get('capital_usd', 0))
        margins = sum(float(p.get('margin_usd', 0)) for p in port.get('positions', []) if p.get('status') == 'open')
        unreal = sum(float(p.get('pnl_usd', 0)) for p in port.get('positions', []) if p.get('status') == 'open')
        equity = cash + margins + unreal
        if equity <= 0:
            return jsonify({"ok": False, "error": "no equity"}), 400
        import sys, time
        from datetime import datetime, timezone
        sys.path.insert(0, str(DATA.parent))
        import martingale_engine as me
        # Read F&G from market
        try:
            mkt = load_json(DATA / "market_latest.json") or {}
            fg_raw = mkt.get("fear_greed", {})
            fg_val = fg_raw.get("value", 50) if isinstance(fg_raw, dict) else int(fg_raw or 50)
        except Exception:
            fg_val = 50
        state = {
            "active": True,
            "target_usd": target,
            "session_id": f"wild_{int(time.time())}",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "starting_equity": round(equity, 4),
            "starting_position_count": len([p for p in port.get('positions', []) if p.get('status') == 'open']),
            "starting_fg": fg_val,
            "martingale_chains": {},
            "decisions_log": []
        }
        # Bootstrap chains: each open position becomes a level-0 chain
        for p in port.get('positions', []):
            if p.get('status') != 'open':
                continue
            sym = p['symbol']
            margin = float(p.get('margin_usd', 0))
            state['martingale_chains'][sym] = {
                "base_position_id": p['id'],
                "base_margin": margin,
                "base_direction": p['direction'],
                "levels": [{
                    "level": 0,
                    "position_id": p['id'],
                    "size_multiplier": 1.0,
                    "margin": margin,
                    "direction": p['direction'],
                    "opened_at": p.get('open_time')
                }],
                "total_margin": margin,
                "max_total_allowed": margin * 4.0
            }
        me.save_state(state)
        # Notify Paperclip — non-blocking
        try:
            import sys as _sys2
            _sys2.path.insert(0, str(DATA.parent))
            import paperclip_client as _pcc
            _pcc.on_wild_mode_session_start(
                session_id=state["session_id"],
                equity=round(equity, 2),
                chains=state["martingale_chains"],
            )
        except Exception:
            pass
        return jsonify({"ok": True, "state": state})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/wild-mode/deactivate', methods=['POST'])
def api_wild_mode_deactivate():
    try:
        import sys
        from datetime import datetime, timezone
        sys.path.insert(0, str(DATA.parent))
        import martingale_engine as me
        state = me.load_state()
        state['active'] = False
        state['deactivated_at'] = datetime.now(timezone.utc).isoformat()
        me.save_state(state)
        # Record outcome
        try:
            port = load_portfolio()
            history = load_trade_history()
            me.record_session_outcome(state, port, history, 'manual_off')
        except Exception:
            pass
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/wild-mode/state', methods=['GET'])
def api_wild_mode_state():
    try:
        import sys
        sys.path.insert(0, str(DATA.parent))
        import martingale_engine as me
        return jsonify(me.load_state())
    except Exception as e:
        return jsonify({"active": False, "error": str(e)})


@app.route('/api/positions')
def api_positions():
    port = load_portfolio()
    port_positions = port.get("positions", [])
    # Check portfolio positions first for "open" status
    open_pos = [t for t in port_positions if t.get("status") == "open"]
    # Fallback: trade_history
    if not open_pos:
        trades_raw = load_trade_history()
        open_pos = [t for t in trades_raw if t.get("status") == "open"]

    # Load wild mode chains to attach hedge level data per position
    wm_chains = {}
    pos_to_chain = {}  # position_id -> {chain_sym, level_num}
    try:
        import json as _json2
        wm_path = DATA / "wild_mode_state.json"
        if wm_path.exists():
            wm_data = _json2.loads(wm_path.read_text())
            if wm_data.get("active"):
                wm_chains = wm_data.get("martingale_chains", {})
                for _sym, _chain in wm_chains.items():
                    for _lv in _chain.get("levels", []):
                        pos_to_chain[_lv["position_id"]] = {
                            "chain_sym": _sym,
                            "level_num": _lv["level"],
                        }
    except Exception:
        pass

    # Build position_id -> raw position map for hedge level PnL lookup
    pos_by_id = {p.get("position_id", p.get("id", p.get("symbol", ""))): p for p in open_pos}

    # Fetch live prices for all open position symbols
    symbols = [t.get("symbol", "") for t in open_pos]
    live_prices = get_live_prices(symbols) if symbols else {}

    # Collect position_ids of hedge levels (level > 0) — skip as top-level rows
    # NOTE: do NOT filter by symbol — same symbol can be both base and hedge (eg. ETH short + ETH hedge long)
    hedge_pos_ids = {pid for pid, info in pos_to_chain.items() if info["level_num"] > 0}

    result = []

    for t in open_pos:
        pos_id = t.get("position_id", t.get("id", t.get("symbol", "")))
        # Skip if this specific position_id is a hedge level
        if pos_id in hedge_pos_ids:
            continue

        ot = t.get("open_time", "")
        symbol = t.get("symbol", "")
        direction = t.get("direction", "long")
        entry_price = safe_float(t.get("entry_price", 0))

        # Use live price if available, fallback to stored price
        current_price = live_prices.get(symbol, safe_float(t.get("current_price", 0)))
        est = estimate_open_position_pnl(t, current_price)
        margin = safe_float(t.get("margin_usd", 0))
        size_usd = est.get("notional_value", safe_float(t.get("notional_value", 0)) or safe_float(t.get("size_usd", 0)))
        pnl_pct = est.get("pnl_pct", safe_float(t.get("pnl_pct", 0)))
        pnl_usd = est.get("pnl_usd", safe_float(t.get("pnl_usd", 0)))

        # Compute time open
        time_open_str = "—"
        if ot:
            try:
                opened = datetime.fromisoformat(ot.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                diff = now - opened
                h = int(diff.total_seconds() // 3600)
                m = int((diff.total_seconds() % 3600) // 60)
                time_open_str = f"{h}h {m}m"
            except Exception:
                pass

        # Build chain sub-level data if this position is in a wild mode chain
        # Fallback to symbol lookup when position_id is None/missing in portfolio
        chain_info = pos_to_chain.get(pos_id, {})
        if not chain_info and symbol in wm_chains:
            chain_info = {"chain_sym": symbol, "level_num": 0}
        chain_sym = chain_info.get("chain_sym", symbol)
        chain_data = wm_chains.get(chain_sym, {})
        chain_levels_out = []
        for _lv in chain_data.get("levels", []):
            if _lv["level"] == 0:
                continue  # skip base level (shown as main row)
            lv_pos_id = _lv["position_id"]
            lv_pos = pos_by_id.get(lv_pos_id, {})
            lv_sym = lv_pos.get("symbol", chain_sym)
            lv_price = live_prices.get(lv_sym, safe_float(lv_pos.get("current_price", 0)))
            lv_est = estimate_open_position_pnl(lv_pos, lv_price) if lv_pos else {}
            chain_levels_out.append({
                "level":           _lv["level"],
                "direction":       _lv.get("direction", ""),
                "margin":          _lv.get("margin", 0),
                "size_multiplier": _lv.get("size_multiplier", 1),
                "entry_price":     safe_float(lv_pos.get("entry_price", 0)),
                "current_price":   round(lv_price, 8),
                "pnl_usd":         round(lv_est.get("pnl_usd", 0), 4),
                "pnl_pct":         round(lv_est.get("pnl_pct", 0), 4),
                "opened_at":       _lv.get("opened_at", ""),
            })

        result.append({
            "symbol":        symbol,
            "direction":     direction,
            "entry_price":   entry_price,
            "current_price": round(current_price, 8),
            "pnl_usd":       round(pnl_usd, 4),
            "pnl_pct":       round(pnl_pct, 4),
            "margin_usd":    margin,
            "size_usd":      round(size_usd, 2),
            "sl_price":      safe_float(t.get("sl_price", 0)),
            "tp_price":      safe_float(t.get("tp_price", 0)),
            "strategy":      t.get("strategy", ""),
            "time_open":     time_open_str,
            "in_chain":      bool(chain_info),
            "chain_levels":  chain_levels_out,
        })

    return jsonify({"positions": result})


@app.route('/api/ai-thinking')
def api_ai_thinking():
    """Aggregates live AI reasoning for the 'What is the AI thinking?' panel."""
    import json as _json

    port = load_portfolio()
    market = load_json(DATA / "market_latest.json") or {}

    # Equity / drawdown
    positions_raw = [p for p in port.get('positions', []) if p.get('status') == 'open']
    cash = safe_float(port.get('capital_usd', 0))
    margin_total = sum(safe_float(p.get('margin_usd', 0)) for p in positions_raw)
    unreal = sum(safe_float(p.get('unrealized_pnl_usd', 0)) for p in positions_raw)
    equity = cash + margin_total + unreal
    initial = safe_float(port.get('initial_capital', equity or 1))
    drawdown_pct = max(0, (initial - equity) / initial * 100) if initial > 0 else 0

    # Fear & Greed
    fg_raw = market.get('fear_greed', {})
    fg = fg_raw if isinstance(fg_raw, dict) else {'value': int(fg_raw or 50), 'label': ''}

    # Position decisions
    pd_data = {}
    try:
        pd_path = DATA / 'position_decisions.json'
        if pd_path.exists():
            pd_raw = _json.loads(pd_path.read_text())
            pd_data = {d['symbol']: d for d in pd_raw.get('decisions', [])}
    except Exception:
        pass

    # Build per-position data
    live_prices = get_live_prices([p.get('symbol', '') for p in positions_raw])
    pos_out = []
    for p in positions_raw:
        sym = p.get('symbol', '')
        dec = pd_data.get(sym, {})
        cur_price = live_prices.get(sym, safe_float(p.get('current_price', 0)))
        est = estimate_open_position_pnl(p, cur_price)

        # Strip Codex header from llm_reasoning if present
        llm_r = dec.get('llm_reasoning', '') or ''
        if '\nassistant\n' in llm_r:
            llm_r = llm_r.split('\nassistant\n')[-1].strip()
        elif 'workdir:' in llm_r:
            llm_r = ''

        pos_out.append({
            'symbol':       sym,
            'direction':    p.get('direction', 'long'),
            'pnl_usd':      round(est.get('pnl_usd', 0), 4),
            'pnl_pct':      round(est.get('pnl_pct', 0), 4),
            'action':       dec.get('action', ''),
            'llm_action':   dec.get('llm_action', ''),
            'quant_action': dec.get('quant_action', ''),
            'alignment':    dec.get('alignment', ''),
            'confidence':   safe_float(dec.get('confidence', 0)),
            'quant_reasons':dec.get('quant_reasons', []),
            'llm_reasoning':llm_r,
            'llm_reasoning_lang': dec.get('llm_reasoning_lang', 'es'),
            'llm_source':   dec.get('llm_source', ''),
            'dist_sl_pct':  safe_float(dec.get('dist_sl_pct', 0)),
            'dist_tp_pct':  safe_float(dec.get('dist_tp_pct', 0)),
            'rr_remaining': safe_float(dec.get('rr_remaining', 0)),
            'hours_open':   safe_float(dec.get('hours_open', 0)),
            # v2.12.15: surface LLM eval freshness — explica por qué pnl mostrado puede diferir del header
            'llm_evaluated_at':    dec.get('evaluated_at') or dec.get('ts') or '',
            'llm_pnl_pct_at_eval': safe_float(dec.get('pnl_pct', 0)),
            'llm_pnl_usd_at_eval': safe_float(dec.get('pnl_usd', 0)),
        })

    # Wild mode chains with last decision
    wild_chains_out = {}
    try:
        wm_path = DATA / 'wild_mode_state.json'
        if wm_path.exists():
            wm = _json.loads(wm_path.read_text())
            if wm.get('active'):
                chains = wm.get('martingale_chains', {})
                dlog = wm.get('decisions_log', [])
                # Last decision per symbol
                last_dec_by_sym = {}
                for entry in dlog:
                    s = entry.get('raw', {}).get('symbol') or entry.get('validated', {}).get('symbol')
                    if s:
                        last_dec_by_sym[s] = entry
                # Chain PnL from portfolio
                chain_pnl_by_sym = {}
                for p in positions_raw:
                    s = p.get('symbol', '')
                    chain_pnl_by_sym[s] = chain_pnl_by_sym.get(s, 0) + safe_float(p.get('unrealized_pnl_usd', 0))
                for sym, chain in chains.items():
                    wild_chains_out[sym] = {
                        'n_levels':     len(chain.get('levels', [])),
                        'total_margin': safe_float(chain.get('total_margin', 0)),
                        'chain_pnl':    round(chain_pnl_by_sym.get(sym, 0), 4),
                        'last_decision':last_dec_by_sym.get(sym, {}),
                    }
    except Exception:
        pass

    return jsonify({
        'timestamp':     datetime.now(timezone.utc).isoformat(),
        'cycle':         port.get('cycle_counter', '—'),
        'equity':        round(equity, 2),
        'drawdown_pct':  round(drawdown_pct, 2),
        'open_positions':len(positions_raw),
        'fear_greed':    fg,
        'positions':     pos_out,
        'wild_chains':   wild_chains_out,
    })


@app.route('/api/reset-history')
def api_reset_history():
    """Get reset history for tracking progress."""
    reset_history_file = DATA / "reset_history.json"
    if reset_history_file.exists():
        try:
            with open(reset_history_file) as f:
                history = json.load(f)
            return jsonify({"history": history})
        except:
            pass
    return jsonify({"history": []})


@app.route('/api/log')
def api_log():
    lines = []
    try:
        if WATCHDOG_LOG.exists():
            with open(WATCHDOG_LOG) as f:
                all_lines = f.readlines()
            lines = [l.rstrip() for l in all_lines[-20:] if l.strip()]
    except Exception as e:
        lines = [f"Error leyendo log: {e}"]
    return jsonify({"lines": lines})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset all bot state to initial values."""
    try:
        data = request.get_json() or {}
        capital = float(data.get("capital", 500.0))
        
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # === GUARDAR SNAPSHOT ANTES DEL RESET ===
        old_portfolio = load_portfolio()
        old_history = load_trade_history()
        
        # Calcular estadísticas del período que termina
        # Fix: trades en history no tienen campo status — identificar por close_time/close_reason
        _closed = [t for t in old_history if t.get("close_time") or t.get("close_reason")]
        total_trades = len(_closed)
        wins   = len([t for t in _closed if safe_float(t.get("pnl_usd", 0)) > 0])
        losses = len([t for t in _closed if safe_float(t.get("pnl_usd", 0)) < 0])
        flat   = len([t for t in _closed if safe_float(t.get("pnl_usd", 0)) == 0])

        total_pnl = sum(safe_float(t.get("pnl_usd", 0)) for t in _closed)

        # Fix: equity real = cash libre + márgenes de posiciones abiertas + PnL no realizado
        old_capital = safe_float(old_portfolio.get("capital_usd", 500))
        _open_pos_old = [p for p in old_portfolio.get("positions", []) if p.get("status") == "open"]
        _margins_old = sum(safe_float(p.get("margin_usd", 0)) for p in _open_pos_old)
        _unreal_old  = sum(safe_float(p.get("pnl_usd", 0))    for p in _open_pos_old)
        old_equity = round(old_capital + _margins_old + _unreal_old, 2)

        old_initial = safe_float(old_portfolio.get("initial_capital", 1000))
        return_pct = ((old_equity - old_initial) / old_initial * 100) if old_initial > 0 else 0

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # Mejor y peor trade
        pnls = [safe_float(t.get("pnl_usd", 0)) for t in _closed]
        best_trade  = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        # Crear entrada de historial
        reset_entry = {
            "reset_date": now,
            "period_start": old_portfolio.get("created_at", "unknown"),
            "period_end": now,
            "initial_capital": old_initial,
            "final_capital": old_equity,
            "return_pct": round(return_pct, 2),
            "total_pnl_usd": round(total_pnl, 2),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "flat": flat,
            "win_rate": round(win_rate, 2),
            "best_trade_usd": round(best_trade, 2),
            "worst_trade_usd": round(worst_trade, 2),
            "new_capital": capital,
            "reason": data.get("reason", "Manual reset"),
            "notes": data.get("notes", "").strip(),
            "attachments": data.get("attachments", [])
        }
        
        # Cargar historial existente y agregar
        reset_history_file = DATA / "reset_history.json"
        reset_history = []
        if reset_history_file.exists():
            try:
                with open(reset_history_file) as f:
                    reset_history = json.load(f)
            except:
                reset_history = []
        
        reset_history.append(reset_entry)
        
        with open(reset_history_file, "w") as f:
            json.dump(reset_history, f, indent=2)
        
        # === CONTINUAR CON EL RESET NORMAL ===
        
        # 1. Portfolio
        portfolio = {
            "capital_usd": capital,
            "initial_capital": capital,
            "positions": [],
            "status": "ACTIVE",
            "mode": "paper_drift",
            "created_at": now,
            "last_updated": now,
            "total_trades": 0,
            "wins": 0,
            "losses": 0
        }
        with open(DATA / "portfolio.json", "w") as f:
            json.dump(portfolio, f, indent=2)
        
        # 2a. Exportar trades al auto_learner.db ANTES de borrar el historial
        #     Asi el aprendizaje acumulado sobrevive a todos los resets
        if _closed:
            try:
                import sqlite3 as _sqlite3
                _db_path = DATA / "auto_learner.db"
                _al_state_path = DATA / "auto_learner_state.json"
                _al_params = {}
                if _al_state_path.exists():
                    try:
                        _al_state_data = json.load(open(_al_state_path))
                        _al_params = _al_state_data.get("params", {})
                    except Exception:
                        pass
                _create_sql = (
                    "CREATE TABLE IF NOT EXISTS trade_results ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
                    "trade_id TEXT UNIQUE, symbol TEXT, direction TEXT, "
                    "sl_pct REAL, tp_pct REAL, leverage REAL, "
                    "pnl_usd REAL, pnl_pct REAL, win INTEGER, "
                    "confidence REAL, holding_time REAL)"
                )
                _insert_sql = (
                    "INSERT OR IGNORE INTO trade_results "
                    "(trade_id, symbol, direction, sl_pct, tp_pct, leverage, "
                    "pnl_usd, pnl_pct, win, confidence, holding_time) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)"
                )
                with _sqlite3.connect(str(_db_path)) as _conn:
                    _conn.execute(_create_sql)
                    _saved = 0
                    for _t in _closed:
                        try:
                            _conn.execute(_insert_sql, (
                                _t.get("id") or _t.get("position_id") or _t.get("trade_id"),
                                _t.get("symbol"), _t.get("direction"),
                                _al_params.get("sl_pct"), _al_params.get("tp_pct"),
                                safe_float(_t.get("leverage", 5.0)),
                                safe_float(_t.get("pnl_usd", 0)),
                                safe_float(_t.get("pnl_pct", 0)),
                                1 if safe_float(_t.get("pnl_usd", 0)) > 0 else 0,
                                safe_float(_t.get("confidence", 0.5)),
                                0.0
                            ))
                            _saved += 1
                        except Exception:
                            pass
                log.info(f"reset: {_saved}/{len(_closed)} trades exportados a auto_learner.db")
            except Exception as _e:
                log.warning(f"reset: no se pudo exportar trades a auto_learner.db: {_e}")

        # 2. Trade History
        with open(DATA / "trade_history.json", "w") as f:
            json.dump([], f)

        # 2b. Wild Mode State — clear any active salvage session on reset
        try:
            import sys
            sys.path.insert(0, str(DATA.parent))
            import martingale_engine as me
            me.save_state({
                "active": False,
                "target_usd": 0.0,
                "martingale_chains": {},
                "decisions_log": [],
                "ended_at": now,
                "end_reason": "reset",
            })
        except Exception as _e:
            log.warning(f'reset: could not clear wild mode state: {_e}')
        
        # 3. Daily Target State
        daily_state = {
            "date": today,
            "starting_capital": capital,
            "target_reached": False,
            "closed_at": None,
            "enabled": True,
            "current_pnl_pct": 0.0,
            "target_pct": 0.05
        }
        with open(DATA / "daily_target_state.json", "w") as f:
            json.dump(daily_state, f, indent=2)
        
        # 4. Daily Report
        daily_report = {
            "timestamp": now,
            "capital_usd": capital,
            "equity": capital,
            "pnl_total_usd": 0.0,
            "pnl_total_pct": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "drawdown_pct": 0.0,
            "positions_open": 0
        }
        with open(DATA / "daily_report.json", "w") as f:
            json.dump(daily_report, f, indent=2)
        
        # 5. Compound State
        compound = {
            "enabled": True,
            "last_compound": None,
            "compound_count": 0,
            "total_compounded": 0.0
        }
        with open(DATA / "compound_state.json", "w") as f:
            json.dump(compound, f, indent=2)
        
        # 6. Auto Learner State — PRESERVAR conocimiento pero RECONCILIAR indices
        learner_file = DATA / "auto_learner_state.json"
        if learner_file.exists():
            with open(learner_file) as f:
                existing_learner = json.load(f)
            existing_learner["last_updated"] = now
            existing_learner["notes"] = f"Reset capital ${capital} - conocimiento preservado"
            # FIX: Reconciliar indices de trades para evitar new_trades negativo
            existing_learner["last_trade_count"] = 0
            existing_learner["total_trades_learned"] = 0
            with open(learner_file, "w") as f:
                json.dump(existing_learner, f, indent=2)
        else:
            learner = {
                "params": {
                    "sl_pct": 0.025,
                    "tp_pct": 0.05,
                    "leverage_tier": 2,
                    "risk_per_trade": 0.02,
                    "max_positions": 5
                },
                "total_trades_learned": 0,
                "last_adaptation": None,
                "adaptation_count": 0,
                "last_updated": now,
                "notes": f"Fresh start - capital ${capital}"
            }
            with open(learner_file, "w") as f:
                json.dump(learner, f, indent=2)
        
        # 7. Alerts State
        alerts = {
            "last_daily_report": "",
            "alerted_trades": [],
            "reset_at": now
        }
        with open(DATA / "alerts_state.json", "w") as f:
            json.dump(alerts, f, indent=2)
        
        return jsonify({
            "success": True,
            "capital": capital,
            "timestamp": now,
            "message": f"Bot reseteado a ${capital}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Agent Notes API ──────────────────────────────────────────────────────────

@app.route('/api/agent-notes', methods=['GET', 'POST'])
def api_agent_notes():
    if request.method == 'POST':
        data = request.get_json(force=True)
        msg_text = data.get("message", "").strip()
        sender = data.get("sender", "user")
        if not msg_text:
            return jsonify({"error": "empty message"}), 400
        notes = load_notes()
        notes.setdefault("messages", []).append({
            "sender": sender,
            "text": msg_text,
            "ts": datetime.now().isoformat()
        })
        save_notes(notes)
        return jsonify({"ok": True})
    return jsonify(load_notes())

@app.route('/api/agent-notes/last')
def api_agent_last_note():
    notes = load_notes()
    agent_msgs = [m for m in notes.get("messages", []) if m.get("sender") == "agent"]
    if agent_msgs:
        return jsonify(agent_msgs[-1])
    return jsonify({"text": None})

@app.route('/api/agent-notes/stream')
def api_agent_notes_stream():
    def generate():
        notes = load_notes()
        yield f"event: init\ndata: {json.dumps({'type': 'init', 'notes': notes}, ensure_ascii=False)}\n\n"
        last_mtime = AGENT_NOTES_FILE.stat().st_mtime if AGENT_NOTES_FILE.exists() else 0
        last_count = len(notes.get("messages", []))
        try:
            while True:
                _time.sleep(1)
                cur_mtime = AGENT_NOTES_FILE.stat().st_mtime if AGENT_NOTES_FILE.exists() else 0
                if cur_mtime != last_mtime:
                    last_mtime = cur_mtime
                    notes = load_notes()
                    msgs = notes.get("messages", [])
                    if len(msgs) != last_count:
                        last_count = len(msgs)
                        yield f"event: update\ndata: {json.dumps({'type': 'update', 'notes': notes}, ensure_ascii=False)}\n\n"
                yield ": keepalive\n\n"
        except GeneratorExit:
            pass
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'})


CHAT_STATE_FILE = DATA / "chat_agent_state.json"

@app.route('/api/reset-upload', methods=['POST'])
def api_reset_upload():
    """Sube archivos adjuntos para un reset y los guarda en reset_attachments/."""
    import werkzeug.utils
    files = request.files.getlist('files')
    if not files:
        return jsonify({'files': []})
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    upload_dir = RESET_ATTACHMENTS_DIR / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files[:10]:
        if f.filename and '.' in f.filename:
            ext = f.filename.rsplit('.', 1)[1].lower()
            if ext in ALLOWED_ATTACH_EXT:
                safe_name = werkzeug.utils.secure_filename(f.filename)
                f.save(str(upload_dir / safe_name))
                saved.append(f'{session_id}/{safe_name}')
    return jsonify({'files': saved, 'session_id': session_id})


@app.route('/api/reset-file/<path:filepath>')
def api_reset_file(filepath):
    """Sirve un archivo adjunto de reset."""
    return send_from_directory(str(RESET_ATTACHMENTS_DIR), filepath)


@app.route('/api/set-language', methods=['POST'])
def api_set_language():
    """Guarda preferencia de idioma en user_profile para el chat agent."""
    data = request.get_json() or {}
    lang = data.get('language', 'es')
    if lang not in ('es', 'en'):
        return jsonify({'error': 'invalid language'}), 400
    profile_file = DATA / "user_profile.json"
    profile = {}
    if profile_file.exists():
        try: profile = json.loads(profile_file.read_text())
        except: pass
    profile['language'] = lang
    profile_file.write_text(json.dumps(profile, ensure_ascii=False, indent=2))
    return jsonify({'ok': True})


@app.route('/api/force-reasoning-refresh', methods=['POST'])
def api_force_reasoning_refresh():
    """v2.8.1: Invalida el cache LLM del WILD mode para que el proximo
    ciclo regenere los reasonings en el idioma actual del usuario.
    Se llama automaticamente desde toggleLang() al cambiar idioma."""
    try:
        wm_path = DATA / 'wild_mode_state.json'
        cleared = []
        if wm_path.exists():
            state = json.loads(wm_path.read_text())
            if state.get('_last_llm_ts', 0) > 0:
                state['_last_llm_ts'] = 0
                wm_path.write_text(json.dumps(state, indent=2))
                cleared.append('wild_mode._last_llm_ts')
        return jsonify({'ok': True, 'cleared': cleared,
                        'ts': datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════════
# v2.9.0-live Sprint 1 — Safety endpoints (kill switch + status)
# ════════════════════════════════════════════════════════════════════
@app.route('/api/safety/status', methods=['GET'])
def api_safety_status():
    """Devuelve el estado actual del modulo safety: kill switch,
    limites, whitelist, y si live trading esta habilitado."""
    try:
        import sys as _sys
        _agents_path = str(Path(__file__).resolve().parent.parent / 'agents')
        if _agents_path not in _sys.path:
            _sys.path.insert(0, _agents_path)
        import importlib
        import safety as _safety
        importlib.reload(_safety)  # refrescar env vars si cambiaron
        return jsonify({'ok': True, **_safety.status_snapshot()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/safety/kill-switch', methods=['POST'])
def api_safety_kill_switch():
    """Activa o desactiva el kill switch.
    Body: {"action": "activate"|"deactivate", "reason": "texto opcional"}"""
    try:
        import sys as _sys
        _agents_path = str(Path(__file__).resolve().parent.parent / 'agents')
        if _agents_path not in _sys.path:
            _sys.path.insert(0, _agents_path)
        import safety as _safety
        data = request.get_json() or {}
        action = (data.get('action') or '').lower()
        if action == 'activate':
            reason = data.get('reason', 'manual_dashboard')
            _safety.activate_kill_switch(reason)
            return jsonify({'ok': True, 'status': 'activated', 'reason': reason})
        elif action == 'deactivate':
            _safety.deactivate_kill_switch()
            return jsonify({'ok': True, 'status': 'deactivated'})
        else:
            return jsonify({'ok': False, 'error': 'action must be activate|deactivate'}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/sol/fuel', methods=['GET'])
def api_sol_fuel():
    """v2.11.0-live: SOL fuel status for auto-replenish visibility."""
    try:
        import sys as _sys
        from pathlib import Path as _P
        _agents = str(_P(__file__).parent.parent / 'agents')
        if _agents not in _sys.path:
            _sys.path.insert(0, _agents)
        import sol_topup as _st, solana_rpc as _srpc, wallet as _wm
        _rpc = _srpc.get_rpc()
        _w = _wm.load_wallet()
        return jsonify({'ok': True, 'fuel': _st.fuel_status(_rpc, _w)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/drift/status', methods=['GET'])
def api_drift_status():
    """v2.12.0-live: Drift Protocol perps status (collateral, leverage, funding, posición)."""
    try:
        import sys as _sys
        from pathlib import Path as _P
        _ag = str(_P(__file__).parent.parent / 'agents')
        if _ag not in _sys.path:
            _sys.path.insert(0, _ag)
        import drift_adapter as _da
        return jsonify({'ok': True, 'status': _da.get_account_snapshot() or {}})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def api_chat_clear():
    """Archiva mensajes actuales y limpia el chat visible."""
    now = datetime.now(timezone.utc).isoformat()
    notes = load_notes()
    msgs = notes.get("messages", [])
    if msgs:
        archive_file = DATA / "agent_notes_archive.json"
        archive = []
        if archive_file.exists():
            try:
                archive = json.loads(archive_file.read_text())
            except Exception:
                archive = []
        archive.append({"archived_at": now, "messages": msgs})
        archive_file.write_text(json.dumps(archive, ensure_ascii=False, indent=2))
    with open(AGENT_NOTES_FILE, "w") as f:
        json.dump({"messages": [], "last_updated": now}, f)
    return jsonify({"ok": True, "archived": len(msgs)})


@app.route('/api/chat-status')
def api_chat_status():
    """Retorna estado del agente de chat (pensando sí/no)."""
    try:
        if CHAT_STATE_FILE.exists():
            state = json.loads(CHAT_STATE_FILE.read_text())
            return jsonify({"thinking": state.get("thinking", False)})
    except:
        pass
    return jsonify({"thinking": False})


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"🚀 Solana Cripto Trader Dashboard")
    print(f"   URL: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=True)
