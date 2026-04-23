#!/usr/bin/env python3
"""Reconcile ALL orphan positions (status=open|needs_manual_reconcile) against on-chain truth.

Handles N positions (1, 2, 3, or more) by iterating portfolio.positions and matching
each to an on-chain close tx via getSignaturesForAddress + decode.

v2.12.16 — generalized from reconcile_both_orphans.py with:
  --dry-run    : preview changes without writing
  --limit N    : tx history window (default 100, was hard-coded 30)
  auto-backup  : portfolio/history copied to .bak_<epoch> before live write
  fix counter  : total_trades incremented correctly
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BOT = Path('/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live')
DATA = BOT / 'agents/data'

# Per-symbol mints used to verify the close tx actually swapped OUT of the
# expected token (not just any tx that happened to increase USDC).
SYMBOL_MINTS = {
    "SOL":  "So11111111111111111111111111111111111111112",
    "JUP":  "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "ETH":  "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
}


def load_env():
    for line in (BOT / '.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview only — no writes, no /tmp cleanup')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of recent txs to fetch (default 100)')
    args = parser.parse_args()

    load_env()
    rpc = os.environ['SOLANA_RPC_URL']
    wallet = os.environ['HOT_WALLET_ADDRESS']
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no files will be modified, no /tmp cleanup")
        print("=" * 60)

    # 1. Current wallet state
    resp = requests.post(rpc, json={"jsonrpc": "2.0", "id": 1,
                                     "method": "getBalance",
                                     "params": [wallet]}).json()
    sol_balance = resp['result']['value'] / 1e9
    resp2 = requests.post(rpc, json={"jsonrpc": "2.0", "id": 2,
                                      "method": "getTokenAccountsByOwner",
                                      "params": [wallet,
                                                 {"mint": USDC_MINT},
                                                 {"encoding": "jsonParsed"}]}).json()
    usdc_balance = 0.0
    for acc in resp2.get('result', {}).get('value', []):
        amt = acc['account']['data']['parsed']['info']['tokenAmount']['uiAmount'] or 0
        usdc_balance += amt
    print(f"ON-CHAIN: SOL={sol_balance:.6f}  USDC=${usdc_balance:.4f}")

    # 2. Recent txs (configurable window)
    resp3 = requests.post(rpc, json={"jsonrpc": "2.0", "id": 3,
                                      "method": "getSignaturesForAddress",
                                      "params": [wallet,
                                                 {"limit": args.limit}]}).json()
    recent_sigs = resp3.get('result', [])
    if recent_sigs:
        oldest = datetime.fromtimestamp(recent_sigs[-1]['blockTime'], tz=timezone.utc)
        newest = datetime.fromtimestamp(recent_sigs[0]['blockTime'], tz=timezone.utc)
        print(f"TX window: {len(recent_sigs)} txs  {oldest.isoformat()} → {newest.isoformat()}")

    # 3. Load portfolio + history
    pf_path = DATA / 'portfolio.json'
    pf = json.loads(pf_path.read_text())
    history_path = DATA / 'trade_history.json'
    history_raw = json.loads(history_path.read_text()) if history_path.exists() else []
    # Normalize: list format preferred; if dict with 'trades', use that
    if isinstance(history_raw, dict) and 'trades' in history_raw:
        history = history_raw['trades']
        history_wrapper = history_raw
    elif isinstance(history_raw, list):
        history = history_raw
        history_wrapper = None
    else:
        history = []
        history_wrapper = None

    reconciled_ids = []
    skipped = []
    prepared_records = []
    claimed_txs = set()  # prevent multiple positions from matching the same close tx

    for pos in list(pf['positions']):
        status = pos.get('status')
        if status not in ('open', 'needs_manual_reconcile'):
            continue
        pid = pos['id']
        open_tx = pos.get('tx_signature', '')
        symbol = pos['symbol']
        token_mint = SYMBOL_MINTS.get(symbol)
        if not token_mint:
            msg = f"{pid}: unknown mint for symbol {symbol} — skip"
            print(f"  ⚠ {msg}")
            skipped.append(msg)
            continue

        # Find open_tx blockTime
        open_block_time = None
        for s in recent_sigs:
            if s['signature'] == open_tx:
                open_block_time = s.get('blockTime')
                break
        if open_block_time is None:
            msg = f"{pid}: open_tx not in last {args.limit} txs — skip"
            print(f"  ⚠ {msg}")
            skipped.append(msg)
            continue

        # Close candidates: 1-3600s after open, not err, not already claimed
        close_candidates = [s for s in recent_sigs
                            if s.get('blockTime', 0) > open_block_time
                            and s.get('blockTime', 0) - open_block_time <= 3600
                            and not s.get('err')
                            and s['signature'] not in claimed_txs]
        close_candidates.sort(key=lambda s: s.get('blockTime', 0))

        matched = None
        for cand in close_candidates:
            close_tx = cand['signature']
            tx_detail = requests.post(rpc, json={
                "jsonrpc": "2.0", "id": 9,
                "method": "getTransaction",
                "params": [close_tx,
                           {"encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0}]
            }).json()
            r = tx_detail.get("result") or {}
            meta = r.get("meta", {})
            pre_tkn = meta.get("preTokenBalances", [])
            post_tkn = meta.get("postTokenBalances", [])

            # USDC delta (wallet owner)
            usdc_pre = usdc_post = 0.0
            for tb in pre_tkn:
                if tb.get("owner") == wallet and tb.get("mint") == USDC_MINT:
                    usdc_pre = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
            for tb in post_tkn:
                if tb.get("owner") == wallet and tb.get("mint") == USDC_MINT:
                    usdc_post = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
            usdc_received = usdc_post - usdc_pre

            # Token-specific delta (must decrease — it's being sold)
            if symbol == "SOL":
                # Native SOL: use pre/post account balances; accountKeys are indexed
                account_keys = r.get("transaction", {}).get("message", {}).get("accountKeys", [])
                pre_sol_lamports = meta.get("preBalances", [])
                post_sol_lamports = meta.get("postBalances", [])
                wallet_idx = None
                for i, ak in enumerate(account_keys):
                    pk = ak.get("pubkey") if isinstance(ak, dict) else ak
                    if pk == wallet:
                        wallet_idx = i
                        break
                if wallet_idx is None or wallet_idx >= len(pre_sol_lamports):
                    token_delta = 0.0
                else:
                    token_delta = (post_sol_lamports[wallet_idx] - pre_sol_lamports[wallet_idx]) / 1e9
                # Also check wrapped SOL in token balances (some swaps use wSOL ATAs)
                wsol_pre = wsol_post = 0.0
                for tb in pre_tkn:
                    if tb.get("owner") == wallet and tb.get("mint") == token_mint:
                        wsol_pre = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
                for tb in post_tkn:
                    if tb.get("owner") == wallet and tb.get("mint") == token_mint:
                        wsol_post = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
                wsol_delta = wsol_post - wsol_pre
                # Effective SOL decrease = loss of native or wSOL (excluding fee)
                effective_decrease = -(token_delta + wsol_delta)
            else:
                tok_pre = tok_post = 0.0
                for tb in pre_tkn:
                    if tb.get("owner") == wallet and tb.get("mint") == token_mint:
                        tok_pre = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
                for tb in post_tkn:
                    if tb.get("owner") == wallet and tb.get("mint") == token_mint:
                        tok_post = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
                effective_decrease = tok_pre - tok_post

            # Require: USDC increased AND target token decreased by ~position.tokens
            position_tokens = float(pos.get('tokens', 0))
            # Tolerate 10% off from exact (dust leftover, fee accounting)
            min_decrease = position_tokens * 0.5  # at least 50% of position sold
            if usdc_received > 0 and effective_decrease >= min_decrease:
                matched = (close_tx, cand.get('blockTime'), usdc_received, effective_decrease)
                break

        if not matched:
            msg = f"{pid} [{symbol}]: no close tx matching token decrease — skip"
            print(f"  ⚠ {msg}")
            skipped.append(msg)
            continue

        close_tx, close_ts, usdc_received, tok_sold = matched
        claimed_txs.add(close_tx)
        close_iso = datetime.fromtimestamp(close_ts, tz=timezone.utc).isoformat()

        margin = float(pos.get('margin_usd', 2.0))
        tokens = float(pos.get('tokens', 0))
        pnl_usd = usdc_received - margin
        exit_price = usdc_received / tokens if tokens else 0.0

        record = {
            'id': pid,
            'symbol': pos['symbol'],
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': round(exit_price, 8),
            'current_price': round(exit_price, 8),
            'size_usd': pos.get('size_usd', 0),
            'notional_value': pos.get('notional_value', 0),
            'margin_usd': margin,
            'leverage': pos.get('leverage', 1),
            'fee_entry': pos.get('fee_entry', 0),
            'fee_exit': 0,
            'funding_accumulated': pos.get('funding_accumulated', 0),
            'pnl_usd': round(pnl_usd, 4),
            'pnl_pct': round(pnl_usd / margin * 100 if margin else 0, 4),
            'open_time': pos['open_time'],
            'close_time': close_iso,
            'close_reason': 'reconcile_orphan_v2.12.16',
            'strategy': pos.get('strategy', 'unknown'),
            'mode': 'live',
            'tx_signature': open_tx,
            'tx_signature_close': close_tx,
        }
        prepared_records.append(record)
        reconciled_ids.append(pid)
        print(f"  ✓ {pid} [{symbol}]: close {close_iso[:19]} tx={close_tx[:16]}... "
              f"sold={tok_sold:.6f} usdc=${usdc_received:.4f} pnl=${pnl_usd:+.4f}")

    print()
    print(f"Summary: {len(reconciled_ids)} reconciled, {len(skipped)} skipped")

    if args.dry_run:
        print()
        print("DRY RUN — would write:")
        print(f"  trade_history.json: +{len(reconciled_ids)} entries")
        new_capital = round(usdc_balance, 4)
        new_trades = pf.get('total_trades', 0) + len(reconciled_ids)
        remaining = [p for p in pf['positions'] if p['id'] not in reconciled_ids]
        print(f"  portfolio.json: capital_usd=${new_capital} "
              f"positions={len(remaining)} trades={new_trades}")
        print(f"  remove: /tmp/solana_live_killswitch, _heartbeat, _orchestrator.lock, _close_failures")
        print(f"  reset: wild_mode_state.json + daily_target_state.json")
        return 0

    # LIVE mode: backups first
    epoch = int(time.time())
    pf_bak = pf_path.with_suffix(f'.json.bak_{epoch}')
    hist_bak = history_path.with_suffix(f'.json.bak_{epoch}')
    shutil.copy2(pf_path, pf_bak)
    if history_path.exists():
        shutil.copy2(history_path, hist_bak)
    print(f"Backup: {pf_bak.name}, {hist_bak.name}")

    # Append history
    history.extend(prepared_records)
    if history_wrapper is not None:
        history_wrapper['trades'] = history
        history_wrapper['last_updated'] = datetime.now(timezone.utc).isoformat()
        history_path.write_text(json.dumps(history_wrapper, indent=2))
    else:
        history_path.write_text(json.dumps(history, indent=2))
    print(f"trade_history.json: +{len(reconciled_ids)} entries")

    # Remove reconciled from portfolio, sync capital
    pf['positions'] = [p for p in pf['positions'] if p['id'] not in reconciled_ids]
    pf['capital_usd'] = round(usdc_balance, 4)
    # v2.12.28: NO sobrescribir initial_capital — es baseline inmutable del primer deposit.
    # Si no existe (nuevo portfolio), establecer; si existe, preservar.
    if not pf.get('initial_capital'):
        pf['initial_capital'] = round(usdc_balance, 4)
    pf['total_trades'] = pf.get('total_trades', 0) + len(reconciled_ids)
    pf['last_updated'] = datetime.now(timezone.utc).isoformat()
    pf_path.write_text(json.dumps(pf, indent=2))
    print(f"portfolio.json: capital_usd=${pf['capital_usd']} "
          f"positions={len(pf['positions'])} trades={pf['total_trades']}")

    # Clear /tmp state
    for f in ['/tmp/solana_live_killswitch',
              '/tmp/solana_live_heartbeat',
              '/tmp/solana_live_orchestrator.lock',
              '/tmp/solana_live_close_failures']:
        p = Path(f)
        if p.exists():
            p.unlink()
            print(f"  removed {f}")

    # Reset wild_mode + daily_target
    now_iso = datetime.now(timezone.utc).isoformat()
    (DATA / 'wild_mode_state.json').write_text(json.dumps({
        "active": False, "target_usd": 0.0, "session_id": "",
        "started_at": None,
        "starting_equity": pf['capital_usd'],
        "starting_position_count": 0, "starting_fg": 0,
        "martingale_chains": {}, "decisions_log": [],
        "ended_at": now_iso, "end_reason": "v2.12.16_recovery",
    }, indent=2))
    (DATA / 'daily_target_state.json').write_text(json.dumps({
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "starting_capital": pf['capital_usd'],
        "target_reached": False, "enabled": True,
        "current_pnl_pct": 0.0, "target_pct": 0.05,
    }, indent=2))
    print("Reset: wild_mode + daily_target")
    print()
    print("✅ RECOVERY DONE")
    return 0


if __name__ == '__main__':
    sys.exit(main())
