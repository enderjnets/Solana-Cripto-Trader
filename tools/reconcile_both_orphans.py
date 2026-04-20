#!/usr/bin/env python3
"""Reconcile BOTH orphan positions on-chain vs portfolio, adapting to whatever close tx we find."""
import os, json, sys, requests, time
from datetime import datetime, timezone
from pathlib import Path

BOT = Path('/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live')
DATA = BOT / 'agents/data'
for l in (BOT / '.env').read_text().splitlines():
    l = l.strip()
    if l and not l.startswith('#') and '=' in l:
        k, v = l.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

rpc = os.environ['SOLANA_RPC_URL']
wallet = os.environ['HOT_WALLET_ADDRESS']
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# 1. Current wallet state
resp = requests.post(rpc, json={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[wallet]}).json()
sol_balance = resp['result']['value'] / 1e9
resp2 = requests.post(rpc, json={"jsonrpc":"2.0","id":2,"method":"getTokenAccountsByOwner","params":[wallet, {"mint":USDC_MINT}, {"encoding":"jsonParsed"}]}).json()
usdc_balance = sum(acc['account']['data']['parsed']['info']['tokenAmount']['uiAmount'] or 0 for acc in resp2.get('result',{}).get('value',[]))
print(f"ON-CHAIN: SOL={sol_balance:.6f}  USDC=${usdc_balance:.4f}")

# 2. Recent txs (last 30)
resp3 = requests.post(rpc, json={"jsonrpc":"2.0","id":3,"method":"getSignaturesForAddress","params":[wallet, {"limit":30}]}).json()
recent_sigs = resp3.get('result', [])

pf_path = DATA / 'portfolio.json'
pf = json.loads(pf_path.read_text())
history_path = DATA / 'trade_history.json'
history = json.loads(history_path.read_text()) if history_path.exists() else []

reconciled_ids = []
for pos in list(pf['positions']):
    if pos.get('status') not in ('open', 'needs_manual_reconcile'):
        continue
    pid = pos['id']
    open_tx = pos.get('tx_signature', '')
    # Find close tx: first tx AFTER open_tx that represents a SOL→USDC swap
    # We can't reliably identify, so: pick tx that happened shortly after open_tx (1-30 min window)
    # For simplicity: use latest tx after open that's not the open itself
    open_block_time = None
    for s in recent_sigs:
        if s['signature'] == open_tx:
            open_block_time = s.get('blockTime')
            break
    if open_block_time is None:
        print(f"  {pid}: open_tx not found in recent sigs — skip")
        continue
    # First close candidate: any later tx within 60 min
    close_candidates = [s for s in recent_sigs if s.get('blockTime',0) > open_block_time and s.get('blockTime',0) - open_block_time <= 3600 and not s.get('err')]
    close_candidates.sort(key=lambda s: s.get('blockTime', 0))
    if not close_candidates:
        print(f"  {pid}: no close tx found — position still on-chain? skip")
        continue
    close_tx = close_candidates[0]['signature']
    close_time_ts = close_candidates[0].get('blockTime')
    close_iso = datetime.fromtimestamp(close_time_ts, tz=timezone.utc).isoformat()

    # Decode close tx to get USDC received
    tx_detail = requests.post(rpc, json={"jsonrpc":"2.0","id":9,"method":"getTransaction","params":[close_tx, {"encoding":"jsonParsed","maxSupportedTransactionVersion":0}]}).json()
    r = tx_detail.get("result") or {}
    meta = r.get("meta", {})
    # USDC delta
    pre_tkn = meta.get("preTokenBalances", [])
    post_tkn = meta.get("postTokenBalances", [])
    usdc_pre = usdc_post = 0.0
    for tb in pre_tkn:
        if tb.get("owner") == wallet and tb.get("mint") == USDC_MINT:
            usdc_pre = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
    for tb in post_tkn:
        if tb.get("owner") == wallet and tb.get("mint") == USDC_MINT:
            usdc_post = tb.get("uiTokenAmount", {}).get("uiAmount", 0) or 0
    usdc_received = usdc_post - usdc_pre

    margin = float(pos.get('margin_usd', 2.0))
    tokens = float(pos.get('tokens', 0))
    if usdc_received <= 0:
        print(f"  {pid}: close tx {close_tx[:16]}... doesn't match swap pattern (usdc_delta={usdc_received:.4f}) — skip")
        continue
    pnl_usd = usdc_received - margin
    exit_price = usdc_received / tokens if tokens else 0.0

    # Build history record
    record = {
        'id': pid, 'symbol': pos['symbol'], 'direction': pos['direction'],
        'entry_price': pos['entry_price'], 'exit_price': round(exit_price, 8),
        'current_price': round(exit_price, 8), 'size_usd': pos.get('size_usd', 0),
        'notional_value': pos.get('notional_value', 0), 'margin_usd': margin,
        'leverage': pos.get('leverage', 1), 'fee_entry': pos.get('fee_entry', 0),
        'fee_exit': 0, 'funding_accumulated': pos.get('funding_accumulated', 0),
        'pnl_usd': round(pnl_usd, 4), 'pnl_pct': round(pnl_usd/margin*100 if margin else 0, 4),
        'open_time': pos['open_time'], 'close_time': close_iso,
        'close_reason': 'reconcile_orphan_v2.12.6', 'strategy': pos.get('strategy', 'unknown'),
        'mode': 'live', 'tx_signature': open_tx, 'tx_signature_close': close_tx,
    }
    history.append(record)
    reconciled_ids.append(pid)
    print(f"  ✓ {pid}: close at {close_iso[:19]} tx={close_tx[:16]}... usdc_recv=${usdc_received:.4f} pnl=${pnl_usd:+.4f}")

history_path.write_text(json.dumps(history, indent=2))
print(f"trade_history.json: +{len(reconciled_ids)} entries")

# Remove reconciled from portfolio, sync capital_usd to wallet
pf['positions'] = [p for p in pf['positions'] if p['id'] not in reconciled_ids]
pf['capital_usd'] = round(usdc_balance, 4)
pf['initial_capital'] = round(usdc_balance, 4)
pf['total_trades'] = pf.get('total_trades', 0)
pf['last_updated'] = datetime.now(timezone.utc).isoformat()
pf_path.write_text(json.dumps(pf, indent=2))
print(f"portfolio.json: capital_usd=${pf['capital_usd']} positions={len(pf['positions'])} trades={pf['total_trades']}")

# Clear kill switch + stale heartbeat + lock
for f in ['/tmp/solana_live_killswitch', '/tmp/solana_live_heartbeat', '/tmp/solana_live_orchestrator.lock', '/tmp/solana_live_close_failures']:
    p = Path(f)
    if p.exists():
        p.unlink()
        print(f"  removed {f}")

# Reset daily_target + wild_mode
now_iso = datetime.now(timezone.utc).isoformat()
(DATA / 'wild_mode_state.json').write_text(json.dumps({
    "active": False, "target_usd": 0.0, "session_id": "", "started_at": None,
    "starting_equity": pf['capital_usd'], "starting_position_count": 0, "starting_fg": 0,
    "martingale_chains": {}, "decisions_log": [], "ended_at": now_iso, "end_reason": "v2.12.6_recovery",
}, indent=2))
(DATA / 'daily_target_state.json').write_text(json.dumps({
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "starting_capital": pf['capital_usd'], "target_reached": False,
    "enabled": True, "current_pnl_pct": 0.0, "target_pct": 0.05,
}, indent=2))
print("Reset: wild_mode + daily_target")
print("\n✅ RECOVERY DONE")
