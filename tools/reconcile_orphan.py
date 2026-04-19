#!/usr/bin/env python3
"""v2.12.3 one-shot recovery: reconcile orphan position SOL_live_1776565594 with on-chain truth.

Usage: python3 tools/reconcile_orphan.py
"""
import os, json, sys, requests
from datetime import datetime, timezone
from pathlib import Path

BOT = Path('/home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live')
DATA = BOT / 'agents/data'

# Load env
for l in (BOT / '.env').read_text().splitlines():
    l = l.strip()
    if l and not l.startswith('#') and '=' in l:
        k, v = l.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

rpc_url = os.environ['SOLANA_RPC_URL']
wallet = os.environ['HOT_WALLET_ADDRESS']

# Step 1: Read current on-chain balances
resp = requests.post(rpc_url, json={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[wallet]}).json()
sol_lamports = resp['result']['value']
sol_balance = sol_lamports / 1e9

# USDC balance via getTokenAccountsByOwner
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
resp = requests.post(rpc_url, json={"jsonrpc":"2.0","id":2,"method":"getTokenAccountsByOwner","params":[wallet, {"mint":USDC_MINT}, {"encoding":"jsonParsed"}]}).json()
usdc_balance = 0.0
for acc in resp.get('result', {}).get('value', []):
    amt = acc['account']['data']['parsed']['info']['tokenAmount']['uiAmount'] or 0
    usdc_balance += amt

print(f"ON-CHAIN: SOL={sol_balance:.6f}  USDC=${usdc_balance:.4f}")

# Step 2: Load portfolio
pf_path = DATA / 'portfolio.json'
pf = json.loads(pf_path.read_text())

open_positions = [p for p in pf.get('positions', []) if p.get('status') == 'open']
print(f"PORTFOLIO: {len(open_positions)} open position(s), capital_usd=${pf.get('capital_usd')}")

# Step 3: For each open position, fetch its close tx if any
# Known: SOL_live_1776565594 opened 2026-04-19T02:26:34, closed on-chain at 02:40:02 via tx 3etBpV4J...
# Actual exit: $2.0016 USDC from on-chain tx decode
orphan_id = 'SOL_live_1776565594'
tx_close_full = "3etBpV4JAKzeU7UWfPM1Naq5A4KgESGCCcRT7U5GmKsimq62X1Xw8FR5rHC8WzMzoNS3K6Qqkeh5rB6SngWZJmkN"

reconciled = []
for pos in list(pf['positions']):
    if pos['id'] != orphan_id or pos.get('status') != 'open':
        continue
    margin = float(pos.get('margin_usd', 0))
    tokens = float(pos.get('tokens', 0))
    entry_price = float(pos.get('entry_price', 0))
    # From on-chain decode: received $2.0016 USDC for 0.023415 SOL sold
    usdc_received = 2.0016
    exit_price = usdc_received / tokens if tokens else 0.0
    pnl_usd = usdc_received - margin
    pnl_pct = (pnl_usd / margin * 100) if margin > 0 else 0.0

    pos['status'] = 'closed'
    pos['close_time'] = '2026-04-19T02:40:02+00:00'
    pos['close_reason'] = 'reconcile_orphan_v2.12.2_close_not_persisted'
    pos['close_price'] = round(exit_price, 8)
    pos['current_price'] = round(exit_price, 8)
    pos['pnl_usd'] = round(pnl_usd, 4)
    pos['pnl_pct'] = round(pnl_pct, 4)
    pos['fee_exit'] = 0
    pos['tx_signature_close'] = tx_close_full
    pos['exit_price'] = round(exit_price, 8)

    # Append to trade_history
    history_path = DATA / 'trade_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if not isinstance(history, list):
        history = []
    # Build record matching existing schema
    record = {
        'id': pos['id'], 'symbol': pos['symbol'], 'direction': pos['direction'],
        'entry_price': pos['entry_price'], 'exit_price': pos['close_price'],
        'current_price': pos['close_price'], 'size_usd': pos.get('size_usd', 0),
        'notional_value': pos.get('notional_value', 0), 'margin_usd': margin,
        'leverage': pos.get('leverage', 1), 'fee_entry': pos.get('fee_entry', 0),
        'fee_exit': 0, 'funding_accumulated': 0,
        'pnl_usd': pos['pnl_usd'], 'pnl_pct': pos['pnl_pct'],
        'open_time': pos['open_time'], 'close_time': pos['close_time'],
        'close_reason': pos['close_reason'], 'strategy': pos.get('strategy', 'unknown'),
        'mode': 'live', 'tx_signature': pos.get('tx_signature'),
        'tx_signature_close': tx_close_full,
    }
    history.append(record)
    history_path.write_text(json.dumps(history, indent=2))
    print(f"✓ Reconciled {pos['id']}: pnl={pnl_usd:+.4f} USD, exit={exit_price:.4f}, tx_close={tx_close_full[:20]}...")
    print(f"  appended to trade_history.json (now {len(history)} trades)")
    reconciled.append(pos)

# Step 4: Remove closed positions from portfolio, update stats
pf['positions'] = [p for p in pf['positions'] if p.get('status') == 'open']
# Sync capital_usd to reality
pf['capital_usd'] = round(usdc_balance, 4)
pf['initial_capital'] = round(usdc_balance, 4)  # rebase baseline to current
pf['total_trades'] = pf.get('total_trades', 0) + len(reconciled)
wins = sum(1 for r in reconciled if r['pnl_usd'] > 0)
pf['wins'] = pf.get('wins', 0) + wins
pf['losses'] = pf.get('losses', 0) + (len(reconciled) - wins)
pf['last_updated'] = datetime.now(timezone.utc).isoformat()

pf_path.write_text(json.dumps(pf, indent=2))
print(f"✓ portfolio.json: capital_usd=${pf['capital_usd']}, positions={len(pf['positions'])}, trades={pf['total_trades']}")

# Step 5: Clean up state files
ks = Path('/tmp/solana_live_killswitch')
if ks.exists():
    ks.unlink()
    print("✓ killed /tmp/solana_live_killswitch")

hb = Path('/tmp/solana_live_heartbeat')
if hb.exists():
    hb.unlink()
    print("✓ removed stale heartbeat")

lock = Path('/tmp/solana_live_orchestrator.lock')
if lock.exists():
    lock.unlink()
    print("✓ removed orchestrator.lock")

# Reset wild_mode
wm_path = DATA / 'wild_mode_state.json'
now_iso = datetime.now(timezone.utc).isoformat()
wm = {
    "active": False, "target_usd": 0.0, "session_id": "",
    "started_at": None, "starting_equity": pf['capital_usd'],
    "starting_position_count": 0, "starting_fg": 0,
    "martingale_chains": {}, "decisions_log": [],
    "ended_at": now_iso, "end_reason": "v2.12.3_recovery_manual_reset",
}
wm_path.write_text(json.dumps(wm, indent=2))
print(f"✓ wild_mode_state reset: active=false")

# Reset daily_target
dt_path = DATA / 'daily_target_state.json'
dt = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "starting_capital": pf['capital_usd'],
    "target_reached": False,
    "enabled": True,
    "current_pnl_pct": 0.0,
    "target_pct": 0.05,
}
dt_path.write_text(json.dumps(dt, indent=2))
print(f"✓ daily_target_state reset: starting_capital=${dt['starting_capital']}, target_reached=false")

print("\n=== RECOVERY COMPLETE ===")
print(f"Wallet on-chain:     SOL={sol_balance:.6f}  USDC=${usdc_balance:.4f}")
print(f"Portfolio synced:    capital_usd=${pf['capital_usd']}  positions={len(pf['positions'])}  trades={pf['total_trades']}")
print(f"Orphan closed:       {len(reconciled)} position(s) moved to trade_history")
