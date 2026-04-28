"""Jupiter Perpetuals smoke test — validates the CLI wrapper path.

Usage:
    python tools/jupiter_perp_smoke.py [--size SOL_SIZE]

This opens a small SOL long (~$10 collateral, 2x leverage) and immediately
closes it, verifying end-to-end connectivity with mainnet via the official
@jup-ag/cli backend.
"""
import argparse
import logging
import os
import sys
import time

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.jupiter_perp_cli_wrapper import (
    close_position,
    get_positions,
    open_position,
    ensure_configured,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jupiter_smoke")


def run_smoke(collateral: float = 10.0, leverage: float = 2.0) -> int:
    log.info("=" * 60)
    log.info("Jupiter Perpetuals Smoke Test (CLI wrapper)")
    log.info("=" * 60)

    # 1. Ensure CLI configured
    if not ensure_configured():
        log.error("Failed to configure jup CLI output to JSON")
        return 1
    log.info("CLI configured for JSON output")

    # 2. Check existing positions
    existing = get_positions()
    log.info(f"Existing positions: {len(existing)}")
    for p in existing:
        log.info(f"  {p.asset} {p.side}: ${p.size_usd:.2f} @ {p.leverage:.1f}x")

    # 3. Dry-run open
    log.info(f"\n[DRY RUN] Open SOL long ${collateral} @ {leverage}x")
    dry = open_position(
        asset="SOL", side="long",
        collateral_usd=collateral, leverage=leverage,
        dry_run=True,
    )
    if not dry.success:
        log.error(f"Dry-run failed: {dry.error}")
        return 1
    log.info(f"  Entry price: ~${dry.entry_price:,.2f}")
    log.info(f"  Notional:    ~${dry.size_usd:,.2f}")
    log.info(f"  Open fee:    ~${dry.fee_usd:.4f}")

    # 4. Live open
    log.info(f"\n[LIVE] Opening SOL long ${collateral} @ {leverage}x ...")
    live = open_position(
        asset="SOL", side="long",
        collateral_usd=collateral, leverage=leverage,
        dry_run=False,
    )
    if not live.success:
        log.error(f"Live open failed: {live.error}")
        return 1
    log.info(f"  Tx: https://solscan.io/tx/{live.tx_signature}")
    log.info(f"  Entry: ${live.entry_price:,.2f} | Notional: ${live.size_usd:,.2f}")

    # 5. Verify position exists
    time.sleep(3)
    positions = get_positions()
    log.info(f"\nOpen positions after trade: {len(positions)}")
    for p in positions:
        log.info(f"  {p.asset} {p.side}: ${p.size_usd:.2f} @ {p.leverage:.1f}x (entry ${p.entry_price:,.2f})")

    if not positions:
        log.error("No positions found after open — investigate!")
        return 1

    # 6. Close all positions
    log.info("\n[LIVE] Closing all positions...")
    for p in positions:
        close_res = close_position(position_pubkey=p.pubkey, dry_run=False)
        if close_res.success:
            log.info(f"  Closed {p.pubkey[:20]}... tx: https://solscan.io/tx/{close_res.tx_signature}")
            log.info(f"  PnL: {close_res.pnl_pct:+.2f}% | Fee: ${close_res.fee_usd:.4f}")
        else:
            log.error(f"  Failed to close {p.pubkey[:20]}: {close_res.error}")
            return 1

    # 7. Final check
    time.sleep(3)
    final = get_positions()
    log.info(f"\nFinal open positions: {len(final)}")
    if final:
        log.error("Positions still open!")
        return 1

    log.info("\n" + "=" * 60)
    log.info("SMOKE TEST PASSED")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jupiter Perps smoke test")
    parser.add_argument("--size", type=float, default=10.0, help="Collateral USD (default 10)")
    parser.add_argument("--leverage", type=float, default=2.0, help="Leverage (default 2x)")
    args = parser.parse_args()
    sys.exit(run_smoke(collateral=args.size, leverage=args.leverage))
