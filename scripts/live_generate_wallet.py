"""Generate a dedicated Solana keypair for Drift live trading.

Writes the keypair as a JSON array of 64 bytes (solana-keygen / solders format)
to ~/.config/solana-drift-bot/id.json with 0600 perms.

Refuses to overwrite an existing file — delete it manually if you really mean to.
"""
import json
import os
import sys
from pathlib import Path

from solders.keypair import Keypair

WALLET_DIR = Path.home() / ".config" / "solana-drift-bot"
WALLET_PATH = WALLET_DIR / "id.json"


def main() -> int:
    WALLET_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(WALLET_DIR, 0o700)

    if WALLET_PATH.exists():
        print(f"refusing to overwrite existing wallet at {WALLET_PATH}", file=sys.stderr)
        return 1

    kp = Keypair()
    WALLET_PATH.write_text(json.dumps(list(bytes(kp))))
    os.chmod(WALLET_PATH, 0o600)

    print(f"wallet   : {WALLET_PATH}")
    print(f"pubkey   : {kp.pubkey()}")
    print(f"network  : devnet (airdrop via https://faucet.solana.com)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
