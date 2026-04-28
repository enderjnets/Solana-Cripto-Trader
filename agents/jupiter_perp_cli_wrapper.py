"""Jupiter Perpetuals — CLI wrapper (official @jup-ag/cli backend).

Replaces the manual Python implementation with the official Jupiter CLI,
which is actively maintained and handles all edge cases (oracles, PDAs,
keeper fulfillment, compute budget, etc.).

Requires: npm/npx + `@jup-ag/cli` installed globally or accessible via npx.
Config: `jup config set --output json` for machine-readable output.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger("jupiter_perp_cli")

JUP_CLI = ["npx", "@jup-ag/cli"]


@dataclass
class PerpPosition:
    pubkey: str
    asset: str
    side: str
    size_usd: float
    entry_price: float
    mark_price: float
    pnl_pct: float
    leverage: float
    liq_price: float


@dataclass
class OpenResult:
    success: bool
    tx_signature: str
    entry_price: float
    size_usd: float
    leverage: float
    fee_usd: float
    error: str = ""


@dataclass
class CloseResult:
    success: bool
    tx_signature: str
    size_reduced_usd: float
    received_usd: float
    pnl_pct: float
    fee_usd: float
    error: str = ""


def _run(args: list, timeout: int = 120) -> dict:
    """Execute Jupiter CLI and return parsed JSON."""
    cmd = JUP_CLI + args
    log.debug(f"jup cli: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.expanduser("~"),
        )
        # CLI outputs JSON to stdout when configured, but may print npm install
        # messages to stderr. We parse the last JSON object from stdout.
        lines = result.stdout.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{") or line.startswith("["):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        # Fallback: try full stdout
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        log.error("jup cli timeout")
        raise RuntimeError("jup cli timeout")
    except Exception as e:
        log.error(f"jup cli error: {e}")
        raise


def ensure_configured() -> bool:
    """Ensure CLI output is JSON and default key exists."""
    try:
        _run(["config", "set", "--output", "json"])
        return True
    except Exception as e:
        log.warning(f"jup cli config failed: {e}")
        return False


def open_position(
    asset: str = "SOL",
    side: str = "long",
    collateral_usd: float = 10.0,
    leverage: float = 2.0,
    input_token: str = "USDC",
    slippage_bps: int = 200,
    dry_run: bool = False,
) -> OpenResult:
    """Open a Jupiter perp position via official CLI."""
    args = [
        "perps", "open",
        "--asset", asset.upper(),
        "--side", side.lower(),
        "--amount", str(collateral_usd),
        "--input", input_token.upper(),
        "--leverage", str(leverage),
        "--slippage", str(slippage_bps),
    ]
    if dry_run:
        args.append("--dry-run")

    try:
        data = _run(args)
        if dry_run:
            return OpenResult(
                success=True,
                tx_signature="",
                entry_price=float(data.get("entryPrice", 0)),
                size_usd=float(data.get("size", 0)),
                leverage=float(data.get("leverage", 0)),
                fee_usd=float(data.get("openFee", 0)),
            )
        return OpenResult(
            success=True,
            tx_signature=data.get("txSignature", ""),
            entry_price=float(data.get("entryPrice", 0)),
            size_usd=float(data.get("size", 0)),
            leverage=float(data.get("leverage", 0)),
            fee_usd=float(data.get("openFee", 0)),
        )
    except Exception as e:
        return OpenResult(success=False, tx_signature="", entry_price=0, size_usd=0, leverage=0, fee_usd=0, error=str(e))


def close_position(
    position_pubkey: str,
    size_usd: Optional[float] = None,
    receive_token: str = "USDC",
    slippage_bps: int = 200,
    dry_run: bool = False,
) -> CloseResult:
    """Close (or partially close) a Jupiter perp position."""
    args = [
        "perps", "close",
        "--position", position_pubkey,
        "--receive", receive_token.upper(),
        "--slippage", str(slippage_bps),
    ]
    if size_usd is not None:
        args += ["--size", str(size_usd)]
    if dry_run:
        args.append("--dry-run")

    try:
        data = _run(args)
        if dry_run:
            return CloseResult(
                success=True,
                tx_signature="",
                size_reduced_usd=float(data.get("sizeReduced", 0)),
                received_usd=float(data.get("receivedUsd", 0) or data.get("received", 0)),
                pnl_pct=float(data.get("pnlPct", 0)),
                fee_usd=float(data.get("fees", 0)),
            )
        return CloseResult(
            success=True,
            tx_signature=data.get("txSignature", ""),
            size_reduced_usd=float(data.get("sizeReduced", 0)),
            received_usd=float(data.get("receivedUsd", 0) or data.get("received", 0)),
            pnl_pct=float(data.get("pnlPct", 0)),
            fee_usd=float(data.get("fees", 0)),
        )
    except Exception as e:
        return CloseResult(success=False, tx_signature="", size_reduced_usd=0, received_usd=0, pnl_pct=0, fee_usd=0, error=str(e))


def get_positions() -> List[PerpPosition]:
    """Fetch all open perp positions."""
    try:
        data = _run(["perps", "positions"])
        positions = []
        for p in data.get("positions", []):
            positions.append(PerpPosition(
                pubkey=p.get("pubkey", ""),
                asset=p.get("asset", ""),
                side=p.get("side", ""),
                size_usd=float(p.get("size", 0)),
                entry_price=float(p.get("entryPrice", 0)),
                mark_price=float(p.get("markPrice", 0)),
                pnl_pct=float(p.get("pnlPct", 0)),
                leverage=float(p.get("leverage", 0)),
                liq_price=float(p.get("liqPrice", 0)),
            ))
        return positions
    except Exception as e:
        log.error(f"get_positions failed: {e}")
        return []
