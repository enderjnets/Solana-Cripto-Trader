# Jupiter Perpetuals Integration — Estado v2.13.0-alpha (2026-04-25)

## TL;DR

**Sprint 1 completo (skeleton + research)**. Sprint 2 (write ops) requiere parse del IDL del programa Drift on-chain — diseño Phase 2.

- ✅ Program ID verificado on-chain: `PERPHjGBqRHArX4DySjwM6UJHiR3sWAatqfdBS2qQJu` (owner BPFLoaderUpgradeable, executable)
- ✅ Skeletons deployados: `agents/jupiter_perp_client.py`, `agents/jupiter_perp_adapter.py`, `tools/jupiter_perp_setup.py`, `tools/jupiter_perp_smoke.py`
- ✅ `.env.example` actualizado con 9 nuevas vars `JUP_PERP_*`
- ✅ Compile clean en todos los archivos
- ⏸ **Phase 2 pendiente**: parse IDL + build instructions Python

## Por qué Jupiter Perps (vs Drift)

Drift Protocol fue **hackeado el 2026-04-01** (pérdida $285M, vector durable_nonces + token CVT falso, atacantes DPRK). Pivotamos a Jupiter Perpetuals que:

- Usa el **mismo Jupiter SDK** que ya usamos para spot
- **JLP pool $2.5B** (más profundo que Drift post-hack $1.5B)
- **Mismo wallet** (`EEmtky...`) ya configurado
- Hasta 100x leverage (capamos a 3x)
- Ecosistema Solana nativo
- No comprometido por hacks recientes

## Hallazgos de la investigación

### SDK Python: NO existe oficial

- `jupiter-python-sdk` (0xTaoDev) y `jup-python-sdk` (Jupiter-DevRel) **solo soportan spot/swap/limit/DCA**
- Perps roadmap "🚀 TO-DO" — sin ETA
- REST API oficial: solo swap (no perps)

### Alternativas

1. **Manual ix building** (elegida) — usar anchorpy/solders + IDL parseado de `https://github.com/julianfssen/jupiter-perps-anchor-idl-parsing`
2. **C# bridge** (`Solnet.JupiterPerps`) — descartado, agrega dependencia .NET
3. **Esperar SDK oficial** — descartado, sin ETA

### IDL del programa: catálogo de instrucciones

Categorías de instructions que necesitaremos:

**Position ops** (foco Phase 2):
- `instantIncreasePosition` — abrir/aumentar posición directamente (lo que necesitamos)
- `instantDecreasePosition` — cerrar/reducir posición
- `createIncreasePositionMarketRequest` + `increasePosition4` — flow request-fulfillment (alternativo)
- `createDecreasePositionRequest2` + `decreasePosition4`
- `liquidateFullPosition4` — liquidación (no la usamos del lado cliente)

**TP/SL**:
- `instantCreateTpsl` — crear TP/SL al abrir
- `instantUpdateTpsl` — actualizar después

**Liquidity (NO usamos)**:
- `addLiquidity2`, `removeLiquidity2` — para ser LP del JLP pool, no para trading

## Arquitectura implementada

### `agents/jupiter_perp_client.py` (~250 líneas)

Async wrapper. API mirror del `drift_client.py` para swap drop-in:

```python
class JupiterPerpClient:
    async def initialize() -> None  # ✅ implementado
    async def get_native_sol_balance() -> float  # ✅
    async def perp_account_exists(market) -> bool  # ⏸ TODO
    async def get_mark_price(market) -> float  # ⏸ TODO Phase 2
    async def get_funding_rate(market) -> float  # ⏸ TODO Phase 2
    async def snapshot() -> AccountSnapshot  # ⏸ stub (sol_balance only)
    async def open_perp_position(market, direction, size_usd, leverage, slippage_bps) -> str  # ⏸ TODO Phase 2
    async def close_perp_position(market, slippage_bps) -> str  # ⏸ TODO Phase 2
    async def deposit_usdc(amount_usd) -> str  # ✅ no-op (Jupiter no requiere pre-deposit)
```

### `agents/jupiter_perp_adapter.py` (~170 líneas)

Sync wrapper con **8 safety gates** (mirror exacto de drift_adapter):
1. `JUP_PERP_ENABLED` env flag
2. `LIVE_TRADING_ENABLED` master switch
3. Kill switch
4. Daily loss cap
5. Market whitelist (`JUP_PERP_MARKET_WHITELIST=SOL`)
6. Confidence threshold (`MIN_CONFIDENCE`)
7. Leverage cap (`JUP_PERP_MAX_LEVERAGE=3`)
8. Free collateral check

API pública: `open_perp_position(signal)`, `close_perp_position(market)`, `get_account_snapshot()`. Devuelve `PerpResult` (mismo shape que `drift_adapter.PerpResult`).

### `tools/jupiter_perp_setup.py`

CLI:
- `--status` — read-only snapshot (✅ funciona, gracias a get_native_sol_balance)
- `--verify-program` — confirma program ID on-chain (✅ funciona)
- `--deposit X` — TODO (Jupiter no necesita deposit separado, write op se hace en open)

### `tools/jupiter_perp_smoke.py`

Smoke test stub. Phase 1 = solo verifica que llamada a `open_perp_position` falla con `NotImplementedError`. Phase 2 implementará el round-trip real.

## Phase 2 plan (próximo sprint)

### Tareas concretas

1. **Descargar IDL JSON** desde `julianfssen/jupiter-perps-anchor-idl-parsing/src/idl/perpetuals.json`
2. **Generate Python bindings** con `anchorpy.workspace.create_workspace_dir`
3. **Implementar `_derive_position_pda(owner, custody, side)`**:
   - Seed: `["position", owner.to_bytes(), custody.to_bytes(), side_byte]`
4. **Implementar `_derive_custody_pda(pool, mint)`**:
   - Seed: `["custody", pool.to_bytes(), mint.to_bytes()]`
5. **Implementar `open_perp_position`**:
   - Args para `instantIncreasePosition`:
     - `collateral_token_delta` (lamports USDC)
     - `size_usd_delta` (lamports * 10^6)
     - `side`: `Side::Long` o `Side::Short` (enum 0|1)
     - `slippage`: bps
   - Accounts:
     - owner (signer)
     - position PDA (writable)
     - pool PDA
     - custody PDA (collateral, writable) — USDC
     - custody PDA (target, writable) — SOL para SOL-PERP
     - pyth oracle (read-only)
     - owner USDC ATA (writable)
     - custody USDC ATA (writable)
     - perpetuals program
     - token program
6. **Implementar `close_perp_position`**: similar con `instantDecreasePosition`
7. **Smoke test mainnet $2** (con autorización explícita del user)

### Riesgos Phase 2

- Falta documentación oficial de account ordering — depender de community repo
- Si IDL cambia upstream → break (similar al issue de driftpy)
- Discriminadores de instrucciones pueden cambiar → mismo problema Drift devnet

### Mitigaciones

- Cache IDL local (no fetch dinámico)
- Smoke test en cada deploy
- Versión de driftpy stale fue 2 meses — probable similar buffer Jupiter

## Estado del bot

- v2.12.32.5 corriendo (skeleton NO afecta — todo gateado detrás de `JUP_PERP_ENABLED=false`)
- Drift code en `agents/drift_*.py` preservado como template (no se elimina)

## Siguientes pasos del usuario

Cuando estés listo para Sprint 2:
```bash
# Verificar skeleton funciona
ssh enderj@10.0.0.240 "cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live && python3 tools/jupiter_perp_setup.py --status"
# Output esperado: snapshot con sol_balance + perp_account_exists=false + Phase 1 stub note

# Cuando autorices Phase 2: implementar IDL parse + ix building
# Estimación: 3-5 días de desarrollo + 1 día smoke test mainnet $2
```

## Sources

- [Jupiter Perpetuals overview](https://dev.jup.ag/docs/perps)
- [julianfssen/jupiter-perps-anchor-idl-parsing](https://github.com/julianfssen/jupiter-perps-anchor-idl-parsing) — IDL + examples
- [DeepWiki: Program Instructions catalog](https://deepwiki.com/pengxuan37/jupiter-perps-anchor-idl-parsing/2.1-program-instructions)
- [Solnet.JupiterPerps (C# reference)](https://github.com/Bifrost-Technologies/Solnet.JupiterPerps)
