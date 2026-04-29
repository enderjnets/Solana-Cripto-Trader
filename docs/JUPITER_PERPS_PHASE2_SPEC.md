# Jupiter Perpetuals Phase 2 — IDL Specs Extraídos (2026-04-26)

Specs precisas para implementación Python sin descubrir nada en runtime.

## Path elegido: `createIncreasePositionMarketRequest`

Cliente crea on-chain request → keeper Jupiter detecta → ejecuta `increasePosition4` ~500ms-1s.

## Discriminator

Anchor estándar: `sha256("global:createIncreasePositionMarketRequest")[:8]`

Python:
```python
import hashlib
disc = hashlib.sha256(b'global:createIncreasePositionMarketRequest').digest()[:8]
```

Mismo patrón para: `global:createDecreasePositionMarketRequest`

## Args struct (`CreateIncreasePositionMarketRequestParams`)

Encoding bytes (Borsh, little-endian):

| Campo | Tipo | Bytes | Notas |
|---|---|---|---|
| `sizeUsdDelta` | u64 | 8 | USD lamports (×10⁶) — para $10 → 10_000_000 |
| `collateralTokenDelta` | u64 | 8 | Lamports del input token (USDC ×10⁶) |
| `side` | enum | 1 | 0=None, 1=Long, 2=Short |
| `priceSlippage` | u64 | 8 | Slippage cap como precio en USD lamports (NO bps) |
| `jupiterMinimumOut` | option<u64> | 1 ó 9 | 0x00 = None; 0x01 + u64 = Some |
| `counter` | u64 | 8 | Random nonce (e.g. `int(time.time()*1000)`) — único por request |

**OJO**: priceSlippage es precio absoluto, no bps. Para SOL @ $100 con 1% slippage:
- Long: priceSlippage = 101 × 10⁶
- Short: priceSlippage = 99 × 10⁶

## 16 accounts (orden estricto)

| # | Nombre | Mut | Signer | Cómo derivar |
|---|---|---|---|---|
| 0 | owner | ✓ | ✓ | Nuestra wallet (Pubkey) |
| 1 | fundingAccount | ✓ | | ATA(owner, USDC mint) — USDC source |
| 2 | perpetuals | | | PDA singleton: seeds=["perpetuals"], program=PERP |
| 3 | pool | | | Hardcoded pool JLP, seeds=["pool", "Crypto.1"] (name del pool principal) |
| 4 | position | ✓ | | PDA: seeds=["position", owner, pool, custody, side_byte] |
| 5 | positionRequest | ✓ | | PDA: seeds=["position_request", position, counter_le_bytes] |
| 6 | positionRequestAta | ✓ | | ATA(positionRequest, USDC mint) — needs creation if not exists |
| 7 | custody | | | PDA: seeds=["custody", pool, target_mint] (SOL para SOL-PERP long) |
| 8 | collateralCustody | | | PDA: seeds=["custody", pool, USDC mint] |
| 9 | inputMint | | | USDC mint: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` |
| 10 | referral | | | Optional — pasar None (encode como Pubkey::default ó skip si IDL lo permite) |
| 11 | tokenProgram | | | `TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA` |
| 12 | associatedTokenProgram | | | `ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL` |
| 13 | systemProgram | | | `11111111111111111111111111111111` |
| 14 | eventAuthority | | | PDA: seeds=["__event_authority"], program=PERP |
| 15 | program | | | Jupiter Perps Program ID: `PERPHjGBqRHArX4DySjwM6UJHiR3sWAatqfdBS2qQJu` |

## Side enum bytes para PDA seed

side=Long → seed byte `0x01`
side=Short → seed byte `0x02`

(El 0x00 None no se usa para abrir posición)

## Mints conocidos (mainnet)

- USDC: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`
- SOL (wrapped, para custody key): `So11111111111111111111111111111111111111112`
- BTC: `3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh` (wrapped BTC en JLP)
- ETH: `7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs` (wrapped ETH en JLP)

## Pool principal Jupiter Perps (mainnet)

Nombre: "Crypto.1" (single-pool design al lanzar). Pubkey hardcodeable después de derive.

## Workflow keeper

1. Cliente envía tx con `createIncreasePositionMarketRequest`
2. PositionRequest PDA queda creada on-chain con `executed=false`
3. Keeper Jupiter polleando programa detecta nueva request
4. Keeper firma `increasePosition4` que setea `executed=true`
5. Cliente debe pollear `PositionRequest.executed` para confirmar fulfillment
6. `maxRequestExecutionSec` (campo de Pool) define timeout — si keeper no fulfila → reembolso

## Plan de implementación (5-7 días)

### Día 1-2: PDA derivations + basic ix
- `agents/jupiter_perp_pdas.py` — funciones puras de derive (testeable offline)
- Test cada PDA contra known-good values del repo julianfssen
- Ix builder bytes (Borsh hand-encode, evitar anchorpy heavy)

### Día 3-4: Tx assembly + signing
- Compose: priority fee (50_000 micro-lamports), compute budget (400k CU)
- Sign con wallet existente (`load_wallet()`)
- Broadcast via solders RPC

### Día 5: Keeper polling + status
- `poll_position_request_executed(request_pda, timeout_sec=15)`
- Read `PositionRequest.executed` field via `getAccountInfo` + decode
- Return final position state

### Día 6: Decrease ix (mirror)
- Mismo patrón con `createDecreasePositionMarketRequest`
- `entirePosition: bool` para close completo

### Día 7: Smoke test mainnet $2
- REQUIERE autorización explícita del user
- Open SOL long $2 collateral 1x → wait 30s → close
- Verify wallet ±5% (fees + slippage)

## Riesgos identificados

1. **`priceSlippage` semantics**: docs comunidad dicen "price" pero algunos PRs sugieren bps. Verificar con caso conocido del repo julianfssen antes de prod.
2. **Pool name hardcoded**: si Jupiter despliega más pools, podría cambiar.
3. **Keeper fulfillment latency**: 500ms-1s típico, pero spike a >5s observado en alta congestión. Setear timeout 15s + retry logic.
4. **Counter collision**: 2 requests con mismo counter del mismo owner → conflicto. Usar timestamp en milisegundos como counter mitiga.
5. **Referral account optional**: IDL marca `isOptional: true` — anchor encoding requiere passing como account writable o usar AccountMeta::new_readonly(Pubkey::default()) — verificar.

## Sources

- IDL local: `agents/jupiter_perps_idl.json`
- Reference: `https://github.com/julianfssen/jupiter-perps-anchor-idl-parsing/blob/main/src/examples/perpetuals.ts`
