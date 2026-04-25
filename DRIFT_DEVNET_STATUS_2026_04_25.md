# Drift Devnet Smoke Test — Estado 2026-04-25 01:00 UTC

## TL;DR
**Read path**: ✅ funciona end-to-end. **Write path**: ❌ bloqueado en `PerpMarketNotFound` (driftpy issue).
**No se modificó nada del bot live** — todo el trabajo en venv aislado + devnet. Live bot v2.12.32 sigue corriendo sano en mainnet spot.

## Lo que está listo (descubierto, no tuve que crear)
- `agents/drift_client.py` (17.7KB) — full async wrapper de driftpy con `open_sol_perp_market`, `open_sol_perp_limit`, `close_sol_perp`, `deposit_usdc`, `initialize_user`, `snapshot`. Phase 1-3.3 done.
- `agents/drift_adapter.py` (10.6KB) — sync wrapper con 8 safety gates (whitelist, leverage cap, daily loss, collateral cap).
- `tools/drift_setup.py` — init Drift user + deposit USDC + write `agents/data/drift_state.json`.
- `tools/drift_devnet_smoke.py` — 1 long + 1 short round-trip on SOL-PERP, asserts ±2% collateral.
- `tools/drift_devnet_mint_usdc.py` — mint devnet test USDC for collateral.
- `requirements.txt` ya tiene `driftpy>=0.8.89`.
- `.env` ya tiene DRIFT_ENABLED, DRIFT_ENV, DRIFT_MAX_LEVERAGE, DRIFT_DEFAULT_LEVERAGE, DRIFT_MAX_COLLATERAL_USD, DRIFT_MIN_USDC_RESERVE, DRIFT_FUNDING_WARN_PCT.
- `.venv-drift/` creado con driftpy 0.8.89 + deps. **No conflicta con .venv del bot.**

## Estado on-chain devnet (validado)
- Wallet (compartida con prod, devnet ≠ mainnet): `EEmtkySNz1SLNZBMBu6EsuqkEhttEKjejsEXdEFT2fMH`
- SOL: 2.464 SOL (suficiente para fees)
- USDC collateral en Drift: $0.50 total / $0.44 free
- Drift user PDA: ya inicializada (sub_account_id=0)
- SOL-PERP base: 0.0 (sin posición abierta)
- SOL-PERP mark: $80.02
- Funding hourly: +0.0029% (positivo)
- Read API `drift_setup.py --status`: ✅ pasa

## Qué falló: `PerpMarketNotFound 6078` en `place_and_take_perp_order`

```
Custom(6078)
Program log: Could not find perp market 0 at programs/drift/src/instructions/optional_accounts.rs:52
Program log: AnchorError occurred. Error Code: PerpMarketNotFound. Error Number: 6078.
```

### Diagnóstico
- `get_perp_market_account(0)` retorna PerpMarket válido con MarketStatus.Active() ✅
- `get_remaining_accounts(writable_perp_market_indexes=[0])` retorna 4 cuentas, 1 writable (`8UJgxaiQx5nTrdDgph5FiahMmzduuLTLf5WmsPegYA6W`) ✅
- Pero el programa Drift on-chain dice que no encuentra perp market 0 en remaining_accounts ❌

### Hipótesis
1. **driftpy 0.8.89 cache mismatch**: la versión del programa Drift en devnet evolucionó y el `writable_perp_market_indexes` ya no incluye el accounting account correcto.
2. **Account ordering bug**: el programa espera perp market en una posición específica de remaining_accounts; driftpy lo pone en otra.
3. **Devnet program migration**: SOL-PERP market puede haber cambiado de pubkey y los constants de driftpy 0.8.89 apuntan al viejo.

### Próximos pasos sugeridos (mañana)
1. Verificar que el pubkey de SOL-PERP en devnet es `8UJgxaiQx5nTrdDgph5FiahMmzduuLTLf5WmsPegYA6W` (lo que usa driftpy 0.8.89). Si Drift devnet hizo upgrade, este puede haber cambiado.
2. Issue en driftpy GitHub: buscar "PerpMarketNotFound" + "0.8.89" en `https://github.com/drift-labs/driftpy/issues`.
3. Patch alternativo en `agents/drift_client.py:open_sol_perp_market`: construir manualmente la ix con `get_remaining_accounts(writable_perp_market_indexes=[0], readable_perp_market_indexes=[0])` para incluir read+write.
4. Probar fork driftpy main HEAD (si el fix está merged pero sin release).
5. Última opción: usar perp orden manual via `get_place_and_take_perp_order_ix` con `remaining_accounts` explícitos forzando el SOL-PERP market PDA al inicio.

## Comandos para reproducir
```bash
cd /home/enderj/.openclaw/workspace/Solana-Cripto-Trader-Live
DRIFT_ENV=devnet DRIFT_RPC_URL=https://api.devnet.solana.com \
  .venv-drift/bin/python tools/drift_setup.py --status
# Output esperado: snapshot con sol_balance=2.46, free_collateral=$0.44, drift_user_exists=True

# Smoke test (esperado fallo en open_sol_perp):
DRIFT_ENV=devnet DRIFT_RPC_URL=https://api.devnet.solana.com \
  .venv-drift/bin/python tools/drift_devnet_smoke.py --env devnet --size 0.01
```

## Recomendación
**No habilitar DRIFT_ENABLED=true** hasta que el smoke test devnet pase. Riesgo si se enciende sin write-path validado: el bot intentará abrir posiciones en mainnet, fallará al transmitir, marcará `needs_manual_reconcile`, y el spot trading sigue funcionando correctamente.

Mientras tanto: spot bot v2.12.32 corre sin tocar drift code (todo está detrás de `DRIFT_ENABLED` gate).

## Trabajo nocturno autónomo realizado
- ✅ Auditoría del código drift en `live` branch (descubrió 95% ya implementado)
- ✅ Setup `.venv-drift/` con driftpy 0.8.89 (no toca .venv del bot)
- ✅ Validó conexión devnet end-to-end (read API)
- ✅ Verificó devnet wallet ya inicializado + collateral
- ❌ Smoke test write op (bloqueado, NO es nuestro código — es driftpy/Drift program)
- ✅ Documentación del bloqueo + plan de acción

## Tiempo estimado para desbloquear (mañana, con contexto fresco)
- Verificación de pubkey + driftpy issue search: 20 min
- Patch del open_sol_perp_market: 30 min  
- Re-run smoke test: 5 min
- Si pasa: enable DRIFT_ENABLED=true en branch separado + plan de mainnet con $20: 1h
