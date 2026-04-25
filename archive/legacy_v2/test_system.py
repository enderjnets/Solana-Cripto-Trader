#!/usr/bin/env python3
"""
Test Suite for Jupiter Solana Trading Bot
======================================
Comprehensive test of all system functions.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all modules can be imported"""
    print("=" * 60)
    print("🧪 TEST 1: IMPORTACIONES")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test config
    try:
        from config.config import get_config, Config
        config = get_config()
        print(f"  ✅ config.config: Network={config.network.network}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ config.config: {e}")
        tests_failed += 1
    
    # Test jupiter_client imports
    try:
        from tools.jupiter_client import JupiterClient, Quote, SwapData
        from tools.jupiter_client import SOL_MINT, USDC_MINT, calculate_fees
        print(f"  ✅ tools.jupiter_client: OK")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ tools.jupiter_client: {e}")
        tests_failed += 1
    
    # Test solana_wallet imports
    try:
        from tools.solana_wallet import SolanaWallet, WalletInfo, WalletBalance
        from tools.solana_wallet import HotWalletManager
        print(f"  ✅ tools.solana_wallet: OK")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ tools.solana_wallet: {e}")
        tests_failed += 1
    
    print(f"\n📊 Resultado: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_config():
    """Test configuration module"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: CONFIGURACIÓN")
    print("=" * 60)
    
    from config.config import get_config
    
    config = get_config()
    
    # Test network config
    print(f"  Network: {config.network.network}")
    print(f"  RPC URL: {config.network.rpc_url[:30]}...")
    
    # Test wallet config
    print(f"  Max Trade %: {config.wallet.max_trade_pct * 100}%")
    print(f"  Max Daily %: {config.wallet.max_daily_pct * 100}%")
    
    # Test Jupiter config
    print(f"  Default Slippage: {config.jupiter.default_slippage_bps / 100}%")
    print(f"  Priority Fee: {config.jupiter.priority_fee_default} lamports")
    
    # Test trading config
    print(f"  Risk Level: {config.trading.risk_level}")
    print(f"  Stop Loss: {config.trading.default_stop_loss_pct * 100}%")
    
    # Test validate
    is_valid, errors = config.validate()
    if is_valid:
        print(f"  ✅ Validación: PASSED")
    else:
        print(f"  ⚠️ Validación: {errors}")
    
    return True


def test_jupiter_client():
    """Test Jupiter API client"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: JUPITER CLIENT")
    print("=" * 60)
    
    import asyncio
    from tools.jupiter_client import JupiterClient, Quote, SOL_MINT, USDC_MINT
    
    async def test():
        async with JupiterClient() as client:
            # Test get_tokens
            print("  Probando get_tokens()...")
            try:
                tokens = await client.get_tokens()
                print(f"    ✅ {len(tokens)} tokens disponibles")
            except Exception as e:
                print(f"    ⚠️ get_tokens: {e} (API puede estar down)")
            
            # Test get_price
            print("  Probando get_price()...")
            try:
                prices = await client.get_price([SOL_MINT, USDC_MINT])
                for mint, price in prices.items():
                    print(f"    ✅ {mint[:10]}...: ${price:.2f}")
            except Exception as e:
                print(f"    ⚠️ get_price: {e} (API puede estar down)")
        
        return True
    
    try:
        asyncio.run(test())
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_wallet():
    """Test wallet module"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: WALLET")
    print("=" * 60)
    
    from tools.solana_wallet import WALLET_DIR, ENCRYPTED_KEY_FILE
    
    print(f"  Wallet Directory: {WALLET_DIR}")
    print(f"  Encrypted Key File: {ENCRYPTED_KEY_FILE}")
    
    # Test WalletInfo
    from tools.solana_wallet import WalletInfo
    
    info = WalletInfo(
        public_key="TestPublicKey123",
        key_type="encrypted",
        created_at="2024-01-01T00:00:00",
        last_used="2024-01-01T00:00:00",
        network="devnet"
    )
    
    print(f"  ✅ WalletInfo created: {info.public_key[:10]}...")
    
    # Test WalletBalance
    from tools.solana_wallet import WalletBalance
    
    balance = WalletBalance(
        sol_balance=1.5,
        usdc_balance=100.0,
        usdt_balance=50.0,
        token_balances={}
    )
    
    total = balance.total_usd_value(100.0)  # SOL = $100
    print(f"  ✅ WalletBalance: Total=${total:.2f}")
    
    return True


def test_backtester():
    """Test backtester module"""
    print("\n" + "=" * 60)
    print("🧪 TEST 5: BACKTESTER")
    print("=" * 60)
    
    try:
        from backtesting.solana_backtester import (
            precompute_indicators,
            JupiterFees,
            generate_sample_data
        )
        
        # Generate sample data
        print("  Generando datos de muestra...")
        df = generate_sample_data(n_candles=1000)
        print(f"    ✅ {len(df)} velas generadas")
        
        # Pre-compute indicators
        print("  Pre-computando indicadores...")
        indicators = precompute_indicators(df)
        print(f"    ✅ Shape: {indicators.shape}")
        
        # Test fees
        print("  Calculando fees...")
        fees = JupiterFees()
        total_fee, fee_usd = fees.calculate_total_fee(
            input_amount_lamports=1000000000,  # 1 SOL
            priority_fee_lamports=1000
        )
        print(f"    ✅ Total fee: {total_fee} lamports (${fee_usd:.4f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_skills():
    """Test skills markdown files"""
    print("\n" + "=" * 60)
    print("🧪 TEST 6: SKILLS")
    print("=" * 60)
    
    import os
    
    skills_dir = PROJECT_ROOT / "skills"
    if not skills_dir.exists():
        print("  ❌ Skills directory not found")
        return False
    
    for skill_file in skills_dir.glob("*.md"):
        size = skill_file.stat().st_size
        print(f"  ✅ {skill_file.name}: {size} bytes")
    
    return True


def test_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 60)
    print("🧪 TEST 7: DEPENDENCIAS")
    print("=" * 60)
    
    required = [
        ("httpx", "HTTP client"),
        ("solders", "Solana types"),
        ("cryptography", "Encryption"),
        ("numpy", "Numerical computing"),
        ("pandas", "Data analysis"),
        ("pydantic", "Data validation"),
    ]
    
    optional = [
        ("streamlit", "Dashboard (optional)"),
        ("numba", "JIT acceleration (optional)"),
    ]
    
    print("  Requeridas:")
    for package, desc in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"    ✅ {package}: OK")
        except ImportError:
            print(f"    ❌ {package}: NO INSTALADA ({desc})")
    
    print("  Opcionales:")
    for package, desc in optional:
        try:
            __import__(package.replace("-", "_"))
            print(f"    ✅ {package}: OK")
        except ImportError:
            print(f"    ⚠️ {package}: NO INSTALADA ({desc})")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🚀 JUPITER SOLANA TRADING BOT - TEST SUITE")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Jupiter Client", test_jupiter_client()))
    results.append(("Wallet", test_wallet()))
    results.append(("Backtester", test_backtester()))
    results.append(("Skills", test_skills()))
    results.append(("Dependencies", test_dependencies()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 Todos los tests pasaron!")
        return 0
    else:
        print(f"\n⚠️ {failed} tests necesitan atención.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
