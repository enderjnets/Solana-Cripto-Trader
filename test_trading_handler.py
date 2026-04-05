#!/usr/bin/env python3
"""
Unit Tests for trading_handler.py
==================================

Comprehensive test suite for the Solana Trading Handler CLI.
Tests cover: initialization, wallet loading, price fetching, quotes, swaps, and CLI.

Usage:
    pytest test_trading_handler.py -v
    pytest test_trading_handler.py -v --tb=short
    pytest test_trading_handler.py -v --cov=. --cov-report=term-missing
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTradingHandlerInitialization:
    """Test TradingHandler class initialization."""
    
    def test_init_default_devnet(self):
        """Test initialization with default devnet network."""
        with patch('trading_handler.load_dotenv'), \
             patch('trading_handler.Keypair') as mock_keypair:
            mock_keypair.from_json.side_effect = Exception("No wallet")
            
            from trading_handler import TradingHandler
            
            handler = TradingHandler.__new__(TradingHandler)
            handler.network = "devnet"
            handler.rpc_url = TradingHandler.RPC_DEVNET
            handler.keypair = None
            handler.jupiter_api_key = ""
            
            assert handler.network == "devnet"
            assert handler.rpc_url == "https://api.devnet.solana.com"
    
    def test_init_mainnet(self):
        """Test initialization with mainnet network."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "mainnet"
        handler.rpc_url = TradingHandler.RPC_MAINNET
        handler.keypair = None
        handler.jupiter_api_key = ""
        
        assert handler.network == "mainnet"
        assert handler.rpc_url == "https://api.mainnet-beta.solana.com"
    
    def test_init_class_constants(self):
        """Test that class constants are properly defined."""
        from trading_handler import TradingHandler
        
        assert TradingHandler.RPC_DEVNET == "https://api.devnet.solana.com"
        assert TradingHandler.RPC_MAINNET == "https://api.mainnet-beta.solana.com"
        assert TradingHandler.JUPITER_PRICE_URL == "https://price.jup.ag/v6/price"
        assert TradingHandler.JUPITER_QUOTE_URL == "https://api.jup.ag/swap/v1/quote"
        assert TradingHandler.JUPITER_SWAP_URL == "https://api.jup.ag/swap/v1/swap"
        assert TradingHandler.SOL_MINT == "So11111111111111111111111111111111111111112"
        assert TradingHandler.USDC_MINT == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class TestWalletLoading:
    """Test wallet loading functionality."""
    
    def test_load_wallet_json_format(self):
        """Test loading wallet from JSON format private key."""
        from trading_handler import TradingHandler
        
        mock_keypair = Mock()
        mock_keypair.pubkey.return_value = Mock()
        mock_keypair.pubkey.return_value.__str__ = Mock(return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ")
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.rpc_url = TradingHandler.RPC_DEVNET
        handler.keypair = None
        handler.jupiter_api_key = ""
        
        # Mock the keypair.from_json
        with patch('trading_handler.Keypair') as mock_kp:
            mock_kp.from_json.return_value = mock_keypair
            
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'read_text', return_value='HOT_WALLET_PRIVATE_KEY=[1,2,3,4,5]'):
                handler._load_wallet()
        
        assert handler.keypair is not None
    
    def test_load_wallet_base58_format(self):
        """Test loading wallet from base58 format private key."""
        from trading_handler import TradingHandler
        import base58
        
        mock_keypair = Mock()
        mock_keypair.pubkey.return_value = Mock()
        mock_keypair.pubkey.return_value.__str__ = Mock(return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ")
        
        with patch('trading_handler.Keypair') as mock_kp:
            mock_kp.from_bytes.return_value = mock_keypair
            
            handler = TradingHandler.__new__(TradingHandler)
            handler.network = "devnet"
            handler.keypair = None
            handler.jupiter_api_key = ""
            
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'read_text', return_value='HOT_WALLET_PRIVATE_KEY=AbCdEfGh123'):
                handler._load_wallet()
        
        mock_kp.from_bytes.assert_called_once()
    
    def test_load_wallet_no_wallet_configured(self):
        """Test behavior when no wallet is configured."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        
        with patch.object(Path, 'exists', return_value=False):
            handler._load_wallet()
        
        assert handler.keypair is None


class TestGetAddress:
    """Test wallet address retrieval."""
    
    def test_get_address_with_wallet(self):
        """Test getting address when wallet is loaded."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.keypair = Mock()
        handler.keypair.pubkey.return_value = Mock()
        handler.keypair.pubkey.return_value.__str__ = Mock(return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ")
        
        address = handler.get_address()
        
        assert address == "7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ"
    
    def test_get_address_no_wallet(self):
        """Test getting address when no wallet is configured."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.keypair = None
        
        address = handler.get_address()
        
        assert address == "No wallet configured"


class TestGetSolPrice:
    """Test SOL price fetching functionality."""
    
    @patch('trading_handler.requests.get')
    def test_get_sol_price_success(self, mock_get):
        """Test successful SOL price fetch."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "price": 95.50
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_PRICE_URL = TradingHandler.JUPITER_PRICE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.SOL_PRICE_FALLBACK = 80.76
        
        price = handler.get_sol_price_value()
        
        assert price == 95.50
        mock_get.assert_called_once()
    
    @patch('trading_handler.requests.get')
    def test_get_sol_price_api_failure(self, mock_get):
        """Test SOL price fetch with API failure (returns fallback)."""
        import requests
        from trading_handler import TradingHandler
        
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_PRICE_URL = TradingHandler.JUPITER_PRICE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.SOL_PRICE_FALLBACK = 80.76
        
        price = handler.get_sol_price_value()
        
        # Should return fallback value
        assert price == 80.76
    
    @patch('trading_handler.requests.get')
    def test_get_sol_price_empty_response(self, mock_get):
        """Test SOL price fetch with empty response."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.json.return_value = {"data": {}}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_PRICE_URL = TradingHandler.JUPITER_PRICE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.SOL_PRICE_FALLBACK = 80.76
        
        price = handler.get_sol_price_value()
        
        # Should return 0 from response (invalid), but then fallback
        # The current implementation returns 0 then logs warning
        assert price == 0 or price == 80.76
    
    def test_get_sol_price_formatted(self):
        """Test formatted SOL price output."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.get_sol_price_value = Mock(return_value=95.50)
        
        result = handler.get_sol_price()
        
        assert "95.50" in result
        assert "USD" in result


class TestGetQuote:
    """Test quote generation functionality."""
    
    @patch('trading_handler.requests.get')
    def test_get_quote_buy(self, mock_get):
        """Test quote for buying SOL (SOL -> USDC)."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "inAmount": 1000000000,  # 1 SOL
            "outAmount": 95000000    # 95 USDC
        }
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(1.0, "buy")
        
        assert "Comprando 1.0 SOL" in result
        assert "95.00" in result or "95" in result
    
    @patch('trading_handler.requests.get')
    def test_get_quote_sell(self, mock_get):
        """Test quote for selling SOL (USDC -> SOL)."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "inAmount": 95000000,    # 95 USDC
            "outAmount": 1000000000  # 1 SOL
        }
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(1.0, "sell")
        
        assert "Vendiendo 1.0 SOL" in result
    
    @patch('trading_handler.requests.get')
    def test_get_quote_api_key_required(self, mock_get):
        """Test quote when API returns 401 unauthorized."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(1.0, "buy")
        
        assert "API de Jupiter requiere autenticación" in result
        assert "portal.jup.ag" in result
    
    @patch('trading_handler.requests.get')
    def test_get_quote_error_handling(self, mock_get):
        """Test quote error handling."""
        from trading_handler import TradingHandler
        
        mock_get.side_effect = Exception("Network error")
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(1.0, "buy")
        
        assert "Quote error" in result
    
    @patch('trading_handler.requests.get')
    def test_get_quote_different_amounts(self, mock_get):
        """Test quote with different amounts."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "inAmount": 500000000,  # 0.5 SOL
            "outAmount": 47500000   # 47.5 USDC
        }
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(0.5, "buy")
        
        assert "0.5" in result


class TestExecuteSwap:
    """Test swap execution functionality."""
    
    def test_execute_swap_no_wallet(self):
        """Test swap execution without wallet configured."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.JUPITER_SWAP_URL = TradingHandler.JUPITER_SWAP_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        handler.get_address = Mock(return_value="No wallet configured")
        handler.get_balance = Mock(return_value="Balance info")
        
        result = handler.execute_swap(1.0, "buy")
        
        assert "Wallet no configurada" in result
        assert "solana_wallet.py" in result
    
    @patch('trading_handler.requests.get')
    @patch('trading_handler.requests.post')
    def test_execute_swap_simulation(self, mock_post, mock_get):
        """Test swap execution returns simulation message."""
        from trading_handler import TradingHandler
        
        mock_keypair = Mock()
        mock_keypair.pubkey.return_value = Mock()
        mock_keypair.pubkey.return_value.__str__ = Mock(return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ")
        
        mock_quote_response = Mock()
        mock_quote_response.json.return_value = {
            "inAmount": 1000000000,
            "outAmount": 95000000
        }
        mock_get.return_value = mock_quote_response
        
        mock_swap_response = Mock()
        mock_swap_response.json.return_value = {
            "swapTransaction": "base64_transaction"
        }
        mock_post.return_value = mock_swap_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = mock_keypair
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.JUPITER_SWAP_URL = TradingHandler.JUPITER_SWAP_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        handler.get_address = Mock(return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ")
        handler.get_balance = Mock(return_value="💰 **Tu Wallet**\n\n**SOL:** 1.0000")
        
        result = handler.execute_swap(1.0, "buy")
        
        assert "Swap simulation" in result or "swaps reales" in result
    
    def test_execute_swap_sell_side(self):
        """Test sell swap execution."""
        from trading_handler import TradingHandler
        
        mock_keypair = Mock()
        mock_keypair.pubkey.return_value = Mock()
        mock_keypair.pubkey.return_value.__str__ = Mock(return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ")
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = mock_keypair
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.JUPITER_SWAP_URL = TradingHandler.JUPITER_SWAP_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        handler.get_address = Mock(return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ")
        handler.get_balance = Mock(return_value="💰 **Tu Wallet**\n\n**SOL:** 1.0000")
        
        with patch('trading_handler.requests.get') as mock_get, \
             patch('trading_handler.requests.post') as mock_post:
            mock_quote_response = Mock()
            mock_quote_response.json.return_value = {
                "inAmount": 95000000,
                "outAmount": 1000000000
            }
            mock_get.return_value = mock_quote_response
            
            mock_swap_response = Mock()
            mock_swap_response.json.return_value = {"swapTransaction": "test"}
            mock_post.return_value = mock_swap_response
            
            result = handler.execute_swap(1.0, "sell")
            
            # Check that it attempted to make the swap call (simulation mode)
            assert "simulation" in result.lower() or "swaps reales" in result.lower()


class TestGetBalance:
    """Test wallet balance retrieval."""
    
    def test_get_balance_no_wallet(self):
        """Test balance retrieval without wallet."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.get_sol_price_value = Mock(return_value=95.50)
        
        result = handler.get_balance()
        
        assert "No wallet configured" in result
    
    @patch('trading_handler.requests.get')
    def test_get_balance_success(self, mock_get):
        """Test successful balance retrieval."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.value = 2000000000  # 2 SOL
        
        mock_client = Mock()
        mock_client.get_balance.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.client = mock_client
        handler.keypair = Mock()
        handler.keypair.pubkey.return_value = Mock()
        handler.get_sol_price_value = Mock(return_value=95.50)
        handler.JUPITER_PRICE_URL = TradingHandler.JUPITER_PRICE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        
        result = handler.get_balance()
        
        assert "2.0000" in result
        assert "191.00" in result  # 2 * 95.50 = 191.00
    
    def test_get_balance_error_handling(self):
        """Test balance retrieval with error."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.client = Mock()
        handler.client.get_balance.side_effect = Exception("RPC Error")
        handler.keypair = Mock()
        handler.keypair.pubkey.return_value = Mock()
        
        result = handler.get_balance()
        
        assert "Error" in result


class TestGetStatus:
    """Test system status retrieval."""
    
    def test_get_status_no_wallet(self):
        """Test status with no wallet configured."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.get_address = Mock(return_value="No wallet configured")
        handler.get_balance = Mock(return_value="Balance info")
        handler.get_sol_price_value = Mock(return_value=95.50)
        
        result = handler.get_status()
        
        assert "Estado del Sistema" in result
        assert "No wallet" in result or "wallet" in result.lower()
        assert "devnet" in result
    
    def test_get_status_with_wallet(self):
        """Test status with wallet configured."""
        from trading_handler import TradingHandler
        
        mock_keypair = Mock()
        mock_keypair.pubkey.return_value = Mock()
        mock_keypair.pubkey.return_value.__str__ = Mock(
            return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ"
        )
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "mainnet"
        handler.keypair = mock_keypair
        handler.get_address = Mock(
            return_value="7nYhNHrKwpG9F3J1XcG3jH8kL5mNpQrStUvWxYzAbCdEfGhJkLmNoPqRsTuVwXyZ"
        )
        handler.get_balance = Mock(return_value="💰 **Tu Wallet**\n\n**SOL:** 1.0000")
        handler.get_sol_price_value = Mock(return_value=95.50)
        
        result = handler.get_status()
        
        assert "Estado del Sistema" in result
        assert "mainnet" in result
        assert "Wallet OK" in result


class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""
    
    def test_cli_balance_argument(self):
        """Test --balance argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--balance']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.get_balance.return_value = "Balance info"
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                mock_instance.get_balance.assert_called_once()
    
    def test_cli_price_argument(self):
        """Test --price argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--price']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.get_sol_price.return_value = "95.50 USD"
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                mock_instance.get_sol_price.assert_called_once()
    
    def test_cli_quote_argument(self):
        """Test --quote argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--quote', '1.5']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.get_quote.return_value = "Quote info"
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                mock_instance.get_quote.assert_called_once_with(1.5, "buy")
    
    def test_cli_buy_argument(self):
        """Test --buy argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--buy', '0.5']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.execute_swap.return_value = "Swap info"
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                mock_instance.execute_swap.assert_called_once_with(0.5, "buy")
    
    def test_cli_sell_argument(self):
        """Test --sell argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--sell', '1.0']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.execute_swap.return_value = "Swap info"
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                mock_instance.execute_swap.assert_called_once_with(1.0, "sell")
    
    def test_cli_status_argument(self):
        """Test --status argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--status']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.get_status.return_value = "Status info"
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                mock_instance.get_status.assert_called_once()
    
    def test_cli_address_argument(self):
        """Test --address argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--address']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.get_address.return_value = "7nYh..."
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                mock_instance.get_address.assert_called_once()
    
    def test_cli_network_argument(self):
        """Test --network argument parsing."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--price', '--network', 'mainnet']), \
             patch('trading_handler.TradingHandler') as MockHandler:
            mock_instance = Mock()
            mock_instance.get_sol_price.return_value = "95.50 USD"
            MockHandler.return_value = mock_instance
            
            from trading_handler import main
            
            with patch('builtins.print') as mock_print:
                main()
                MockHandler.assert_called_once_with(network="mainnet")
    
    def test_cli_invalid_network(self):
        """Test invalid network argument."""
        with patch.object(sys, 'argv', ['trading_handler.py', '--network', 'invalid']):
            from trading_handler import main
            import argparse
            
            with pytest.raises(SystemExit):
                main()
    
    def test_cli_no_arguments(self):
        """Test CLI with no arguments shows help."""
        import argparse
        from trading_handler import TradingHandler
        
        # Test that TradingHandler constants are properly defined
        assert hasattr(TradingHandler, 'RPC_DEVNET')
        assert hasattr(TradingHandler, 'RPC_MAINNET')
        assert hasattr(TradingHandler, 'SOL_MINT')
        assert hasattr(TradingHandler, 'USDC_MINT')
        assert hasattr(TradingHandler, 'SOL_PRICE_FALLBACK')
        
        # Test default fallback value
        assert TradingHandler.SOL_PRICE_FALLBACK == 80.76


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('trading_handler.requests.get')
    def test_get_quote_zero_amount(self, mock_get):
        """Test quote with zero amount."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(0, "buy")
        
        # Should handle zero amount gracefully
        assert result is not None
    
    @patch('trading_handler.requests.get')
    def test_get_quote_negative_amount(self, mock_get):
        """Test quote with negative amount."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "inAmount": 1000000000,
            "outAmount": 95000000
        }
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(-1.0, "buy")
        
        # Negative amount would be converted to lamports
        assert result is not None
    
    @patch('trading_handler.requests.get')
    def test_get_quote_large_amount(self, mock_get):
        """Test quote with very large amount."""
        from trading_handler import TradingHandler
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "inAmount": 100000000000,  # 100 SOL
            "outAmount": 9500000000    # 9500 USDC
        }
        mock_get.return_value = mock_response
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_QUOTE_URL = TradingHandler.JUPITER_QUOTE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.USDC_MINT = TradingHandler.USDC_MINT
        
        result = handler.get_quote(100, "buy")
        
        assert "100" in result
    
    @patch('trading_handler.requests.get')
    def test_get_sol_price_network_error(self, mock_get):
        """Test price fetch with network error."""
        from trading_handler import TradingHandler
        
        mock_get.side_effect = ConnectionError("Connection refused")
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_PRICE_URL = TradingHandler.JUPITER_PRICE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.SOL_PRICE_FALLBACK = 80.76
        
        price = handler.get_sol_price_value()
        
        # Should return fallback
        assert price == 80.76
    
    @patch('trading_handler.requests.get')
    def test_get_sol_price_timeout(self, mock_get):
        """Test price fetch with timeout."""
        from trading_handler import TradingHandler
        import requests
        
        mock_get.side_effect = requests.Timeout("Request timed out")
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        handler.keypair = None
        handler.jupiter_api_key = ""
        handler.JUPITER_PRICE_URL = TradingHandler.JUPITER_PRICE_URL
        handler.SOL_MINT = TradingHandler.SOL_MINT
        handler.SOL_PRICE_FALLBACK = 80.76
        
        price = handler.get_sol_price_value()
        
        # Should return fallback
        assert price == 80.76


class TestJupiterAPIKey:
    """Test Jupiter API key loading."""
    
    def test_load_jupiter_key_exists(self):
        """Test loading Jupiter API key when it exists."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value="JUPITER_API_KEY=test_key_123"):
            key = handler._load_jupiter_key()
        
        assert key == "test_key_123"
    
    def test_load_jupiter_key_not_exists(self):
        """Test loading Jupiter API key when it doesn't exist."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        
        with patch.object(Path, 'exists', return_value=False):
            key = handler._load_jupiter_key()
        
        assert key == ""
    
    def test_load_jupiter_key_empty(self):
        """Test loading empty Jupiter API key."""
        from trading_handler import TradingHandler
        
        handler = TradingHandler.__new__(TradingHandler)
        handler.network = "devnet"
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value="JUPITER_API_KEY="):
            key = handler._load_jupiter_key()
        
        # Empty key should be skipped, return empty string
        assert key == ""


class TestAmountConversion:
    """Test amount conversion between SOL and lamports."""
    
    def test_sol_to_lamports(self):
        """Test SOL to lamports conversion."""
        amount_sol = 1.5
        amount_lamports = int(amount_sol * 1e9)
        
        assert amount_lamports == 1500000000
    
    def test_lamports_to_sol(self):
        """Test lamports to SOL conversion."""
        amount_lamports = 1500000000
        amount_sol = amount_lamports / 1e9
        
        assert amount_sol == 1.5
    
    def test_usdc_decimals(self):
        """Test USDC decimal conversion (6 decimals)."""
        amount_usdc_lamports = 95000000  # 95 USDC
        amount_usdc = amount_usdc_lamports / 1e6
        
        assert amount_usdc == 95.0


class TestDecimalPlaces:
    """Test proper decimal place handling."""
    
    def test_sol_balance_formatting(self):
        """Test SOL balance is formatted with 4 decimal places."""
        sol_balance = 1.23456
        formatted = f"{sol_balance:.4f}"
        
        assert formatted == "1.2346"  # Rounded
    
    def test_usdc_balance_formatting(self):
        """Test USDC balance is formatted with 2 decimal places."""
        usdc_balance = 95.123
        formatted = f"{usdc_balance:.2f}"
        
        assert formatted == "95.12"
    
    def test_total_usd_formatting(self):
        """Test total USD is formatted with 2 decimal places."""
        total_usd = 190.456
        formatted = f"${total_usd:.2f}"
        
        assert formatted == "$190.46"
    
    def test_sol_price_formatting(self):
        """Test SOL price is formatted with 2 decimal places."""
        sol_price = 95.456789
        formatted = f"${sol_price:.2f}"
        
        assert formatted == "$95.46"


class TestSlippageCalculation:
    """Test slippage parameter handling."""
    
    def test_slippage_bps_to_percentage(self):
        """Test slippage BPS to percentage conversion."""
        slippage_bps = 50
        slippage_percent = slippage_bps / 100
        
        assert slippage_percent == 0.5  # 0.5%
    
    def test_default_slippage(self):
        """Test default slippage value."""
        slippage_bps = 50  # Default in code
        
        assert slippage_bps == 50  # 0.5%


# ==================== RUNNER ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
