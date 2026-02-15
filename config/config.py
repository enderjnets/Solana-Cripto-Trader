#!/usr/bin/env python3
"""
Solana Jupiter Trading Bot Configuration
Centralized configuration management with environment variable support
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


# Base paths
PROJECT_ROOT = Path(__file__).parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class NetworkConfig:
    """Network configuration"""
    network: str = "devnet"  # mainnet, testnet, devnet
    
    @property
    def rpc_url(self) -> str:
        urls = {
            "mainnet": os.environ.get("SOLANA_RPC_MAINNET", "https://api.mainnet-beta.solana.com"),
            "testnet": os.environ.get("SOLANA_RPC_TESTNET", "https://api.testnet.solana.com"),
            "devnet": os.environ.get("SOLANA_RPC_DEVNET", "https://api.devnet.solana.com")
        }
        return urls.get(self.network, urls["devnet"])
    
    @property
    def explorer_url(self) -> str:
        urls = {
            "mainnet": "https://explorer.solana.com",
            "testnet": "https://explorer.solana.com/?cluster=testnet",
            "devnet": "https://explorer.solana.com/?cluster=devnet"
        }
        return urls.get(self.network, urls["devnet"])


@dataclass
class WalletConfig:
    """Wallet configuration"""
    hot_wallet_address: str = os.environ.get("HOT_WALLET_ADDRESS", "")
    hot_wallet_path: str = os.environ.get("HOT_WALLET_PATH", "~/.config/solana-jupiter-bot/wallet.enc")
    cold_wallet_address: str = os.environ.get("COLD_WALLET_ADDRESS", "")
    max_trade_pct: float = 0.10  # 10% del hot wallet
    max_daily_pct: float = 0.30  # 30% del hot wallet daily
    min_reserve_sol: float = 0.05  # Reserve para fees
    min_trade_sol: float = 0.01  # Mínimo trade


@dataclass
class JupiterConfig:
    """Jupiter API configuration"""
    base_url: str = "https://api.jup.ag/swap/v6"
    default_slippage_bps: int = 50  # 0.5%
    priority_fee_default: int = 1000  # lamports
    use_jito_tips: bool = True
    jito_tip_default: int = 500  # lamports
    
    # Token mints
    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenuNYW"
    JUP_MINT = "JUPyiwrYJFskUPiHa7hkeR8VUtkqjberbSOWd91pbT2"
    BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoZGVixqUi5iA2ztETHuJXJP"
    
    # Trading pairs
    TRADING_PAIRS = {
        "SOL-USDC": {"base": SOL_MINT, "quote": USDC_MINT, "decimals": 9},
        "SOL-USDT": {"base": SOL_MINT, "quote": USDT_MINT, "decimals": 9},
        "JUP-SOL": {"base": JUP_MINT, "quote": SOL_MINT, "decimals": 6},
        "BONK-USDC": {"base": BONK_MINT, "quote": USDC_MINT, "decimals": 6},
    }


@dataclass
class TradingConfig:
    """Trading parameters"""
    # Risk management
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    max_position_pct: float = 0.10  # 10% del capital
    max_concurrent_positions: int = 5
    max_daily_loss_pct: float = 0.10  # 10% daily loss limit
    
    # Stop loss / Take profit
    default_stop_loss_pct: float = 0.03  # 3%
    default_take_profit_pct: float = 0.06  # 6%
    
    # Timeframes
    analysis_timeframe: str = "1h"
    confirmation_timeframe: str = "15m"
    
    # Rebalancing
    rebalance_threshold: float = 0.10  # 10% drift
    rebalance_interval_hours: int = 24


@dataclass
class StrategyConfig:
    """Strategy mining configuration"""
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_count: int = 2
    
    # Indicators
    INDICATORS = ["RSI", "SMA", "EMA", "VOLSMA"]
    PERIODS = [10, 14, 20, 50, 100, 200]
    
    # Operators
    OPERATORS = [">", "<", "crosses_above", "crosses_below"]


@dataclass
class CoordinatorConfig:
    """Coordinator server configuration"""
    host: str = "0.0.0.0"
    port: int = 5001
    database_path: str = "config/coordinator.db"
    max_workers: int = 8
    work_unit_timeout_minutes: int = 30


@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration"""
    host: str = "0.0.0.0"
    port: int = 8501
    debug_mode: bool = False
    refresh_interval_seconds: int = 5


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    commands: list = field(default_factory=lambda: [
        "/status", "/workers", "/progress", "/help"
    ])


@dataclass
class MiniMaxConfig:
    """MiniMax M2.5 configuration"""
    api_url: str = os.environ.get("MINIMAX_API_URL", "http://localhost:8090/v1")
    api_key: str = os.environ.get("MINIMAX_API_KEY", "")
    model: str = "MiniMax-M2.5"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "config/trading_bot.log"
    max_size_mb: float = 10.0
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_key_path: str = "~/.config/solana-jupiter-bot/encryption.key"
    audit_enabled: bool = True
    sandbox_enabled: bool = False
    max_failed_attempts: int = 3
    lockout_duration_minutes: int = 15


class Config:
    """
    Main configuration class that combines all config sections
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.network = NetworkConfig()
        self.wallet = WalletConfig()
        self.jupiter = JupiterConfig()
        self.trading = TradingConfig()
        self.strategy = StrategyConfig()
        self.coordinator = CoordinatorConfig()
        self.dashboard = DashboardConfig()
        self.telegram = TelegramConfig()
        self.minimax = MiniMaxConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        path = Path(config_path)
        if not path.exists():
            print(f"⚠️ Config file not found: {config_path}")
            return
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Apply overrides (simplified)
        print(f"✅ Configuration loaded from: {config_path}")
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        data = {
            "network": {"network": self.network.network},
            "wallet": {
                "max_trade_pct": self.wallet.max_trade_pct,
                "max_daily_pct": self.wallet.max_daily_pct
            },
            "trading": {
                "risk_level": self.trading.risk_level,
                "max_position_pct": self.trading.max_position_pct
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_path}")
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check wallet
        if not self.wallet.hot_wallet_address:
            errors.append("HOT_WALLET_ADDRESS not configured")
        
        # Check RPC
        if not self.network.rpc_url:
            errors.append("No RPC URL configured")
        
        # Check Telegram (optional but warn)
        if not self.telegram.enabled:
            errors.append("⚠️ Telegram not configured (notifications disabled)")
        
        # Check MiniMax (optional but warn)
        if not self.minimax.api_key:
            errors.append("⚠️ MiniMax API key not configured (AI disabled)")
        
        return len(errors) == 0, errors
    
    def get_trading_pair(self, pair: str) -> Dict[str, Any]:
        """Get trading pair configuration"""
        return self.jupiter.TRADING_PAIRS.get(pair, {})
    
    def is_major_pair(self, pair: str) -> bool:
        """Check if pair is a major pair"""
        return pair in ["SOL-USDC", "SOL-USDT"]
    
    def get_slippage_max(self, pair: str) -> float:
        """Get max slippage for pair"""
        if self.is_major_pair(pair):
            return 0.01  # 1%
        return 0.02  # 2%


# Singleton config instance
_config: Optional[Config] = None

def get_config(config_file: Optional[str] = None) -> Config:
    """Get or create config instance"""
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


# Environment file template
ENV_TEMPLATE = """# Solana Jupiter Trading Bot - Environment Configuration
# Copy this file to .env and fill in your values

# Network
SOLANA_RPC_DEVNET=https://api.devnet.solana.com
SOLANA_RPC_MAINNET=https://api.mainnet-beta.solana.com
SOLANA_RPC_TESTNET=https://api.testnet.solana.com

# Wallet (use ONLY addresses, not private keys!)
HOT_WALLET_ADDRESS=
COLD_WALLET_ADDRESS=

# Jupiter API (optional - uses public API by default)
JUPITER_API_KEY=

# MiniMax M2.1
MINIMAX_API_URL=http://localhost:8090/v1
MINIMAX_API_KEY=sk-your-api-key

# Telegram Bot
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Logging
LOG_LEVEL=INFO
"""


def create_env_template():
    """Create .env.example file"""
    env_file = PROJECT_ROOT / ".env.example"
    with open(env_file, "w") as f:
        f.write(ENV_TEMPLATE)
    print(f"✅ Created .env.example template")


def create_sample_config():
    """Create sample config.json"""
    config = get_config()
    config.save_to_file("config/config.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration management")
    parser.add_argument("--validate", action="store_true", help="Validate current configuration")
    parser.add_argument("--create-env", action="store_true", help="Create .env.example template")
    parser.add_argument("--create-config", action="store_true", help="Create sample config.json")
    
    args = parser.parse_args()
    
    if args.create_env:
        create_env_template()
    elif args.create_config:
        create_sample_config()
    elif args.validate:
        config = get_config()
        is_valid, errors = config.validate()
        if is_valid:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"  - {error}")
    else:
        print("📋 Solana Jupiter Bot Configuration")
        print("=" * 50)
        config = get_config()
        print(f"Network: {config.network.network}")
        print(f"RPC URL: {config.network.rpc_url}")
        print(f"Risk Level: {config.trading.risk_level}")
        print(f"Max Trade: {config.wallet.max_trade_pct * 100}%")
        print(f"Telegram: {'✅' if config.telegram.enabled else '❌'}")
