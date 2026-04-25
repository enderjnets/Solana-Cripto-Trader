#!/bin/bash
# Switch between devnet and mainnet

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║         SOLANA JUPITER BOT - NETWORK SWITCH        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

NETWORK=$1

if [ -z "$NETWORK" ]; then
    echo "Usage: ./switch_network.sh [devnet|mainnet|testnet]"
    echo ""
    echo "Current network:"
    grep "^NETWORK=" .env | cut -d= -f2
    exit 1
fi

case $NETWORK in
    devnet|mainnet|testnet)
        echo "Switching to: $NETWORK"
        ;;
    *)
        echo "Error: Invalid network '$NETWORK'"
        echo "Use: devnet, mainnet, or testnet"
        exit 1
        ;;
esac

# Backup current .env
cp .env .env.backup
echo "✅ Backup created: .env.backup"

# Update .env
sed -i "s/^NETWORK=.*/NETWORK=$NETWORK/" .env

# Update RPC URL
case $NETWORK in
    devnet)
        sed -i 's|^SOLANA_RPC_URL=.*|SOLANA_RPC_URL=https://api.devnet.solana.com|' .env
        sed -i 's|^HOT_WALLET_ADDRESS=.*|HOT_WALLET_ADDRESS=65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3|' .env
        echo "🔵 DEVNET ACTIVATED"
        echo "   Wallet: 65YqSYGwR6UNCUmeaKt1V1HV99Ky1tii2bgg6jwJSGN3"
        echo "   Balance: 5.0 SOL (test)"
        ;;
    mainnet)
        sed -i 's|^SOLANA_RPC_URL=.*|SOLANA_RPC_URL=https://api.mainnet-beta.solana.com|' .env
        echo "🟢 MAINNET ACTIVATED"
        echo "   Wallet: Ht3J5crwQoMgJ77K2y2V7BPo6F4Ld6pRyMBCCCKGgSTw"
        echo "   Balance: TU ELIGES"
        echo ""
        echo "⚠️  WARNING: Using real funds!"
        ;;
    testnet)
        sed -i 's|^SOLANA_RPC_URL=.*|SOLANA_RPC_URL=https://api.testnet.solana.com|' .env
        echo "🟡 TESTNET ACTIVATED"
        ;;
esac

echo ""
echo "✅ Network switched to: $NETWORK"
echo ""
echo "To verify:"
echo "   grep '^NETWORK=' .env"
echo ""
echo "Restart dashboard:"
echo "   streamlit run dashboard/solana_dashboard.py"
