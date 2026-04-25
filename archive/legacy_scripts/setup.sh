#!/bin/bash
#
# Jupiter Solana Trading Bot - Quick Setup Script
# ===============================================
# This script sets up the project for development.
#

set -e  # Exit on error

echo "========================================"
echo "🚀 JUPITER SOLANA TRADING BOT SETUP"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        exit 1
    fi
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# 1. Create virtual environment
echo "1️⃣  Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status $? "Virtual environment created"
else
    echo "   ℹ️  venv already exists"
fi

# 2. Activate virtual environment
echo ""
echo "2️⃣  Activating virtual environment..."
source venv/bin/activate
print_status $? "Virtual environment activated"

# 3. Upgrade pip
echo ""
echo "3️⃣  Upgrading pip..."
pip install --upgrade pip --quiet
print_status $? "pip upgraded"

# 4. Install dependencies
echo ""
echo "4️⃣  Installing dependencies..."
pip install -r requirements.txt --quiet 2>/dev/null || {
    echo "   ⚠️  Some packages may have failed. Installing core packages..."
    pip install httpx solders cryptography numpy pandas pydantic streamlit --quiet
}
print_status $? "Dependencies installed"

# 5. Create __init__.py files
echo ""
echo "5️⃣  Creating Python package files..."
for dir in tools config backtesting workers dashboard skills; do
    if [ ! -f "$dir/__init__.py" ]; then
        touch "$dir/__init__.py"
        echo "   ✅ $dir/__init__.py"
    fi
done

# 6. Create .env file
echo ""
echo "6️⃣  Creating .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_status $? ".env created from template"
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env with your settings!${NC}"
else
    echo "   ℹ️  .env already exists"
fi

# 7. Create config directory
echo ""
echo "7️⃣  Creating config directory..."
mkdir -p config
if [ ! -f "config/__init__.py" ]; then
    touch "config/__init__.py"
fi
print_status $? "Config directory ready"

# 8. Create data directory
echo ""
echo "8️⃣  Creating data directory..."
mkdir -p data
print_status $? "Data directory ready"

# Summary
echo ""
echo "========================================"
echo "🎉 SETUP COMPLETE!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env with your configuration:"
echo "   nano .env"
echo ""
echo "2. Test the installation:"
echo "   source venv/bin/activate"
echo "   python test_system.py"
echo ""
echo "3. Run the dashboard:"
echo "   source venv/bin/activate"
echo "   cd dashboard"
echo "   streamlit run solana_dashboard.py"
echo ""
echo "========================================"
