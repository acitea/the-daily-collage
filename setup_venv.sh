#!/bin/bash
# Setup virtual environment and install dependencies for fine-tuning

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

echo "
╔════════════════════════════════════════════════════════════════╗
║     THE DAILY COLLAGE - DEPENDENCY SETUP                       ║
║     Creating Python virtual environment...                     ║
╚════════════════════════════════════════════════════════════════╝
"

# Step 1: Create virtual environment
echo "📦 Step 1: Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "   → Virtual environment already exists"
else
    python3 -m venv .venv
    if [ $? -eq 0 ]; then
        echo "   ✅ Virtual environment created"
    else
        echo "   ❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Step 2: Activate virtual environment
echo "📦 Step 2: Activating virtual environment..."
source .venv/bin/activate
if [ $? -eq 0 ]; then
    echo "   ✅ Virtual environment activated"
else
    echo "   ❌ Failed to activate virtual environment"
    exit 1
fi

# Step 3: Upgrade pip
echo "📦 Step 3: Upgrading pip..."
python3 -m pip install --upgrade pip > /dev/null 2>&1

# Step 4: Install dependencies
echo "📦 Step 4: Installing dependencies..."
echo "   This may take 10-15 minutes on first run..."
python3 -m pip install torch transformers datasets tqdm polars pydantic gdeltdoc pillow requests > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "   ✅ Dependencies installed"
else
    echo "   ❌ Failed to install dependencies"
    exit 1
fi

# Step 5: Verify installation
echo "📦 Step 5: Verifying installation..."
python3 -c "
import sys
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'polars': 'Polars',
    'gdeltdoc': 'GDELT',
    'PIL': 'Pillow',
    'requests': 'Requests',
}

all_ok = True
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f'  ✅ {name}')
    except ImportError as e:
        print(f'  ❌ {name}: {e}')
        all_ok = False

if all_ok:
    print('\n✅ All dependencies verified!')
else:
    print('\n❌ Some dependencies failed')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo "
╔════════════════════════════════════════════════════════════════╗
║              ✅ SETUP COMPLETE!                               ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Virtual environment activated with all dependencies!          ║
║                                                                ║
║  To start training:                                            ║
║                                                                ║
║    source .venv/bin/activate    (if not already)               ║
║    ./train_one_day.sh                                          ║
║                                                                ║
║  Or in one step:                                               ║
║                                                                ║
║    source .venv/bin/activate && ./train_one_day.sh             ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"
