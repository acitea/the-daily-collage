#!/bin/bash
# Install dependencies for The Daily Collage fine-tuning

echo "📦 Installing fine-tuning dependencies..."
echo "   This may take 5-10 minutes on first run..."
echo ""

# Upgrade pip first
echo "→ Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install all dependencies
echo "→ Installing PyTorch..."
python3 -m pip install torch

echo "→ Installing Transformers & ML dependencies..."
python3 -m pip install transformers datasets tqdm polars pydantic gdeltdoc

echo "→ Verifying installation..."
python3 -c "
import sys
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'polars': 'Polars',
    'gdeltdoc': 'GDELT',
}

missing = []
for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f'  ✅ {name}')
    except ImportError:
        print(f'  ❌ {name}')
        missing.append(pkg)

if missing:
    print(f'\n❌ Failed to install: {missing}')
    sys.exit(1)
else:
    print('\n✅ All dependencies installed!')
"

if [ $? -eq 0 ]; then
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║           ✅ Dependencies Installed Successfully!           ║"
  echo "╠════════════════════════════════════════════════════════════╣"
  echo "║  You can now run:                                          ║"
  echo "║                                                            ║"
  echo "║    ./train_one_day.sh                                      ║"
  echo "║                                                            ║"
  echo "╚════════════════════════════════════════════════════════════╝"
  echo ""
else
  echo ""
  echo "❌ Installation failed. Please check the errors above."
  exit 1
fi
