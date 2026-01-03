#!/bin/bash

# One-day fine-tuning sprint for The Daily Collage
# Run this to train a complete model in one day

set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

echo "
╔════════════════════════════════════════════════════════════════╗
║     THE DAILY COLLAGE - ONE-DAY FINE-TUNING SPRINT             ║
╚════════════════════════════════════════════════════════════════╝
"

# Step 1: Environment
echo "📦 Step 1: Setting up environment..."
echo "   Installing dependencies (torch, transformers, polars, etc)..."
python3 -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
python3 -m pip install torch transformers datasets tqdm polars pydantic gdeltdoc > /dev/null 2>&1

if ! python3 -c "import polars, transformers, torch" 2>/dev/null; then
  echo "❌ Failed to install dependencies. Please run manually:"
  echo "   python3 -m pip install torch transformers polars tqdm gdeltdoc"
  exit 1
fi
echo "   ✅ All dependencies installed"

# Step 2: Bootstrap data
echo ""
echo "📰 Step 2: Bootstrapping training data (500 articles via batched GDELT)..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from ml.data.quick_bootstrap import quick_bootstrap
print('  Fetching 6250 articles from Sweden (25 x 250-article batches with delays)...')
quick_bootstrap(
    countries=['sweden'],
    articles_per_country=6250,
    use_batching=True,
    batch_size=250,
    days_lookback=180
)
" || {
  echo "❌ Data bootstrap failed"
  exit 1
}
echo "   ✅ 6250 articles collected and classified"
echo "
📰 Step 2: Collecting and labeling ~6250 articles from Sweden (~2 min)..."
python3 ml/data/quick_bootstrap.py \
    --articles-per-country 6250 \
    --output-dir ml/data

# Step 3: Train
echo "
🚀 Step 3: Training fine-tuned BERT model (8 epochs, ~5-20 min on GPU/CPU)..."
python3 ml/models/quick_finetune.py \
    --train ml/data/train_bootstrap.parquet \
    --val ml/data/val_bootstrap.parquet \
    --output ml/models/checkpoints \
    --epochs 8

# Step 4: Verify model
echo "
✅ Step 4: Verifying model..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from ml.models.inference import get_fine_tuned_classifier

model = get_fine_tuned_classifier('ml/models/checkpoints/best_model.pt')
print('✓ Model loaded successfully!')

# Test inference
result = model.classify('Fire breaks out in Stockholm', 'A major fire has broken out in central Stockholm')
print('\n🧪 Test classification:')
for cat, (score, tag) in result.items():
    print(f'  {cat:20s}: score={score:+.2f}, tag={tag}')
"

echo "
╔════════════════════════════════════════════════════════════════╗
║                    ✨ TRAINING COMPLETE! ✨                    ║
╠════════════════════════════════════════════════════════════════╣
║ Model saved to: ml/models/checkpoints/best_model.pt           ║
║ Next: Update hopsworks_pipeline.py to use the fine-tuned model║
║ See: ml/models/inference.py for integration instructions       ║
╚════════════════════════════════════════════════════════════════╝
"
