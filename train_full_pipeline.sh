#!/bin/bash
set -e

echo "================================================================================"
echo "FULL TRAINING PIPELINE: LLM Labeling → Model Training → Validation"
echo "================================================================================"

# Phase 1: Generate training data
echo ""
echo "📰 Phase 1: Generating 500+ labeled articles with LLM..."
echo "================================================================================"
python ml/utils/label_dataset.py \
  --articles 500 \
  --country sweden \
  --max-llm-calls 50 \
  --llm-max-categories 3 \
  --llm-batch-size 15

# Phase 2: Train model
echo ""
echo "🤖 Phase 2: Training fine-tuned BERT model..."
echo "================================================================================"
python ml/models/quick_finetune.py \
  --train data/train.parquet \
  --val data/val.parquet \
  --epochs 10

# Phase 3: Validate
echo ""
echo "✅ Phase 3: Running smoke tests..."
echo "================================================================================"
python tests/model_inference_smoke.py

echo ""
echo "================================================================================"
echo "✓ TRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review test results above"
echo "  2. Check history: cat ml/models/checkpoints/history.json | python -m json.tool"
echo "  3. Deploy to backend: cp ml/models/checkpoints/best_model.pt backend/models/"
echo ""
