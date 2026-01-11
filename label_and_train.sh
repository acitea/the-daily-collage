#!/bin/bash

# Quick-start script for labeling GDELT data and training BERT model
# This demonstrates the recommended workflow using embedding-based labeling

set -e  # Exit on error

echo "=========================================="
echo "GDELT DATA LABELING & MODEL TRAINING"
echo "=========================================="
echo ""

# Configuration
ARTICLES=500
COUNTRY="sweden"
OUTPUT_DIR="data"

echo "Configuration:"
echo "  Articles:  $ARTICLES"
echo "  Country:   $COUNTRY"
echo "  Output:    $OUTPUT_DIR"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "   Run: source .venv/bin/activate"
    exit 1
fi

# Step 1: Label dataset using embeddings
echo "=========================================="
echo "STEP 1: Fetch & Auto-Label Articles"
echo "=========================================="
echo ""
echo "Using embedding-based classification..."
echo ""

python ml/utils/label_dataset.py \
    --articles $ARTICLES \
    --country $COUNTRY \
    --method embedding \
    --threshold 0.35 \
    --output $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Labeling failed!"
    exit 1
fi

echo ""
echo "✓ Step 1 complete!"
echo ""

# Step 2: Optional manual review
echo "=========================================="
echo "STEP 2: Review Labels (OPTIONAL)"
echo "=========================================="
echo ""
echo "Would you like to review and correct uncertain labels?"
echo "This is recommended for production models."
echo ""
read -p "Review labels? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting interactive review..."
    echo ""
    
    python ml/utils/review_labels.py \
        --input $OUTPUT_DIR/train.parquet \
        --threshold 0.5
    
    # Update train file path if reviewed
    TRAIN_FILE="$OUTPUT_DIR/train_reviewed.parquet"
    
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "⚠️  No corrections saved, using original labels"
        TRAIN_FILE="$OUTPUT_DIR/train.parquet"
    fi
else
    echo "Skipping review, using auto-generated labels..."
    TRAIN_FILE="$OUTPUT_DIR/train.parquet"
fi

echo ""
echo "✓ Step 2 complete!"
echo ""

# Step 3: Train model
echo "=========================================="
echo "STEP 3: Train BERT Model"
echo "=========================================="
echo ""
echo "Training with:"
echo "  Train: $TRAIN_FILE"
echo "  Val:   $OUTPUT_DIR/val.parquet"
echo ""

python ml/models/quick_finetune.py \
    --train $TRAIN_FILE \
    --val $OUTPUT_DIR/val.parquet \
    --epochs 3 \
    --output ml/models/checkpoints

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✓ Step 3 complete!"
echo ""

# Step 4: Test inference
echo "=========================================="
echo "STEP 4: Test Model Inference"
echo "=========================================="
echo ""

python ml/models/inference.py \
    --text "Brand utbryter i centrum av Stockholm" \
    --model ml/models/checkpoints/best_model.pt

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Inference test failed (model may not be ready yet)"
else
    echo ""
    echo "✓ Step 4 complete!"
fi

echo ""
echo "=========================================="
echo "✅ ALL DONE!"
echo "=========================================="
echo ""
echo "Your model is trained and ready!"
echo ""
echo "Next steps:"
echo "  • Test inference: python ml/models/inference.py --text 'your text'"
echo "  • Use in pipeline: The model will auto-load in hopsworks_pipeline.py"
echo "  • Generate vibes: python backend/utils/generate_layout.py"
echo ""
