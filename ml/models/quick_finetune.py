"""
Quick fine-tuning script for one-day MVP.
Streamlined, minimal configuration, production-ready.
Supports stratified k-fold cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import polars as pl
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES, TAG_VOCAB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Model config
BASE_MODEL = "KB/bert-base-swedish-cased"
MAX_LENGTH = 512
BATCH_SIZE = 32 if torch.cuda.is_available() else 8
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
SCORE_LOSS_WEIGHT = 0.4
TAG_LOSS_WEIGHT = 0.6
DROPOUT_RATE = 0.2


class QuickNewsDataset(Dataset):
    """Minimal dataset implementation."""

    def __init__(self, parquet_path: str, tokenizer, max_length: int = 512):
        self.df = pl.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        # Combine title + description
        text = f"{row['title']} [SEP] {row.get('description', '')}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract labels
        labels = {}
        for category in SIGNAL_CATEGORIES:
            score = float(row.get(f"{category}_score", 0.0))
            tag = row.get(f"{category}_tag", "")
            tag_idx = TAG_VOCAB[category].index(tag) if tag in TAG_VOCAB[category] else 0

            labels[category] = {
                'score': torch.tensor(score, dtype=torch.float32),
                'tag_idx': torch.tensor(tag_idx, dtype=torch.long),
            }

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
        }


class QuickNewsClassifier(nn.Module):
    """Minimal multi-head classifier."""

    def __init__(self, base_model: str = BASE_MODEL):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model)
        hidden_size = 768

        # Score heads (regression)
        self.score_heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(128, 1),
                nn.Tanh()
            )
            for cat in SIGNAL_CATEGORIES
        })

        # Tag heads (classification)
        self.tag_heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(128, len(TAG_VOCAB[cat]))
            )
            for cat in SIGNAL_CATEGORIES
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output

        results = {}
        for cat in SIGNAL_CATEGORIES:
            score = self.score_heads[cat](pooled).squeeze(-1)
            tag_logits = self.tag_heads[cat](pooled)
            results[cat] = (score, tag_logits)

        return results


def train_epoch(model, loader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {
            cat: {
                'score': batch['labels'][cat]['score'].to(device),
                'tag_idx': batch['labels'][cat]['tag_idx'].to(device),
            }
            for cat in SIGNAL_CATEGORIES
        }

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)

        # Compute loss with weighted multi-task learning
        loss = 0.0
        for cat in SIGNAL_CATEGORIES:
            pred_score, pred_tag = predictions[cat]
            true_score = labels[cat]['score']
            true_tag = labels[cat]['tag_idx']

            loss += SCORE_LOSS_WEIGHT * mse_loss(pred_score, true_score)
            loss += TAG_LOSS_WEIGHT * ce_loss(pred_tag, true_tag)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(loader)


def validate(model, loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {
                cat: {
                    'score': batch['labels'][cat]['score'].to(device),
                    'tag_idx': batch['labels'][cat]['tag_idx'].to(device),
                }
                for cat in SIGNAL_CATEGORIES
            }

            predictions = model(input_ids, attention_mask)

            loss = 0.0
            for cat in SIGNAL_CATEGORIES:
                pred_score, pred_tag = predictions[cat]
                true_score = labels[cat]['score']
                true_tag = labels[cat]['tag_idx']

                loss += SCORE_LOSS_WEIGHT * mse_loss(pred_score, true_score)
                loss += TAG_LOSS_WEIGHT * ce_loss(pred_tag, true_tag)

            total_loss += loss.item()

    return total_loss / len(loader)


def train_quick(train_path: str, val_path: str, output_dir: str = "ml/models/checkpoints"):
    """Quick training loop."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    train_dataset = QuickNewsDataset(train_path, tokenizer)
    val_dataset = QuickNewsDataset(val_path, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    logger.info("Loading model...")
    model = QuickNewsClassifier(BASE_MODEL)
    model.to(DEVICE)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Training
    best_val_loss = float('inf')
    history = []
    patience = 3
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate(model, val_loader, DEVICE)

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
        })

        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = output_path / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint)
            logger.info(f"✓ Saved best model: {checkpoint}")
        else:
            patience_counter += 1
            logger.warning(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Step scheduler
        scheduler.step()

    # Save history
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f)

    logger.info(f"\n✓ Training complete! Best val_loss: {best_val_loss:.4f}")
    logger.info(f"✓ Model saved to: {output_path / 'best_model.pt'}")

    return model


def train_with_stratified_kfold(data_path: str, k: int = 5, output_dir: str = "ml/models/checkpoints"):
    """
    Train model using stratified k-fold cross-validation.
    
    Args:
        data_path: Path to labeled dataset parquet file
        k: Number of folds (default: 5)
        output_dir: Output directory for checkpoints and results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data for stratified {k}-fold cross-validation...")
    df = pl.read_parquet(data_path)
    
    # Create stratification key based on primary signal
    # Data has flattened columns: emergencies_score, emergencies_tag, etc.
    def get_primary_signal_from_row(row) -> str:
        """Get primary signal category from flattened columns."""
        max_score = 0.0
        max_cat = "none"
        
        for cat in SIGNAL_CATEGORIES:
            score_col = f"{cat}_score"
            if score_col in row:
                score = abs(float(row[score_col]))
                if score > 0.01 and score > max_score:
                    max_score = score
                    max_cat = cat
        
        return max_cat
    
    # Create stratification labels
    strat_labels = []
    for row in df.iter_rows(named=True):
        strat_labels.append(get_primary_signal_from_row(row))
    
    # Convert to numpy for sklearn
    import numpy as np
    indices = np.arange(len(df))
    y = np.array(strat_labels)
    
    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    fold_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, y)):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{k}")
        logger.info(f"{'='*60}")
        logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
        
        # Create fold-specific datasets
        train_df = df[train_idx]
        val_df = df[val_idx]
        
        # Save fold data temporarily
        fold_train_path = output_path / f"fold_{fold_idx}_train.parquet"
        fold_val_path = output_path / f"fold_{fold_idx}_val.parquet"
        
        train_df.write_parquet(fold_train_path)
        val_df.write_parquet(fold_val_path)
        
        # Train on this fold
        logger.info(f"Training fold {fold_idx + 1}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        train_dataset = QuickNewsDataset(str(fold_train_path), tokenizer)
        val_dataset = QuickNewsDataset(str(fold_val_path), tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Create and train model
        model = QuickNewsClassifier(BASE_MODEL)
        model.to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        
        best_val_loss = float('inf')
        fold_history = []
        patience_counter = 0
        patience = 3
        
        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
            val_loss = validate(model, val_loader, DEVICE)
            
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
            })
            
            logger.info(f"  Epoch {epoch + 1}/{NUM_EPOCHS} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            # Save best model for this fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                fold_checkpoint = output_path / f"fold_{fold_idx}_best_model.pt"
                torch.save({
                    'fold': fold_idx + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                }, fold_checkpoint)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break
            
            scheduler.step()
        
        # Record fold results
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_loss': float(best_val_loss),
            'history': fold_history,
            'num_epochs': epoch + 1,
        })
        fold_models.append(model)
        
        logger.info(f"Fold {fold_idx + 1} complete - Best Val Loss: {best_val_loss:.4f}")
        
        # Cleanup fold data files
        fold_train_path.unlink()
        fold_val_path.unlink()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"{k}-FOLD CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    val_losses = [r['best_val_loss'] for r in fold_results]
    logger.info(f"Fold Results (Val Loss):")
    for fold_result in fold_results:
        logger.info(f"  Fold {fold_result['fold']}: {fold_result['best_val_loss']:.4f}")
    
    logger.info(f"\nMean Val Loss: {sum(val_losses) / len(val_losses):.4f}")
    logger.info(f"Std Dev:      {(sum((x - sum(val_losses)/len(val_losses))**2 for x in val_losses) / len(val_losses))**0.5:.4f}")
    logger.info(f"Best Fold:    Fold {fold_results[val_losses.index(min(val_losses))]['fold']} ({min(val_losses):.4f})")
    
    # Save cross-validation results
    cv_results = {
        'k': k,
        'fold_results': fold_results,
        'mean_val_loss': float(sum(val_losses) / len(val_losses)),
        'best_fold': fold_results[val_losses.index(min(val_losses))]['fold'],
    }
    
    with open(output_path / "kfold_results.json", 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    logger.info(f"\n✓ K-fold results saved to: {output_path / 'kfold_results.json'}")
    
    return fold_models, fold_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Training data path (for standard train/val split)")
    parser.add_argument("--val", help="Validation data path (for standard train/val split)")
    parser.add_argument("--data", help="Data path for stratified k-fold cross-validation")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for k-fold CV (default: 5)")
    parser.add_argument("--output", default="ml/models/checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")

    args = parser.parse_args()
    NUM_EPOCHS = args.epochs
    
    if args.data:
        # Stratified k-fold cross-validation
        logger.info(f"Running stratified {args.k}-fold cross-validation on {args.data}")
        train_with_stratified_kfold(args.data, k=args.k, output_dir=args.output)
    elif args.train and args.val:
        # Standard train/val split
        logger.info(f"Running standard training with fixed train/val split")
        train_quick(args.train, args.val, args.output)
    else:
        parser.error("Either provide --data (for k-fold) or both --train and --val (for standard split)")
