"""
Quick fine-tuning script for one-day MVP.
Streamlined, minimal configuration, production-ready.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import polars as pl
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import json
from datetime import datetime

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
BATCH_SIZE = 32 if torch.cuda.is_available() else 12
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500


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
                nn.Dropout(0.1),
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
                nn.Dropout(0.1),
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

        # Compute loss
        loss = 0.0
        for cat in SIGNAL_CATEGORIES:
            pred_score, pred_tag = predictions[cat]
            true_score = labels[cat]['score']
            true_tag = labels[cat]['tag_idx']

            loss += 0.5 * mse_loss(pred_score, true_score)
            loss += 0.5 * ce_loss(pred_tag, true_tag)

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

                loss += 0.5 * mse_loss(pred_score, true_score)
                loss += 0.5 * ce_loss(pred_tag, true_tag)

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
    
    # Training
    best_val_loss = float('inf')
    history = []

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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = output_path / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint)
            logger.info(f"✓ Saved best model: {checkpoint}")

    # Save history
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f)

    logger.info(f"\n✓ Training complete! Best val_loss: {best_val_loss:.4f}")
    logger.info(f"✓ Model saved to: {output_path / 'best_model.pt'}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training data path")
    parser.add_argument("--val", required=True, help="Validation data path")
    parser.add_argument("--output", default="ml/models/checkpoints")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)

    args = parser.parse_args()
    NUM_EPOCHS = args.epochs

    train_quick(args.train, args.val, args.output)
