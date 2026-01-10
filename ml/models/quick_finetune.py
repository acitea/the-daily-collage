"""
Quick fine-tuning script for one-day MVP.
Streamlined, minimal configuration, production-ready.
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
from typing import Optional, Tuple

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
NUM_EPOCHS = 8
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


def _filter_labeled_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only rows that have at least one non-empty tag or non-zero score."""
    label_masks = []
    for cat in SIGNAL_CATEGORIES:
        label_masks.append(pl.col(f"{cat}_tag").cast(pl.Utf8).fill_null("") != "")
        label_masks.append(pl.col(f"{cat}_score").cast(pl.Float64).fill_null(0.0) != 0.0)

    combined_mask = pl.any_horizontal(label_masks)
    return df.filter(combined_mask)


def load_hopsworks_training_df(
    api_key: str,
    project: str,
    host: Optional[str],
    fg_name: str,
    fg_version: int,
    city: Optional[str] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """Load labeled headlines from Hopsworks feature group and return a polars DataFrame."""
    import hopsworks
    
    logger.info(f"Connecting to Hopsworks project: {project}")
    
    # Direct Hopsworks login
    login_kwargs = {
        "api_key_value": api_key,
        "project": project,
        "engine": "python"
    }
    if host:
        login_kwargs["host"] = host
    
    hops_project = hopsworks.login(**login_kwargs)
    fs = hops_project.get_feature_store()
    logger.info(f"Connected to feature store: {fs.name}")

    # Get feature group and create/use a feature view to stabilize column projection
    logger.info(f"Fetching feature group: {fg_name} v{fg_version}")
    try:
        fg = fs.get_feature_group(name=fg_name, version=fg_version)
        logger.info("Feature group fetched successfully")

        # Columns expected from label_dataset.py
        base_cols = ["title", "description", "tone", "url", "source", "date"]
        score_cols = [f"{c}_score" for c in SIGNAL_CATEGORIES]
        tag_cols = [f"{c}_tag" for c in SIGNAL_CATEGORIES]
        all_cols = base_cols + score_cols + tag_cols

        fv_name = f"{fg_name}_view"
        logger.info(f"Ensuring feature view: {fv_name} v{fg_version}")
        fv = fs.get_or_create_feature_view(
            name=fv_name,
            version=fg_version,
            description=f"View over {fg_name} for training",
            query=fg.select(all_cols),
            labels=[],
        )

        # Read via feature view to avoid binder errors
        logger.info("Reading feature view with use_hive=False …")
        df_pd = fv.get_batch_data(read_options={"use_hive": False})
        logger.info(f"Feature view read returned {len(df_pd) if df_pd is not None else 0} rows")

        # Fallback: direct FG read if feature view is empty
        if df_pd is None or (hasattr(df_pd, '__len__') and len(df_pd) == 0):
            logger.warning("Feature view returned no rows; falling back to feature group read")
            df_pd = fg.read(read_options={"use_hive": False})
            if df_pd is None or (hasattr(df_pd, '__len__') and len(df_pd) == 0):
                logger.warning("FG read empty; retrying select_all().read(use_hive=False)")
                df_pd = fg.select_all().read(read_options={"use_hive": False})

    except Exception as e:
        logger.error(f"Error reading feature view/group: {e}")
        raise ValueError(f"Failed to read feature group/view '{fg_name}': {str(e)}")

    if df_pd is None or df_pd.empty:
        raise ValueError("No labeled rows returned from Hopsworks feature group or feature view")

    # Convert to polars
    df = pl.from_pandas(df_pd)
    
    # Log available columns for debugging
    logger.info(f"Available columns: {df.columns}")
    
    # Handle column name prefixes (Hopsworks sometimes adds prefixes)
    # e.g., "read.parquet_headline_labels_title" -> "title"
    column_mapping = {}
    for col in df.columns:
        clean_name = col
        
        # Remove feature group prefix and parquet prefix
        # Pattern: "read.parquet_<fg_name>_<actual_col>" or similar
        if "." in col:
            # Take last part after dot
            clean_name = col.split(".")[-1]
        
        # Remove "parquet_" prefix if present
        if clean_name.startswith("parquet_"):
            clean_name = clean_name.replace("parquet_", "", 1)
        
        # Remove feature group name prefix if present
        if fg_name and clean_name.startswith(fg_name + "_"):
            clean_name = clean_name.replace(fg_name + "_", "", 1)
        
        if clean_name != col:
            column_mapping[col] = clean_name
            logger.info(f"  Mapping: {col} -> {clean_name}")
    
    if column_mapping:
        logger.info(f"Renaming {len(column_mapping)} columns")
        df = df.rename(column_mapping)
    
    # Ensure all expected columns are present and handle nulls
    expected_cols = ["title", "description", "tone", "url", "source"]
    for col_name in expected_cols:
        if col_name not in df.columns:
            logger.warning(f"Column '{col_name}' not found, adding empty column")
            if col_name == "tone":
                df = df.with_columns(pl.lit(0.0).alias(col_name))
            else:
                df = df.with_columns(pl.lit("").alias(col_name))
    
    df = df.with_columns([
        pl.col("title").fill_null("").alias("title"),
        pl.col("description").fill_null("").alias("description"),
        pl.col("tone").fill_null(0.0).alias("tone"),
        pl.col("url").fill_null("").alias("url"),
        pl.col("source").fill_null("").alias("source"),
    ])
    
    # Fill null scores and tags for all categories
    for category in SIGNAL_CATEGORIES:
        score_col = f"{category}_score"
        tag_col = f"{category}_tag"
        
        if score_col not in df.columns:
            logger.warning(f"Column '{score_col}' not found, adding zero column")
            df = df.with_columns(pl.lit(0.0).alias(score_col))
        else:
            df = df.with_columns(pl.col(score_col).fill_null(0.0))
            
        if tag_col not in df.columns:
            logger.warning(f"Column '{tag_col}' not found, adding empty column")
            df = df.with_columns(pl.lit("").alias(tag_col))
        else:
            df = df.with_columns(pl.col(tag_col).fill_null(""))
    
    df = _filter_labeled_rows(df)

    if df.is_empty():
        raise ValueError("All rows from Hopsworks were unlabeled; nothing to train on")

    return df


def materialize_train_val(
    df: pl.DataFrame,
    output_dir: str,
    val_ratio: float = 0.2,
    prefix: str = "hopsworks",
) -> Tuple[str, str]:
    """Shuffle, split, and persist train/val parquet files from a polars DataFrame."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shuffled = df.sample(fraction=1.0, shuffle=True, with_replacement=False)
    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 0 else 0

    val_df = shuffled.head(val_size) if val_size > 0 else pl.DataFrame()
    train_df = shuffled.tail(len(shuffled) - val_size) if val_size > 0 else shuffled

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Not enough rows to create a non-empty train/val split")

    train_path = output_path / f"{prefix}_train.parquet"
    val_path = output_path / f"{prefix}_val.parquet"

    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)

    logger.info(f"Materialized train/val to {train_path} and {val_path}")
    return str(train_path), str(val_path)


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


def train_quick(train_path: str, val_path: str, output_dir: str = "ml/models/checkpoints") -> Tuple[QuickNewsClassifier, float, Path]:
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
    best_checkpoint = output_path / "best_model.pt"
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
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, best_checkpoint)
            logger.info(f"✓ Saved best model: {best_checkpoint}")
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
    logger.info(f"✓ Model saved to: {best_checkpoint}")

    return model, best_val_loss, best_checkpoint


def train_with_stratified_kfold(*args, **kwargs):
    """Deprecated: stratified k-fold removed. Use standard train/val split."""
    raise NotImplementedError("Stratified k-fold has been removed. Use --train and --val.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=False, help="Training data path (parquet)")
    parser.add_argument("--val", required=False, help="Validation data path (parquet)")
    parser.add_argument("--output", default="ml/models/checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--from-hopsworks", action="store_true", help="Load train/val from Hopsworks feature group")
    parser.add_argument("--hopsworks-api-key", type=str, default=None, help="Hopsworks API key")
    parser.add_argument("--hopsworks-project", type=str, default="daily_collage", help="Hopsworks project name")
    parser.add_argument("--hopsworks-host", type=str, default=None, help="Optional Hopsworks host override")
    parser.add_argument("--fg-name", type=str, default="headline_labels", help="Feature group name for labels")
    parser.add_argument("--fg-version", type=int, default=1, help="Feature group version")
    parser.add_argument("--city", type=str, default=None, help="Optional city filter for training rows")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit when pulling from Hopsworks")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio when pulling from Hopsworks")
    parser.add_argument("--register-to-hopsworks", action="store_true", help="Register trained model to Hopsworks Model Registry")
    parser.add_argument("--model-name", type=str, default="daily_collage_classifier", help="Model registry name")
    parser.add_argument("--model-version", type=int, default=None, help="Optional model version for registry")

    args = parser.parse_args()
    NUM_EPOCHS = args.epochs
    
    train_path = args.train
    val_path = args.val

    if args.from_hopsworks:
        if not args.hopsworks_api_key:
            parser.error("--hopsworks-api-key is required when --from-hopsworks is set")

        hops_df = load_hopsworks_training_df(
            api_key=args.hopsworks_api_key,
            project=args.hopsworks_project,
            host=args.hopsworks_host,
            fg_name=args.fg_name,
            fg_version=args.fg_version,
            city=args.city,
            limit=args.limit,
        )

        train_path, val_path = materialize_train_val(
            hops_df,
            output_dir=args.output,
            val_ratio=args.val_ratio,
            prefix="hopsworks",
        )

    if not train_path or not val_path:
        parser.error("Provide --train and --val paths or use --from-hopsworks")

    logger.info(f"Running training with train={train_path} val={val_path}")
    model, best_val_loss, best_checkpoint = train_quick(train_path, val_path, args.output)

    if args.register_to_hopsworks:
        if not args.hopsworks_api_key:
            parser.error("--hopsworks-api-key is required when --register-to-hopsworks is set")

        import hopsworks
        
        logger.info(f"Registering model to Hopsworks project: {args.hopsworks_project}")
        
        # Direct Hopsworks login
        login_kwargs = {
            "api_key_value": args.hopsworks_api_key,
            "project": args.hopsworks_project,
            "engine": "python"
        }
        if args.hopsworks_host:
            login_kwargs["host"] = args.hopsworks_host
        
        hops_project = hopsworks.login(**login_kwargs)
        mr = hops_project.get_model_registry()
        
        # Create model
        model_metadata = mr.python.create_model(
            name=args.model_name,
            version=args.model_version,
            metrics={"val_loss": float(best_val_loss)},
            description="Fine-tuned multi-head news classifier",
        )
        
        # Save model files
        model_metadata.save(str(best_checkpoint))
        logger.info(f"✓ Model registered to Hopsworks: {args.model_name}")
