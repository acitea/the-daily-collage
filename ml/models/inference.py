"""
Production inference module for fine-tuned model.
Integrates with existing hopsworks_pipeline.py classification system.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import sys
import logging
from typing import Dict, Tuple, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES, TAG_VOCAB

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NewsSignalClassifierInference:
    """
    Production inference wrapper for fine-tuned model.
    Can be used as a drop-in replacement for classify_article().
    """

    def __init__(
        self,
        model_path: str,
        base_model: str = "KB/bert-base-swedish-cased",
        device: Optional[torch.device] = None,
    ):
        """
        Load fine-tuned model for inference.

        Args:
            model_path: Path to best_model.pt checkpoint
            base_model: Base model name
            device: torch device (auto-detected if None)
        """
        self.device = device or DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load model architecture
        self.model = self._build_model(base_model)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"✓ Loaded fine-tuned model from {model_path}")

    def _build_model(self, base_model: str) -> nn.Module:
        """Build model architecture (mirrors training model)."""
        
        class QuickNewsClassifier(nn.Module):
            """Exact replica of training model."""
            def __init__(self, base_model: str = "KB/bert-base-swedish-cased"):
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

        return QuickNewsClassifier(base_model)

    @torch.no_grad()
    def classify(self, title: str, description: str = "") -> Dict[str, Tuple[float, str]]:
        """
        Classify article using fine-tuned model.

        Returns:
            Dict mapping category -> (score, tag)
            e.g., {"emergencies": (0.8, "fire"), "crime": (0.0, "")}
        """
        text = f"{title} [SEP] {description}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Forward pass
        predictions = self.model(input_ids, attention_mask)

        # Process outputs
        results = {}
        for cat in SIGNAL_CATEGORIES:
            score_tensor, tag_logits = predictions[cat]
            score = float(score_tensor.squeeze().cpu())

            # Get tag
            tag_idx = torch.argmax(tag_logits, dim=-1).item()
            tag = TAG_VOCAB[cat][tag_idx] if tag_idx < len(TAG_VOCAB[cat]) else ""

            # Include if relevant (lowered threshold for sparse training data)
            # For limited training data, use 0.0 threshold to show all non-neutral predictions
            if abs(score) > 0.01 or tag != "":
                results[cat] = (score, tag)

        return results
    
    @torch.no_grad()
    def classify_with_confidence(self, title: str, description: str = "") -> Dict[str, Tuple[float, str, float]]:
        """
        Classify article with category relevance scores.
        
        Returns:
            Dict mapping category -> (score, tag, confidence)
            - score: Sentiment/intensity (-1 to 1, Tanh output)
            - tag: Predicted tag within category
            - confidence: Category relevance (0 to 1, from tag head probabilities)
        """
        text = f"{title} [SEP] {description}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Forward pass
        predictions = self.model(input_ids, attention_mask)

        # Process outputs with confidence scores
        results = {}
        for cat in SIGNAL_CATEGORIES:
            score_tensor, tag_logits = predictions[cat]
            score = float(score_tensor.squeeze().cpu())

            # Get tag from tag head
            tag_probs = torch.softmax(tag_logits, dim=-1)
            tag_idx = torch.argmax(tag_probs, dim=-1).item()
            tag = TAG_VOCAB[cat][tag_idx] if tag_idx < len(TAG_VOCAB[cat]) else ""
            
            # Use absolute value of score as confidence/relevance
            # Rational: Score represents intensity (how strong the signal is)
            # Articles irrelevant to a category should have scores near 0
            # Relevant articles (positive or negative events) should have high |score|
            confidence = abs(score)

            # Include if relevant
            if confidence > 0.01:  # Very low threshold - include anything non-zero
                results[cat] = (score, tag, confidence)

        return results
    
    def get_top_category(self, title: str, description: str = "", 
                        exclude_categories: Optional[List[str]] = None) -> Optional[Tuple[str, float, str, float]]:
        """
        Get the most relevant category for an article.
        
        Args:
            title: Article title
            description: Article description
            exclude_categories: List of categories to exclude (e.g., ['sports'] if untrained)
            
        Returns:
            (category, score, tag, confidence) or None if no relevant category
            Selects based on confidence (relevance), not score (sentiment)
        """
        results = self.classify_with_confidence(title, description)
        
        if not results:
            return None
        
        # Filter out excluded categories
        if exclude_categories:
            results = {k: v for k, v in results.items() if k not in exclude_categories}
        
        if not results:
            return None
        
        # Find category with highest confidence (relevance)
        top_cat = max(results.items(), key=lambda x: x[1][2])  # x[1][2] is confidence
        category, (score, tag, confidence) = top_cat
        
        return category, score, tag, confidence


# Global classifier instance
_classifier: Optional[NewsSignalClassifierInference] = None


def get_fine_tuned_classifier(
    model_path: str = "ml/models/checkpoints/best_model.pt",
) -> NewsSignalClassifierInference:
    """Get or create global classifier instance."""
    global _classifier

    if _classifier is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        _classifier = NewsSignalClassifierInference(model_path)

    return _classifier
