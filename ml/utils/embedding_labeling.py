"""
Embedding-based labeling for GDELT articles.

This module provides semantic/embedding-based classification of Swedish news articles
using sentence transformers and cosine similarity, replacing the keyword-matching approach
with something more robust and semantically aware.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

from sentence_transformers import SentenceTransformer
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Global model cache
_model = None
_signal_embeddings = None


# Define semantic templates for each signal category
# These are diverse examples that capture the semantic space of each signal
SIGNAL_TEMPLATES = {
    "emergencies": [
        "En stor brand utbröt i centrum",
        "Jordbävningen orsakade omfattande skador",
        "Explosionen evakuerade hela området",
        "Nödsituation med många skadade",
        "Räddningstjänsten rycker ut till olycka",
        "Katastrofberedskap aktiverad",
    ],
    "crime": [
        "Rån på bensinstation i nackan",
        "Polisen söker misstänkt mördare",
        "Stöld från varuhuset anmäld",
        "Misshandling på torget sent på kvällen",
        "Inbrott i privatbostaden på natten",
        "Brottsplats spärrad av polis",
    ],
    "festivals": [
        "Konserthändelsen lockar tusentals besökare",
        "Firande på gatan med musik och dans",
        "Festivalen börjar nästa vecka",
        "Världscupen attraktion för fans",
        "Kulturell manifestation påbörjas",
        "Fest och firande på torget",
    ],
    "transportation": [
        "Trafikstörning orsakar långa köer",
        "Trafikolycka på motorvägen",
        "Tung lastbil försenar kollektivtrafiken",
        "Vägen är stängd för reparation",
        "Gränsöverväxlingen orsakar försinkningar",
        "Busshållplatsen fylld av väntande passagerare",
    ],
    "weather_temp": [
        "Värmebölja slår värmerekord",
        "Temperaturerna stiger till extrema nivåer",
        "Kallvågorna kommer att påverka regionen",
        "Frysande väder på väg",
        "Hettan blir outhärdlig i städerna",
        "Kylan skapar problem på vägen",
    ],
    "weather_wet": [
        "Kraftiga regn orsakar översvämningar",
        "Snöstormen lamslår trafiken",
        "Översvämningen hotar husen",
        "Skyfall och blixt på väg",
        "Snön täcker hela landet",
        "Blötregn under hela dagen",
    ],
    "sports": [
        "Fotbollslaget vinner matchen",
        "Hockeymästerskapet avgörs i dag",
        "Idrottscupfinalen lockar folkmassa",
        "Segeröl kommer att flyta",
        "Landslaget spelar viktigt derbyn",
        "Atleterna tävlar om guldet",
    ],
    "economics": [
        "Börsen stiger på goda nyheter",
        "Arbetslösheten ökar i landet",
        "Företagen rapporterar svagt resultat",
        "Handel ökar mellan länderna",
        "Inflation påverkar konsumenternas köpkraft",
        "Marknaden väntar på centralbankens beslut",
    ],
    "politics": [
        "Regeringen presenterar ny politik",
        "Valet är närmare än någonsin",
        "Protester mot regeringspolitiken",
        "Parlamentet debatterar ny lag",
        "Oppositionen critiserar regeringen",
        "Politiska förhandlingar pågår",
    ],
}


def get_embedding_model(model_name: str = "KBLab/sentence-bert-swedish-cased") -> SentenceTransformer:
    """
    Load and cache the Swedish sentence transformer model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        SentenceTransformer model
    """
    global _model
    
    if _model is None:
        logger.info(f"Loading Swedish embedding model: {model_name}")
        try:
            _model = SentenceTransformer(model_name)
            logger.info("✓ Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    return _model


def get_signal_embeddings(
    model: SentenceTransformer = None,
    templates: Dict[str, List[str]] = SIGNAL_TEMPLATES,
) -> Dict[str, np.ndarray]:
    """
    Pre-compute embeddings for all signal templates.
    
    Args:
        model: SentenceTransformer model
        templates: Dict mapping signal category to list of semantic templates
        
    Returns:
        Dict mapping category to averaged embedding vector
    """
    global _signal_embeddings
    
    if _signal_embeddings is not None:
        return _signal_embeddings
    
    if model is None:
        model = get_embedding_model()
    
    logger.info("Computing signal embeddings from templates...")
    _signal_embeddings = {}
    
    for category, category_templates in templates.items():
        # Encode all templates for this category
        embeddings = model.encode(category_templates, convert_to_numpy=True)
        
        # Average the embeddings to get a single representative vector
        avg_embedding = np.mean(embeddings, axis=0)
        _signal_embeddings[category] = avg_embedding
        
        logger.debug(f"  {category:20s}: averaged {len(category_templates)} templates")
    
    return _signal_embeddings


def classify_article_embedding(
    title: str,
    description: str = "",
    similarity_threshold: float = 0.35,
    model: SentenceTransformer = None,
    signal_embeddings: Dict[str, np.ndarray] = None,
) -> Dict[str, Tuple[float, str]]:
    """
    Classify article using embedding-based semantic similarity.
    
    Instead of keyword matching, this computes embeddings for the article
    and compares them to semantic templates for each signal category.
    Similarity scores are directly used as confidence scores.
    
    Args:
        title: Article title
        description: Article description/body
        similarity_threshold: Only include signals above this similarity score
        model: SentenceTransformer model (uses cached if None)
        signal_embeddings: Pre-computed signal embeddings (uses cached if None)
        
    Returns:
        Dict mapping category to (score, tag) tuple
        e.g., {"emergencies": (0.72, "fire"), "crime": (0.41, "police")}
    """
    if model is None:
        model = get_embedding_model()
    
    if signal_embeddings is None:
        signal_embeddings = get_signal_embeddings(model)
    
    # Combine title and description for better semantic context
    combined_text = f"{title} {description}".strip()
    
    if not combined_text:
        return {}
    
    # Encode the article
    article_embedding = model.encode(combined_text, convert_to_numpy=True)
    
    # Compute cosine similarity to each signal category
    results = {}
    
    for category, signal_emb in signal_embeddings.items():
        # Cosine similarity: (A·B) / (|A||B|)
        # Returns value in [-1, 1], but for similar texts typically [0, 1]
        similarity = np.dot(article_embedding, signal_emb) / (
            np.linalg.norm(article_embedding) * np.linalg.norm(signal_emb)
        )
        
        # Use raw similarity as confidence score (already in 0-1 range for similar texts)
        # Clamp negative similarities to 0
        confidence_score = max(0.0, similarity)
        
        if confidence_score >= similarity_threshold:
            # Select tag based on highest template similarity
            category_embeddings = model.encode(
                SIGNAL_TEMPLATES[category],
                convert_to_numpy=True
            )
            template_similarities = [
                np.dot(article_embedding, templ_emb) / (
                    np.linalg.norm(article_embedding) * np.linalg.norm(templ_emb)
                )
                for templ_emb in category_embeddings
            ]
            best_template_idx = np.argmax(template_similarities)
            best_template = SIGNAL_TEMPLATES[category][best_template_idx]
            
            # Infer tag from best template
            tag = infer_tag_from_template(category, best_template)
            
            results[category] = (confidence_score, tag)
    
    return results


def infer_tag_from_template(category: str, template: str) -> str:
    """
    Infer a tag value based on the best-matching template for a category.
    
    This is a simple heuristic that extracts key signal words from templates.
    
    Args:
        category: Signal category
        template: The best-matching template text
        
    Returns:
        Tag string (empty if no specific tag detected)
    """
    template_lower = template.lower()
    
    # Category-specific tag extraction
    TAG_KEYWORDS = {
        "emergencies": {
            "brand": "fire",
            "explosion": "explosion",
            "earthquake": "earthquake",
            "evacuation": "evacuation",
            "olycka": "accident",
        },
        "crime": {
            "rån": "robbery",
            "stöld": "theft",
            "misshandling": "assault",
            "mord": "assault",
            "inbrott": "theft",
            "polis": "police",
        },
        "festivals": {
            "konsert": "concert",
            "firande": "celebration",
            "fest": "celebration",
            "mästerskap": "event",
        },
        "transportation": {
            "trafik": "traffic",
            "trafikolycka": "accident",
            "kö": "congestion",
            "stängd": "closure",
            "vägen": "closure",
        },
        "weather_temp": {
            "varme": "hot",
            "hettan": "hot",
            "kylan": "cold",
            "frysande": "cold",
            "temperatur": "hot",
        },
        "weather_wet": {
            "regn": "rain",
            "snö": "snow",
            "översvämning": "flood",
            "skyfall": "rain",
            "blöt": "rain",
            "storm": "rain",
        },
        "sports": {
            "fotboll": "football",
            "hockey": "hockey",
            "seger": "victory",
            "vinna": "victory",
            "match": "football",
            "tävl": "event",
        },
        "economics": {
            "börsen": "market",
            "marknad": "market",
            "arbetslöshet": "employment",
            "företag": "business",
            "inflation": "inflation",
            "handel": "trade",
        },
        "politics": {
            "val": "election",
            "protest": "protest",
            "regering": "government",
            "parlament": "government",
            "lag": "policy",
            "politiker": "government",
        },
    }
    
    # Check for keywords specific to this category
    if category in TAG_KEYWORDS:
        for keyword, tag in TAG_KEYWORDS[category].items():
            if keyword in template_lower:
                return tag
    
    # Fallback to empty tag if no match
    return ""


# For backwards compatibility, wrap this in classify_article if needed
def classify_article_with_fallback(
    title: str,
    description: str = "",
    use_embedding: bool = True,
) -> Dict[str, Tuple[float, str]]:
    """
    Classify article, trying embedding-based first, then keyword fallback.
    
    Args:
        title: Article title
        description: Article description
        use_embedding: If True, use embedding-based; otherwise use keywords only
        
    Returns:
        Dict mapping category to (score, tag) tuple
    """
    if use_embedding:
        try:
            result = classify_article_embedding(title, description)
            if result:
                logger.debug(f"Embedding classification succeeded: {result}")
                return result
        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}")
    
    # Fallback to keyword-based
    from ml.ingestion.hopsworks_pipeline import classify_article as keyword_classify
    return keyword_classify(title, description)
