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
# Using REAL excerpts from GDELT articles for accurate semantic matching
SIGNAL_TEMPLATES = {
    "emergencies": [
        "Villabrand i Kramfors",
        "Polisen : Unga tar på sig mordbranduppdrag",
        "Oväder i Afghanistan – många döda",
        "Tusentals fortsatt strömlösa efter Johannes",
        "Kommuner hjälper strömlösa invånare",
        "SMHI utvidgar snövarningarna inför nyår",
        "Brand sprider sig i byggnader",
        "100 - tal larm om smällar i Jönköping",
        "Explosioner rapporterade i området",
        "Räddningstjänsten på plats",
        "Evakuering av bostäder pågår",
        "Nödsituation utlyst",
        "Jordbävning registrerad",
        "Översvämningar hotar området",
        "Katastrofberedskap aktiverad",
        "Gasläcka orsakar panik",
        "Tågolycka med skador",
        "Bridge collapses in storm",
        "Major fire outbreak reported",
        "Chemical leak forces evacuation",
        "Building collapse reported",
        "Emergency services respond",
        "Rescue operations underway",
        "Wildfire spreads rapidly",
        "Flooding threatens homes",
        "Earthquake damage reported",
        "Disaster declared",
        "Power outages widespread",
        "Infrastructure damaged",
        "Infrastructure collapse",
        "Storm damage extensive",
        "Weather emergency declared",
        "Hazardous conditions reported",
        "Mass casualty event",
        "Critical infrastructure affected",
    ],
    "crime": [
        "Fängelse för våldtäkt i park i Örebro",
        "Säpotillslag mot man – kopplas till rikets säkerhet",
        "Polis söker efter hotfull man i Västerhaninge",
        "Ny attack mot misstänkt knarkbåt",
        "Mordmisstänkt tidigare dömd – försökte kidnappa barn",
        "Slagsmål i Helsingborg",
        "Man och kvinna beskjutna med fyrverkerier",
        "Tidningsbud misshandlat – vägrade öppna portdörr",
        "Tidningsbud misshandlad vid trappuppgång",
        "Rönninge - mordet : Misstänkta Vilma Andersson",
        "Fem punkter om mordet på kvinnan",
        "Åtal väckt för dubbelmord i Klippan",
        "Kvinna hittade okänd man på soffan",
        "Man åtalas efter brutalt dubbelmord",
        "Gripen för brott",
        "Polisen söker misstänkt",
        "Överfall rapporterat",
        "Theft from store",
        "Robbery at shop",
        "Assault victim hospitalized",
        "Criminal arrested",
        "Police investigation ongoing",
        "Murder investigation",
        "Homicide reported",
        "Stabbing incident",
        "Shooting reported",
        "Breaking and entering",
        "Gang violence erupts",
        "Drug trafficking bust",
        "Fraud investigation",
        "Cybercrime uncovered",
        "Witness protection",
        "Crime rate increases",
        "Law enforcement operation",
        "Arrest made",
    ],
    "festivals": [
        "Premiärministern vill ha fler damtoaletter",
        "Kungar och kejsares möte",
        "Kulturell event",
        "Konsert arrangerad",
        "Musik framförd",
        "Fest och firande",
        "Parade genom gatorna",
        "Festival börjar",
        "Publiksamling",
        "Kulturprogram",
        "Artistframträdande",
        "Scenuppställning",
        "Underhålningsevenemang",
        "Sportuttagning",
        "Tävling arrangeras",
        "Mästerskap spelas",
        "Cup final",
        "Concert series",
        "Music festival",
        "Live performance",
        "Street parade",
        "Public celebration",
        "Community event",
        "Cultural fair",
        "Theater show",
        "Comedy performance",
        "Dance recital",
        "Art exhibition",
        "Games tournament",
        "Sports event",
        "Opening ceremony",
        "Awards ceremony",
        "Gathering of people",
        "Entertainment show",
        "Celebration event",
    ],
    "transportation": [
        "SMHI utvidgar snövarningarna inför nyår",
        "Tusentals fortsatt strömlösa efter Johannes",
        "Träden skyddar Winfridas gård mot det nya vädret",
        "Wall Street föll – teknikaktier tyngde",
        "Chattbottar fick väljare att ändra sig",
        "Vägen blockerad",
        "Tågtrafiken försenad",
        "Flygtrafiken påverkad",
        "Väg stängd för arbete",
        "Busslinje inställd",
        "Tunnelbanan störd",
        "Metro stopped",
        "Train delayed",
        "Flight cancelled",
        "Road closed",
        "Traffic jam reported",
        "Accident blocks road",
        "Commute disrupted",
        "Transit delayed",
        "Vehicle collision",
        "Truck accident",
        "Car crash",
        "Accident on highway",
        "Traffic congestion",
        "Commuter impact",
        "Travel delay",
        "Transport disruption",
        "Infrastructure failure",
        "Service interrupted",
        "Closure announced",
        "Detour established",
        "Alternative route",
        "Transit impact",
        "Commuter warning",
        "Travel alert",
    ],
    "weather_temp": [
        "Gnistrande beskedet inför nyår – först sol sedan snö",
        "Träden skyddar Winfridas gård mot det nya vädret",
        "SMHI utvidgar snövarningarna inför nyår",
        "Temperaturer stiger",
        "Temperaturer sjunker",
        "Värmebölja varning",
        "Kallvåg närmar sig",
        "Hetta ökar",
        "Kyla drabbar",
        "Frost varning",
        "Heatwave advisory",
        "Cold warning issued",
        "Temperature record",
        "Extreme heat",
        "Freezing conditions",
        "Temperature drop",
        "Unseasonable warmth",
        "Cold snap",
        "Heat index high",
        "Ice forming",
        "Frost expected",
        "Snow warning",
        "Cold front",
        "Warm front",
        "Weather shift",
        "Temperature swing",
        "Record heat",
        "Record cold",
        "Dangerous heat",
        "Bitter cold",
        "Unusual weather",
        "Season change",
        "Weather alert",
        "Climate concern",
        "Temperature alert",
    ],
    "weather_wet": [
        "SMHI utvidgar snövarningarna inför nyår",
        "Oväder i Afghanistan – många döda",
        "Tusentals fortsatt strömlösa efter Johannes",
        "Snöfall varning",
        "Regn förväntas",
        "Storm närmar sig",
        "Översvämning risk",
        "Hagel varning",
        "Snöstorm varning",
        "Tornadovarning",
        "Heavy rain forecast",
        "Snow expected",
        "Storm warning issued",
        "Flooding alert",
        "Blizzard conditions",
        "Sleet warning",
        "Hail expected",
        "Wind advisory",
        "River flooding",
        "Flash flood warning",
        "Winter storm",
        "Severe weather alert",
        "Weather emergency",
        "Precipitation warning",
        "Precipitation forecast",
        "Rain advisory",
        "Snow advisory",
        "Storm alert",
        "Weather system",
        "Precipitation event",
        "Rainfall warning",
        "Wet conditions",
        "Slippery roads",
        "Water damage",
        "Precipitation event",
    ],
    "sports": [
        "Australisk polis : Misstänkta terroristerna agerade själva",
        "Chattbottar fick väljare att ändra sig",
        "Wall Street föll – teknikaktier tyngde",
        "Fotbollsmatch spelas",
        "Hockeymatch arrangeras",
        "Tennisturneringen börjar",
        "Seger för hemmalaget",
        "Förlust för bortalaget",
        "Matchresultat",
        "Tävling pågår",
        "Spelare presterar",
        "Träningsfas",
        "Qualification round",
        "Championship match",
        "Tournament begins",
        "Team advances",
        "Player scores",
        "Victory announced",
        "Game result",
        "Match outcome",
        "Player performance",
        "League standings",
        "Playoff game",
        "Final match",
        "Semi-final",
        "Quarter-final",
        "Sports event",
        "Athletic competition",
        "Tournament announced",
        "Season starts",
        "Game scheduled",
        "Match scheduled",
        "Competition",
        "Athletic event",
        "Sports league",
    ],
    "economics": [
        "Wall Street föll – teknikaktier tyngde",
        "Protester i Iran mot skenande ekonomi",
        "Börsindex faller",
        "Börsindex stiger",
        "Aktiekurser sjunker",
        "Aktiekurser stiger",
        "Jobben ökar",
        "Jobben minskar",
        "Arbetslösheten stigande",
        "Tillväxten sakta",
        "Inflationen stiger",
        "Inflationen faller",
        "Stock market falls",
        "Stock market rises",
        "Economic growth",
        "Economic decline",
        "Job losses",
        "Job creation",
        "Unemployment rate",
        "Wage growth",
        "Inflation concerns",
        "Interest rate",
        "Currency movement",
        "Trade balance",
        "Business confidence",
        "Consumer spending",
        "Market volatility",
        "Economic forecast",
        "Earnings report",
        "GDP report",
        "Industry performance",
        "Company results",
        "Financial news",
        "Economic data",
    ],
    "politics": [
        "Premiärministern vill ha fler damtoaletter",
        "Chattbottar fick väljare att ändra sig",
        "Protester i Iran mot skenande ekonomi",
        "Ryssland : Skärper tonen efter påstådd attack",
        "Kina avfyrar raketer runt Taiwan",
        "Bolsonaro lämnar sjukhuset",
        "Trump : Attackerat stor anläggning",
        "Regeringen presenterar lagförslag",
        "Riksdagen röstar",
        "Parlamentet debatterar",
        "Val annonseratt",
        "Regering bildad",
        "Opposition protesterar",
        "Minister avgår",
        "Politisk kris",
        "Lagförslag presenteras",
        "Government announced",
        "Election called",
        "Parliament vote",
        "Cabinet reshuffle",
        "Policy debate",
        "Budget passed",
        "Reform proposed",
        "Opposition party",
        "Political alliance",
        "Vote of confidence",
        "Legislative session",
        "Political agreement",
        "Coalition talks",
        "Minister appointed",
        "Political statement",
        "Government policy",
        "Legislative vote",
        "Political development",
        "Government decision",
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
        Dict mapping category to array of template embeddings (not averaged)
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
        
        # Store all template embeddings (will use max/top-k similarity later)
        _signal_embeddings[category] = embeddings
        
        logger.debug(f"  {category:20s}: stored {len(category_templates)} template embeddings")
    
    return _signal_embeddings


def classify_article_embedding(
    title: str,
    description: str = "",
    similarity_threshold: float = 0.35,
    model: SentenceTransformer = None,
    signal_embeddings: Dict[str, np.ndarray] = None,
    top_k_signals: int = None,
    relative_threshold: float = 0.70,
) -> Dict[str, Tuple[float, str]]:
    """
    Classify article using embedding-based semantic similarity.
    
    Instead of keyword matching, this computes embeddings for the article
    and compares them to semantic templates for each signal category.
    Similarity scores are directly used as confidence scores.
    
    Args:
        title: Article title
        description: Article description/body
        similarity_threshold: Base similarity threshold (minimum to consider)
        model: SentenceTransformer model (uses cached if None)
        signal_embeddings: Pre-computed signal embeddings (uses cached if None)
        top_k_signals: If set, only return top K most confident signals (default: None = all above threshold)
        relative_threshold: Only keep signals within this ratio of max similarity (default: 0.70)
            e.g., if max_sim=0.9, only keep signals with sim >= 0.9 * 0.70 = 0.63
        
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
    all_scores = []  # Track all scores for relative filtering

    for category, template_embeddings in signal_embeddings.items():
        # Compute similarity to each template in this category
        similarities = []
        for template_emb in template_embeddings:
            # Cosine similarity: (A·B) / (|A||B|)
            similarity = np.dot(article_embedding, template_emb) / (
                np.linalg.norm(article_embedding) * np.linalg.norm(template_emb)
            )
            similarities.append(similarity)
        
        # Use max similarity (best match across all templates in category)
        max_similarity = max(similarities)
        confidence_score = max(0.0, max_similarity)
        
        # Also track which template matched best for tag inference
        best_template_idx = np.argmax(similarities)

        if confidence_score >= similarity_threshold:
            # Select tag based on the best-matching template
            best_template = SIGNAL_TEMPLATES[category][best_template_idx]
            
            # Infer tag from best template
            tag = infer_tag_from_template(category, best_template)
            
            results[category] = (confidence_score, tag)
            all_scores.append(confidence_score)
    
    # Filter to keep only high-confidence signals
    if results and (top_k_signals or relative_threshold):
        max_score = max(all_scores)
        
        # Apply relative threshold: keep signals within X% of the max
        if relative_threshold and max_score > 0:
            min_relative_score = max_score * relative_threshold
            results = {
                cat: (score, tag) 
                for cat, (score, tag) in results.items() 
                if score >= min_relative_score
            }
        
        # Apply top-K filtering if specified
        if top_k_signals and len(results) > top_k_signals:
            sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
            results = dict(sorted_results[:top_k_signals])
    
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
            "fire": "fire",
            "explosion": "explosion",
            "earthquake": "earthquake",
            "evacuation": "evacuation",
            "olycka": "accident",
            "flood": "flood",
            "storm": "storm",
        },
        "crime": {
            "rån": "robbery",
            "stöld": "theft",
            "misshandling": "assault",
            "mord": "assault",
            "inbrott": "theft",
            "polis": "police",
            "robbery": "robbery",
            "burglary": "theft",
            "assault": "assault",
            "vandalism": "vandalism",
        },
        "festivals": {
            "konsert": "concert",
            "firande": "celebration",
            "fest": "celebration",
            "mästerskap": "event",
            "concert": "concert",
            "festival": "event",
            "parade": "celebration",
        },
        "transportation": {
            "trafik": "traffic",
            "trafikolycka": "accident",
            "kö": "congestion",
            "stängd": "closure",
            "vägen": "closure",
            "traffic": "traffic",
            "accident": "accident",
            "delay": "delay",
            "closure": "closure",
            "cancelled": "delay",
        },
        "weather_temp": {
            "varme": "hot",
            "hettan": "hot",
            "kylan": "cold",
            "frysande": "cold",
            "temperatur": "hot",
            "heat": "hot",
            "heatwave": "hot",
            "cold": "cold",
        },
        "weather_wet": {
            "regn": "rain",
            "snö": "snow",
            "översvämning": "flood",
            "skyfall": "rain",
            "blöt": "rain",
            "storm": "rain",
            "rain": "rain",
            "snow": "snow",
            "flood": "flood",
            "hail": "rain",
        },
        "sports": {
            "fotboll": "football",
            "hockey": "hockey",
            "seger": "victory",
            "vinna": "victory",
            "match": "football",
            "tävl": "event",
            "football": "football",
            "hockey": "hockey",
            "victory": "victory",
            "championship": "championship",
        },
        "economics": {
            "börsen": "market",
            "marknad": "market",
            "arbetslöshet": "employment",
            "företag": "business",
            "inflation": "inflation",
            "handel": "trade",
            "market": "market",
            "inflation": "inflation",
            "unemployment": "employment",
            "growth": "growth",
            "recession": "inflation",
        },
        "politics": {
            "val": "election",
            "protest": "protest",
            "regering": "government",
            "parlament": "government",
            "lag": "policy",
            "politiker": "government",
            "election": "election",
            "protest": "protest",
            "government": "government",
            "parliament": "government",
            "strike": "protest",
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
