"""
Simplified template generation using existing classification results.

This script:
1. Fetches GDELT articles
2. Uses existing embedding classification to categorize them
3. Extracts the article titles/descriptions as templates directly
4. Generates keywords from the classified articles

This is faster and doesn't require loading a separate LLM model.

Usage:
    python ml/utils/generate_templates_simple.py --articles 200
"""

import argparse
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news_batched
from ml.utils.embedding_labeling import classify_article_embedding

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SIGNAL_CATEGORIES = [
    "emergencies", "crime", "festivals", "transportation",
    "weather_temp", "weather_wet", "sports", "economics", "politics"
]


def extract_keywords_simple(text: str, min_length: int = 4) -> list:
    """Extract potential keywords from text (simple word tokenization)."""
    # Simple keyword extraction: split and filter
    words = text.lower().split()
    # Remove common Swedish stop words and short words
    stop_words = {"och", "att", "det", "som", "för", "med", "den", "är", "på", "en", "av", "till", "i", "har", "från", "ett", "om", "vid", "kan", "men", "efter", "inte", "var", "får", "han", "ska", "när", "nu"}
    
    keywords = []
    for word in words:
        # Clean punctuation
        word = word.strip('.,!?:;–—()[]{}"\'"')
        if len(word) >= min_length and word not in stop_words and word.isalpha():
            keywords.append(word)
    
    return keywords


def generate_templates_simple(
    num_articles: int,
    country: str,
    templates_per_category: int,
    output_dir: Path,
    confidence_threshold: float = 0.45,
):
    """
    Generate templates from articles using embedding classification.
    
    Args:
        num_articles: Number of articles to fetch
        country: Country filter
        templates_per_category: Max templates per category
        output_dir: Output directory for JSON files
        confidence_threshold: Minimum confidence score to include article
    """
    logger.info(f"Fetching {num_articles} articles from GDELT (country={country})...")
    
    # Fetch articles
    articles_df = fetch_news_batched(
        country=country,
        total_articles=num_articles,
        batch_size=min(250, num_articles),
        days_lookback=7,
    )
    
    if articles_df is None or len(articles_df) == 0:
        logger.error("No articles fetched. Exiting.")
        return
    
    articles = articles_df.to_dicts()
    logger.info(f"✓ Fetched {len(articles)} articles")
    
    # Classify articles using embedding
    logger.info("Classifying articles with embeddings...")
    categorized_articles = defaultdict(list)
    categorized_keywords = defaultdict(list)
    
    for article in tqdm(articles, desc="Classifying"):
        title = article.get("title", "")
        description = article.get("description", "")
        
        if not title:
            continue
        
        # Classify
        classification = classify_article_embedding(
            title=title,
            description=description,
            similarity_threshold=0.35,
        )
        
        # Add to categories above threshold
        for category, (score, tag) in classification.items():
            if score >= confidence_threshold:
                # Use title as template (it's more concise than description)
                categorized_articles[category].append(title)
                
                # Extract keywords from title and description
                text = f"{title} {description}"
                keywords = extract_keywords_simple(text)
                categorized_keywords[category].extend(keywords)
    
    logger.info("✓ Classification complete")
    
    # Generate templates (use top unique titles per category)
    logger.info("\nGenerating signal templates...")
    signal_templates = {}
    
    for category in SIGNAL_CATEGORIES:
        if category in categorized_articles:
            # Get unique titles, sorted by frequency
            title_counter = Counter(categorized_articles[category])
            unique_titles = [title for title, _ in title_counter.most_common(templates_per_category * 2)]
            
            # Take top N unique templates
            templates = unique_titles[:templates_per_category]
            signal_templates[category] = templates
            
            logger.info(f"  {category:20s}: {len(templates)} templates (from {len(categorized_articles[category])} articles)")
        else:
            logger.warning(f"  {category:20s}: No articles found")
            signal_templates[category] = []
    
    # Generate keywords (most common words per category)
    logger.info("\nGenerating tag keywords...")
    tag_keywords = {}
    
    for category in SIGNAL_CATEGORIES:
        if category in categorized_keywords:
            # Count keyword frequency
            keyword_counter = Counter(categorized_keywords[category])
            
            # Generate simple keyword -> tag mappings
            # For now, use the keyword itself as the tag (simplified)
            keywords_dict = {}
            for keyword, count in keyword_counter.most_common(25):
                if count >= 2:  # Must appear at least twice
                    # Simple heuristic: use keyword as tag
                    # In a real system, you'd map to semantic tags
                    keywords_dict[keyword] = keyword
            
            tag_keywords[category] = keywords_dict
            logger.info(f"  {category:20s}: {len(keywords_dict)} keywords")
        else:
            logger.warning(f"  {category:20s}: No keywords found")
            tag_keywords[category] = {}
    
    # Save to JSON files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    templates_file = output_dir / "signal_templates.json"
    keywords_file = output_dir / "tag_keywords.json"
    
    with open(templates_file, 'w', encoding='utf-8') as f:
        json.dump(signal_templates, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ Saved signal templates to {templates_file}")
    
    with open(keywords_file, 'w', encoding='utf-8') as f:
        json.dump(tag_keywords, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved tag keywords to {keywords_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Articles processed: {len(articles)}")
    logger.info(f"Templates generated: {sum(len(t) for t in signal_templates.values())}")
    logger.info(f"Keywords generated: {sum(len(k) for k in tag_keywords.values())}")
    
    logger.info("\nCategory breakdown:")
    for cat in SIGNAL_CATEGORIES:
        templates_count = len(signal_templates.get(cat, []))
        keywords_count = len(tag_keywords.get(cat, {}))
        articles_count = len(categorized_articles.get(cat, []))
        logger.info(f"  {cat:20s}: {articles_count:3d} articles -> {templates_count:2d} templates, {keywords_count:2d} keywords")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review the generated JSON files")
    logger.info("  2. Test classification with new templates:")
    logger.info("     python test_json_loading.py")


def main():
    parser = argparse.ArgumentParser(description="Generate templates from classified articles")
    parser.add_argument("--articles", type=int, default=200, help="Number of articles to fetch")
    parser.add_argument("--country", type=str, default="sweden", help="Country filter")
    parser.add_argument("--templates-per-category", type=int, default=25, 
                        help="Max templates per category")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), 
                        help="Output directory for JSON files")
    parser.add_argument("--confidence", type=float, default=0.45, 
                        help="Minimum confidence score to include article")
    
    args = parser.parse_args()
    
    generate_templates_simple(
        num_articles=args.articles,
        country=args.country,
        templates_per_category=args.templates_per_category,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
