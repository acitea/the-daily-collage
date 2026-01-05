"""
Complete labeling pipeline: Fetch → Auto-Label → Review → Export

This script provides an end-to-end solution for creating high-quality
training data for the fine-tuned BERT model.

Usage:
    python ml/utils/label_dataset.py --articles 500 --country sweden
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import polars as pl

# Optional dependencies for fetching article body
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional runtime dep
    requests = None
    BeautifulSoup = None

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news, fetch_news_batched
from ml.utils.embedding_labeling import classify_article_embedding
from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES
from ml.utils.intensity_calibration import calibrate_article_labels

def fetch_and_label(
    country: str = "sweden",
    num_articles: int = 500,
    method: str = "embedding",
    similarity_threshold: float = 0.20,
    auto_calibrate: bool = True,
    fetch_body: bool = False,
    llm_verify: bool = False,
    max_llm_calls: int = None,
) -> pl.DataFrame:
    """
    Fetch articles and auto-label them.
    
    Args:
        country: Country code (sweden, us, etc.)
        num_articles: Number of articles to fetch
        method: "embedding" or "keywords"
        similarity_threshold: Minimum similarity for embedding method
        auto_calibrate: If True, automatically convert similarity to intensity
        fetch_body: If True, fetch article body from URL when description is missing (slower, may add noise)
        llm_verify: If True, use LLM to verify and correct classifications (selective usage)
        max_llm_calls: Maximum LLM API calls allowed (None = unlimited, default: 50)
        
    Returns:
        Labeled DataFrame
    """
    # Set default API limit if not specified
    if llm_verify and max_llm_calls is None:
        max_llm_calls = 50  # Default safe limit (~$0.10)
    
    # Track API usage
    llm_call_count = {'calls': 0}
    print(f"📰 Fetching {num_articles} articles from {country}...")
    
    # Use batched fetching if more than 250 articles requested
    if num_articles > 250:
        df = fetch_news_batched(
            country=country,
            total_articles=num_articles,
            batch_size=250,
            days_lookback=30
        )
    else:
        df = fetch_news(
            country=country,
            max_articles=num_articles
        )
    
    print(f"✓ Fetched {len(df)} articles\n")
    
    if method == "embedding":
        print("🤖 Auto-labeling with embedding-based classification...")
        print(f"   (similarity threshold: {similarity_threshold})")
        if auto_calibrate:
            print("   (with automatic intensity calibration)")
        if llm_verify:
            print(f"   (LLM verification: selective (0 or >2 signals), max {max_llm_calls} calls)")
    else:
        print("🔤 Auto-labeling with keyword-based classification...")
    
    labeled_rows = []
    
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Labeling"):
        title = row.get("title", "")
        # Prefer detailed text if available; some GDELT responses omit description
        description = (
            row.get("description")
            or row.get("content")
            or row.get("summary")
            or ""
        )
        tone = row.get("tone")
        # Optionally fetch body from source URL if description is missing
        if not description and fetch_body:
            fetched_body = _fetch_page_content(row.get("url", ""))
            if fetched_body:
                description = fetched_body
        
        # Classify based on method
        if method == "embedding":
            try:
                signals = classify_article_embedding(
                    title=title,
                    description=description,
                    similarity_threshold=similarity_threshold,
                    relative_threshold=0.70  # Only keep signals within 70% of max confidence
                )
                
                # Automatically calibrate intensity from similarity scores
                if auto_calibrate:
                    signals = calibrate_article_labels(
                        signals=signals,
                        title=title,
                        description=description
                    )
                
                # Optional LLM verification (selective: 0 or >2 signals only)
                if llm_verify:
                    signals = classify_article_with_llm_verification(
                        title=title,
                        description=description,
                        embedding_signals=signals,
                        llm_call_count=llm_call_count,
                        max_llm_calls=max_llm_calls,
                    )
            except Exception as e:
                print(f"\n⚠️  Embedding failed: {e}")
                print("   Falling back to keywords...")
                method = "keywords"
                signals = {}
        
        if method == "keywords":
            from ml.ingestion.hopsworks_pipeline import classify_article
            signals = classify_article(title, description, method="keywords")
        
        # Build labeled row
        labeled_row = {
            "title": title,
            "description": description,
            "tone": float(tone) if tone is not None else None,
            "url": row.get("url", ""),
            "source": row.get("source", ""),
            "date": str(row.get("date", "")),
        }
        
        # Add category scores and tags
        for category in SIGNAL_CATEGORIES:
            if category in signals:
                score, tag = signals[category]
                labeled_row[f"{category}_score"] = float(score)
                labeled_row[f"{category}_tag"] = str(tag)
            else:
                labeled_row[f"{category}_score"] = 0.0
                labeled_row[f"{category}_tag"] = ""
        
        labeled_rows.append(labeled_row)
    
    labeled_df = pl.DataFrame(labeled_rows)
    print(f"\n✓ Labeled {len(labeled_df)} articles")
    
    # Show API usage if LLM was used
    if llm_verify and llm_call_count['calls'] > 0:
        estimated_cost = llm_call_count['calls'] * 0.002  # ~$0.002 per call
        print(f"📊 LLM API Usage: {llm_call_count['calls']} calls (estimated cost: ${estimated_cost:.2f})")
    
    return labeled_df


def _fetch_page_content(url: str) -> str:
    """
    Best-effort fetch of article body, filtering out navigation and boilerplate.
    
    Prioritizes common article content tags and removes navigation/menu elements.
    """
    if not url or requests is None or BeautifulSoup is None:
        return ""
    try:
        resp = requests.get(url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove navigation, ads, and boilerplate elements
        for tag in soup([
            "script", "style", "noscript", 
            "nav", "header", "footer", "aside",
            "iframe", "form", "button"
        ]):
            tag.decompose()
        
        # Remove common navigation/menu class patterns
        for pattern in ["nav", "menu", "sidebar", "footer", "header", "ad", "banner", "cookie"]:
            for elem in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                elem.decompose()
            for elem in soup.find_all(id=lambda x: x and pattern in x.lower()):
                elem.decompose()
        
        # Try to find main article content (prioritize semantic tags)
        article_content = None
        for selector in ["article", "main", '[role="main"]', ".article-body", ".post-content"]:
            article_content = soup.select_one(selector)
            if article_content:
                break
        
        # If found specific article container, use it; otherwise use body
        text_source = article_content if article_content else soup.body
        if text_source:
            text = text_source.get_text(separator=" ", strip=True)
            # Clean up excessive whitespace
            text = " ".join(text.split())
            # Keep reasonable length (first 1500 chars to focus on article start)
            return text[:1500]
        
        return ""
    except Exception:
        return ""


def classify_article_with_llm_verification(
    title: str,
    description: str,
    embedding_signals: dict,
    llm_call_count: dict = None,
    max_llm_calls: int = None,
) -> dict:
    """
    Use LLM to verify and correct article classifications (selective usage).
    
    Args:
        title: Article title
        description: Article description
        embedding_signals: Dict of {category: (score, tag)} from embedding classification
        llm_call_count: Dict tracking {'calls': N} for cost control
        max_llm_calls: Maximum LLM API calls allowed (None = unlimited)
        
    Returns:
        Updated signals dict with LLM corrections (scores, tags, and categories)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Count detected signals (those with non-zero scores)
    detected_count = sum(1 for cat, (score, tag) in embedding_signals.items() if abs(score) > 0.01)
    
    # Only use LLM for edge cases: no signals (0) or too many signals (>2)
    should_verify = detected_count == 0 or detected_count > 2
    
    if not should_verify:
        # 1-2 signals: embedding is usually reliable, skip LLM
        return embedding_signals
    
    # Check API usage limit
    if llm_call_count is None:
        llm_call_count = {'calls': 0}
    
    if max_llm_calls is not None and llm_call_count['calls'] >= max_llm_calls:
        logger.warning(f"LLM API limit reached ({max_llm_calls} calls). Skipping verification.")
        return embedding_signals
    
    # Increment counter
    llm_call_count['calls'] += 1
    
    try:
        from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES
        import json
        from openai import OpenAI
        
        # Build current classification summary
        current_detections = {}
        for category in SIGNAL_CATEGORIES:
            if category in embedding_signals:
                score, tag = embedding_signals[category]
                current_detections[category] = {"score": float(score), "tag": str(tag)}
        
        # Create prompt for LLM
        prompt = f"""Analyze this news article and verify the signal detections with scores.

Article Title: {title}
Description: {description}

Current Embedding-Based Classifications:
{json.dumps(current_detections, indent=2)}

Signal Categories to consider:
- emergencies: natural disasters, accidents, extreme weather impacts (score: -1 to -0.5 for severe, -0.3 to 0 for minor)
- crime: criminal incidents, violence, arrests (score: -1 to -0.5 for serious, -0.3 to 0 for minor)
- festivals: cultural events, celebrations, entertainment (score: 0.3 to 1 for positive events)
- transportation: traffic, accidents, transit issues, infrastructure (score: -0.5 to 0 for problems, 0.3+ for improvements)
- weather_temp: temperature extremes, heat waves, cold snaps (score: -0.8 to 0 for extremes)
- weather_wet: rain, flooding, snow, storms (score: -0.8 to 0 for severe, -0.3 to 0 for minor)
- sports: sporting events, competitions, athletic achievements (score: -0.5 to 0 for disappointments, 0.3 to 1 for successes)
- economics: markets, business, employment, inflation, trade (score: -0.8 to 0 for downturns, 0.3 to 1 for growth)
- politics: elections, policy, governance, political events (score: -0.5 to 0 for negative, 0.3 to 1 for positive)

For each detected signal:
1. Does the category make sense for this article?
2. Is the score magnitude reasonable? (closer to -1/1 for major events, closer to 0 for minor)
3. Is the direction correct? (negative for problems/bad news, positive for good news/achievements)

Identify any signals that should be:
- Removed (wrong category detected)
- Corrected with different score (wrong magnitude/direction)
- Added (missed signals)

Return ONLY valid JSON (no markdown, no explanation):
{{
  "corrections": {{"category_name": {{"score": -0.7, "tag": "corrected_tag"}}, ...}},
  "removals": ["category_name", ...],
  "additions": {{"category_name": {{"score": 0.8, "tag": "new_tag"}}, ...}}
}}

If no changes needed, return: {{"corrections": {{}}, "removals": [], "additions": {{}}}}"""

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.debug(f"LLM verification #{llm_call_count['calls']}: {detected_count} signals detected")
        
        # Parse response
        result = json.loads(result_text)
        verified_signals = embedding_signals.copy()
        
        # Apply corrections (update both scores and tags)
        for category, correction in result.get("corrections", {}).items():
            if category in verified_signals:
                new_score = correction.get("score", verified_signals[category][0])
                new_tag = correction.get("tag", verified_signals[category][1])
                verified_signals[category] = (float(new_score), str(new_tag))
                logger.debug(f"  → Corrected {category}: {verified_signals[category]}")
        
        # Remove signals that LLM thinks are wrong
        for category in result.get("removals", []):
            if category in verified_signals:
                verified_signals[category] = (0.0, "")
                logger.debug(f"  → Removed {category}")
        
        # Add any new detections with LLM-provided scores
        for category, signal_info in result.get("additions", {}).items():
            if category not in verified_signals or abs(verified_signals[category][0]) < 0.01:
                score = signal_info.get("score", 0.5)
                tag = signal_info.get("tag", category)
                verified_signals[category] = (float(score), str(tag))
                logger.debug(f"  → Added {category}: ({score}, {tag})")
        
        return verified_signals
        
    except Exception as e:
        logger.warning(f"LLM verification failed: {e}")
        return embedding_signals


def show_label_statistics(df: pl.DataFrame):
    """Display labeling statistics."""
    print("\n" + "="*80)
    print("LABEL STATISTICS")
    print("="*80)
    
    print(f"\nTotal articles: {len(df)}")
    
    print("\nSignal distribution:")
    for category in SIGNAL_CATEGORIES:
        # Count articles with significant signal (positive OR negative impact)
        non_zero = df.filter(pl.col(f"{category}_score").abs() > 0.01).height
        percentage = (non_zero / len(df)) * 100 if len(df) > 0 else 0
        
        avg_score = 0
        scores = df[f"{category}_score"].to_list()
        # Include both positive and negative scores
        non_zero_scores = [s for s in scores if abs(s) > 0.01]
        if non_zero_scores:
            avg_score = sum(non_zero_scores) / len(non_zero_scores)
        
        print(f"  {category:20s}: {non_zero:4d} ({percentage:5.1f}%) | avg score: {avg_score:.3f}")
    
    print("\nTop tags per category:")
    for category in SIGNAL_CATEGORIES:
        tags = df[f"{category}_tag"].to_list()
        tags_nonzero = [t for t in tags if t]
        
        if tags_nonzero:
            from collections import Counter
            tag_counts = Counter(tags_nonzero)
            top_tags = tag_counts.most_common(3)
            tag_str = ", ".join([f"{tag}({count})" for tag, count in top_tags])
            print(f"  {category:20s}: {tag_str}")


def split_train_val(
    df: pl.DataFrame,
    train_ratio: float = 0.8,
    output_dir: str = "data"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split dataset into train and validation sets.
    
    Args:
        df: Labeled DataFrame
        train_ratio: Ratio of training data (0-1)
        output_dir: Directory to save files
        
    Returns:
        (train_df, val_df)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
    
    # Split
    train_size = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:]
    
    # Save
    train_path = output_path / "train.parquet"
    val_path = output_path / "val.parquet"
    
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)
    
    print(f"\n✓ Train: {len(train_df)} articles → {train_path}")
    print(f"✓ Val:   {len(val_df)} articles → {val_path}")
    
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(
        description="Label GDELT articles for BERT fine-tuning"
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=500,
        help="Number of articles to fetch (default: 500)"
    )
    parser.add_argument(
        "--country",
        type=str,
        default="sweden",
        help="Country to fetch from (default: sweden)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["embedding", "keywords"],
        default="embedding",
        help="Labeling method (default: embedding)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Similarity threshold for embedding method (default: 0.20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Don't split into train/val, just save labeled dataset"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable automatic intensity calibration (use raw similarity scores)"
    )
    parser.add_argument(
        "--fetch-body",
        action="store_true",
        help="Fetch article body from URLs when description is missing (slower, may add noise)"
    )
    parser.add_argument(
        "--llm-verify",
        action="store_true",
        help="Enable LLM verification for signal corrections (selective: 0 or >2 signals only)"
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=50,
        help="Maximum LLM API calls allowed (default: 50, set 0 for unlimited)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("GDELT TRAINING DATA LABELING")
    print("="*80)
    print(f"Country:         {args.country}")
    print(f"Articles:        {args.articles}")
    print(f"Method:          {args.method}")
    if args.method == "embedding":
        print(f"Threshold:       {args.threshold}")
        print(f"Auto-calibrate:  {not args.no_calibrate}")
    print(f"Output:          {args.output}")
    print("="*80 + "\n")
    
    # Fetch and label
    labeled_df = fetch_and_label(
        country=args.country,
        num_articles=args.articles,
        method=args.method,
        similarity_threshold=args.threshold,
        auto_calibrate=not args.no_calibrate,
        fetch_body=args.fetch_body,
        llm_verify=args.llm_verify,
        max_llm_calls=args.max_llm_calls if args.max_llm_calls > 0 else None,
    )
    
    # Show statistics
    show_label_statistics(labeled_df)
    
    # Save labeled dataset
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    labeled_path = output_path / "labeled_dataset.parquet"
    labeled_df.write_parquet(labeled_path)
    print(f"\n✓ Saved labeled dataset to {labeled_path}")
    
    # Split train/val
    if not args.no_split:
        split_train_val(labeled_df, output_dir=args.output)
    
    # Final instructions
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review labels (especially low-confidence ones):")
    print(f"   python ml/utils/review_labels.py --input {output_path / 'train.parquet'}")
    print("\n2. Train the model:")
    print(f"   python ml/models/quick_finetune.py \\")
    print(f"     --train {output_path / 'train.parquet'} \\")
    print(f"     --val {output_path / 'val.parquet'} \\")
    print(f"     --epochs 3")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
