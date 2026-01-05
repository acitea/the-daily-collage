"""
Complete labeling pipeline with BATCH LLM verification

This script provides an end-to-end solution for creating high-quality
training data for the fine-tuned BERT model, with efficient batch LLM processing.

Usage:
    python ml/utils/label_dataset.py --articles 500 --country sweden --llm-verify
"""

import argparse
import sys
import json
import logging
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

# Optional LLM dependency
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news, fetch_news_batched
from ml.utils.embedding_labeling import classify_article_embedding
from ml.ingestion.hopsworks_pipeline import SIGNAL_CATEGORIES
from ml.utils.intensity_calibration import calibrate_article_labels

logger = logging.getLogger(__name__)


def classify_articles_batch_with_llm(
    articles: list,
    llm_call_count: dict = None,
    max_llm_calls: int = None,
) -> dict:
    """
    Use LLM to verify and correct multiple articles in a single batch call.
    
    Args:
        articles: List of {title, description, signals, detected_count} dicts
        llm_call_count: Dict tracking {'calls': N} for cost control
        max_llm_calls: Maximum LLM API calls allowed
        
    Returns:
        Dict mapping article index to corrected signals
    """
    if llm_call_count is None:
        llm_call_count = {'calls': 0}
    
    if max_llm_calls is not None and llm_call_count['calls'] >= max_llm_calls:
        logger.warning(f"LLM API limit reached ({max_llm_calls} calls). Returning uncorrected signals.")
        return {i: article["signals"] for i, article in enumerate(articles)}
    
    if not articles:
        return {}
    
    # Increment call counter
    llm_call_count['calls'] += 1
    
    try:
        from openai import OpenAI
        
        # Build batch prompt with all articles
        articles_text = ""
        for i, article in enumerate(articles):
            title = article["title"]
            desc = article["description"][:500]  # Truncate to 500 chars
            signals = article["signals"]
            
            current_signals = {}
            for cat in SIGNAL_CATEGORIES:
                if cat in signals:
                    score, tag = signals[cat]
                    if abs(score) > 0.01:
                        current_signals[cat] = {"score": float(score), "tag": str(tag)}
            
            articles_text += f"""
ARTICLE {i+1}:
Title: {title}
Description: {desc}...
Current Classifications: {json.dumps(current_signals) if current_signals else '(no signals)'}
---"""
        
        prompt = f"""Analyze these {len(articles)} news articles and verify all signal detections.

{articles_text}

Signal Categories:
- emergencies: natural disasters, accidents, extreme weather impacts (score: -1 to -0.5 for severe)
- crime: criminal incidents, violence, arrests (score: -1 to -0.5 for serious)
- festivals: cultural events, celebrations, entertainment (score: 0.3 to 1 for positive)
- transportation: traffic accidents, transit issues (score: -0.5 to 0 for problems)
- weather_temp: temperature extremes, heat waves, cold snaps (score: -0.8 to 0)
- weather_wet: rain, flooding, snow, storms (score: -0.8 to 0)
- sports: sporting events, competitions, achievements (score: -0.5 to 1)
- economics: markets, business, employment (score: -0.8 to 1)
- politics: elections, policy, governance (score: -0.5 to 1)

For each article:
1. Does each detected category make sense?
2. Is the score magnitude correct? (major events: -0.8 to -1 or 0.8 to 1; minor: -0.3 to 0 or 0.3)
3. Is the direction correct? (negative=bad/problem, positive=good/success)
4. Are any signals completely wrong?
5. Are any major signals missed?

Return ONLY valid JSON (one object per article):
{{
  "results": [
    {{
      "article": 1,
      "corrections": {{"category": {{"score": -0.8, "tag": "tag"}}, ...}},
      "removals": ["wrong_category", ...],
      "additions": {{"category": {{"score": 0.7, "tag": "tag"}}, ...}}
    }},
    ...
  ]
}}

If no changes for an article, use empty dicts: {{"corrections": {{}}, "removals": [], "additions": {{}}}}"""

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.debug(f"LLM batch verification #{llm_call_count['calls']}: {len(articles)} articles")
        
        # Parse response
        result = json.loads(result_text)
        batch_corrections = {}
        
        for item in result.get("results", []):
            article_idx = item.get("article", 1) - 1  # Convert to 0-indexed
            
            if article_idx < len(articles):
                article = articles[article_idx]
                verified_signals = article["signals"].copy()
                
                # Apply corrections
                for category, correction in item.get("corrections", {}).items():
                    if category in verified_signals:
                        new_score = correction.get("score", verified_signals[category][0])
                        new_tag = correction.get("tag", verified_signals[category][1])
                        verified_signals[category] = (float(new_score), str(new_tag))
                        logger.debug(f"  Article {article_idx+1} → Corrected {category}")
                
                # Apply removals
                for category in item.get("removals", []):
                    if category in verified_signals:
                        verified_signals[category] = (0.0, "")
                        logger.debug(f"  Article {article_idx+1} → Removed {category}")
                
                # Apply additions
                for category, signal_info in item.get("additions", {}).items():
                    if category not in verified_signals or abs(verified_signals[category][0]) < 0.01:
                        score = signal_info.get("score", 0.5)
                        tag = signal_info.get("tag", category)
                        verified_signals[category] = (float(score), str(tag))
                        logger.debug(f"  Article {article_idx+1} → Added {category}")
                
                batch_corrections[article_idx] = verified_signals
        
        # Fill in missing articles (no corrections)
        for i in range(len(articles)):
            if i not in batch_corrections:
                batch_corrections[i] = articles[i]["signals"]
        
        return batch_corrections
        
    except Exception as e:
        logger.warning(f"LLM batch verification failed: {e}")
        return {i: article["signals"] for i, article in enumerate(articles)}


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
    Fetch articles and auto-label them with optional batch LLM verification.
    
    Args:
        country: Country code (sweden, us, etc.)
        num_articles: Number of articles to fetch
        method: "embedding" or "keywords"
        similarity_threshold: Minimum similarity for embedding method
        auto_calibrate: If True, automatically convert similarity to intensity
        fetch_body: If True, fetch article body from URL when description is missing
        llm_verify: If True, use LLM to verify and correct classifications (batch processing)
        max_llm_calls: Maximum LLM API calls allowed (None = unlimited, default: 50)
        
    Returns:
        Labeled DataFrame
    """
    # Set default API limit
    if llm_verify and max_llm_calls is None:
        max_llm_calls = 50
    
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
            print(f"   (LLM verification: batch processing, max {max_llm_calls} batch calls)")
    else:
        print("🔤 Auto-labeling with keyword-based classification...")
    
    # Phase 1: Classify all articles
    print("\n📊 Phase 1: Embedding-based classification...")
    all_articles = []
    articles_needing_verification = []
    
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Classifying"):
        title = row.get("title", "")
        description = (
            row.get("description")
            or row.get("content")
            or row.get("summary")
            or ""
        )
        
        if not description and fetch_body:
            fetched_body = _fetch_page_content(row.get("url", ""))
            if fetched_body:
                description = fetched_body
        
        signals = {}
        if method == "embedding":
            try:
                signals = classify_article_embedding(
                    title=title,
                    description=description,
                    similarity_threshold=similarity_threshold,
                    relative_threshold=0.70
                )
                
                if auto_calibrate:
                    signals = calibrate_article_labels(
                        signals=signals,
                        title=title,
                        description=description
                    )
            except Exception as e:
                print(f"\n⚠️  Embedding failed: {e}")
                method = "keywords"
        
        if method == "keywords":
            from ml.ingestion.hopsworks_pipeline import classify_article
            signals = classify_article(title, description, method="keywords")
        
        # Check if needs LLM verification
        detected_count = sum(1 for cat, (score, tag) in signals.items() if abs(score) > 0.01)
        needs_verification = llm_verify and (detected_count == 0 or detected_count > 2)
        
        article_data = {
            "row": row,
            "title": title,
            "description": description,
            "signals": signals,
            "detected_count": detected_count,
        }
        
        all_articles.append(article_data)
        
        if needs_verification:
            articles_needing_verification.append(article_data)
    
    # Phase 2: Batch LLM verification
    if articles_needing_verification:
        print(f"\n🤖 Phase 2: LLM batch verification ({len(articles_needing_verification)} articles)...")
        batch_results = classify_articles_batch_with_llm(
            articles=articles_needing_verification,
            llm_call_count=llm_call_count,
            max_llm_calls=max_llm_calls,
        )
        
        # Apply batch results back
        for article_idx, corrected_signals in batch_results.items():
            articles_needing_verification[article_idx]["signals"] = corrected_signals
    
    # Phase 3: Build output
    print("\n📝 Building output...")
    labeled_rows = []
    
    for article_data in tqdm(all_articles, desc="Building rows"):
        row = article_data["row"]
        signals = article_data["signals"]
        tone = row.get("tone")
        
        labeled_row = {
            "title": row.get("title", ""),
            "description": row.get("description", ""),
            "tone": float(tone) if tone is not None else None,
            "url": row.get("url", ""),
            "source": row.get("source", ""),
            "date": str(row.get("date", "")),
        }
        
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
    
    # Show API usage
    if llm_verify and llm_call_count['calls'] > 0:
        estimated_cost = llm_call_count['calls'] * 0.05  # ~$0.05 per batch call
        print(f"📊 LLM API Usage: {llm_call_count['calls']} batch call(s) (estimated cost: ${estimated_cost:.2f})")
    
    return labeled_df


def _fetch_page_content(url: str) -> str:
    """
    Best-effort fetch of article body, filtering out navigation and boilerplate.
    """
    if not url or requests is None or BeautifulSoup is None:
        return ""
    try:
        resp = requests.get(url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "iframe", "form", "button"]):
            tag.decompose()
        
        for pattern in ["nav", "menu", "sidebar", "footer", "header", "ad", "banner", "cookie"]:
            for elem in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                elem.decompose()
            for elem in soup.find_all(id=lambda x: x and pattern in x.lower()):
                elem.decompose()
        
        article_content = None
        for selector in ["article", "main", '[role="main"]', ".article-body", ".post-content"]:
            article_content = soup.select_one(selector)
            if article_content:
                break
        
        text_source = article_content if article_content else soup.body
        if text_source:
            text = text_source.get_text(separator=" ", strip=True)
            text = " ".join(text.split())
            return text[:1500]
        
        return ""
    except Exception:
        return ""


def show_label_statistics(df: pl.DataFrame):
    """Display labeling statistics."""
    print("\n" + "="*80)
    print("LABEL STATISTICS")
    print("="*80)
    
    print(f"\nTotal articles: {len(df)}")
    
    print("\nSignal distribution:")
    for category in SIGNAL_CATEGORIES:
        non_zero = df.filter(pl.col(f"{category}_score").abs() > 0.01).height
        percentage = (non_zero / len(df)) * 100 if len(df) > 0 else 0
        
        scores = df[f"{category}_score"].to_list()
        non_zero_scores = [s for s in scores if abs(s) > 0.01]
        avg_score = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0
        
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
    """Split dataset into train and validation sets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
    
    train_size = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:]
    
    train_path = output_path / "train.parquet"
    val_path = output_path / "val.parquet"
    
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)
    
    print(f"\n✓ Train: {len(train_df)} articles → {train_path}")
    print(f"✓ Val:   {len(val_df)} articles → {val_path}")
    
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Label GDELT articles for BERT fine-tuning")
    parser.add_argument("--articles", type=int, default=500, help="Number of articles to fetch (default: 500)")
    parser.add_argument("--country", type=str, default="sweden", help="Country to fetch from (default: sweden)")
    parser.add_argument("--method", type=str, choices=["embedding", "keywords"], default="embedding", help="Labeling method (default: embedding)")
    parser.add_argument("--threshold", type=float, default=0.20, help="Similarity threshold (default: 0.20)")
    parser.add_argument("--output", type=str, default="data", help="Output directory (default: data/)")
    parser.add_argument("--no-split", action="store_true", help="Don't split into train/val")
    parser.add_argument("--no-calibrate", action="store_true", help="Disable automatic intensity calibration")
    parser.add_argument("--fetch-body", action="store_true", help="Fetch article body from URLs")
    parser.add_argument("--llm-verify", action="store_true", help="Enable LLM verification (batch processing)")
    parser.add_argument("--max-llm-calls", type=int, default=50, help="Maximum LLM batch calls (default: 50)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("GDELT TRAINING DATA LABELING (BATCH LLM)")
    print("="*80)
    print(f"Country:         {args.country}")
    print(f"Articles:        {args.articles}")
    print(f"Method:          {args.method}")
    if args.method == "embedding":
        print(f"Threshold:       {args.threshold}")
        print(f"Auto-calibrate:  {not args.no_calibrate}")
    if args.llm_verify:
        max_calls = args.max_llm_calls if args.max_llm_calls > 0 else "unlimited"
        print(f"LLM verification: enabled (batch, max {max_calls} calls)")
    else:
        print(f"LLM verification: disabled")
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
    print("\n1. Review labels:")
    print(f"   python ml/utils/review_labels.py --input {output_path / 'train.parquet'}")
    print("\n2. Train the model:")
    print(f"   python ml/models/quick_finetune.py \\")
    print(f"     --train {output_path / 'train.parquet'} \\")
    print(f"     --val {output_path / 'val.parquet'} \\")
    print(f"     --epochs 3")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
