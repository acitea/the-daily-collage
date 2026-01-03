"""
Quick bootstrap data collection in under 1 hour.
Fetches articles from GDELT and auto-labels with keywords.
"""

import sys
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news, fetch_news_batched
from ml.ingestion.hopsworks_pipeline import classify_article, SIGNAL_CATEGORIES

def quick_bootstrap(
    countries: list = ["sweden"],
    articles_per_country: int = 500,
    output_dir: str = "ml/data",
    use_batching: bool = True,
    batch_size: int = 250,
    days_lookback: int = 360,
):
    """
    Bootstrap 500+ labeled articles using batched GDELT fetching.
    Uses multiple requests to overcome the 250-article-per-request GDELT limit.
    
    Args:
        countries: List of countries to fetch from (default: ["sweden"])
        articles_per_country: Target articles per country (e.g., 500)
        output_dir: Output directory for parquet files
        use_batching: If True, use fetch_news_batched to get more articles
        batch_size: Articles per GDELT request (max 250)
        days_lookback: Days to look back for articles (spreads requests across this range)
    """
    print("🚀 Starting quick bootstrap with batched fetching...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_articles = []
    
    for country in countries:
        print(f"\n📰 Fetching {articles_per_country} articles from {country}...")
        try:
            if use_batching:
                # Use batched fetching to overcome 250-article limit
                df = fetch_news_batched(
                    country=country,
                    total_articles=articles_per_country,
                    batch_size=batch_size,
                    days_lookback=days_lookback,
                    batch_delay=0.5  # 500ms delay between requests
                )
            else:
                # Fallback to single request (max 250 articles)
                articles_to_fetch = min(articles_per_country, 250)
                df = fetch_news(country=country, max_articles=articles_to_fetch)
        except Exception as e:
            print(f"⚠️  Error fetching {country}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if df.is_empty():
            print(f"⚠️  No articles found for {country}")
            continue

        print(f"✓ Got {len(df)} articles, classifying...")

        for idx, row in enumerate(df.iter_rows(named=True)):
            title = row.get("title", "")
            description = row.get("description", "")

            # Auto-classify using keywords
            signals = classify_article(title, description)

            article_data = {
                "title": title,
                "description": description,
                "url": row.get("url", ""),
                "source": row.get("source", ""),
            }

            # Add labels (0.0 / "" for missing categories)
            for category in SIGNAL_CATEGORIES:
                if category in signals:
                    score, tag = signals[category]
                    article_data[f"{category}_score"] = score
                    article_data[f"{category}_tag"] = tag
                else:
                    article_data[f"{category}_score"] = 0.0
                    article_data[f"{category}_tag"] = ""

            all_articles.append(article_data)

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(df)} classified")

    print(f"\n✓ Total articles: {len(all_articles)}")

    # Create DataFrame
    if not all_articles:
        print(f"\n❌ No articles were collected. Check GDELT API and network connectivity.")
        # Create empty parquet files
        empty_df = pl.DataFrame({"title": [], "description": []})
        empty_df.write_parquet(output_path / "train_bootstrap.parquet")
        empty_df.write_parquet(output_path / "val_bootstrap.parquet")
        return output_path / "train_bootstrap.parquet", output_path / "val_bootstrap.parquet"

    df_all = pl.DataFrame(all_articles)

    if df_all.is_empty():
        print(f"\n❌ No articles were collected. Check GDELT API and network connectivity.")
        return output_path / "train_bootstrap.parquet", output_path / "val_bootstrap.parquet"

    # Shuffle
    df_all = df_all.sample(fraction=1.0, shuffle=True)

    # Split: 70% train, 30% val
    n = len(df_all)
    split_idx = int(0.7 * n)

    train_df = df_all[:split_idx]
    val_df = df_all[split_idx:]

    # Save
    train_path = output_path / "train_bootstrap.parquet"
    val_path = output_path / "val_bootstrap.parquet"

    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)

    print(f"\n✓ Saved training data: {train_path}")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"\n✓ Saved validation data: {val_path}")

    # Print sample statistics
    if len(df_all) > 0:
        print("\n📊 Signal distribution:")
        for cat in SIGNAL_CATEGORIES:
            non_zero = df_all.filter(pl.col(f"{cat}_score") != 0.0).height
            pct = (non_zero / len(df_all)) * 100
            print(f"  {cat:20s}: {non_zero:4d} articles ({pct:5.1f}%)")
    else:
        print("\n⚠️  No data to show distribution")

    return train_path, val_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--articles-per-country", type=int, default=250, help="Articles to fetch (max 250 per GDELT API limit)")
    parser.add_argument("--output-dir", default="ml/data")

    args = parser.parse_args()

    # Always use Sweden only
    train_path, val_path = quick_bootstrap(
        countries=["sweden"],
        articles_per_country=args.articles_per_country,
        output_dir=args.output_dir,
    )

    print(f"\n✅ Bootstrap complete!")
    print(f"   Train: {train_path}")
    print(f"   Val:   {val_path}")
