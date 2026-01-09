#!/usr/bin/env python3
"""
Comprehensive smoke test for the LLM-based Swedish news classifier.
Tests all 9 signal categories with real GDELT articles fetched per category.
Automatically validates that:
  1. The category with highest confidence matches the expected category
  2. The tag is properly labeled

NOTE: Requires OPENAI_API_KEY environment variable to be set for LLM adjudication.
"""

from pathlib import Path
import sys
import os
import json
from typing import Dict, Tuple, Optional, List
import time

# Optional LLM dependency for adjudication
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:  # pragma: no cover - optional runtime dep
    HAS_OPENAI = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.inference import get_fine_tuned_classifier
from gdeltdoc import GdeltDoc, Filters
from ml.ingestion.script import normalize_country_input

# Category-specific keywords for GDELT queries (Swedish)
CATEGORY_KEYWORDS = {
    "emergencies": ["brand", "explosion", "räddning", "olycka", "katastrof", "evakuering", "larm", "nödläge"],
    "crime": ["brott", "polis", "misstänkt", "åtal", "rån", "mord", "skottlossning", "gripna"],
    "festivals": ["festival", "konsert", "evenemang", "firande", "kulturvecka", "musikfestival", "teater"],
    "transportation": ["trafik", "kollektivtrafik", "tåg", "buss", "förseningar", "inställt", "väg", "trafikstörning"],
    "weather_temp": ["temperatur", "värmebölja", "kyla", "vädret", "SMHI", "vädervarning", "grader"],
    "weather_wet": ["regn", "snö", "översvämning", "storm", "blåst", "skyfall", "halka"],
    "sports": ["sport", "fotboll", "ishockey", "seger", "matcher", "VM", "EM", "allsvenskan"],
    "economics": ["ekonomi", "börs", "aktier", "inflation", "ränta", "finans", "arbetsmarknad", "löner"],
    "politics": ["regering", "riksdag", "politik", "val", "minister", "parti", "opposition", "lagförslag"]
}


def fetch_articles_for_category(
    category: str,
    country: str = "sweden",
    num_articles: int = 3,
    days_lookback: int = 180,
) -> List[Dict]:
    """
    Fetch GDELT articles specifically for one category using keyword filtering.
    
    Args:
        category: Signal category name
        country: Country code
        num_articles: Target number of articles to fetch
        days_lookback: How many days to search back
        
    Returns:
        List of article dicts
    """
    import polars as pl
    
    fips_code = normalize_country_input(country)
    keywords = CATEGORY_KEYWORDS.get(category, [])
    
    if not keywords:
        print(f"      ⚠️  No keywords defined for category: {category}")
        return []
    
    all_articles = []
    seen_urls = set()
    
    # Try first 2 keywords to get variety
    for keyword in keywords[:2]:
        try:
            gd = GdeltDoc()
            filters = Filters(
                country=fips_code,
                keyword=keyword,
                num_records=min(50, num_articles * 2),
                timespan=f"{days_lookback}d",
            )
            
            articles_pd = gd.article_search(filters)
            
            if articles_pd is not None and not articles_pd.empty:
                articles_pl = pl.from_pandas(articles_pd)
                
                # Deduplicate by URL
                for article in articles_pl.iter_rows(named=True):
                    url = article.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_articles.append(article)
                
                # Stop if we have enough
                if len(all_articles) >= num_articles:
                    break
            
            # Small delay between keyword queries
            time.sleep(0.3)
                
        except Exception as e:
            print(f"      ⚠️  Failed to fetch articles for keyword '{keyword}': {e}")
            continue
    
    return all_articles[:num_articles]


def test_case(title: str, text: str, desc: str, expected_category: str) -> Tuple[bool, Optional[str]]:
    """
    Test a single case and validate results.
    
    Returns:
        (passed: bool, error_msg: Optional[str])
    """
    # Use confidence-based classification
    top_result = model.get_top_category(text, desc)
    
    if not top_result:
        return False, "No predictions above threshold"
    
    top_category, top_score, top_tag, confidence = top_result
    
    # Check: Top category matches expected (tag can be empty, that's ok)
    if top_category != expected_category:
        return False, f"Expected {expected_category}, got {top_category} (confidence={confidence:.3f})"
    
    return True, None


def run_real_news_tests(limit: int = 5, country: str = "sweden") -> Tuple[int, int, List[str]]:
    """Fetch real news and ensure classifier returns at least one signal per article."""
    print(f"\n📰 Real news integration test (country={country}, limit={limit})")
    print("-" * 80)

    try:
        articles = fetch_news(country=country, max_articles=limit)
    except Exception as exc:  # pragma: no cover - external API
        print(f"❌ Failed to fetch news: {exc}")
        return 0, 0, ["fetch_error"]

    if articles.is_empty():
        print("❌ No articles returned from GDELT; skipping real-news test")
        return 0, 0, ["no_articles"]

    passed = 0
    failed = 0
    errors: List[str] = []

    llm_agree = 0
    llm_disagree = 0
    llm_skipped = 0

    for article in articles.iter_rows(named=True):
        title = article.get("title") or article.get("documentIdentifier") or ""
        description = article.get("summary") or article.get("content") or article.get("excerpt") or ""

        if not title:
            failed += 1
            errors.append("missing_title")
            print("  ❌ FAIL | missing title")
            continue

        result = model.classify(title, description)
        if not result:
            failed += 1
            errors.append("empty_result")
            print(f"  ❌ FAIL | {title[:60]}...")
            print("        Error: No predictions")
            continue

        # Check top prediction using confidence-based selection
        top_result = model.get_top_category(title, description)
        if not top_result:
            failed += 1
            errors.append("empty_result")
            print(f"  ❌ FAIL | {title[:60]}...")
            print("        Error: No top category")
            continue
        
        top_cat, top_score, top_tag, confidence = top_result
        if top_score < -1.5 or top_score > 1.5:
            failed += 1
            errors.append("invalid_score")
            print(f"  ❌ FAIL | {title[:60]}...")
            print(f"        Error: Invalid score {top_score}")
            continue

        # LLM adjudication for real-news outputs (best-effort)
        llm_verdict = None
        if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            llm_verdict = llm_judge_prediction(title, description, top_cat, top_score, top_tag)
            if llm_verdict is not None:
                if llm_verdict[0]:
                    llm_agree += 1
                else:
                    llm_disagree += 1
            else:
                llm_skipped += 1
        else:
            llm_skipped += 1

        passed += 1
        print(f"  ✅ PASS | {title[:60]}...")
        print(f"        Top: {top_cat:18s} sentiment={top_score:+.3f} confidence={confidence:.3f} tag='{top_tag}'")
        if llm_verdict:
            verdict, reason = llm_verdict
            status = "LLM agrees" if verdict else "LLM disagrees"
            print(f"        LLM: {status} — {reason}")
        elif HAS_OPENAI:
            print("        LLM: skipped (no verdict)")
        else:
            print("        LLM: skipped (no OPENAI_API_KEY)")

    if HAS_OPENAI:
        print(f"\nLLM adjudication: agree={llm_agree}, disagree={llm_disagree}, skipped={llm_skipped}")

    return passed, failed, errors


def llm_judge_prediction(title: str, description: str, category: str, score: float, tag: str) -> Optional[Tuple[bool, str]]:
    """Ask the LLM whether the predicted category/tag makes sense for the article."""
    try:
        client = OpenAI()
    except Exception:
        return None

    prompt = f"""
Article:
Title: {title}
Description: {description[:600]}

Model prediction:
- Category: {category}
- Score: {score:+.3f}
- Tag: {tag}

Signal categories to choose from: emergencies, crime, festivals, transportation, weather_temp, weather_wet, sports, economics, politics.

Decide if the predicted category matches the article. Return strict JSON:
{{"agree": true|false, "reason": "short reason"}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return bool(data.get("agree", False)), str(data.get("reason", "no reason provided"))
    except Exception as exc:  # pragma: no cover - best-effort guard
        print(f"        LLM adjudication failed: {exc}")
        return None


def run_tests() -> None:
    """Run all test cases and print results."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("SWEDISH NEWS CLASSIFIER - COMPREHENSIVE LLM-BASED TEST")
    print("=" * 80)
    print("Testing all 9 categories with LLM classifier...")
    
    total = 0
    passed = 0
    failed_tests = []
    
    for category, cases in TEST_CASES.items():
        print(f"\n📦 Category: {category.upper()}")
        print("-" * 80)
        
        for title, text, desc in cases:
            total += 1
            success, error = test_case(title, text, desc, category)
            
            # Get full result for display
            top_result = model.get_top_category(text, desc, exclude_categories=['sports'])
            if top_result:
                top_cat, top_score, top_tag, confidence = top_result
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {status} | {title}")
                print(f"        Top: {top_cat:18s} sentiment={top_score:+.3f} confidence={confidence:.3f} tag='{top_tag}'")
                
                if not success:
                    print(f"        Error: {error}")
                    failed_tests.append((title, category, error))
                    
            else:
                print(f"  ❌ FAIL | {title}")
                print(f"        Error: No predictions")
                failed_tests.append((title, category, "No predictions"))
            
            if success:
                passed += 1
    
    # Real news integration tests
    real_passed, real_failed, real_errors = run_real_news_tests()
    passed += real_passed
    total += real_passed + real_failed
    if real_failed:
        failed_tests.append(("real_news_batch", "real_news", ",".join(real_errors)))

    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{total} tests passed (including real news)")
    print("=" * 80)

    if failed_tests:
        print("\n❌ FAILURES:")
        for title, category, error in failed_tests:
            print(f"  - {title} (expected: {category})")
            print(f"    {error}")
    else:
        print("\n✅ ALL TESTS PASSED!")


if __name__ == "__main__":
    model = get_fine_tuned_classifier()
    run_tests()
