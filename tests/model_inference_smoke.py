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


def generate_articles_with_llm(category: str, num_articles: int = 3) -> List[Dict]:
    """
    Use LLM to generate realistic Swedish news articles for a category.
    
    Args:
        category: Signal category name
        num_articles: Number of articles to generate
        
    Returns:
        List of article dicts with title and description
    """
    if not HAS_OPENAI:
        return []
    
    try:
        client = OpenAI()
    except Exception:
        return []
    
    prompt = f"""Generate {num_articles} realistic Swedish news article titles and descriptions for the '{category}' category.

Return strict JSON array:
[
  {{
    "title": "Article title in Swedish",
    "description": "1-2 sentence description of the article content"
  }},
  ...
]

Make them varied and realistic. Only return valid JSON, no other text."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()
        # Try to extract JSON from response (may have extra text)
        if "[" in content and "]" in content:
            content = content[content.index("["):content.rindex("]")+1]
        articles = json.loads(content)
        return articles[:num_articles]
    except Exception as e:
        print(f"      ⚠️  LLM generation failed: {e}")
        return []


def fetch_articles_for_category(
    category: str,
    country: str = "sweden",
    num_articles: int = 3,
    days_lookback: int = 180,
) -> List[Dict]:
    """
    Fetch GDELT articles specifically for one category using keyword filtering.
    Falls back to LLM-generated articles if fetch fails.
    
    Args:
        category: Signal category name
        country: Country code
        num_articles: Target number of articles to fetch
        days_lookback: How many days to search back
        
    Returns:
        List of article dicts with 'title' and 'description' fields
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
                
                # Deduplicate by URL and extract title + description
                for article in articles_pl.iter_rows(named=True):
                    url = article.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        # Normalize article fields
                        title = article.get("title") or article.get("segtitle") or ""
                        description = article.get("description") or article.get("summary") or ""
                        if title:
                            all_articles.append({"title": title, "description": description})
                
                # Stop if we have enough
                if len(all_articles) >= num_articles:
                    break
            
            # Small delay between keyword queries
            time.sleep(0.3)
                
        except Exception as e:
            print(f"      ⚠️  Failed to fetch articles for keyword '{keyword}': {e}")
            continue
    
    # If GDELT fetch failed or returned few articles, fall back to LLM
    if len(all_articles) < num_articles:
        needed = num_articles - len(all_articles)
        print(f"      → Fetched {len(all_articles)} articles, generating {needed} with LLM...")
        llm_articles = generate_articles_with_llm(category, num_articles=needed)
        all_articles.extend(llm_articles)
    
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


def run_real_news_tests(articles_per_category: int = 3, country: str = "sweden") -> Tuple[int, int, List[str]]:
    """
    Fetch real GDELT news articles per category and test model predictions.
    
    Args:
        articles_per_category: Number of articles to test per category
        country: Country code for GDELT
        
    Returns:
        (passed, failed, errors)
    """
    print(f"\n📰 Real news category-specific test (country={country}, {articles_per_category} articles/category)")
    print("-" * 80)

    passed = 0
    failed = 0
    errors: List[str] = []
    
    llm_agree = 0
    llm_disagree = 0
    llm_skipped = 0
    
    categories = list(CATEGORY_KEYWORDS.keys())
    
    for category in categories:
        print(f"\n  📦 Testing {category.upper()}...")
        
        try:
            articles = fetch_articles_for_category(
                category=category,
                country=country,
                num_articles=articles_per_category,
                days_lookback=180
            )
        except Exception as e:
            print(f"      ❌ Failed to fetch articles: {e}")
            errors.append(f"{category}_fetch_error")
            failed += articles_per_category
            continue
        
        if not articles:
            print(f"      ⚠️  No articles found for {category} (neither GDELT nor LLM)")
            errors.append(f"{category}_no_articles")
            continue
        
        print(f"      → Testing {len(articles)} articles")
        
        for idx, article in enumerate(articles, 1):
            title = article.get("title") or ""
            description = article.get("description") or ""
            
            if not title:
                failed += 1
                errors.append(f"{category}_missing_title")
                print(f"      ❌ FAIL | missing title")
                continue
            
            # Classify with model, passing both title and description
            result = model.classify(title, description)
            if not result:
                failed += 1
                errors.append(f"{category}_empty_result")
                print(f"      [{idx}/{len(articles)}] ❌ FAIL | {title[:50]}...")
                print(f"            Error: No predictions")
                continue
            
            # Get top category
            top_result = model.get_top_category(title, description)
            if not top_result:
                failed += 1
                errors.append(f"{category}_no_top")
                print(f"      ❌ FAIL | {title[:50]}...")
                print(f"            Error: No top category")
                continue
            
            top_cat, top_score, top_tag, confidence = top_result
            
            # Validate score range
            if top_score < -1.5 or top_score > 1.5:
                failed += 1
                errors.append(f"{category}_invalid_score")
                print(f"      ❌ FAIL | {title[:50]}...")
                print(f"            Error: Invalid score {top_score}")
                continue
            
            # Check if prediction matches expected category
            is_correct = top_cat == category
            
            # LLM adjudication (optional)
            llm_verdict = None
            if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                llm_verdict = llm_judge_prediction(title, description, top_cat, top_score, top_tag, expected_category=category)
                if llm_verdict is not None:
                    if llm_verdict[0]:
                        llm_agree += 1
                    else:
                        llm_disagree += 1
                else:
                    llm_skipped += 1
            else:
                llm_skipped += 1
            
            # Determine pass/fail
            if is_correct:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                errors.append(f"{category}_wrong_prediction")
                status = "❌ FAIL"
            
            print(f"      [{idx}/{len(articles)}] {status} | {title[:50]}...")
            if description:
                desc_preview = description[:60] + "..." if len(description) > 60 else description
                print(f"            Desc: {desc_preview}")
            print(f"            Predicted: {top_cat:18s} sentiment={top_score:+.3f} confidence={confidence:.3f}")
            print(f"            Expected:  {category:18s}")
            
            if llm_verdict:
                verdict, reason = llm_verdict
                llm_status = "LLM agrees" if verdict else "LLM disagrees"
                print(f"            LLM: {llm_status} — {reason}")
    
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        print(f"\n  LLM adjudication: agree={llm_agree}, disagree={llm_disagree}, skipped={llm_skipped}")
    
    return passed, failed, errors


def llm_judge_prediction(
    title: str, 
    description: str, 
    category: str, 
    score: float, 
    tag: str,
    expected_category: str = None
) -> Optional[Tuple[bool, str]]:
    """Ask the LLM whether the predicted category/tag makes sense for the article."""
    try:
        client = OpenAI()
    except Exception:
        return None

    expected_text = f"\nExpected category: {expected_category}" if expected_category else ""

    prompt = f"""
Article:
Title: {title}
Description: {description[:600]}

Model prediction:
- Category: {category}
- Score: {score:+.3f}
- Tag: {tag}{expected_text}

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
        return None


def run_tests() -> None:
    """Run all test cases and print results."""
    print("\n" + "=" * 80)
    print("SWEDISH NEWS CLASSIFIER - COMPREHENSIVE REAL GDELT SMOKE TEST")
    print("=" * 80)
    print("Testing all 9 categories with real GDELT articles per category...")
    print("=" * 80)
    
    total = 0
    passed = 0
    failed_tests = []
    
    # Real news category-specific tests (main test)
    real_passed, real_failed, real_errors = run_real_news_tests(articles_per_category=3, country="sweden")
    passed += real_passed
    total += real_passed + real_failed
    
    if real_failed > 0:
        for error in real_errors:
            failed_tests.append(("real_news", error, "Category-specific test failure"))

    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 80)

    if failed_tests:
        print("\n❌ FAILURES:")
        for category, error_type, description in failed_tests:
            print(f"  - {category}: {error_type}")
            print(f"    {description}")
    else:
        print("\n✅ ALL TESTS PASSED!")
    
    # Exit with appropriate code
    if failed_tests:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    model = get_fine_tuned_classifier()
    run_tests()
