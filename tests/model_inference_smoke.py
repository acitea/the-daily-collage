#!/usr/bin/env python3
"""
Comprehensive smoke test for the LLM-based Swedish news classifier.
Tests all 9 signal categories with multiple examples per category.
Automatically validates that:
  1. The category with highest score is correctly identified
  2. The tag is not null (properly labeled)

NOTE: Requires OPENAI_API_KEY environment variable to be set.
"""

from pathlib import Path
import sys
import os
from typing import Dict, Tuple, Optional, List

# Optional LLM dependency for adjudication
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:  # pragma: no cover - optional runtime dep
    HAS_OPENAI = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.inference import get_fine_tuned_classifier
from ml.ingestion.script import fetch_news


# Test cases: (title, description, expected_category)
TEST_CASES = {
    "emergencies": [
        (
            "Brand i Stockholm",
            "Stor brand på Kungsholmen i Stockholm",
            "Räddningstjänsten bekräftar kraftig rökutveckling och evakueringar",
        ),
        (
            "Jordskalv i Jämtland",
            "Jordskalv med magnitud 4.2 orsakar skador",
            "Flera hus har spruckit och människor evakueras från området",
        ),
    ],
    "crime": [
        (
            "Rån på bensinstation",
            "Beväpnat rån på OKQ8 i Västerås",
            "Polisen söker två män som flydde med okänd summa pengar",
        ),
        (
            "Inbrott i villa",
            "Inbrott i ett bostadshus i Nacka",
            "Tjuvar stal elektronik och smycken för flera hundra tusen kronor",
        ),
    ],
    "festivals": [
        (
            "Summerburst musikfestival",
            "Summerburst startar denna helg med internationella artister",
            "Över 50,000 besökare förväntas till festivalen på Gärdet",
        ),
        (
            "Stockholm Pride parade",
            "Pride-paraden marscherar genom Stockholm",
            "Tusentals människor samlas för att fira mångfald och inkludering",
        ),
    ],
    "transportation": [
        (
            "Trafikstörning på E4",
            "Tung lastbil orsakar köer på E4 norr om Uppsala",
            "Trafikverket rapporterar långsamma köer och inställda bussar",
        ),
        (
            "Tågtrafik försenad",
            "Signalfel orsakar förseningar på flera tåglinjer",
            "Pendlare uppmanas att planera längre restid",
        ),
    ],
    "weather_temp": [
        (
            "Värmebölja i Sverige",
            "Temperaturerna stiger till över 30 grader",
            "SMHI varnar för extrem värme i hela landet",
        ),
        (
            "Kall vinter i norra Sverige",
            "Temperaturer sjunker till minus 35 grader i Kiruna",
            "Invånare varnas att stanna inomhus",
        ),
    ],
    "weather_wet": [
        (
            "Kraftigt regn och översvämning",
            "SMHI varnar för skyfall i Göteborg",
            "SMHI varnar för översvämningar och störningar i trafiken",
        ),
        (
            "Snöstorm i fjällen",
            "Kraftig snöfall orsakar lavinvarning",
            "Vägar stängs och evakueringar genomförs i området",
        ),
    ],
    "sports": [
        (
            "Fotbollsmatch Sverige-Norge",
            "Sverige spelar hemmakvinnor mot Norge i VM-kval",
            "Omkring 35,000 fans väntas fylla Stockholms Stadion",
        ),
        (
            "Djurgårdens SM-titel",
            "Djurgården vinner SM-guld i ishockey",
            "Jubilande fans fyller gatorna i Stockholm efter segern",
        ),
    ],
    "economics": [
        (
            "Riksbanken höjer räntan",
            "Riksbanken höjer räntan med 0.5 procent",
            "Hypotekslån och lån för konsumenter blir dyrare",
        ),
        (
            "Arbetslöshet sjunker",
            "Arbetslösheten faller till 6.5 procent",
            "Arbetsmarknaden visar stark utveckling",
        ),
    ],
    "politics": [
        (
            "Regeringen presenterar ny klimatpolitik",
            "Statsminister presenterar ambitiös klimatplan",
            "Oppositionen kräver mer drastiska åtgärder",
        ),
        (
            "Partiledardebatt inför valet",
            "Ledarna från alla riksdagspartier debatterar",
            "Välfärd och skatter är huvudsakliga teman",
        ),
    ],
}


def test_case(title: str, text: str, desc: str, expected_category: str) -> Tuple[bool, Optional[str]]:
    """
    Test a single case and validate results.
    
    Returns:
        (passed: bool, error_msg: Optional[str])
    """
    result = model.classify(text, desc)
    
    if not result:
        return False, "No predictions above threshold"
    
    # Find category with highest score
    top_category, (top_score, top_tag) = max(result.items(), key=lambda x: x[1][0])
    
    # Check 1: Top category matches expected
    if top_category != expected_category:
        return False, f"Expected {expected_category}, got {top_category}"
    
    # Check 2: Tag is not null
    if top_tag is None or top_tag.strip() == "":
        return False, f"Tag is null or empty for category {top_category}"
    
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

        # Check top prediction exists and score within range
        top_cat, (top_score, top_tag) = max(result.items(), key=lambda x: x[1][0])
        if top_cat not in result or top_score < -1.5 or top_score > 1.5:
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
        print(f"        Top: {top_cat:18s} ({top_score:+.3f}) tag='{top_tag}'")
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
            result = model.classify(text, desc)
            if result:
                top_cat, (top_score, top_tag) = max(result.items(), key=lambda x: x[1][0])
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  {status} | {title}")
                print(f"        Top: {top_cat:18s} ({top_score:+.3f}) tag='{top_tag}'")
                
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
