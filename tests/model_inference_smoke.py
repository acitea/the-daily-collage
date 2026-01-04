#!/usr/bin/env python3
"""
Comprehensive smoke test for the fine-tuned Swedish news classifier.
Tests all 9 signal categories with multiple examples per category.
Automatically validates that:
  1. The category with highest score is correctly identified
  2. The tag is not null (properly labeled)
"""

from pathlib import Path
import sys
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.inference import get_fine_tuned_classifier


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
        (
            "Översvämning i västra Sverige",
            "Kraftigt regn orsakar översvämningar",
            "Vatten strömmar in i bostäder och vägar är blockerade",
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
            "Tjuvar stals elektronik och smycken för flera hundra tusen kronor",
        ),
        (
            "Grov misshandling",
            "Man misshandlad på Stureplan",
            "Fyra personer gripen misstänkta för grov misshandling",
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
        (
            "Lucia-firande på torget",
            "Traditionell lucia-tåg genom gamla stan",
            "Hundratals barn deltar i det årliga lucia-marchandet",
        ),
    ],
    "transportation": [
        (
            "Trafikstörning på E4",
            "Tung lastbil orsakar köer på E4 norr om Uppsala",
            "Trafikverket rapporterar långsamma köer och inställda bussar",
        ),
        (
            "Bågöbroöppning efter renovering",
            "Gamla Bågöbron öppnades igen efter två års renovering",
            "Trafikflödet genom bron förbättrades avsevärt",
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
        (
            "Mild höst i södra regionen",
            "Oväntligt varm höstväder i maj",
            "Växterna blommar tidigare än normalt",
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
            "Kraftig snöfall orsakar avalancher",
            "Vägar stängs och evakueringar genomförs i området",
        ),
        (
            "Hagelstorm i Skåne",
            "Stora hagel förstör grödor och bilar",
            "Jordbrukare rapporterar stora ekonomiska förluster",
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
        (
            "Tennismästerskapet i Båstad",
            "ATP-tennisen i Båstad attraherar världseliten",
            "Försäljningen av biljetter slog nytt rekord",
        ),
    ],
    "economics": [
        (
            "Riksbanken höjer räntan",
            "Riksbanken höjer räntan med 0.5 procent",
            "Hypotek och lån för konsumenter blir dyrare",
        ),
        (
            "Arbetslöshet sjunker",
            "Arbetslösheten faller till 6.5 procent",
            "Arbetsmarknaden visar stark utveckling",
        ),
        (
            "Börsuppgång",
            "Stockholmsbörsen stiger kraftigt på positiv global data",
            "Teknikaktier leder uppgången med över 5 procent",
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
            "Välfärd och skattar är huvudsakliga teman",
        ),
        (
            "Ny EU-förordning om dataöverföring",
            "Sverige implementerar ny EU-regel",
            "Kritiker säger att reglerna är för strikta",
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


def run_tests() -> None:
    """Run all test cases and print results."""
    print("\n" + "=" * 80)
    print("SWEDISH NEWS CLASSIFIER - COMPREHENSIVE SMOKE TEST")
    print("=" * 80)
    
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
    
    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{total} tests passed")
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
