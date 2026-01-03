#!/usr/bin/env python3
"""
Quick smoke test for the fine-tuned Swedish news classifier.
Runs a few Swedish-language prompts and prints scored categories.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.inference import get_fine_tuned_classifier


def pretty_print(title: str, text: str, desc: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    result = model.classify(text, desc)
    if not result:
        print("(no predictions above threshold)")
        return
    for category, (score, tag) in sorted(result.items()):
        print(f"{category:18s}: {score:+.3f}  tag={tag}")


if __name__ == "__main__":
    model = get_fine_tuned_classifier()

    pretty_print(
        "Brand i Stockholm",
        "Stor brand på Kungsholmen i Stockholm",
        "Räddningstjänsten bekräftar kraftig rökutveckling och evakueringar",
    )

    pretty_print(
        "Kraftigt regn och översvämning",
        "Metrologerna varnarn för skyfall i Göteborg",
        "SMHI varnar för översvämningar och störningar i trafiken",
    )

    pretty_print(
        "Trafikstörning på E4",
        "Tung lastbil orsakar köer på E4 norr om Uppsala",
        "Trafikverket rapporterar långsamma köer och inställda bussar",
    )

    pretty_print(
        "Ekonomi och politik",
        "Riksbanken höjer räntan igen",
        "Oppositionen kräver stöd till hushall och företag",
    )
