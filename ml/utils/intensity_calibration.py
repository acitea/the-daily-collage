"""
Automatic intensity calibration for embedding-based labels.

Converts semantic similarity scores (0-1) into proper intensity scores (-1 to +1)
that reflect the actual severity/impact of incidents.
"""

import re
from typing import Dict, Tuple

# Severity indicators that boost intensity
HIGH_SEVERITY_KEYWORDS = {
    # Universal high-severity words
    "devastat", "catastroph", "massive", "major", "severe", "critical", "emergency",
    "widespread", "extensive", "significant", "serious", "large-scale",
    # Swedish equivalents
    "allvarlig", "omfattande", "stor", "massiv", "kritisk", "akut", "betydande",
}

MEDIUM_SEVERITY_KEYWORDS = {
    "moderate", "notable", "considerable", "substantial",
    "medel", "märkbar", "betydlig",
}

LOW_SEVERITY_KEYWORDS = {
    "minor", "small", "slight", "limited", "brief", "minimal", "negligible",
    "mindre", "liten", "begränsad", "kort", "minimal",
}

# Response magnitude indicators (suggest high intensity)
RESPONSE_INDICATORS = {
    # Emergency response scale
    r"\d{2,}\s*(?:firefighters|police|officers|units|trucks|crews)": 0.15,  # "50 firefighters"
    r"(?:multi-alarm|multiple\s+units|full\s+response)": 0.2,
    r"(?:evacuation|evacuated|evacuate)": 0.15,
    r"(?:state of emergency|disaster declared)": 0.25,
    # Traffic/transport impact
    r"(?:highway|motorway|freeway)\s+(?:closed|blocked|shutdown)": 0.15,
    r"(?:major|significant)\s+(?:delays|disruption)": 0.1,
    # Casualty indicators
    r"(?:fatal|fatality|death|deaths|killed)": 0.2,
    r"\d+\s+(?:injured|hurt|wounded)": 0.15,
    r"(?:serious|critical)\s+(?:injuries|condition)": 0.15,
    # Property damage
    r"(?:destroyed|demolished|gutted|leveled)": 0.2,
    r"(?:extensive|major|significant)\s+damage": 0.15,
    # Weather severity
    r"(?:hurricane|tornado|earthquake|tsunami)": 0.25,
    r"(?:record|historic|unprecedented)": 0.15,
}

# Mitigation indicators (reduce intensity)
MITIGATION_INDICATORS = {
    r"(?:no\s+(?:injuries|casualties|damage))": -0.15,
    r"(?:quickly\s+(?:contained|extinguished|resolved))": -0.1,
    r"(?:minor|minimal)\s+(?:damage|impact)": -0.1,
    r"(?:under control|contained)": -0.1,
}


def extract_severity_modifiers(text: str) -> float:
    """
    Extract severity adjustment from text content.
    
    Args:
        text: Combined article text (title + description)
        
    Returns:
        Adjustment value (-0.3 to +0.3)
    """
    text_lower = text.lower()
    adjustment = 0.0
    
    # Check for explicit severity keywords
    if any(kw in text_lower for kw in HIGH_SEVERITY_KEYWORDS):
        adjustment += 0.15
    elif any(kw in text_lower for kw in MEDIUM_SEVERITY_KEYWORDS):
        adjustment += 0.05
    elif any(kw in text_lower for kw in LOW_SEVERITY_KEYWORDS):
        adjustment -= 0.15
    
    # Check for response magnitude indicators
    for pattern, boost in RESPONSE_INDICATORS.items():
        if re.search(pattern, text_lower):
            adjustment += boost
    
    # Check for mitigation indicators
    for pattern, reduction in MITIGATION_INDICATORS.items():
        if re.search(pattern, text_lower):
            adjustment += reduction  # reduction is already negative
    
    # Clamp to reasonable range
    return max(-0.3, min(0.3, adjustment))


def calibrate_intensity_score(
    category: str,
    similarity_score: float,
    tag: str,
    text: str
) -> float:
    """
    Convert embedding similarity (0-1) to intensity score (-1 to +1).
    
    Strategy:
    1. Base score from similarity (scaled and centered)
    2. Adjust based on severity keywords
    3. Apply category-specific scaling
    4. Determine positive vs negative impact
    
    Args:
        category: Signal category (e.g., "emergencies")
        similarity_score: Raw cosine similarity (0-1)
        tag: Event tag (e.g., "fire")
        text: Article text for context analysis
        
    Returns:
        Intensity score (-1 to +1)
    """
    # Step 1: Base intensity from similarity
    # Similarity 0.35 (threshold) → intensity 0.3
    # Similarity 0.6 → intensity 0.6
    # Similarity 0.85+ → intensity 0.85+
    base_intensity = similarity_score
    
    # Step 2: Extract severity modifiers from text
    severity_adjustment = extract_severity_modifiers(text)
    
    # Step 3: Apply adjustment
    adjusted_intensity = base_intensity + severity_adjustment
    
    # Step 4: Category-specific adjustments
    intensity = apply_category_scaling(category, tag, adjusted_intensity, text)
    
    # Step 5: Determine sign (positive vs negative impact)
    # Most categories are negative events, but some have positive signals
    if category in ["festivals", "sports"]:
        # Positive events - check for negative context
        if any(word in text.lower() for word in ["cancel", "postpone", "riot", "violence", "clash"]):
            intensity = -abs(intensity)  # Make negative
        else:
            intensity = abs(intensity)   # Keep positive
    elif category == "economics":
        # Can be positive or negative
        if any(word in text.lower() for word in ["growth", "gain", "rise", "recovery", "boom", "success"]):
            intensity = abs(intensity)
        else:
            intensity = -abs(intensity)  # Default negative (problems)
    else:
        # Emergencies, crime, transportation issues, etc. are negative
        intensity = -abs(intensity)
    
    # Clamp to valid range
    return max(-1.0, min(1.0, intensity))


def apply_category_scaling(
    category: str,
    tag: str,
    intensity: float,
    text: str
) -> float:
    """
    Apply category-specific intensity scaling.
    
    Different categories have different intensity distributions:
    - Emergencies: Can reach very high intensity (0.9+)
    - Crime: Usually moderate (0.5-0.7)
    - Weather: Highly variable based on type
    - etc.
    """
    text_lower = text.lower()
    
    if category == "emergencies":
        # Emergencies can be very severe
        if tag == "fire":
            if any(kw in text_lower for kw in ["structure", "building", "multi-alarm", "major"]):
                intensity = min(0.95, intensity + 0.1)
        elif tag == "earthquake":
            # Always high intensity
            intensity = max(0.75, intensity)
        elif tag == "explosion":
            intensity = max(0.8, intensity)
        elif tag == "flood":
            intensity = max(0.85, intensity)  # Floods are very impactful
    
    elif category == "crime":
        # Crime varies but rarely exceeds 0.8
        if tag in ["assault", "robbery"]:
            intensity = min(0.75, intensity)
        elif tag == "theft":
            intensity = min(0.6, intensity)
    
    elif category == "transportation":
        # Transportation issues usually moderate
        if tag == "accident":
            # Accidents can be severe if fatal
            if any(kw in text_lower for kw in ["fatal", "death", "killed"]):
                intensity = min(0.9, intensity + 0.15)
            else:
                intensity = min(0.7, intensity)
        elif tag in ["traffic", "congestion"]:
            intensity = min(0.65, intensity)
    
    elif category == "weather_wet":
        # Precipitation events vary widely
        if tag == "flood":
            intensity = max(0.8, intensity)  # Always serious
        elif tag == "storm":
            intensity = max(0.65, intensity)
        elif tag == "rain":
            intensity = min(0.5, intensity)  # Usually mild
    
    elif category == "weather_temp":
        # Temperature extremes
        if any(kw in text_lower for kw in ["record", "extreme", "dangerous"]):
            intensity = min(0.8, intensity + 0.1)
        else:
            intensity = min(0.6, intensity)
    
    elif category in ["festivals", "sports"]:
        # Positive events - scale based on magnitude
        if any(kw in text_lower for kw in ["championship", "final", "victory", "win"]):
            intensity = max(0.7, intensity)
        else:
            intensity = min(0.6, intensity)
    
    elif category == "economics":
        # Economic news usually moderate impact
        if any(kw in text_lower for kw in ["crash", "collapse", "crisis"]):
            intensity = max(0.75, intensity)
        else:
            intensity = min(0.65, intensity)
    
    elif category == "politics":
        # Political events vary
        if tag == "protest":
            if any(kw in text_lower for kw in ["violent", "riot", "clash"]):
                intensity = min(0.8, intensity + 0.1)
            else:
                intensity = min(0.65, intensity)
        elif tag == "election":
            intensity = min(0.75, intensity)
    
    return intensity


def calibrate_article_labels(
    signals: Dict[str, Tuple[float, str]],
    title: str,
    description: str
) -> Dict[str, Tuple[float, str]]:
    """
    Calibrate all signal intensities for an article.
    
    Args:
        signals: Raw signals from embedding classification
        title: Article title
        description: Article description
        
    Returns:
        Calibrated signals with proper intensity scores
    """
    text = f"{title} {description}"
    calibrated = {}
    
    for category, (similarity_score, tag) in signals.items():
        # Convert similarity to intensity
        intensity = calibrate_intensity_score(
            category=category,
            similarity_score=similarity_score,
            tag=tag,
            text=text
        )
        
        calibrated[category] = (intensity, tag)
    
    return calibrated


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_articles = [
        {
            "title": "Major fire destroys historic building in Stockholm",
            "description": "A devastating blaze required 50 firefighters and multiple units. The building was completely destroyed.",
            "expected": {"emergencies": -0.9}  # High negative intensity
        },
        {
            "title": "Small trash fire quickly extinguished",
            "description": "Minor fire in a trash can was quickly contained by firefighters. No damage reported.",
            "expected": {"emergencies": -0.35}  # Low negative intensity
        },
        {
            "title": "Sweden wins hockey championship",
            "description": "National team defeats Finland in thrilling final match to claim the gold medal.",
            "expected": {"sports": 0.8}  # High positive intensity
        },
        {
            "title": "Minor traffic delays on highway",
            "description": "Slight congestion reported on E4 due to increased traffic volume.",
            "expected": {"transportation": -0.4}  # Low negative intensity
        },
    ]
    
    print("Testing intensity calibration:")
    print("="*80)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nTest {i}: {article['title'][:50]}...")
        
        # Simulate embedding classification (assume 0.6 similarity)
        raw_signals = {}
        for category, expected_score in article['expected'].items():
            raw_signals[category] = (0.6, "fire" if "fire" in article['title'].lower() else "event")
        
        # Calibrate
        calibrated = calibrate_article_labels(
            raw_signals,
            article['title'],
            article['description']
        )
        
        # Show results
        for category, (intensity, tag) in calibrated.items():
            expected = article['expected'][category]
            diff = abs(intensity - expected)
            status = "✓" if diff < 0.15 else "⚠️"
            print(f"  {status} {category}: {intensity:.2f} (expected: {expected:.2f}, diff: {diff:.2f})")
