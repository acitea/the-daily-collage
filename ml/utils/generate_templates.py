"""
Generate SIGNAL_TEMPLATES and TAG_KEYWORDS from real GDELT data using LLM.

This script:
1. Fetches GDELT articles per category using category-specific keywords
2. Preprocesses articles (Swedish text normalization)
3. Extracts representative example phrases as SIGNAL_TEMPLATES from category-specific articles
4. Extracts observed keywords/tags from titles and descriptions as TAG_KEYWORDS
5. Generates generic templates/keywords using LLM to ensure comprehensive coverage
6. Saves both to JSON files for use by embedding-based classification

Usage:
    # Generate templates from 100 Swedish articles per category (900 total)
    python ml/utils/generate_templates.py --articles-per-category 100 --country sweden
    
    # Search further back in time (up to 360 days)
    python ml/utils/generate_templates.py --articles-per-category 150 --days-lookback 180
    
    # Use GPT-4 for better quality
    python ml/utils/generate_templates.py --articles-per-category 100 --model gpt-4
    
    # Generate more templates per category (default 30)
    python ml/utils/generate_templates.py --articles-per-category 100 --templates-per-category 50
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import os
from datetime import datetime, timedelta, timezone
from itertools import chain
import math

from openai import OpenAI
from tqdm import tqdm
import re

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import normalize_country_input
from ml.utils.embedding_labeling import preprocess_swedish_text
from gdeltdoc import GdeltDoc, Filters

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Signal categories (must match system)
SIGNAL_CATEGORIES = [
    "emergencies", "crime", "festivals", "transportation",
    "weather_temp", "weather_wet", "sports", "economics", "politics"
]

# Category-specific keywords for GDELT queries (Swedish)
CATEGORY_KEYWORDS = {
    "emergencies": [
        "brand", "explosion", "räddning", "olycka", "katastrof", "evakuering", "larm", "nödläge",
        "jordbävning", "skogsbrand", "storbrand", "evakueras"
    ],
    "crime": [
        "brott", "polis", "misstänkt", "åtal", "rån", "mord", "skottlossning", "gripna",
        "rånförsök", "överfall", "bedrägeri"
    ],
    "festivals": ["festival", "konsert", "evenemang", "firande", "kulturvecka", "musikfestival", "teater", "kulturkalas"],
    "transportation": [
        "trafik", "kollektivtrafik", "tåg", "buss", "förseningar", "inställt", "väg", "trafikstörning",
        "trafikolycka", "bilolycka", "kö", "trafikkaos", "tunnelbana", "pendeltåg", "spårfel", "vägstängd", "färja", "brostängning"
    ],
    "weather_temp": [
        "temperatur", "värmebölja", "kyla", "vädret", "smhi", "vädervarning", "grader",
        "värme", "hetta", "kallfront", "köldknäpp", "frost", "minusgrader", "plusgrader", "värmerekord", "köldrekord"
    ],
    "weather_wet": [
        "regn", "snö", "översvämning", "storm", "blåst", "skyfall", "halka",
        "regnoväder", "snöfall", "snökaos", "slask", "stormvindar", "orkan", "blixt", "hagel", "vädervarning"
    ],
    "sports": [
        "sport", "fotboll", "ishockey", "seger", "matcher", "vm", "em", "allsvenskan",
        "derby", "guld", "silver", "brons", "turnering", "cupen", "landslaget", "mål", "förlust", "poäng"
    ],
    "economics": [
        "ekonomi", "börs", "aktier", "inflation", "ränta", "finans", "arbetsmarknad", "löner",
        "bnp", "tillväxt", "budget", "underskott", "nedskärningar", "krona", "växelkurs", "arbetslöshet", "konkurs"
    ],
    "politics": ["regering", "riksdag", "politik", "val", "minister", "parti", "opposition", "lagförslag", "protest", "demonstration"]
}

# Expansions for short/ambiguous keywords to avoid GDELT "phrase too short" errors
SHORT_KEYWORD_EXPANSIONS = {
    "sports": {
        "vm": ["vm fotboll", "vm ishockey"],
        "em": ["em fotboll"],
    },
    "transportation": {
        "väg": ["vägstängd", "väg avstängd"],
        "kö": ["kö trafik", "trafikkö"],
    },
    "crime": {
        "åtal": ["åtal väcks", "åtal mot"],
        "rån": ["butiksrån", "vapenrån", "rån mot bank"],
        "mord": ["misstänkt mord"],
    },
    "weather_wet": {
        "snö": ["snöoväder", "snöfall"],
        "regn": ["kraftigt regn"],
    },
    "weather_temp": {
        "värme": ["värmebölja"],
        "kyla": ["sträng kyla"],
    },
}

# Minimum required items per category (strictly from real articles)
MIN_ITEMS_PER_CATEGORY = 50
MIN_KEYWORD_LEN = 5


def normalize_category(raw: str) -> str:
    """Map common variants/synonyms to canonical SIGNAL_CATEGORIES.
    Returns canonical category or None if no match.
    """
    if not raw:
        return None
    c = raw.strip().lower().replace('-', '_')
    # Exact match first
    if c in SIGNAL_CATEGORIES:
        return c
    # Simple singular/plural and common synonyms
    alias = {
        "emergency": "emergencies",
        "emergencies": "emergencies",
        "crime": "crime",
        "crimes": "crime",
        "festival": "festivals",
        "festivals": "festivals",
        "transport": "transportation",
        "traffic": "transportation",
        "transportation": "transportation",
        "weather": "weather_temp",  # generic weather -> temp bucket by default
        "temperature": "weather_temp",
        "heat": "weather_temp",
        "cold": "weather_temp",
        "rain": "weather_wet",
        "snow": "weather_wet",
        "wet": "weather_wet",
        "sports": "sports",
        "sport": "sports",
        "economy": "economics",
        "economics": "economics",
        "economic": "economics",
        "politic": "politics",
        "politics": "politics",
    }
    return alias.get(c)


class TemplateLLM:
    """Wrapper for OpenAI API to generate templates and keywords."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        logger.info(f"Using OpenAI API with model: {model_name}")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"✓ OpenAI client initialized")
    
    def call(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.5) -> str:
        """Generate response from OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise news classification assistant. Follow output format instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""
    
    def classify_article_batch(self, articles: List[Dict]) -> Dict[int, Dict[str, float]]:
        """
        Classify multiple articles in one LLM call.
        
        Args:
            articles: List of {title, description} dicts
            
        Returns:
            Dict mapping article index to {category: confidence_score}
        """
        # Build batch prompt
        articles_text = []
        for i, article in enumerate(articles):
            articles_text.append(f"Article {i+1}:\nTitle: {article['title']}\nDescription: {article.get('description', '')[:300]}")
        
        prompt = f"""Task: Classify each article into categories. Output ONLY the format below. NO explanations, NO code, NO extra text.

Categories: {', '.join(SIGNAL_CATEGORIES)}

{chr(10).join(articles_text)}

OUTPUT FORMAT (EXACTLY THIS FORMAT, NOTHING ELSE):
Article 1: crime=0.85
Article 2: sports=0.90 economics=0.40

YOUR OUTPUT:"""
        
        try:
            response = self.call(prompt, max_tokens=800, temperature=0.1)
            
            # Log response excerpt for debugging
            logger.debug(f"LLM Response (first 500 chars):\n{response[:500]}")
            
            # Parse simple text format: "Article 1: category=0.85 category2=0.40"
            result = {}
            
            # Match lines like "Article 1: crime=0.85 emergencies=0.40"
            pattern = r'Article\s+(\d+)\s*:?\s*(.+)'
            
            for line in response.split('\n'):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    article_num = int(match.group(1))
                    categories_str = match.group(2)
                    
                    # Extract category=score pairs (accept comma decimal too)
                    score_pattern = r'([a-z_]+)\s*[=:]\s*([0-9.,]+)'
                    scores = {}
                    
                    for cat, score_str in re.findall(score_pattern, categories_str, re.IGNORECASE):
                        cat_norm = normalize_category(cat)
                        if cat_norm:
                            try:
                                # Support European decimal comma
                                score_val = float(score_str.replace(',', '.'))
                                scores[cat_norm] = score_val
                            except ValueError:
                                pass
                    
                    if scores:
                        result[article_num] = scores
            
            # If no results, log full response for debugging
            if not result:
                logger.warning(f"⚠️  No classifications parsed from LLM. Full response:\n{response[:800]}")
            else:
                logger.debug(f"✓ Parsed {len(result)} article classifications from LLM")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {}
    
    def extract_templates(self, articles: List[Dict[str, str]], category: str, max_templates: int = 15) -> List[str]:
        """
        Extract representative example phrases for a specific category.
        
        Args:
            articles: List of articles classified as this category
            category: Signal category name
            max_templates: Maximum number of templates to generate
            
        Returns:
            List of template strings
        """
        # Limit to first 30 articles for context window
        sample_articles = articles[:30]
        
        articles_text = [f"{i+1}. {a['title']}" for i, a in enumerate(sample_articles)]
        
        prompt = f"""Extract {max_templates} short Swedish news phrases about {category}.

Articles:
{chr(10).join(articles_text)}

OUTPUT (NO EXPLANATIONS, JUST THE LIST):
1. 
2. 
3. """
        
        try:
            response = self.call(prompt, max_tokens=600, temperature=0.6)
            
            # Parse numbered list
            templates = []
            for line in response.split('\n'):
                line = line.strip()
                # Match patterns like "1. phrase" or "1) phrase" or "- phrase"
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    phrase = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                    if phrase and len(phrase) > 10:
                        templates.append(phrase)
            
            return templates[:max_templates]
        except Exception as e:
            logger.error(f"Template extraction failed for {category}: {e}")
            return []
    
    def extract_keywords(self, articles: List[Dict[str, str]], category: str, max_keywords: int = 20) -> Dict[str, str]:
        """
        Extract common keywords/tags observed in articles.
        
        Args:
            articles: List of articles classified as this category
            category: Signal category name
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            Dict mapping keyword to tag (e.g., {"brand": "fire", "polis": "police"})
        """
        # Limit to first 40 articles for context
        sample_articles = articles[:40]
        
        titles_and_desc = []
        for a in sample_articles:
            text = f"{a['title']} {a.get('description', '')[:150]}"
            titles_and_desc.append(text)
        
        combined_text = "\n".join(titles_and_desc)
        
        prompt = f"""You are analyzing Swedish news about "{category}".

From these article titles/descriptions, extract {max_keywords} COMMON KEYWORDS that frequently appear.
For each keyword, assign a SHORT tag (1-2 words) that categorizes it within "{category}".

Text:
{combined_text[:2000]}

Respond in JSON format ONLY:
{{
  "keyword1": "tag1",
  "keyword2": "tag2",
  ...
}}

Example for "crime": {{"rån": "robbery", "polis": "police", "misshandel": "assault"}}

JSON:"""
        
        try:
            response = self.call(prompt, max_tokens=500, temperature=0.4)
            
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                
                # Clean common JSON issues
                json_str = re.sub(r'(?<!\\)\\(?!["\\/$bfnrtu])', r'\\\\', json_str)
                json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
                # Replace single quotes with double quotes
                json_str = re.sub(r"'", '"', json_str)
                # Fix missing commas
                json_str = re.sub(r'"\s+"', '", "', json_str)
                
                try:
                    keywords = json.loads(json_str)
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse error for {category}: {je}. Trying regex extraction...")
                    # Extract key-value pairs with more flexible regex
                    pattern = r'"?([^":]+)"?\s*:\s*"?([^",}]+)"?'
                    matches = re.findall(pattern, json_str)
                    keywords = dict(matches)
                
                # Filter out invalid entries
                valid_keywords = {}
                for k, v in keywords.items():
                    if isinstance(k, str) and isinstance(v, str) and len(k) > 2 and len(v) > 0:
                        valid_keywords[k.lower().strip()] = v.lower().strip()
                
                return valid_keywords
            else:
                logger.warning(f"No JSON found in keyword extraction for {category}")
                return {}
        except Exception as e:
            logger.error(f"Keyword extraction failed for {category}: {e}")
            return {}
    
    def generate_fallback_templates(self, category: str, max_templates: int = 15) -> List[str]:
        """
        Generate templates for a category when no articles are available.
        Uses LLM to create representative Swedish news phrases.
        
        Args:
            category: Signal category name
            max_templates: Number of templates to generate
            
        Returns:
            List of template strings
        """
        prompt = f"""Generate {max_templates} representative Swedish news headlines/phrases about "{category}".
These should be realistic and varied examples that appear in Swedish news.
Output only the numbered list, one phrase per line.

Example format for "crime":
1. Rån på bensinstation - två män gripna
2. Misstänkt mord under utredning
3. Polisen ökar patrulleringen i området
etc.

Swedish {category} headlines:
"""
        
        try:
            response = self.call(prompt, max_tokens=600, temperature=0.7)
            
            # Parse numbered list
            templates = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    phrase = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                    if phrase and len(phrase) > 5:
                        templates.append(phrase)
            
            logger.info(f"  Generated {len(templates)} fallback templates for {category}")
            return templates[:max_templates]
        except Exception as e:
            logger.error(f"Fallback template generation failed for {category}: {e}")
            return []
    
    def generate_fallback_keywords(self, category: str, max_keywords: int = 20) -> Dict[str, str]:
        """
        Generate keywords for a category when no articles are available.
        Uses LLM to create representative Swedish keywords.
        
        Args:
            category: Signal category name
            max_keywords: Number of keywords to generate
            
        Returns:
            Dict mapping keyword to tag
        """
        prompt = f"""Generate {max_keywords} common Swedish keywords related to "{category}" news.
For each keyword, assign a SHORT tag (1-2 words) that categorizes it within the topic.

Respond in JSON format ONLY:
{{
  "keyword1": "tag1",
  "keyword2": "tag2",
  ...
}}

Example for "emergencies": {{"brand": "fire", "explosion": "blast", "jordbävning": "earthquake"}}

For "{category}", generate relevant Swedish keywords:
JSON:"""
        
        try:
            response = self.call(prompt, max_tokens=500, temperature=0.5)
            
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                
                # Clean common JSON issues
                json_str = re.sub(r'(?<!\\)\\(?!["\\/$bfnrtu])', r'\\\\', json_str)
                json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
                json_str = re.sub(r"'", '"', json_str)
                json_str = re.sub(r'"\s+"', '", "', json_str)
                
                try:
                    keywords = json.loads(json_str)
                except json.JSONDecodeError:
                    pattern = r'"?([^":]+)"?\s*:\s*"?([^",}]+)"?'
                    matches = re.findall(pattern, json_str)
                    keywords = dict(matches)
                
                # Filter valid entries
                valid_keywords = {}
                for k, v in keywords.items():
                    if isinstance(k, str) and isinstance(v, str) and len(k) > 2 and len(v) > 0:
                        valid_keywords[k.lower().strip()] = v.lower().strip()
                
                logger.info(f"  Generated {len(valid_keywords)} fallback keywords for {category}")
                return valid_keywords
            else:
                return {}
        except Exception as e:
            logger.error(f"Fallback keyword generation failed for {category}: {e}")
            return {}


def classify_articles_by_category(llm: TemplateLLM, articles: List[Dict], batch_size: int = 20) -> Dict[str, List[Dict]]:
    """
    Run LLM classification over all articles and bucket them into categories by best score.
    """
    categorized: Dict[str, List[Dict]] = {cat: [] for cat in SIGNAL_CATEGORIES}
    seen_urls = set()

    # Clamp to avoid accidental per-article calls
    batch_size = max(5, batch_size)
    logger.info(f"Classifying {len(articles)} articles in batches of {batch_size}...")

    for start_idx in range(0, len(articles), batch_size):
        batch = articles[start_idx:start_idx + batch_size]
        resp = llm.classify_article_batch(batch)

        for local_idx, scores in resp.items():
            if not scores:
                continue
            best_cat = max(scores.items(), key=lambda kv: kv[1])[0]
            global_idx = start_idx + (local_idx - 1)
            if 0 <= global_idx < len(articles):
                art = articles[global_idx]
                url = art.get("url")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                categorized[best_cat].append(art)

    return categorized


def fetch_articles_for_category(
    category: str,
    country: str,
    num_articles: int = 100,
    days_lookback: int = 360,
    allow_global: bool = False,
    window_days: int = 30,
) -> list:
    """
    Fetch GDELT articles specifically for one category using keyword filtering.
    
    Args:
        category: Signal category name
        country: Country code
        num_articles: Target number of articles to fetch
        days_lookback: How many days to search back (max 360)
        
    Returns:
        List of article dicts
    """
    from gdeltdoc import GdeltDoc, Filters
    import polars as pl
    import time
    
    fips_code = normalize_country_input(country)
    keywords = CATEGORY_KEYWORDS.get(category, [])
    
    if not keywords:
        logger.warning(f"No keywords defined for category: {category}")
        return []
    
    # Build query terms; expand too-short keywords to avoid GDELT "phrase too short" errors
    query_terms = []
    expansions = SHORT_KEYWORD_EXPANSIONS.get(category, {})
    for kw in keywords:
        extra = expansions.get(kw)
        if extra:
            query_terms.extend(extra)
        elif len(kw) < MIN_KEYWORD_LEN:
            logger.debug(f"Skipping too-short keyword '{kw}' for {category}; no expansion found")
            continue
        else:
            query_terms.append(kw)

    if not query_terms:
        query_terms = keywords  # fallback to original list if everything was skipped

    logger.info(f"Fetching articles for {category} with keywords: {query_terms[:3]} (country={country})...")
    
    all_articles = []
    seen_urls = set()

    # Use more keywords for sparse categories to improve recall; collapse into one OR clause per window
    max_kw = 4
    if category in {"transportation", "weather_temp", "weather_wet", "sports", "economics"}:
        max_kw = min(len(query_terms), 8)
    window_keywords = query_terms[:max_kw]

    def _iter_windows(total_days: int, step_days: int):
        now = datetime.now(timezone.utc)
        for offset in range(0, total_days, step_days):
            end = now - timedelta(days=offset)
            start = end - timedelta(days=step_days)
            yield start, end

    def _filters_kwargs(use_country: bool) -> dict:
        return {
            "country": fips_code if use_country else None,
            # Language filter removed to avoid GDELT invalid/unsupported language errors; rely on country + keywords
            "language": None,
            "keyword": window_keywords,
        }

    # Query by windows to bypass per-call caps
    for start_dt, end_dt in _iter_windows(days_lookback, window_days):
        start_str = start_dt.strftime('%Y%m%d%H%M%S')
        end_str = end_dt.strftime('%Y%m%d%H%M%S')

        try:
            gd = GdeltDoc()
            filters = Filters(
                **_filters_kwargs(use_country=True),
                start_date=start_dt,
                end_date=end_dt,
                num_records=min(250, num_articles),
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

                logger.debug(
                    f"  Window {start_str}-{end_str} keywords OR: {len(articles_pl)} articles, unique: {len(all_articles)}"
                )

            time.sleep(0.15)

            if len(all_articles) >= num_articles:
                break

        except Exception as e:
            logger.error(f"Failed to fetch articles window {start_str}-{end_str}: {e}")
            continue
    
    logger.info(f"  → Fetched {len(all_articles)} unique articles for {category}")

    # If insufficient and allowed, retry with global (no country filter) using same windowed strategy
    if allow_global and len(all_articles) < num_articles:
        logger.info(f"  Insufficient articles for {category} (have {len(all_articles)} < {num_articles}); retrying without country filter")
        try:
            for start_dt, end_dt in _iter_windows(days_lookback, window_days):
                start_str = start_dt.strftime('%Y%m%d%H%M%S')
                end_str = end_dt.strftime('%Y%m%d%H%M%S')

                gd = GdeltDoc()
                filters = Filters(
                    **_filters_kwargs(use_country=False),
                    start_date=start_dt,
                    end_date=end_dt,
                    num_records=min(250, num_articles),
                )
                articles_pd = gd.article_search(filters)
                if articles_pd is not None and not articles_pd.empty:
                    articles_pl = pl.from_pandas(articles_pd)
                    for article in articles_pl.iter_rows(named=True):
                        url = article.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_articles.append(article)
                time.sleep(0.15)
                if len(all_articles) >= num_articles:
                    break
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Global fetch fallback failed for {category}: {e}")

    return all_articles


def generate_templates_and_keywords(
    articles_per_category: int,
    country: str,
    llm_model_name: str,
    templates_per_category: int,
    keywords_per_category: int,
    output_dir: Path,
    days_lookback: int = 360,
    total_articles: int = 1000,
    llm_batch_size: int = 20,
):
    """
    Main pipeline to generate templates and keywords by fetching articles per category.
    
    Args:
        articles_per_category: Number of articles to fetch per category
        country: Country filter for GDELT
        llm_model_name: OpenAI model name
        templates_per_category: Number of template phrases per category
        keywords_per_category: Number of keywords per category
        output_dir: Directory to save JSON files
        days_lookback: How many days to search back (max 360)
    """
    # Initialize LLM (only for parsing/cleanup if needed)
    llm = TemplateLLM(model_name=llm_model_name)

    # Step 1: Fetch articles per category (keyword seeds)
    per_category_target = max(articles_per_category, math.ceil(total_articles / len(SIGNAL_CATEGORIES)))
    logger.info(
        f"Fetching ~{per_category_target} articles per category (~{per_category_target * len(SIGNAL_CATEGORIES)} total target) "
        f"from GDELT (country={country}, lookback={days_lookback} days)..."
    )
    seeded_articles = {}

    for category in SIGNAL_CATEGORIES:
        logger.info(f"\n📰 Category seed fetch: {category}")
        articles = fetch_articles_for_category(
            category=category,
            country=country,
            num_articles=per_category_target,
            days_lookback=days_lookback,
            allow_global=True,
        )
        seeded_articles[category] = articles

    all_seeded = list(chain.from_iterable(seeded_articles.values()))
    logger.info(f"\n✓ Seed fetch complete — total articles collected: {len(all_seeded)}")

    if not all_seeded:
        logger.warning("⚠️  No articles fetched; generation will rely entirely on fallbacks")

    # Step 2: LLM classification of all fetched articles
    llm = TemplateLLM(model_name=llm_model_name)
    logger.info("\nClassifying fetched articles into signal categories via LLM...")
    classified_articles = classify_articles_by_category(llm, all_seeded, batch_size=llm_batch_size)

    logger.info("Classified distribution:")
    for cat in SIGNAL_CATEGORIES:
        logger.info(f"  {cat:20s}: {len(classified_articles.get(cat, [])):3d} articles")

    # Preprocess classified articles (remove stopwords, punctuation, stem)
    for category, articles in classified_articles.items():
        if not articles:
            continue
        logger.info(f"  Preprocessing {len(articles)} articles for {category}...")
        for article in articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            combined = f"{title} {desc}"
            article['preprocessed'] = preprocess_swedish_text(combined, remove_stopwords=True, stem=True)
    
    # Step 3: Generate templates for each category (with fallback top-up)
    logger.info("\nGenerating signal templates (real articles only)...")
    signal_templates = {}
    
    for category in SIGNAL_CATEGORIES:
        all_templates = []
        
        # Extract templates from LLM-classified articles
        if category in classified_articles and len(classified_articles[category]) >= 2:
            logger.info(f"  {category}: extracting from classified articles...")
            article_templates = llm.extract_templates(
                classified_articles[category],
                category,
                max_templates=max(templates_per_category, MIN_ITEMS_PER_CATEGORY)
            )
            all_templates.extend(article_templates)
            logger.info(f"    ✓ Extracted {len(article_templates)} templates from articles")
        else:
            logger.warning(f"  {category}: no classified articles available to extract templates")
        
        # Deduplicate while preserving order (case-insensitive)
        seen = set()
        unique_templates = []
        for template in all_templates:
            template_lower = template.lower()
            if template_lower not in seen:
                seen.add(template_lower)
                unique_templates.append(template)

        # Fallback top-up to reach minimum coverage
        if len(unique_templates) < MIN_ITEMS_PER_CATEGORY:
            need = MIN_ITEMS_PER_CATEGORY - len(unique_templates)
            logger.info(f"    ↳ Adding {need} fallback templates for {category} to reach minimum {MIN_ITEMS_PER_CATEGORY}")
            fallback_templates = llm.generate_fallback_templates(category, max_templates=need + 5)
            for tmpl in fallback_templates:
                if tmpl.lower() not in seen:
                    seen.add(tmpl.lower())
                    unique_templates.append(tmpl)
        
        signal_templates[category] = unique_templates
        logger.info(f"    → Total (after dedup): {len(unique_templates)} templates")
    
    # Step 4: Generate keywords for each category (with fallback top-up)
    logger.info("\nGenerating tag keywords (real articles only)...")
    tag_keywords = {}
    
    for category in SIGNAL_CATEGORIES:
        all_keywords = {}
        
        # Extract keywords from LLM-classified articles
        if category in classified_articles and len(classified_articles[category]) >= 2:
            logger.info(f"  {category}: extracting from classified articles...")
            article_keywords = llm.extract_keywords(
                classified_articles[category],
                category,
                max_keywords=max(keywords_per_category, MIN_ITEMS_PER_CATEGORY)
            )
            all_keywords.update(article_keywords)
            logger.info(f"    ✓ Extracted {len(article_keywords)} keywords from articles")
        else:
            logger.warning(f"  {category}: no classified articles available to extract keywords")

        # Fallback top-up to reach minimum coverage
        if len(all_keywords) < MIN_ITEMS_PER_CATEGORY:
            need = MIN_ITEMS_PER_CATEGORY - len(all_keywords)
            logger.info(f"    ↳ Adding {need} fallback keywords for {category} to reach minimum {MIN_ITEMS_PER_CATEGORY}")
            fallback_keywords = llm.generate_fallback_keywords(category, max_keywords=need + 10)
            for k, v in fallback_keywords.items():
                if k not in all_keywords and len(all_keywords) < MIN_ITEMS_PER_CATEGORY:
                    all_keywords[k] = v
        
        tag_keywords[category] = all_keywords
        logger.info(f"    → Total (after dedup): {len(all_keywords)} keywords")
    
    # Step 4: Save to JSON files
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
    logger.info(f"Articles fetched per category target: {per_category_target}")
    logger.info(f"Days searched: {days_lookback}")
    logger.info(f"Total articles collected (post-classification pool): {len(all_seeded)}")
    logger.info(f"Templates generated: {sum(len(t) for t in signal_templates.values())}")
    logger.info(f"Keywords generated: {sum(len(k) for k in tag_keywords.values())}")
    
    logger.info("\nTemplate breakdown per category:")
    for category in SIGNAL_CATEGORIES:
        count = len(signal_templates.get(category, []))
        logger.info(f"  {category:20s}: {count:3d} templates")
    
    logger.info("\nKeyword breakdown per category:")
    for category in SIGNAL_CATEGORIES:
        count = len(tag_keywords.get(category, {}))
        logger.info(f"  {category:20s}: {count:3d} keywords")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review the generated JSON files")
    logger.info("  2. Run embedding classification with new templates:")
    logger.info(f"     python ml/utils/label_dataset.py --articles 50 --method embedding")
    
    return signal_templates, tag_keywords


def main():
    parser = argparse.ArgumentParser(description="Generate signal templates and keywords from GDELT per category")
    parser.add_argument("--articles-per-category", type=int, default=100, 
                        help="Number of articles to fetch per category (default: 100)")
    parser.add_argument("--country", type=str, default="sweden", help="Country filter")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                        help="OpenAI model name (default: gpt-3.5-turbo, or use gpt-4 for better quality)")
    parser.add_argument("--templates-per-category", type=int, default=30, 
                        help="Number of template phrases per category")
    parser.add_argument("--keywords-per-category", type=int, default=25, 
                        help="Number of keywords per category")
    parser.add_argument("--days-lookback", type=int, default=360,
                        help="How many days to search back (max 360)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), 
                        help="Output directory for JSON files")
    parser.add_argument("--total-articles", type=int, default=1000,
                        help="Approximate total articles to fetch across all categories (default: 1000)")
    parser.add_argument("--llm-batch-size", type=int, default=20,
                        help="Number of articles per LLM classification call (default: 20; higher = fewer calls)")
    
    args = parser.parse_args()
    
    generate_templates_and_keywords(
        articles_per_category=args.articles_per_category,
        country=args.country,
        llm_model_name=args.model,
        templates_per_category=args.templates_per_category,
        keywords_per_category=args.keywords_per_category,
        output_dir=args.output_dir,
        days_lookback=args.days_lookback,
        total_articles=args.total_articles,
        llm_batch_size=args.llm_batch_size,
    )


if __name__ == "__main__":
    main()
