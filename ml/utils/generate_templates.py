"""
Generate SIGNAL_TEMPLATES and TAG_KEYWORDS from real GDELT data using LLM.

This script:
1. Fetches a large batch of GDELT articles (titles + descriptions)
2. Uses local LLM to classify articles into signal categories
3. Extracts representative example phrases as SIGNAL_TEMPLATES
4. Extracts observed keywords/tags from titles and descriptions as TAG_KEYWORDS
5. Saves both to JSON files for use by embedding-based classification

Usage:
    # Generate templates from 500 Swedish articles
    python ml/utils/generate_templates.py --articles 500 --country sweden
    
    # Use specific LLM model
    python ml/utils/generate_templates.py --articles 300 --model AI-Sweden-Models/gpt-sw3-1.3b-instruct
    
    # Generate more templates per category (default 30)
    python ml/utils/generate_templates.py --articles 1000 --templates-per-category 50
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ingestion.script import fetch_news_batched

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Signal categories (must match system)
SIGNAL_CATEGORIES = [
    "emergencies", "crime", "festivals", "transportation",
    "weather_temp", "weather_wet", "sports", "economics", "politics"
]


class TemplateLLM:
    """Wrapper for local LLM to generate templates and keywords."""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        logger.info(f"Loading LLM: {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Phi-2 is 2.7B (much faster than 7B), fits in 12GB without quantization
        if torch.cuda.is_available():
            logger.info("Using float16 on GPU (Phi-2 2.7B is fast & memory-efficient)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"✓ LLM loaded on {self.device}")
    
    def call(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.5) -> str:
        """Generate response from LLM."""
        # Phi-2 uses this format
        formatted_prompt = f"Instruct: {prompt}\nOutput:"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
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
        
        prompt = f"""You are classifying Swedish news articles into these categories:
{', '.join(SIGNAL_CATEGORIES)}

For each article, identify which categories apply (0-3 categories per article). Assign confidence scores from 0.0 to 1.0.

Articles to classify:
{chr(10).join(articles_text)}

Respond in JSON format ONLY:
{{
  "1": {{"category": "confidence", ...}},
  "2": {{"category": "confidence", ...}},
  ...
}}

Example: {{"1": {{"emergencies": 0.85, "crime": 0.40}}, "2": {{"sports": 0.95}}}}

JSON:"""
        
        try:
            response = self.call(prompt, max_tokens=800, temperature=0.3)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                
                # Clean common JSON issues from LLM output
                # Replace unescaped backslashes (but not already escaped ones)
                json_str = re.sub(r'(?<!\\)\\(?!["\\/$bfnrtu])', r'\\\\', json_str)
                # Remove control characters that break JSON
                json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
                
                try:
                    classifications = json.loads(json_str)
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse error: {je}. Trying lenient parse...")
                    # Try to extract just the structure we need with regex
                    pattern = r'"(\d+)"\s*:\s*\{([^}]+)\}'
                    matches = re.findall(pattern, json_str)
                    classifications = {}
                    for article_num, scores_str in matches:
                        # Extract category:score pairs
                        score_pattern = r'"([^"]+)"\s*:\s*([0-9.]+)'
                        scores = dict(re.findall(score_pattern, scores_str))
                        classifications[article_num] = {k: float(v) for k, v in scores.items()}
                
                # Ensure all scores are floats (LLM might return strings like "0.85")
                result = {}
                for k, v in classifications.items():
                    if isinstance(v, dict):
                        result[int(k)] = {cat: float(score) if isinstance(score, (int, float, str)) else 0.0 
                                         for cat, score in v.items()}
                    else:
                        result[k] = v
                
                return result
            else:
                logger.warning("No JSON found in LLM response")
                return {}
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
        
        prompt = f"""You are analyzing Swedish news articles about "{category}".

Extract {max_templates} SHORT representative phrases (5-15 words each) that describe typical "{category}" news.
These phrases should be SPECIFIC and DESCRIPTIVE, capturing the semantic essence of this category.

Articles:
{chr(10).join(articles_text)}

Respond with ONLY a numbered list of phrases (no explanations):
1. [phrase]
2. [phrase]
...

List:"""
        
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
                
                try:
                    keywords = json.loads(json_str)
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse error for {category}: {je}. Trying regex extraction...")
                    # Extract key-value pairs with regex
                    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
                    matches = re.findall(pattern, json_str)
                    keywords = dict(matches)
                
                # Filter out invalid entries
                valid_keywords = {}
                for k, v in keywords.items():
                    if isinstance(k, str) and isinstance(v, str) and len(k) > 2 and len(v) > 0:
                        valid_keywords[k.lower()] = v.lower()
                
                return valid_keywords
            else:
                logger.warning(f"No JSON found in keyword extraction for {category}")
                return {}
        except Exception as e:
            logger.error(f"Keyword extraction failed for {category}: {e}")
            return {}


def generate_templates_and_keywords(
    num_articles: int,
    country: str,
    llm_model_name: str,
    templates_per_category: int,
    keywords_per_category: int,
    output_dir: Path,
):
    """
    Main pipeline to generate templates and keywords.
    
    Args:
        num_articles: Number of GDELT articles to fetch
        country: Country filter for GDELT
        llm_model_name: HuggingFace model name
        templates_per_category: Number of template phrases per category
        keywords_per_category: Number of keywords per category
        output_dir: Directory to save JSON files
    """
    logger.info(f"Fetching {num_articles} articles from GDELT (country={country})...")
    
    # Fetch articles
    articles_df = fetch_news_batched(
        country=country,
        total_articles=num_articles,
        batch_size=min(250, num_articles),
        days_lookback=7,  # Last week for variety
    )
    
    if articles_df is None or len(articles_df) == 0:
        logger.error("No articles fetched. Exiting.")
        return
    
    # Convert to list of dicts
    articles = articles_df.to_dicts()
    logger.info(f"✓ Fetched {len(articles)} articles")
    
    # Initialize LLM
    llm = TemplateLLM(model_name=llm_model_name)
    
    # Step 1: Classify all articles
    logger.info("Classifying articles into categories...")
    categorized_articles = defaultdict(list)
    
    # Phi-2 is fast and lightweight, can use larger batch size
    batch_size = 16
    for i in tqdm(range(0, len(articles), batch_size), desc="Classifying"):
        batch = articles[i:i+batch_size]
        classifications = llm.classify_article_batch(batch)
        
        for local_idx, category_scores in classifications.items():
            global_idx = i + local_idx - 1  # LLM uses 1-indexed
            if global_idx < len(articles):
                article = articles[global_idx]
                
                # Add article to each applicable category
                for category, score in category_scores.items():
                    if score >= 0.4:  # Confidence threshold
                        categorized_articles[category].append(article)
        
        # Clear CUDA cache between batches to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("✓ Classification complete")
    for cat, arts in categorized_articles.items():
        logger.info(f"  {cat:20s}: {len(arts):3d} articles")
    
    # Step 2: Generate templates for each category
    logger.info("\nGenerating signal templates...")
    signal_templates = {}
    
    for category in SIGNAL_CATEGORIES:
        if category in categorized_articles and len(categorized_articles[category]) >= 5:
            logger.info(f"  {category}...")
            templates = llm.extract_templates(
                categorized_articles[category],
                category,
                max_templates=templates_per_category
            )
            signal_templates[category] = templates
            logger.info(f"    ✓ Generated {len(templates)} templates")
        else:
            logger.warning(f"  {category}: Not enough articles (<5), skipping")
            signal_templates[category] = []
    
    # Step 3: Generate keywords for each category
    logger.info("\nGenerating tag keywords...")
    tag_keywords = {}
    
    for category in SIGNAL_CATEGORIES:
        if category in categorized_articles and len(categorized_articles[category]) >= 5:
            logger.info(f"  {category}...")
            keywords = llm.extract_keywords(
                categorized_articles[category],
                category,
                max_keywords=keywords_per_category
            )
            tag_keywords[category] = keywords
            logger.info(f"    ✓ Generated {len(keywords)} keywords")
        else:
            logger.warning(f"  {category}: Not enough articles (<5), skipping")
            tag_keywords[category] = {}
    
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
    logger.info(f"Articles processed: {len(articles)}")
    logger.info(f"Templates generated: {sum(len(t) for t in signal_templates.values())}")
    logger.info(f"Keywords generated: {sum(len(k) for k in tag_keywords.values())}")
    logger.info("\nNext steps:")
    logger.info("  1. Review the generated JSON files")
    logger.info("  2. Run embedding classification with new templates:")
    logger.info(f"     python ml/utils/label_dataset.py --articles 50 --method embedding")


def main():
    parser = argparse.ArgumentParser(description="Generate signal templates and keywords from GDELT")
    parser.add_argument("--articles", type=int, default=500, help="Number of articles to fetch")
    parser.add_argument("--country", type=str, default="sweden", help="Country filter")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", 
                        help="LLM model name (default: Phi-2 2.7B, much faster than Mistral-7B)")
    parser.add_argument("--templates-per-category", type=int, default=30, 
                        help="Number of template phrases per category")
    parser.add_argument("--keywords-per-category", type=int, default=25, 
                        help="Number of keywords per category")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), 
                        help="Output directory for JSON files")
    
    args = parser.parse_args()
    
    generate_templates_and_keywords(
        num_articles=args.articles,
        country=args.country,
        llm_model_name=args.model,
        templates_per_category=args.templates_per_category,
        keywords_per_category=args.keywords_per_category,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
