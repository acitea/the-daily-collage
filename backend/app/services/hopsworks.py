"""
Hopsworks integration service for The Daily Collage.

Handles:
1. Feature Store: Vibe vectors and signal data
2. Model Registry: ML classification models
3. Artifact Store: Generated visualizations
"""

import logging, hopsworks
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from hopsworks.project import Project
from hopsworks.core.dataset_api import DatasetApi
from hsfs.feature_store import FeatureStore
from hsfs.feature import Feature

from _types import SignalCategory

logger = logging.getLogger(__name__)
instance = None

class HopsworksService:
    """
    Service for interacting with Hopsworks Feature Store and Artifact Store.
    
    Responsibilities:
    - Store/retrieve vibe vectors in feature groups
    - Store/retrieve generated images in artifact registry
    - Query historical vibes for comparison
    """
    
    # Signal categories from centralized enum
    SIGNAL_CATEGORIES = [cat.value for cat in SignalCategory]
    
    def __init__(
        self,
        api_key: str,
        project_name: str,
        host: Optional[str] = None,
    ):
        """
        Initialize Hopsworks service.
        
        Args:
            api_key: Hopsworks API key
            project_name: Project name in Hopsworks
            host: Hopsworks host (optional, uses default if not provided)
        """
        self.api_key = api_key
        self.project_name = project_name
        self.host = host
        self._project: Project = None
        self._fs: FeatureStore = None
        self._dataset_api: DatasetApi = None
        
    def connect(self):
        """Establish connection to Hopsworks."""
        try:
            logger.info(f"Connecting to Hopsworks project: {self.project_name}")
            
            self._project = hopsworks.login(
                host=self.host,
                project=self.project_name,
                api_key_value=self.api_key,
            )
            
            # Get feature store from project
            self._fs = self._project.get_feature_store()
            logger.info(f"Connected to feature store: {self._fs.name}")
            
            # Get model registry
            try:
                self._mr = self._project.get_model_registry()
                logger.info("Model registry connected")
            except Exception as e:
                logger.warning(f"Could not access model registry: {e}")
                self._mr = None
            
            # Get dataset API for artifact storage
            self._dataset_api = self._project.get_dataset_api()
            logger.info(f"Successfully connected to Hopsworks project: {self.project_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            logger.error(f"Project: {self.project_name}")
            logger.error(f"Host: {self.host or 'c.app.hopsworks.ai (default)'}")
            logger.error("Make sure 'hopsworks' package is installed: pip install hopsworks")
            logger.error("For managed Hopsworks, ensure your API key is valid and the project exists")
            raise
            
    def get_or_create_headline_feature_group(self, fg_name: str = "headline_classifications", version: int = 1):
        """
        Get or create feature group for storing individual headline classifications.
        
        Schema:
        - article_id: string (primary key)
        - city: string
        - timestamp: timestamp (event_time)
        - title: string
        - description: string
        - url: string
        - source: string
        - emergencies_score: float
        - emergencies_tag: string
        - crime_score: float
        - crime_tag: string
        - festivals_score: float
        - festivals_tag: string
        - transportation_score: float
        - transportation_tag: string
        - weather_temp_score: float
        - weather_temp_tag: string
        - weather_wet_score: float
        - weather_wet_tag: string
        - sports_score: float
        - sports_tag: string
        - economics_score: float
        - economics_tag: string
        - politics_score: float
        - politics_tag: string
        
        Args:
            fg_name: Feature group name
            version: Feature group version
            
        Returns:
            Feature group object
        """
        if not self._fs:
            self.connect()
            
        try:
            fg = self._fs.get_feature_group(name=fg_name, version=version)
            if not fg: raise Exception("Feature group not found")
            logger.info(f"Using existing feature group: {fg_name} v{version}")
            return fg
        except Exception:
            logger.info(f"Feature group {fg_name} v{version} not found, creating new one")
            
        try:
            fg = self._fs.create_feature_group(
                name=fg_name,
                version=version,
                description="Individual headline classifications with scores and tags per category",
                primary_key=["article_id"],
                event_time="timestamp",
                online_enabled=True,
            )
            logger.info(f"Created feature group: {fg_name} v{version}")
            return fg
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            raise
    
    def get_or_create_vibe_feature_group(self, fg_name: str = "vibe_vectors", version: int = 1):
        """
        Get or create feature group for storing aggregated vibe vectors.
        
        Schema:
        - city: string (partition key)
        - window_key: string (partition key, format YYYY-MM-DD_HH-HH)
        - timestamp: timestamp (event_time)
        - emergencies_score: float
        - emergencies_tag: string
        - emergencies_count: int
        - crime_score: float
        - crime_tag: string
        - crime_count: int
        - festivals_score: float
        - festivals_tag: string
        - festivals_count: int
        - transportation_score: float
        - transportation_tag: string
        - transportation_count: int
        - weather_temp_score: float
        - weather_temp_tag: string
        - weather_temp_count: int
        - weather_wet_score: float
        - weather_wet_tag: string
        - weather_wet_count: int
        - sports_score: float
        - sports_tag: string
        - sports_count: int
        - economics_score: float
        - economics_tag: string
        - economics_count: int
        - politics_score: float
        - politics_tag: string
        - politics_count: int
        
        Args:
            fg_name: Feature group name
            version: Feature group version
            
        Returns:
            Feature group object
        """
        if not self._fs:
            self.connect()
            
        try:
            # Try to get existing feature group
            fg = self._fs.get_feature_group(name=fg_name, version=version)
            if not fg: raise Exception("Feature group not found")
            logger.info(f"Using existing feature group: {fg_name} v{version}")
            return fg
        except Exception:
            logger.info(f"Feature group {fg_name} v{version} not found, creating new one")
            
        # Define schema
        try:
            fg = self._fs.create_feature_group(
                name=fg_name,
                version=version,
                description="Aggregated vibe vectors per location and time window with frequency counts",
                primary_key=["city", "window_key"],
                event_time="timestamp",
                partition_key=["city", "window_key"],
            )
            logger.info(f"Created feature group: {fg_name} v{version}")
            return fg
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            raise

    def get_or_create_templates_feature_group(self, fg_name: str = "signal_templates", version: int = 1):
        """
        Feature group for storing signal templates.

        Schema:
        - category: string(100)
        - template: string(1000)
        - created_at: timestamp
        """
        if not self._fs:
            self.connect()
        
        if not self._fs:
            raise RuntimeError("Failed to connect to Hopsworks feature store")

        try:
            fg = self._fs.get_feature_group(name=fg_name, version=version)
            if not fg: raise Exception("Feature group not found")
            logger.info(f"Using existing feature group: {fg_name} v{version}")
            return fg
        except Exception:
            logger.info(f"Feature group {fg_name} v{version} not found, creating new one")

        try:
            # Import pandas for schema definition
            import pandas as pd
            
            # Create empty DataFrame with proper types to define schema
            schema_df = pd.DataFrame({
                'category': pd.Series(dtype='string'),
                'template': pd.Series(dtype='string'),
                'created_at': pd.Series(dtype='datetime64[ns]')
            })
            
            fg = self._fs.create_feature_group(
                name=fg_name,
                version=version,
                description="Signal templates per category",
                primary_key=["category"],
                event_time="created_at",
                online_enabled=False,
            )
            logger.info(f"Created feature group: {fg_name} v{version}")
            return fg
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            raise

    def get_or_create_keywords_feature_group(self, fg_name: str = "tag_keywords", version: int = 1):
        """
        Feature group for storing tag keywords.

        Schema:
        - category: string(100)
        - keyword: string(200)
        - tag: string(100)
        - created_at: timestamp
        """
        if not self._fs:
            self.connect()
        
        if not self._fs:
            raise RuntimeError("Failed to connect to Hopsworks feature store")

        try:
            fg = self._fs.get_feature_group(name=fg_name, version=version)
            if not fg: raise Exception("Feature group not found")
            logger.info(f"Using existing feature group: {fg_name} v{version}")
            return fg
        except Exception:
            logger.info(f"Feature group {fg_name} v{version} not found, creating new one")

        try:
            # Import pandas for schema definition
            import pandas as pd
            
            # Create empty DataFrame with proper types to define schema
            schema_df = pd.DataFrame({
                'category': pd.Series(dtype='string'),
                'keyword': pd.Series(dtype='string'),
                'tag': pd.Series(dtype='string'),
                'created_at': pd.Series(dtype='datetime64[ns]')
            })
            
            fg = self._fs.create_feature_group(
                name=fg_name,
                version=version,
                description="Tag keywords per category",
                primary_key=["category", "keyword"],
                event_time="created_at",
                online_enabled=False,
            )
            logger.info(f"Created feature group: {fg_name} v{version}")
            return fg
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            raise

    def store_signal_templates(self, templates: Dict[str, List[str]], fg_name: str = "signal_templates", version: int = 1):
        """Insert signal templates into the feature group."""
        if not self._fs:
            self.connect()

        fg = self.get_or_create_templates_feature_group(fg_name, version)
        if fg is None:
            raise RuntimeError(f"Failed to get or create feature group: {fg_name}")
        
        rows = []
        now = datetime.utcnow()
        for category, tmpl_list in templates.items():
            for tmpl in tmpl_list:
                rows.append({
                    "category": category,
                    "template": tmpl,
                    "created_at": now,
                })
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            fg.insert(df)
            logger.info(f"Stored {len(rows)} templates into {fg_name}")
        except Exception as e:
            logger.error(f"Failed to store templates: {e}")
            raise

    def store_tag_keywords(self, keywords: Dict[str, Dict[str, str]], fg_name: str = "tag_keywords", version: int = 1):
        """Insert tag keywords into the feature group."""
        if not self._fs:
            self.connect()

        fg = self.get_or_create_keywords_feature_group(fg_name, version)
        if fg is None:
            raise RuntimeError(f"Failed to get or create feature group: {fg_name}")
        
        rows = []
        now = datetime.utcnow()
        for category, kv in keywords.items():
            for k, v in kv.items():
                rows.append({
                    "category": category,
                    "keyword": k,
                    "tag": v,
                    "created_at": now,
                })
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            fg.insert(df)
            logger.info(f"Stored {len(rows)} tag keywords into {fg_name}")
        except Exception as e:
            logger.error(f"Failed to store tag keywords: {e}")
            raise

    def get_or_create_headline_labels_feature_group(self, fg_name: str = "headline_labels", version: int = 1):
        """
        Feature group for storing labeled dataset rows.
        Similar to headline_classifications but intended for training sets.
        """
        if not self._fs:
            self.connect()

        try:
            fg = self._fs.get_feature_group(name=fg_name, version=version)
            if not fg: raise Exception("Feature group not found")
            logger.info(f"Using existing feature group: {fg_name} v{version}")
            return fg
        except Exception:
            logger.info(f"Feature group {fg_name} v{version} not found, creating new one")

        try:
            fg = self._fs.create_feature_group(
                name=fg_name,
                version=version,
                description="Labeled dataset rows with scores/tags per category",
                primary_key=["url"],
                event_time="date",
                online_enabled=False,
            )
            logger.info(f"Created feature group: {fg_name} v{version}")
            return fg
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            raise

    def store_labeled_dataset(self, labeled_rows: List[Dict], fg_name: str = "headline_labels", version: int = 1):
        """Insert labeled dataset rows into Hopsworks."""
        if not self._fs:
            self.connect()

        fg = self.get_or_create_headline_labels_feature_group(fg_name, version)
        try:
            import pandas as pd
            df = pd.DataFrame(labeled_rows)
            fg.insert(df)
            logger.info(f"Stored {len(labeled_rows)} labeled rows into {fg_name}")
        except Exception as e:
            logger.error(f"Failed to store labeled dataset: {e}")
            raise
    
    def store_headline_classifications(
        self,
        headlines: List[Dict],
        city: str,
        timestamp: datetime,
        fg_name: str = "headline_classifications",
        version: int = 1,
    ):
        """
        Store individual headline classifications in the feature store.
        
        Args:
            headlines: List of dicts with article data and classifications
                      Each dict should have: article_id, title, url, source, and classifications
            city: City name
            timestamp: Time window timestamp
            fg_name: Feature group name
            version: Feature group version
        """
        if not self._fs:
            self.connect()
            
        fg = self.get_or_create_headline_feature_group(fg_name, version)
        
        rows = []
        for headline in headlines:
            row = {
                "article_id": headline["article_id"],
                "city": city,
                # Store publication time if available; fallback to provided timestamp
                "timestamp": headline.get("published_at", timestamp),
                "title": headline.get("title", ""),
                "description": headline.get("description", ""),
                "url": headline.get("url", ""),
                "source": headline.get("source", ""),
            }
            
            # Add classifications
            classifications = headline.get("classifications", {})
            for category in self.SIGNAL_CATEGORIES:
                if category in classifications:
                    score, tag = classifications[category]
                    row[f"{category}_score"] = score
                    row[f"{category}_tag"] = tag
                else:
                    row[f"{category}_score"] = 0.0
                    row[f"{category}_tag"] = ""
            
            rows.append(row)
        
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            fg.insert(df)
            logger.info(f"Stored {len(rows)} headline classifications for {city} at {timestamp}")
        except Exception as e:
            logger.error(f"Failed to store headline classifications: {e}")
            raise
    
    def store_vibe_vector(
        self,
        city: str,
        timestamp: datetime,
        vibe_vector: Dict[str, Tuple[float, str, int]],
        fg_name: str = "vibe_vectors",
        version: int = 1,
    ):
        """
        Store an aggregated vibe vector in the feature store.
        
        Args:
            city: City name
            timestamp: Time window timestamp
            vibe_vector: Dict mapping category to (score, tag, count)
                        e.g., {"emergencies": (0.8, "fire", 5), "crime": (0.3, "theft", 2)}
            fg_name: Feature group name
            version: Feature group version
        """
        if not self._fs:
            self.connect()
            
        fg = self.get_or_create_vibe_feature_group(fg_name, version)
        
        # Flatten vibe vector into row format

        window_end = timestamp + timedelta(hours=6)
        window_key = f"{timestamp:%Y-%m-%d}_{timestamp:%H}-{window_end:%H}"

        row = {
            "city": city,
            "window_key": window_key,
            "timestamp": timestamp,
        }
        
        # Add each category's score, tag, and count
        for category in self.SIGNAL_CATEGORIES:
            if category in vibe_vector:
                score, tag, count = vibe_vector[category]
                row[f"{category}_score"] = score
                row[f"{category}_tag"] = tag
                row[f"{category}_count"] = count
            else:
                row[f"{category}_score"] = 0.0
                row[f"{category}_tag"] = ""
                row[f"{category}_count"] = 0
        
        try:
            import pandas as pd
            df = pd.DataFrame([row])
            fg.insert(df)
            logger.info(f"Stored vibe vector for {city} at {timestamp}")
        except Exception as e:
            logger.error(f"Failed to store vibe vector: {e}")
            raise
    
    def get_latest_vibe_vector(
        self,
        city: str,
        fg_name: str = "vibe_vectors",
        version: int = 1,
    ) -> Optional[Dict[str, Tuple[float, str, int]]]:
        """
        Retrieve the latest vibe vector for a city.
        
        Args:
            city: City name
            fg_name: Feature group name
            version: Feature group version
            
        Returns:
            Vibe vector dict or None if not found
        """
        if not self._fs:
            self.connect()
            
        try:
            fg = self.get_or_create_vibe_feature_group(fg_name, version)
            
            # Query latest entry for city
            query = fg.select_all().filter(fg.city == city).order_by(fg.timestamp.desc()).limit(1)
            df = query.read()
            
            if df.empty:
                return None
                
            row = df.iloc[0]
            
            # Reconstruct vibe vector
            vibe_vector = {}
            for category in self.SIGNAL_CATEGORIES:
                score = row[f"{category}_score"]
                tag = row[f"{category}_tag"]
                count = row.get(f"{category}_count", 0)
                if score != 0.0 or tag != "":
                    vibe_vector[category] = (score, tag, count)
                    
            return vibe_vector
            
        except Exception as e:
            logger.error(f"Failed to retrieve vibe vector: {e}")
            return None
    
    def get_vibe_vector_at_time(
        self,
        city: str,
        timestamp: datetime,
        fg_name: str = "vibe_vectors",
        version: int = 1,
    ) -> Optional[Dict[str, Tuple[float, str, int]]]:
        """
        Retrieve the vibe vector for a city at a specific timestamp.
        
        Args:
            city: City name
            timestamp: Specific timestamp to query
            fg_name: Feature group name
            version: Feature group version
            
        Returns:
            Vibe vector dict or None if not found
        """
        if not self._fs:
            self.connect()
            
        try:
            fg = self.get_or_create_vibe_feature_group(fg_name, version)
            
            # Compute window_key (matches VibeHash minus city prefix)
            window_index = timestamp.hour // 6
            window_str = f"{window_index * 6:02d}-{(window_index + 1) * 6:02d}"
            window_key = f"{timestamp:%Y-%m-%d}_{window_str}"

            logger.info(f"Querying vibe vector for {city} at {timestamp} (window_key={window_key})")
            # Query for exact city + window_key match (expected single row)
            query = fg.select_all().filter(
                (Feature('city').like(city)) & (Feature('window_key').like(window_key))
            )
            df = query.read()
            
            if df.empty:
                logger.warning(f"No vibe vector found for {city} at {timestamp} (window_key={window_key})")
                return None
                
            row = df.iloc[0]
            
            # Reconstruct vibe vector
            vibe_vector = {}
            for category in self.SIGNAL_CATEGORIES:
                score = row[f"{category}_score"]
                tag = row[f"{category}_tag"]
                count = row.get(f"{category}_count", 0)
                if score != 0.0 or tag != "":
                    vibe_vector[category] = (score, tag, count)
            
            logger.info(f"Retrieved vibe vector for {city} at {timestamp}")
            return vibe_vector
            
        except Exception as e:
            logger.error(f"Failed to retrieve vibe vector at time: {e}")
            return None

    
    def get_headlines_for_city(
        self,
        cache_key: str,
        fg_name: str = "headline_classifications",
        version: int = 1,
    ) -> List[Dict]:
        """
        Retrieve individual headlines from the feature store for a cache_key.
        
        Args:
            cache_key: Cache key (format: city_YYYY-MM-DD_HH-HH)
            fg_name: Feature group name
            version: Feature group version
            
        Returns:
            List of article dicts with title, url, source, and category scores/tags
        """
        if not self._fs:
            self.connect()
        
        try:
            # Parse cache_key to extract city and time window
            from storage.core import VibeHash
            
            cache_info = VibeHash.extract_info(cache_key)
            if not cache_info:
                logger.error(f"Invalid cache_key format: {cache_key}")
                return []
            
            city = cache_info["city"].title()
            date = cache_info["date"]
            window = cache_info["window"]
            
            # Parse window to get start and end hours (e.g., "00-06" -> 0 and 6)
            window_start_hour, window_end_hour = map(int, window.split("-"))
            
            # Create timestamp range for the 6-hour window
            timestamp_start = date.replace(hour=window_start_hour, minute=0, second=0, microsecond=0)
            timestamp_end = date.replace(hour=window_end_hour, minute=0, second=0, microsecond=0)
            
            fg = self.get_or_create_headline_feature_group(fg_name, version)
            
            # Query for city and timestamp range
            query = fg.select_all().filter(
                (Feature('city').like(city)) & 
                (fg.timestamp >= timestamp_start) & 
                (fg.timestamp <= timestamp_end)
            )
            
            df = query.read()
            
            if df.empty:
                logger.warning(f"No headlines found for cache_key {cache_key} (city={city}, window={window})")
                return []
            
            # Convert to list of dicts with structured classifications
            headlines = []
            
            for _, row in df.iterrows():
                headline = {
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "url": row["url"],
                    "source": row["source"],
                    "classifications": {}
                }
                
                # Add category scores and tags
                for category in self.SIGNAL_CATEGORIES:
                    score = row.get(f"{category}_score", 0.0)
                    tag = row.get(f"{category}_tag", "")
                    if score != 0.0 or tag:  # Only include if there's data
                        headline["classifications"][category] = {
                            "score": score,
                            "tag": tag
                        }
                
                headlines.append(headline)
            
            logger.info(f"Retrieved {len(headlines)} headlines for cache_key {cache_key}")
            return headlines
            
        except Exception as e:
            logger.error(f"Failed to retrieve headlines: {e}")
            return []



    def register_model(
        self,
        model_dir: str,
        name: str,
        metrics: Optional[Dict] = None,
        description: str = "",
        version: Optional[int] = None,
    ):
        """
        Register a trained model in the Hopsworks Model Registry.

        Args:
            model_dir: Local directory containing the serialized model artifacts
            name: Model name in the registry
            metrics: Optional evaluation metrics to log
            description: Optional description
            version: Optional version to set; if None, Hopsworks will auto-increment
        """
        if not self._mr:
            self.connect()

        try:
            model = self._mr.python.create_model(
                name=name,
                metrics=metrics or {},
                description=description,
                version=version,
            )
            model.save(model_dir)
            logger.info(f"Registered model '{name}' from {model_dir}")
            return model
        except Exception as e:
            logger.error(f"Failed to register model '{name}': {e}")
            raise


class MockHopsworksService:
    """
    Mock Hopsworks service for development/testing without Hopsworks credentials.
    """
    
    def __init__(self, **kwargs):
        """Initialize mock service (accepts any args for compatibility)."""
        self.storage: Dict[str, Tuple[bytes, Dict]] = {}
        self.vibes: Dict[str, Dict] = {}
        self.headlines: Dict[str, List[Dict]] = {}
        logger.info("Using MockHopsworksService")
    
    def connect(self):
        """Mock connect."""
        logger.info("Mock: Connected to Hopsworks")
        
    def get_or_create_headline_feature_group(self, fg_name: str = "headline_classifications", version: int = 1):
        """Mock headline feature group getter."""
        logger.info(f"Mock: Using headline feature group {fg_name} v{version}")
        return None
        
    def get_or_create_vibe_feature_group(self, fg_name: str = "vibe_vectors", version: int = 1):
        """Mock vibe feature group getter."""
        logger.info(f"Mock: Using vibe feature group {fg_name} v{version}")
        return None
    
    def get_or_create_templates_feature_group(self, fg_name: str = "signal_templates", version: int = 1):
        """Mock templates feature group getter."""
        logger.info(f"Mock: Using templates feature group {fg_name} v{version}")
        return self
    
    def get_or_create_keywords_feature_group(self, fg_name: str = "tag_keywords", version: int = 1):
        """Mock keywords feature group getter."""
        logger.info(f"Mock: Using keywords feature group {fg_name} v{version}")
        return self
    
    def get_or_create_headline_labels_feature_group(self, fg_name: str = "headline_labels", version: int = 1):
        """Mock headline labels feature group getter."""
        logger.info(f"Mock: Using headline labels feature group {fg_name} v{version}")
        return self
        
    def store_headline_classifications(self, headlines: List[Dict], city: str, timestamp: datetime, **kwargs):
        """Mock headline storage."""
        key = f"{city}_{timestamp.isoformat()}"
        self.headlines[key] = headlines
        logger.info(f"Mock: Stored {len(headlines)} headline classifications for {city}")
    
    def store_signal_templates(self, templates: Dict[str, List[str]], **kwargs):
        """Mock signal templates storage."""
        total = sum(len(t) for t in templates.values())
        logger.info(f"Mock: Stored {total} signal templates")
    
    def store_tag_keywords(self, keywords: Dict[str, Dict[str, str]], **kwargs):
        """Mock tag keywords storage."""
        total = sum(len(k) for k in keywords.values())
        logger.info(f"Mock: Stored {total} tag keywords")
    
    def store_labeled_dataset(self, df, **kwargs):
        """Mock labeled dataset storage."""
        logger.info(f"Mock: Stored {len(df)} labeled headlines")
        
    def store_vibe_vector(self, city: str, timestamp: datetime, vibe_vector: Dict, **kwargs):
        """Mock vibe vector storage."""
        key = f"{city}_{timestamp.isoformat()}"
        self.vibes[key] = vibe_vector
        logger.info(f"Mock: Stored vibe vector for {city}")
        
    def get_latest_vibe_vector(self, city: str, **kwargs) -> Optional[Dict]:
        """Mock vibe vector retrieval."""
        # Return most recent for city
        matching = [k for k in self.vibes.keys() if k.startswith(city)]
        if not matching:
            return None
        latest_key = sorted(matching)[-1]
        return self.vibes[latest_key]
        
    def store_visualization(self, vibe_hash: str, image_data: bytes, metadata: Dict):
        """Mock visualization storage."""
        self.storage[vibe_hash] = (image_data, metadata)
        logger.info(f"Mock: Stored visualization {vibe_hash}")
        
    def get_visualization(self, vibe_hash: str) -> Optional[Tuple[bytes, Dict]]:
        """Mock visualization retrieval."""
        return self.storage.get(vibe_hash)
    
    def insert(self, df):
        """Mock insert method (for when mock is returned as feature group)."""
        logger.info(f"Mock: Inserted {len(df)} rows")
        return None
        
    def visualization_exists(self, vibe_hash: str) -> bool:
        """Mock visualization existence check."""
        return vibe_hash in self.storage


def get_or_create_hopsworks_service(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    host: Optional[str] = None,
) -> Optional[HopsworksService]:
    """
    Factory function to get or create HopsworksService.
    
    Args:
        api_key: Hopsworks API key
        project_name: Project name
        host: Hopsworks host
        
    Returns:
        HopsworksService instance or None if disabled/missing credentials
    """
    global instance
    if instance:
        return instance

    if not api_key or not project_name:
        logger.warning("Hopsworks API key or project name not configured")
        logger.info("Using MockHopsworksService")
        instance = MockHopsworksService()
        return instance

    try:
        instance = HopsworksService(
            api_key=api_key,
            project_name=project_name,
            host=host,
        )
        logger.info(f"HopsworksService created for project: {project_name}")
        return instance
    except Exception as e:
        logger.error(f"Failed to create HopsworksService: {e}")
        raise
