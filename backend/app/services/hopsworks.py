"""
Hopsworks integration service for The Daily Collage.

Handles:
1. Feature Store: Vibe vectors and signal data
2. Model Registry: ML classification models
3. Artifact Store: Generated visualizations
"""

import logging, hopsworks
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from hopsworks.project import Project
from hopsworks.core.dataset_api import DatasetApi
from hsfs.feature_store import FeatureStore

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
            connection_args = {
                "project": self.project_name,
                "api_key_value": self.api_key,
            }
            
            if self.host:
                connection_args["host"] = self.host
                    
            self._project = hopsworks.login(**connection_args)
            self._fs = self._project.get_feature_store()
            self._dataset_api = self._project.get_dataset_api()
            
            logger.info(f"Connected to Hopsworks project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            raise
            
    def get_or_create_headline_feature_group(self, fg_name: str = "headline_classifications", version: int = 1):
        """
        Get or create feature group for storing individual headline classifications.
        
        Schema:
        - article_id: string (primary key)
        - city: string
        - timestamp: timestamp (event_time)
        - title: string
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
        - city: string
        - timestamp: timestamp
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
                primary_key=["city", "timestamp"],
                event_time="timestamp",
                online_enabled=True,
            )
            logger.info(f"Created feature group: {fg_name} v{version}")
            return fg
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
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
                "timestamp": timestamp,
                "title": headline.get("title", ""),
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
        row = {
            "city": city,
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
            
            # Query for exact city and timestamp match
            query = fg.select_all().filter(
                (fg.city == city) & (fg.timestamp == timestamp)
            )
            df = query.read()
            
            if df.empty:
                logger.warning(f"No vibe vector found for {city} at {timestamp}")
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
    
    def store_visualization(
        self,
        vibe_hash: str,
        image_data: bytes,
        metadata: Dict,
        artifact_collection: str = "vibe_images",
    ):
        """
        Store a generated visualization in Hopsworks artifact registry.
        
        Args:
            vibe_hash: Unique hash for this visualization
            image_data: PNG image bytes
            metadata: Metadata dict (hitboxes, vibe_vector, etc.)
            artifact_collection: Name of artifact collection to store in
        """
        if not self._project:
            self.connect()
            
        try:
            import io
            import json
            
            # Get artifacts API
            artifacts_api = self._project.get_artifacts_api()
            
            # Upload image as artifact
            image_file = io.BytesIO(image_data)
            image_file.name = f"{vibe_hash}.png"
            artifacts_api.upload(
                artifact=image_file,
                name=f"{vibe_hash}.png",
                collection=artifact_collection,
                description=f"Vibe visualization for {vibe_hash}",
            )
            
            # Upload metadata as JSON artifact
            metadata_json = json.dumps(metadata, indent=2)
            metadata_file = io.BytesIO(metadata_json.encode())
            metadata_file.name = f"{vibe_hash}_metadata.json"
            artifacts_api.upload(
                artifact=metadata_file,
                name=f"{vibe_hash}_metadata.json",
                collection=artifact_collection,
                description=f"Metadata for vibe visualization {vibe_hash}",
            )
            
            logger.info(f"Stored visualization {vibe_hash} in Hopsworks artifacts")
            
        except Exception as e:
            logger.error(f"Failed to store visualization: {e}")
            raise
    
    def get_visualization(
        self,
        vibe_hash: str,
        artifact_collection: str = "vibe_images",
    ) -> Optional[Tuple[bytes, Dict]]:
        """
        Retrieve a visualization from Hopsworks artifact registry.
        
        Args:
            vibe_hash: Unique hash for the visualization
            artifact_collection: Name of artifact collection to retrieve from
            
        Returns:
            Tuple of (image_bytes, metadata_dict) or None if not found
        """
        if not self._project:
            self.connect()
            
        try:
            import json
            
            # Get artifacts API
            artifacts_api = self._project.get_artifacts_api()
            
            # Download image artifact
            image_path = artifacts_api.download(
                name=f"{vibe_hash}.png",
                collection=artifact_collection,
            )
            
            # Download metadata artifact
            metadata_path = artifacts_api.download(
                name=f"{vibe_hash}_metadata.json",
                collection=artifact_collection,
            )
            
            if not image_path or not metadata_path:
                return None
            
            # Read image data
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Read metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            logger.info(f"Retrieved visualization from Hopsworks: {vibe_hash}")
            return image_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve visualization: {e}")
            return None
    
    def get_headlines_for_city(
        self,
        city: str,
        timestamp: datetime,
        fg_name: str = "headline_classifications",
        version: int = 1,
    ) -> List[Dict]:
        """
        Retrieve individual headlines from the feature store for a city and time.
        
        Args:
            city: City name
            timestamp: Time window timestamp
            fg_name: Feature group name
            version: Feature group version
            
        Returns:
            List of article dicts with title, url, source, and category scores/tags
        """
        if not self._fs:
            self.connect()
        
        try:
            fg = self.get_or_create_headline_feature_group(fg_name, version)
            
            # Query for city and timestamp
            query = fg.select_all().filter(
                (fg.city == city) & (fg.timestamp == timestamp)
            )
            
            df = query.read()
            
            if df.empty:
                logger.warning(f"No headlines found for {city} at {timestamp}")
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
            
            logger.info(f"Retrieved {len(headlines)} headlines for {city} at {timestamp}")
            return headlines
            
        except Exception as e:
            logger.error(f"Failed to retrieve headlines: {e}")
            return []


def get_or_create_hopsworks_service(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    host: Optional[str] = None,
) -> Optional[HopsworksService]:
    """
    Factory function to create HopsworksService.
    
    Args:
        enabled: Whether Hopsworks integration is enabled
        api_key: Hopsworks API key
        project_name: Project name
        host: Hopsworks host
        
    Returns:
        HopsworksService instance or None if disabled/missing credentials
    """
    if not api_key or not project_name:
        logger.warning("Hopsworks API key or project name not configured")
        return None
    
    try:
        global instance
        if instance:
            return instance

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
