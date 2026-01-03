"""
Hopsworks integration service for The Daily Collage.

Handles:
1. Feature Store: Vibe vectors and signal data
2. Model Registry: ML classification models
3. Artifact Store: Generated visualizations
"""

import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io

from PIL import Image

logger = logging.getLogger(__name__)


class HopsworksService:
    """
    Service for interacting with Hopsworks Feature Store and Artifact Store.
    
    Responsibilities:
    - Store/retrieve vibe vectors in feature groups
    - Store/retrieve generated images in artifact registry
    - Query historical vibes for comparison
    """
    
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
        self._project = None
        self._fs = None
        self._mr = None
        
    def connect(self):
        """Establish connection to Hopsworks."""
        try:
            import hopsworks
            
            connection_args = {
                "project": self.project_name,
                "api_key_value": self.api_key,
            }
            
            if self.host:
                # Parse host to extract region if it contains "c.app.hopsworks.ai"
                if "c.app.hopsworks.ai" in self.host or "cloud.hopsworks.ai" in self.host:
                    # Extract region from host (e.g., "c.app.hopsworks.ai" or specific region)
                    connection_args["host"] = "c.app.hopsworks.ai"
                else:
                    connection_args["host"] = self.host
                    
            self._project = hopsworks.login(**connection_args)
            self._fs = self._project.get_feature_store()
            self._mr = self._project.get_model_registry()
            
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
            for category in [
                "emergencies", "crime", "festivals", "transportation",
                "weather_temp", "weather_wet", "sports", "economics", "politics"
            ]:
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
        for category in [
            "emergencies", "crime", "festivals", "transportation",
            "weather_temp", "weather_wet", "sports", "economics", "politics"
        ]:
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
            for category in [
                "emergencies", "crime", "festivals", "transportation",
                "weather_temp", "weather_wet", "sports", "economics", "politics"
            ]:
                score = row[f"{category}_score"]
                tag = row[f"{category}_tag"]
                count = row.get(f"{category}_count", 0)
                if score != 0.0 or tag != "":
                    vibe_vector[category] = (score, tag, count)
                    
            return vibe_vector
            
        except Exception as e:
            logger.error(f"Failed to retrieve vibe vector: {e}")
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
            
            return image_data, metadata
            
        except Exception as e:
            logger.debug(f"Visualization {vibe_hash} not found in Hopsworks: {e}")
            return None
    
    def visualization_exists(self, vibe_hash: str, artifact_collection: str = "vibe_images") -> bool:
        """
        Check if a visualization exists in Hopsworks artifact registry.
        
        Args:
            vibe_hash: Unique hash for the visualization
            artifact_collection: Name of artifact collection to check in
            
        Returns:
            True if exists, False otherwise
        """
        if not self._project:
            self.connect()
            
        try:
            # Get artifacts API
            artifacts_api = self._project.get_artifacts_api()
            
            # List artifacts and check if our artifact exists
            artifacts = artifacts_api.list(collection=artifact_collection)
            
            for artifact in artifacts:
                if artifact.name == f"{vibe_hash}.png":
                    return True
            
            return False
            
        except Exception:
            return False


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
        
    def store_headline_classifications(self, headlines: List[Dict], city: str, timestamp: datetime, **kwargs):
        """Mock headline storage."""
        key = f"{city}_{timestamp.isoformat()}"
        self.headlines[key] = headlines
        logger.info(f"Mock: Stored {len(headlines)} headline classifications for {city}")
        
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
        
    def visualization_exists(self, vibe_hash: str) -> bool:
        """Mock visualization existence check."""
        return vibe_hash in self.storage


def create_hopsworks_service(
    enabled: bool = True,
    api_key: Optional[str] = None,
    project_name: str = "daily_collage",
    host: Optional[str] = None,
) -> HopsworksService:
    """
    Factory function to create appropriate Hopsworks service.
    
    Args:
        enabled: If False, use mock service
        api_key: Hopsworks API key
        project_name: Project name
        host: Hopsworks host
        
    Returns:
        HopsworksService or MockHopsworksService
    """
    if not enabled or not api_key:
        logger.info("Using MockHopsworksService")
        return MockHopsworksService()
        
    logger.info("Using real HopsworksService")
    return HopsworksService(
        api_key=api_key,
        project_name=project_name,
        host=host,
    )