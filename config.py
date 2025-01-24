import os
from dotenv import load_dotenv
from dataclasses import dataclass
import logging

@dataclass
class Config:
    """Configuration settings for the application."""
    
    def __init__(self):
        """Initialize configuration with environment variables."""
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        # API Keys
        self.google_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.keywords_everywhere_api_key = os.getenv('KEYWORDS_EVERYWHERE_API_KEY')
        
        # Default thresholds
        self.similarity_threshold = 0.85
        self.entity_threshold = 0.85
        self.confidence_threshold = 0.85
        
        # Scraping settings
        self.default_scraping_rate = 3
        self.max_scraping_rate = 5
        self.js_timeout = 10
        
        # Content analysis settings
        self.max_content_length = 100000  # Maximum content length to analyze
        self.min_content_length = 100     # Minimum content length to analyze
        
        # Link optimization settings
        self.max_links_per_page = 10
        self.max_anchor_text_length = 100
        self.min_anchor_text_length = 3
        
        # Cache settings
        self.cache_duration = 7 * 24 * 60 * 60  # 7 days in seconds
        
        self._validate_config()
        
    def _validate_config(self):
        """Validate the configuration settings."""
        missing_keys = []
        
        if not self.google_credentials:
            missing_keys.append('GOOGLE_APPLICATION_CREDENTIALS')
        if not self.openai_api_key:
            missing_keys.append('OPENAI_API_KEY')
        if not self.keywords_everywhere_api_key:
            missing_keys.append('KEYWORDS_EVERYWHERE_API_KEY')
            
        if missing_keys:
            self.logger.warning(
                f"Missing environment variables: {', '.join(missing_keys)}"
            )
            
    def update(self, **kwargs):
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
                
    def get_api_headers(self) -> dict:
        """Get API headers for Keywords Everywhere."""
        return {
            'Authorization': f'Bearer {self.keywords_everywhere_api_key}'
        }
