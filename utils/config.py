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
        
        # Default thresholds
        self.similarity_threshold = 0.3
        
        # Scraping settings
        self.default_scraping_rate = 3
        self.max_scraping_rate = 5
        self.request_timeout = 10
        
        # Content analysis settings
        self.max_content_length = 100000  # Maximum content length to analyze
        self.min_content_length = 100     # Minimum content length to analyze
        
        # Link optimization settings
        self.max_links_per_page = 10
        self.max_anchor_text_length = 100
        self.min_anchor_text_length = 3
        
        # Cache settings
        self.cache_duration = 7 * 24 * 60 * 60  # 7 days in seconds
