import os
import json
import time
from typing import Dict, Any
import logging
from pathlib import Path

class CacheManager:
    """Manages caching of scraped content and analysis results."""
    
    def __init__(self, cache_dir: str = None, cache_duration: int = 7 * 24 * 60 * 60):
        """Initialize the cache manager."""
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'cache')
        self.cache_duration = cache_duration  # 7 days in seconds
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        # Create a safe filename from the key
        safe_key = "".join(c for c in key if c.isalnum() or c in ('-', '_')).rstrip()
        return os.path.join(self.cache_dir, f"{safe_key}.json")
        
    def get(self, key: str) -> Dict[str, Any]:
        """Get data from cache if it exists and is not expired."""
        try:
            cache_path = self._get_cache_path(key)
            
            if not os.path.exists(cache_path):
                return None
                
            # Check if cache is expired
            if time.time() - os.path.getmtime(cache_path) > self.cache_duration:
                os.remove(cache_path)
                return None
                
            with open(cache_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error reading from cache: {str(e)}")
            return None
            
    def set(self, key: str, data: Dict[str, Any]) -> bool:
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(key)
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to cache: {str(e)}")
            return False
            
    def clear(self, key: str = None) -> bool:
        """Clear specific cache entry or all cache."""
        try:
            if key:
                cache_path = self._get_cache_path(key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            else:
                # Clear all cache files
                for file in os.listdir(self.cache_dir):
                    os.remove(os.path.join(self.cache_dir, file))
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
            
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        cleaned = 0
        try:
            current_time = time.time()
            for file in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file)
                if current_time - os.path.getmtime(file_path) > self.cache_duration:
                    os.remove(file_path)
                    cleaned += 1
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {str(e)}")
            
        return cleaned
