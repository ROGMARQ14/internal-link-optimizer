import pandas as pd
from typing import Dict, List
import logging
from urllib.parse import urlparse

class URLProcessor:
    """Processes a list of URLs for internal linking analysis."""
    
    def __init__(self, urls: List[str]):
        """Initialize the processor with a list of URLs."""
        self.urls = urls
        self.logger = logging.getLogger(__name__)
        
    def validate_urls(self, urls: List[str]) -> List[str]:
        """Validate URLs and ensure they are properly formatted."""
        valid_urls = []
        for url in urls:
            try:
                # Clean the URL
                url = url.strip().lower()
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                # Parse URL to validate
                parsed = urlparse(url)
                if parsed.netloc and parsed.scheme:
                    valid_urls.append(url.rstrip('/'))
                else:
                    self.logger.warning(f"Invalid URL format: {url}")
            except Exception as e:
                self.logger.error(f"Error processing URL {url}: {str(e)}")
                
        return valid_urls
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc
    
    def group_by_domain(self, urls: List[str]) -> Dict[str, List[str]]:
        """Group URLs by their domain."""
        domains = {}
        for url in urls:
            domain = self.extract_domain(url)
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(url)
        return domains
    
    def process(self) -> Dict:
        """Process the URLs and return processed information."""
        try:
            # Validate URLs
            valid_urls = self.validate_urls(self.urls)
            
            if not valid_urls:
                raise ValueError("No valid URLs provided")
            
            # Group URLs by domain
            domain_groups = self.group_by_domain(valid_urls)
            
            # Create basic structure for each URL
            url_data = {}
            for url in valid_urls:
                url_data[url] = {
                    'domain': self.extract_domain(url),
                    'path': urlparse(url).path
                }
            
            return {
                'urls': valid_urls,
                'domains': list(domain_groups.keys()),
                'url_data': url_data,
                'domain_groups': domain_groups
            }
            
        except Exception as e:
            self.logger.error(f"Error processing URLs: {str(e)}")
            raise
