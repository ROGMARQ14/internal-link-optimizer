from typing import Dict, List, Callable
import logging
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoTokenizer, AutoModel
import torch
from google.cloud import language_v2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests

class ContentAnalyzer:
    """Analyzes webpage content using various NLP techniques."""
    
    def __init__(self, similarity_threshold: float = 0.85, entity_threshold: float = 0.85):
        """Initialize the content analyzer with configuration."""
        self.similarity_threshold = similarity_threshold
        self.entity_threshold = entity_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Initialize Google Cloud Language client
        self.language_client = language_v2.LanguageServiceClient()
        
        # Setup Chrome options for Selenium
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        
    def setup_webdriver(self):
        """Setup and return a Chrome WebDriver instance."""
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=self.chrome_options)
        
    def scrape_content(self, url: str, js_timeout: int = 10) -> str:
        """Scrape content from a URL, handling JavaScript rendering."""
        try:
            driver = self.setup_webdriver()
            driver.set_page_load_timeout(js_timeout)
            driver.get(url)
            time.sleep(2)  # Allow for any dynamic content to load
            
            # Get the page source after JavaScript execution
            html_content = driver.page_source
            driver.quit()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'header', 'footer', 'nav', 'sidebar']):
                element.decompose()
                
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                # Remove image alt text
                for img in main_content.find_all('img'):
                    img.decompose()
                    
                return main_content.get_text(strip=True)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return ""
            
    def analyze_entities(self, text: str) -> List[Dict]:
        """Analyze entities using Google Cloud Natural Language API."""
        try:
            document = language_v2.Document(
                content=text,
                type_=language_v2.Document.Type.PLAIN_TEXT,
                language='en'
            )
            
            response = self.language_client.analyze_entities(
                request={'document': document}
            )
            
            entities = []
            for entity in response.entities:
                if entity.salience >= self.entity_threshold:
                    entities.append({
                        'name': entity.name,
                        'type': entity.type_.name,
                        'salience': entity.salience,
                        'mentions': [mention.text.content for mention in entity.mentions]
                    })
                    
            return entities
            
        except Exception as e:
            self.logger.error(f"Error analyzing entities: {str(e)}")
            return []
            
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for text."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
        
    def analyze_style(self, text: str) -> Dict:
        """Analyze writing style of the content."""
        sentences = text.split('.')
        words = text.split()
        
        return {
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'vocabulary_richness': len(set(words)) / len(words) if words else 0,
            'formality_score': self._calculate_formality_score(text),
            'readability_score': self._calculate_readability_score(text)
        }
        
    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score based on various indicators."""
        # Implement formality scoring logic
        return 0.0
        
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score."""
        # Implement readability scoring logic
        return 0.0
        
    def analyze_url(self, url: str, js_timeout: int) -> Dict:
        """Analyze a single URL."""
        content = self.scrape_content(url, js_timeout)
        if not content:
            return None
            
        return {
            'url': url,
            'content': content,
            'entities': self.analyze_entities(content),
            'embedding': self.get_bert_embedding(content),
            'style_metrics': self.analyze_style(content)
        }
        
    def analyze(self, gsc_data: Dict, scraping_rate: int = 3,
                js_timeout: int = 10, progress_callback: Callable = None) -> Dict:
        """Analyze all URLs and their content."""
        urls = gsc_data['unique_urls']
        total_urls = len(urls)
        analyzed_data = {}
        
        with ThreadPoolExecutor(max_workers=scraping_rate) as executor:
            future_to_url = {
                executor.submit(self.analyze_url, url, js_timeout): url
                for url in urls
            }
            
            completed = 0
            for future in futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        analyzed_data[url] = result
                except Exception as e:
                    self.logger.error(f"Error analyzing {url}: {str(e)}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed / total_urls)
                
                # Respect scraping rate
                time.sleep(1 / scraping_rate)
        
        return analyzed_data
