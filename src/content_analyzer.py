from typing import Dict, List, Callable
import logging
from bs4 import BeautifulSoup
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import Counter
import multiprocessing
from functools import lru_cache

class ContentAnalyzer:
    """Analyzes webpage content using efficient NLP techniques."""
    
    def __init__(self, similarity_threshold: float = 0.80, entity_threshold: float = 0.80):
        """Initialize the content analyzer with configuration.
        
        Args:
            similarity_threshold: Threshold for semantic similarity (0.0 to 1.0)
            entity_threshold: Threshold for entity relevance (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.entity_threshold = entity_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy with only necessary components
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
            # Optimize pipeline for better performance
            self.nlp.select_pipes(enable=['tagger', 'ner'])
        except OSError:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
            self.nlp.select_pipes(enable=['tagger', 'ner'])
        
        # Initialize TF-IDF vectorizer with optimized settings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=3000,  # Reduced from 5000 for better performance
            ngram_range=(1, 2),
            max_df=0.95,  # Ignore terms that appear in >95% of documents
            min_df=2      # Ignore terms that appear in <2 documents
        )
        
        # Configure session for efficient requests
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1)
        self.session.mount('http://', HTTPAdapter(max_retries=retries, pool_maxsize=20))
        self.session.mount('https://', HTTPAdapter(max_retries=retries, pool_maxsize=20))
        
        # Initialize cache for processed content
        self.content_cache = {}
        
    @lru_cache(maxsize=100)
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text[:100000] if len(text) > 100000 else text
        
    def scrape_content(self, url: str) -> str:
        """Scrape content from a URL efficiently."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Use lxml parser for better performance
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
                
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                # Remove image alt text
                for img in main_content.find_all('img'):
                    img.decompose()
                    
                return self.clean_text(main_content.get_text(strip=True))
            return ""
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return ""
            
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using spaCy efficiently."""
        # Process text in chunks for better memory usage
        max_length = self.nlp.max_length
        if len(text) > max_length:
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        else:
            chunks = [text]
            
        entities = []
        total_words = len(text.split())
        entity_counts = Counter()
        
        for chunk in chunks:
            doc = self.nlp(chunk)
            # Combine named entities and noun phrases
            chunk_entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
            entity_counts.update(ent[0] for ent in chunk_entities)
        
        # Filter and process entities
        for text, count in entity_counts.items():
            salience = count / total_words
            if salience >= self.entity_threshold and len(text.split()) <= 5:
                entities.append({
                    'text': text,
                    'count': count,
                    'salience': salience
                })
        
        return entities
            
    def analyze_style(self, text: str) -> Dict:
        """Analyze writing style efficiently."""
        # Use basic string operations instead of full parsing
        sentences = text.split('.')
        words = text.split()
        unique_words = set(words)
        
        return {
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'vocabulary_richness': len(unique_words) / len(words) if words else 0
        }
        
    def process_url(self, url: str) -> Dict:
        """Process a single URL."""
        # Check cache first
        if url in self.content_cache:
            return self.content_cache[url]
            
        content = self.scrape_content(url)
        if not content:
            return None
            
        # Process content
        entities = self.extract_entities(content)
        style_metrics = self.analyze_style(content)
        
        result = {
            'url': url,
            'content': content,
            'entities': entities,
            'style_metrics': style_metrics
        }
        
        # Cache the result
        self.content_cache[url] = result
        return result
        
    def analyze(self, url_data: Dict, scraping_rate: int = 3,
                progress_callback: Callable = None) -> Dict:
        """Analyze all URLs efficiently using parallel processing."""
        urls = url_data['urls']
        total_urls = len(urls)
        analyzed_data = {}
        
        # Calculate optimal batch size
        batch_size = min(20, max(1, total_urls // multiprocessing.cpu_count()))
        
        # Process URLs in batches
        for i in range(0, total_urls, batch_size):
            batch_urls = urls[i:i + batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(batch_size, scraping_rate)) as executor:
                future_to_url = {
                    executor.submit(self.process_url, url): url
                    for url in batch_urls
                }
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        if result:
                            analyzed_data[url] = result
                    except Exception as e:
                        self.logger.error(f"Error analyzing {url}: {str(e)}")
                    
                    if progress_callback:
                        progress_callback((i + len(analyzed_data)) / total_urls)
            
            # Small delay between batches to prevent rate limiting
            time.sleep(1 / scraping_rate)
        
        return analyzed_data
