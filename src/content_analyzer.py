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

class ContentAnalyzer:
    """Analyzes webpage content using efficient NLP techniques."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize the content analyzer with configuration."""
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            # Download if not available
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Configure session for efficient requests
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1)
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
    def scrape_content(self, url: str) -> str:
        """Scrape content from a URL efficiently."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'header', 'footer', 'nav']):
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
            
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        ner_entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Count frequencies
        entity_freq = Counter(ent[0].lower() for ent in ner_entities)
        phrase_freq = Counter(phrase.lower() for phrase in noun_phrases)
        
        # Combine and filter entities
        for text, count in {**entity_freq, **phrase_freq}.items():
            if count >= 2:  # Only include entities that appear multiple times
                entities.append({
                    'text': text,
                    'count': count,
                    'salience': count / len(doc)  # Simple salience score
                })
        
        return entities
            
    def analyze_style(self, text: str) -> Dict:
        """Analyze writing style efficiently."""
        doc = self.nlp(text)
        
        sentences = list(doc.sents)
        return {
            'avg_sentence_length': len(doc) / len(sentences) if sentences else 0,
            'vocabulary_richness': len(set(token.text.lower() for token in doc)) / len(doc) if doc else 0
        }
        
    def process_url(self, url: str) -> Dict:
        """Process a single URL."""
        content = self.scrape_content(url)
        if not content:
            return None
            
        # Process content in parallel
        with multiprocessing.Pool() as pool:
            entities_future = pool.apply_async(self.extract_entities, (content,))
            style_future = pool.apply_async(self.analyze_style, (content,))
            
            entities = entities_future.get()
            style_metrics = style_future.get()
        
        return {
            'url': url,
            'content': content,
            'entities': entities,
            'style_metrics': style_metrics
        }
        
    def analyze(self, url_data: Dict, scraping_rate: int = 3,
                progress_callback: Callable = None) -> Dict:
        """Analyze all URLs efficiently using parallel processing."""
        urls = url_data['urls']
        total_urls = len(urls)
        analyzed_data = {}
        
        # Process URLs in parallel
        with ThreadPoolExecutor(max_workers=scraping_rate) as executor:
            future_to_url = {
                executor.submit(self.process_url, url): url
                for url in urls
            }
            
            completed = 0
            for future in as_completed(future_to_url):
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
