from typing import Dict, List
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class LinkSuggestion:
    source_url: str
    target_url: str
    relevance_score: float
    anchor_text: str
    context: str

class LinkOptimizer:
    """Optimizes internal linking structure using efficient algorithms."""
    
    def __init__(self, min_similarity: float = 0.80, use_ai_features: bool = False):
        self.min_similarity = min_similarity
        self.use_ai_features = use_ai_features
        self.logger = logging.getLogger(__name__)
        
        # Initialize TF-IDF vectorizer with optimized settings
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=3000,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        # Initialize results cache
        self.similarity_cache = {}
        
    @lru_cache(maxsize=1000)
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts efficiently."""
        try:
            # Transform texts to TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0
            
    def find_best_anchor_text(self, source_entities: List[Dict], 
                             target_entities: List[Dict]) -> str:
        """Find the best anchor text based on entity overlap."""
        # Sort entities by salience score
        source_entities = sorted(source_entities, key=lambda x: x['salience'], reverse=True)
        target_entities = sorted(target_entities, key=lambda x: x['salience'], reverse=True)
        
        # Find overlapping entities
        source_texts = {e['text'].lower() for e in source_entities}
        target_texts = {e['text'].lower() for e in target_entities}
        common_entities = source_texts.intersection(target_texts)
        
        if common_entities:
            # Get the most salient common entity
            best_entity = max(
                (e for e in source_entities if e['text'].lower() in common_entities),
                key=lambda x: x['salience']
            )
            return best_entity['text']
            
        # Fallback to most salient target entity
        return target_entities[0]['text'] if target_entities else ""
        
    def process_url_pair(self, source_url: str, target_url: str, 
                        analyzed_data: Dict) -> LinkSuggestion:
        """Process a pair of URLs to generate link suggestions."""
        try:
            source_data = analyzed_data[source_url]
            target_data = analyzed_data[target_url]
            
            # Calculate similarity score
            cache_key = (source_url, target_url)
            if cache_key not in self.similarity_cache:
                similarity = self.calculate_similarity(
                    source_data['content'],
                    target_data['content']
                )
                self.similarity_cache[cache_key] = similarity
            else:
                similarity = self.similarity_cache[cache_key]
            
            if similarity >= self.min_similarity:
                # Find best anchor text
                anchor_text = self.find_best_anchor_text(
                    source_data['entities'],
                    target_data['entities']
                )
                
                # Get context (first paragraph containing the anchor text)
                context = ""
                if anchor_text:
                    paragraphs = source_data['content'].split('\n')
                    for para in paragraphs:
                        if anchor_text.lower() in para.lower():
                            context = para[:200] + "..." if len(para) > 200 else para
                            break
                
                return LinkSuggestion(
                    source_url=source_url,
                    target_url=target_url,
                    relevance_score=similarity,
                    anchor_text=anchor_text,
                    context=context
                )
                
        except Exception as e:
            self.logger.error(f"Error processing {source_url} -> {target_url}: {str(e)}")
            
        return None
        
    def optimize(self, analyzed_data: Dict) -> List[LinkSuggestion]:
        """Generate internal linking suggestions efficiently."""
        urls = list(analyzed_data.keys())
        suggestions = []
        
        # Group URLs by domain for more relevant suggestions
        domain_groups = defaultdict(list)
        for url in urls:
            domain = url.split('/')[2]  # Extract domain from URL
            domain_groups[domain].append(url)
        
        # Process each domain group in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            for domain_urls in domain_groups.values():
                # Only process URLs within the same domain
                for i, source_url in enumerate(domain_urls):
                    for target_url in domain_urls[i+1:]:
                        futures.append(
                            executor.submit(
                                self.process_url_pair,
                                source_url,
                                target_url,
                                analyzed_data
                            )
                        )
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        suggestions.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing URL pair: {str(e)}")
        
        # Sort suggestions by relevance score
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit suggestions per URL
        url_suggestions = defaultdict(list)
        for sugg in suggestions:
            if len(url_suggestions[sugg.source_url]) < 10:  # Max 10 suggestions per URL
                url_suggestions[sugg.source_url].append(sugg)
        
        # Flatten and return final suggestions
        return [
            sugg for suggestions in url_suggestions.values()
            for sugg in suggestions
        ]
