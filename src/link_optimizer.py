from typing import Dict, List
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import multiprocessing
from itertools import combinations
import openai
from dataclasses import dataclass
import json

@dataclass
class LinkSuggestion:
    source_url: str
    target_url: str
    anchor_text: str
    context: str
    similarity_score: float
    content_suggestion: str = None
    enhanced_entities: List[str] = None

class LinkOptimizer:
    """Optimizes internal linking using efficient similarity calculations with optional AI enhancements."""
    
    def __init__(self, min_similarity: float = 0.3, use_ai_features: bool = False):
        """Initialize the optimizer.
        
        Args:
            min_similarity: Minimum similarity score for link suggestions
            use_ai_features: Whether to use OpenAI-powered enhancements
        """
        self.min_similarity = min_similarity
        self.use_ai_features = use_ai_features
        self.logger = logging.getLogger(__name__)
        
    def calculate_similarity_score(self, source_entities: List[Dict], 
                                 target_entities: List[Dict]) -> float:
        """Calculate similarity score between two pages based on their entities."""
        if not source_entities or not target_entities:
            return 0.0
            
        # Create entity sets with weights
        source_dict = {e['text']: e['salience'] for e in source_entities}
        target_dict = {e['text']: e['salience'] for e in target_entities}
        
        # Find common entities
        common_entities = set(source_dict.keys()) & set(target_dict.keys())
        if not common_entities:
            return 0.0
            
        # Calculate weighted similarity
        similarity = sum(source_dict[entity] * target_dict[entity] 
                        for entity in common_entities)
        
        # Normalize
        max_possible = max(
            sum(source_dict.values()),
            sum(target_dict.values())
        )
        
        return similarity / max_possible if max_possible > 0 else 0.0
        
    def enhance_entities_with_ai(self, content: str, existing_entities: List[Dict]) -> List[str]:
        """Use OpenAI to discover additional relevant entities and keywords."""
        if not self.use_ai_features:
            return []
            
        try:
            prompt = f"""
            Content excerpt: {content[:1000]}...
            
            Existing entities: {[e['text'] for e in existing_entities]}
            
            Task: Identify 5-10 additional relevant entities, keywords, or phrases that could be 
            good candidates for internal linking. Focus on:
            1. Industry-specific terms
            2. Related concepts
            3. Synonyms or alternative phrasings
            4. Broader/narrower terms
            
            Return as a JSON array of strings.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an SEO expert specializing in content analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response as JSON array
            enhanced_entities = json.loads(response.choices[0].message.content)
            return enhanced_entities
            
        except Exception as e:
            self.logger.error(f"Error enhancing entities with AI: {str(e)}")
            return []
            
    def generate_content_suggestion(self, source_content: str, target_content: str, 
                                  anchor_text: str) -> str:
        """Generate a natural content suggestion for adding the link."""
        if not self.use_ai_features:
            return None
            
        try:
            prompt = f"""
            Source content excerpt: {source_content[:500]}...
            Target page topic: {target_content[:200]}...
            Desired anchor text: {anchor_text}
            
            Task: Generate a natural, 1-2 sentence suggestion for adding a link using the anchor text.
            The suggestion should:
            1. Flow naturally with the existing content
            2. Provide value to readers
            3. Make sense in the context
            4. Be concise and focused
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert content editor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating content suggestion: {str(e)}")
            return None
        
    def find_anchor_text(self, source_content: str, target_entities: List[Dict]) -> str:
        """Find the best anchor text from the source content."""
        best_anchor = None
        best_score = 0
        
        # Sort entities by salience
        target_entities = sorted(target_entities, key=lambda x: x['salience'], reverse=True)
        
        # Try to find entity mentions in the source content
        for entity in target_entities:
            entity_text = entity['text'].lower()
            if entity_text in source_content.lower():
                score = entity['salience']
                if score > best_score:
                    best_score = score
                    best_anchor = entity_text
                    
        return best_anchor.title() if best_anchor else target_entities[0]['text'].title()
        
    def process_url_pair(self, pair: tuple, analyzed_data: Dict) -> LinkSuggestion:
        """Process a pair of URLs for linking opportunities."""
        source_url, target_url = pair
        
        # Skip self-linking
        if source_url == target_url:
            return None
            
        source_data = analyzed_data[source_url]
        target_data = analyzed_data[target_url]
        
        # Calculate similarity
        similarity = self.calculate_similarity_score(
            source_data['entities'],
            target_data['entities']
        )
        
        if similarity >= self.min_similarity:
            anchor_text = self.find_anchor_text(
                source_data['content'],
                target_data['entities']
            )
            
            # Optional AI enhancements
            enhanced_entities = None
            content_suggestion = None
            
            if self.use_ai_features:
                enhanced_entities = self.enhance_entities_with_ai(
                    target_data['content'],
                    target_data['entities']
                )
                content_suggestion = self.generate_content_suggestion(
                    source_data['content'],
                    target_data['content'],
                    anchor_text
                )
            
            return LinkSuggestion(
                source_url=source_url,
                target_url=target_url,
                anchor_text=anchor_text,
                context=source_data['content'][:200],  # Preview
                similarity_score=similarity,
                content_suggestion=content_suggestion,
                enhanced_entities=enhanced_entities
            )
            
        return None
        
    def optimize(self, analyzed_data: Dict) -> List[LinkSuggestion]:
        """Generate internal linking suggestions efficiently."""
        urls = list(analyzed_data.keys())
        suggestions = []
        
        # Generate URL pairs
        url_pairs = list(combinations(urls, 2))
        
        # Process pairs in parallel
        with multiprocessing.Pool() as pool:
            results = pool.starmap(
                self.process_url_pair,
                [(pair, analyzed_data) for pair in url_pairs]
            )
            
        # Filter and sort results
        suggestions = [r for r in results if r is not None]
        suggestions.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return suggestions
