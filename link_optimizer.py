from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dataclasses import dataclass
import logging

@dataclass
class LinkSuggestion:
    source_url: str
    target_url: str
    anchor_text: str
    context: str
    confidence_score: float
    suggestion_type: str  # 'existing', 'new_paragraph', 'modified'
    modification_extent: str  # 'none', 'minor', 'major'
    section_position: str  # 'top_25', 'upper_middle_25', 'lower_middle_25', 'bottom_25'

class LinkOptimizer:
    """Generates and optimizes internal linking suggestions."""
    
    def __init__(self):
        """Initialize the link optimizer."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
        
    def find_existing_phrase(self, content: str, query: str) -> Tuple[str, str, float]:
        """Find existing phrases that could be used as anchor text."""
        # Implement fuzzy matching to find variations of the query in content
        # Return (found_phrase, context, confidence_score)
        return None
        
    def generate_content_modification(self, 
                                    source_content: str,
                                    target_content: str,
                                    query: str,
                                    style_metrics: Dict) -> Tuple[str, str, float]:
        """Generate or modify content to include a new internal link."""
        try:
            # Prepare the prompt for GPT
            prompt = f"""
            Original content excerpt: {source_content[:500]}...
            
            Target page topic: {target_content[:200]}...
            
            Writing style metrics:
            - Average sentence length: {style_metrics['avg_sentence_length']}
            - Vocabulary richness: {style_metrics['vocabulary_richness']}
            - Formality score: {style_metrics['formality_score']}
            
            Task: Generate a natural, contextually relevant modification or new paragraph
            that includes a link to the target page using '{query}' or a variation as anchor text.
            The modification should match the original content's style and tone.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content editor."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_content = response.choices[0].message.content
            
            # Calculate confidence based on style matching and relevance
            confidence_score = self._calculate_content_confidence(
                generated_content,
                source_content,
                style_metrics
            )
            
            return generated_content, confidence_score
            
        except Exception as e:
            self.logger.error(f"Error generating content modification: {str(e)}")
            return None, 0.0
            
    def _calculate_content_confidence(self,
                                    generated_content: str,
                                    original_content: str,
                                    style_metrics: Dict) -> float:
        """Calculate confidence score for generated content."""
        # Implement confidence scoring based on style matching and relevance
        return 0.0
        
    def analyze_link_placement(self, content: str) -> Dict[str, List[str]]:
        """Analyze existing link placement in content sections."""
        # Split content into quarters
        quarters = np.array_split(content.split('\n'), 4)
        
        placement = {
            'top_25': [],
            'upper_middle_25': [],
            'lower_middle_25': [],
            'bottom_25': []
        }
        
        # Analyze each quarter for existing links
        # Implementation here
        
        return placement
        
    def optimize(self, analyzed_data: Dict) -> List[LinkSuggestion]:
        """Generate linking suggestions based on analyzed content."""
        suggestions = []
        
        # Get all URL pairs
        urls = list(analyzed_data.keys())
        
        for i, source_url in enumerate(urls):
            source_data = analyzed_data[source_url]
            source_embedding = source_data['embedding']
            source_content = source_data['content']
            source_style = source_data['style_metrics']
            
            # Analyze existing link placement
            current_links = self.analyze_link_placement(source_content)
            
            for j, target_url in enumerate(urls):
                if i == j:
                    continue
                    
                target_data = analyzed_data[target_url]
                target_embedding = target_data['embedding']
                target_content = target_data['content']
                
                # Calculate content similarity
                similarity = self.calculate_similarity(source_embedding, target_embedding)
                
                if similarity >= 0.85:  # Configurable threshold
                    # Look for existing phrases first
                    for entity in target_data['entities']:
                        existing_phrase = self.find_existing_phrase(
                            source_content,
                            entity['name']
                        )
                        
                        if existing_phrase:
                            phrase, context, conf = existing_phrase
                            suggestions.append(LinkSuggestion(
                                source_url=source_url,
                                target_url=target_url,
                                anchor_text=phrase,
                                context=context,
                                confidence_score=conf,
                                suggestion_type='existing',
                                modification_extent='none',
                                section_position=self._determine_section(context, source_content)
                            ))
                            continue
                        
                        # Try content modification if no existing phrase
                        modified_content, conf = self.generate_content_modification(
                            source_content,
                            target_content,
                            entity['name'],
                            source_style
                        )
                        
                        if modified_content and conf >= 0.85:
                            suggestions.append(LinkSuggestion(
                                source_url=source_url,
                                target_url=target_url,
                                anchor_text=entity['name'],
                                context=modified_content,
                                confidence_score=conf,
                                suggestion_type='modified',
                                modification_extent='minor' if len(modified_content) < 100 else 'major',
                                section_position='top_25'  # Prioritize top placement
                            ))
        
        return suggestions
        
    def _determine_section(self, context: str, full_content: str) -> str:
        """Determine which section of the content the context appears in."""
        position = full_content.find(context)
        content_length = len(full_content)
        
        if position < content_length * 0.25:
            return 'top_25'
        elif position < content_length * 0.5:
            return 'upper_middle_25'
        elif position < content_length * 0.75:
            return 'lower_middle_25'
        else:
            return 'bottom_25'
