from typing import Dict, List
import pandas as pd
from io import BytesIO
import logging
from dataclasses import asdict
from collections import defaultdict

class ReportGenerator:
    """Generates Excel reports with internal linking suggestions."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.logger = logging.getLogger(__name__)
        
    def create_main_tab(self, gsc_data: Dict, optimization_results: List) -> pd.DataFrame:
        """Create the main overview tab."""
        records = []
        
        for suggestion in optimization_results:
            suggestion_dict = asdict(suggestion)
            
            # Get GSC metrics for the target URL
            target_metrics = gsc_data['url_data'][suggestion.target_url]['metrics']
            
            records.append({
                'Source URL': suggestion.source_url,
                'Target URL': suggestion.target_url,
                'Suggested Anchor Text': suggestion.anchor_text,
                'Confidence Score': suggestion.confidence_score,
                'Suggestion Type': suggestion.suggestion_type,
                'Modification Extent': suggestion.modification_extent,
                'Section Position': suggestion.section_position,
                'Target URL Clicks': target_metrics['clicks'],
                'Target URL Impressions': target_metrics['impressions'],
                'Target URL Avg Position': target_metrics['avg_position']
            })
        
        return pd.DataFrame(records)
        
    def create_suggestions_tab(self, optimization_results: List) -> pd.DataFrame:
        """Create detailed suggestions tab."""
        records = []
        
        for suggestion in optimization_results:
            records.append({
                'Source URL': suggestion.source_url,
                'Target URL': suggestion.target_url,
                'Anchor Text': suggestion.anchor_text,
                'Context Preview': suggestion.context[:200] + '...',  # Preview first 200 chars
                'Full Context': suggestion.context,
                'Confidence Score': suggestion.confidence_score,
                'Suggestion Type': suggestion.suggestion_type,
                'Modification Extent': suggestion.modification_extent,
                'Section Position': suggestion.section_position
            })
        
        return pd.DataFrame(records)
        
    def create_style_analysis_tab(self, content_data: Dict) -> pd.DataFrame:
        """Create style analysis tab."""
        records = []
        
        for url, data in content_data.items():
            style_metrics = data['style_metrics']
            records.append({
                'URL': url,
                'Average Sentence Length': style_metrics['avg_sentence_length'],
                'Vocabulary Richness': style_metrics['vocabulary_richness'],
                'Formality Score': style_metrics['formality_score'],
                'Readability Score': style_metrics['readability_score']
            })
        
        return pd.DataFrame(records)
        
    def create_cluster_analysis_tab(self, content_data: Dict, optimization_results: List) -> pd.DataFrame:
        """Create cluster analysis tab."""
        # Group suggestions by URL clusters
        clusters = defaultdict(list)
        for suggestion in optimization_results:
            clusters[suggestion.source_url].append(suggestion.target_url)
        
        records = []
        for url, related_urls in clusters.items():
            records.append({
                'URL': url,
                'Cluster Size': len(related_urls),
                'Related URLs': ', '.join(related_urls),
                'Suggested Links': len([s for s in optimization_results 
                                     if s.source_url == url]),
                'Average Confidence': sum(s.confidence_score 
                                        for s in optimization_results 
                                        if s.source_url == url) / len(related_urls)
                                        if related_urls else 0
            })
        
        return pd.DataFrame(records)
        
    def generate(self, gsc_data: Dict, content_data: Dict, optimization_results: List) -> bytes:
        """Generate the complete Excel report."""
        try:
            # Create Excel writer object
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Create and write each tab
                self.create_main_tab(gsc_data, optimization_results).to_excel(
                    writer, sheet_name='Overview', index=False
                )
                
                self.create_suggestions_tab(optimization_results).to_excel(
                    writer, sheet_name='Detailed Suggestions', index=False
                )
                
                self.create_style_analysis_tab(content_data).to_excel(
                    writer, sheet_name='Style Analysis', index=False
                )
                
                self.create_cluster_analysis_tab(content_data, optimization_results).to_excel(
                    writer, sheet_name='Cluster Analysis', index=False
                )
            
            return output.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
