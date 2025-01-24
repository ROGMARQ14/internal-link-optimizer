import pandas as pd
from typing import Dict, List
import logging

class GSCDataProcessor:
    """Processes Google Search Console data from CSV exports."""
    
    def __init__(self, file_path: str):
        """Initialize the processor with the CSV file path."""
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        
    def validate_csv(self, df: pd.DataFrame) -> bool:
        """Validate if the CSV has the required columns."""
        required_columns = ['Query', 'Landing Page', 'Clicks', 'Impressions', 'CTR', 'Position']
        return all(col in df.columns for col in required_columns)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data."""
        # Remove any duplicate rows
        df = df.drop_duplicates()
        
        # Remove rows with missing values
        df = df.dropna(subset=['Query', 'Landing Page'])
        
        # Clean URLs (remove trailing slashes, normalize to lowercase)
        df['Landing Page'] = df['Landing Page'].str.lower().str.rstrip('/')
        
        # Clean queries (remove extra whitespace, normalize to lowercase)
        df['Query'] = df['Query'].str.lower().str.strip()
        
        return df
    
    def extract_unique_urls(self, df: pd.DataFrame) -> List[str]:
        """Extract list of unique URLs from the data."""
        return df['Landing Page'].unique().tolist()
    
    def extract_unique_queries(self, df: pd.DataFrame) -> List[str]:
        """Extract list of unique queries from the data."""
        return df['Query'].unique().tolist()
    
    def process(self) -> Dict:
        """Process the GSC data and return processed information."""
        try:
            # Read CSV file
            df = pd.read_csv(self.file_path)
            
            # Validate CSV structure
            if not self.validate_csv(df):
                raise ValueError("Invalid CSV format: missing required columns")
            
            # Clean the data
            df = self.clean_data(df)
            
            # Extract unique URLs and queries
            unique_urls = self.extract_unique_urls(df)
            unique_queries = self.extract_unique_queries(df)
            
            # Group data by Landing Page for easier processing
            url_data = {}
            for url in unique_urls:
                url_data[url] = {
                    'queries': df[df['Landing Page'] == url]['Query'].tolist(),
                    'metrics': {
                        'clicks': df[df['Landing Page'] == url]['Clicks'].sum(),
                        'impressions': df[df['Landing Page'] == url]['Impressions'].sum(),
                        'avg_position': df[df['Landing Page'] == url]['Position'].mean()
                    }
                }
            
            return {
                'dataframe': df,
                'unique_urls': unique_urls,
                'unique_queries': unique_queries,
                'url_data': url_data
            }
            
        except Exception as e:
            self.logger.error(f"Error processing GSC data: {str(e)}")
            raise
