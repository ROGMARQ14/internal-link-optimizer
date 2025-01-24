import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables
load_dotenv()

# Import local modules
from src.data_processor import URLProcessor
from src.content_analyzer import ContentAnalyzer
from src.link_optimizer import LinkOptimizer
from src.report_generator import ReportGenerator
from utils.cache_manager import CacheManager
from utils.config import Config

# Set page config
st.set_page_config(
    page_title="Internal Linking Optimizer",
    page_icon="🔗",
    layout="wide"
)

class App:
    def __init__(self):
        self.config = Config()
        self.cache_manager = CacheManager()
        
    def parse_urls(self, text: str) -> List[str]:
        """Parse URLs from text input."""
        # Split by common separators and clean
        urls = []
        for line in text.split('\n'):
            # Split by comma if present
            parts = line.split(',') if ',' in line else [line]
            urls.extend([url.strip() for url in parts if url.strip()])
        return urls
        
    def main(self):
        st.title("Internal Linking Optimizer")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Thresholds
            st.subheader("Thresholds")
            similarity_threshold = st.slider(
                "Semantic Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.01
            )
            
            entity_threshold = st.slider(
                "Entity Relevance Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.01
            )
            
            # Scraping settings
            st.subheader("Scraping Settings")
            scraping_rate = st.slider(
                "URLs per second",
                min_value=1,
                max_value=10,
                value=3,
                step=1
            )
            
            js_timeout = st.slider(
                "JavaScript Timeout (seconds)",
                min_value=5,
                max_value=20,
                value=10,
                step=1
            )
        
        # Main content area
        st.write("""
        ## URL Input
        Enter your URLs below (one per line) or upload a text file containing URLs.
        You can also paste URLs separated by commas.
        """)
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload a text file with URLs",
            type=['txt', 'csv']
        )
        
        # Text input option
        url_text = st.text_area(
            "Or paste your URLs here",
            height=200,
            help="Enter URLs (one per line or comma-separated)"
        )
        
        # Process URLs
        urls = []
        if uploaded_file:
            content = uploaded_file.getvalue().decode()
            urls = self.parse_urls(content)
        elif url_text:
            urls = self.parse_urls(url_text)
        
        if urls:
            try:
                # Initialize processors
                url_processor = URLProcessor(urls)
                content_analyzer = ContentAnalyzer(
                    similarity_threshold=similarity_threshold,
                    entity_threshold=entity_threshold
                )
                link_optimizer = LinkOptimizer()
                report_generator = ReportGenerator()
                
                # Process URLs
                with st.spinner("Processing URLs..."):
                    url_data = url_processor.process()
                    
                    # Show basic stats
                    st.write(f"Found {len(url_data['urls'])} valid URLs across {len(url_data['domains'])} domains")
                
                # Analyze content
                if st.button("Start Analysis"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Content analysis
                    status_text.text("Analyzing content...")
                    content_data = content_analyzer.analyze(
                        url_data,
                        scraping_rate=scraping_rate,
                        js_timeout=js_timeout,
                        progress_callback=lambda x: progress_bar.progress(x)
                    )
                    
                    # Link optimization
                    status_text.text("Generating linking suggestions...")
                    optimization_results = link_optimizer.optimize(content_data)
                    
                    # Generate report
                    status_text.text("Generating report...")
                    report = report_generator.generate(
                        url_data,
                        content_data,
                        optimization_results
                    )
                    
                    # Offer download
                    st.success("Analysis complete!")
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="internal_linking_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = App()
    app.main()
