# Internal Linking Optimizer

A powerful tool for optimizing website internal linking structure using advanced NLP techniques and semantic analysis.

## Features

- Process Google Search Console data
- Analyze content semantically using BERT and Google Cloud Natural Language API
- Generate intelligent internal linking suggestions
- Provide style-matched content modifications
- Export detailed reports in Excel format

## Requirements

- Python 3.8+
- Google Cloud credentials (for Natural Language API)
- OpenAI API key (for content generation)
- Keywords Everywhere API key
- Chrome/Firefox WebDriver (for JavaScript-rendered content)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
   OPENAI_API_KEY=your_openai_api_key
   KEYWORDS_EVERYWHERE_API_KEY=your_keywords_everywhere_api_key
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload your Google Search Console CSV export
3. Configure parameters as needed
4. Process and analyze your data
5. Download the generated report

## Configuration

- Semantic Similarity Threshold: 85% (default)
- Scraping Rate: 3-5 URLs/second
- Entity Analysis Threshold: 85%
- Style Matching: Enabled
- Content Generation: OpenAI GPT

## Output

The tool generates an Excel report with:
- Main overview tab with all data
- Detailed linking suggestions
- Style analysis results
- Confidence scores
- Preview of suggested changes
