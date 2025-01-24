from setuptools import setup, find_packages

setup(
    name="internal-linking-optimizer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit>=1.24.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'requests>=2.31.0',
        'beautifulsoup4>=4.12.0',
        'selenium>=4.10.0',
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'google-cloud-language>=2.11.0',
        'python-dotenv>=1.0.0',
        'openai>=1.0.0',
        'scikit-learn>=1.3.0',
        'spacy>=3.6.0',
        'webdriver-manager>=4.0.0',
        'openpyxl>=3.1.0'
    ],
)
