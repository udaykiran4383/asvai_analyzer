# AI Search Readiness Analyzer

A Streamlit application that analyzes website content for search readiness using AI.

## Features

- Website crawling and content extraction
- SEO analysis
- Search readiness scoring
- Content optimization suggestions
- Performance visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/udaykiran4383/asvai_analyzer.git
cd asvai_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main/web_interface_v2.py
```

## Usage

1. Enter the URL of the website you want to analyze
2. Click "Analyze" to start the process
3. View the results and recommendations
4. Download the analysis report

## Live Demo

[View the live demo](https://asvai-analyzer.streamlit.app)

## Deployment

This application is deployed on Streamlit Cloud. To deploy your own version:

1. Fork this repository
2. Sign up for Streamlit Cloud
3. Connect your GitHub repository
4. Set the main file path to `main/web_interface_v2.py`
5. Deploy!

## Project Structure

```
asvai_analyzer/
├── main/
│   ├── web_interface_v2.py      # Main Streamlit application
│   ├── seo_auditor.py           # SEO analysis module
│   ├── serach_readiness_analyser.py  # Search readiness analysis
│   ├── web_crawler_v2.py        # Web crawling module
│   └── llm_analyser.py          # LLM analysis module
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
