# AI Search Readiness Analyzer

A Streamlit application that analyzes websites for AI search readiness and optimization.

## Features

- Website crawling and analysis
- Content quality assessment
- Technical optimization scoring
- Authority signal analysis
- Q&A capability evaluation
- Interactive visualizations
- Detailed recommendations

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Running Locally

```bash
streamlit run web_interface_v2.py
```

## Deployment

This application can be deployed on Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the environment variables in the Streamlit Cloud dashboard
5. Deploy!

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## License

MIT License

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
