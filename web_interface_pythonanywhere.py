import streamlit as st
import json
import pandas as pd
import time
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import threading
import queue
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import traceback
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
if 'OPENAI_API_KEY' in os.environ:
    openai_api_key = os.environ['OPENAI_API_KEY']
else:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from web_crawler_v2 import WebsiteCrawler
from search_readiness_analyser import AIReadinessAnalyzer
from llm_analyser import AISearchOptimizer
from seo_auditor import SEOAuditor

# Configure Streamlit
st.set_page_config(
    page_title="AI Search Readiness Analyzer", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_in_progress' not in st.session_state:
        st.session_state.analysis_in_progress = False
    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = 0
    if 'current_status' not in st.session_state:
        st.session_state.current_status = ""
    if 'report_data' not in st.session_state:
        st.session_state.report_data = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

# Create the Streamlit app
app = st

def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()
    load_analysis_history()
    
    st.title("🔍 AI Search Readiness Analyzer")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        url = st.text_input("Enter website URL", "https://")
        max_pages = st.slider("Maximum pages to analyze", 1, 50, 10)
        use_js = st.checkbox("Enable JavaScript", value=True)
        wait_time = st.slider("Page load wait time (seconds)", 1, 10, 3)
        use_llm = st.checkbox("Use AI Analysis", value=True)
        check_security = st.checkbox("Check Security", value=True)
        check_mobile = st.checkbox("Check Mobile Optimization", value=True)
        check_accessibility = st.checkbox("Check Accessibility", value=True)
        
        if st.button("Start Analysis", type="primary"):
            if not url.startswith(('http://', 'https://')):
                st.error("Please enter a valid URL starting with http:// or https://")
                return
                
            st.session_state.analysis_in_progress = True
            st.session_state.current_progress = 0
            st.session_state.current_status = "Starting analysis..."
            
            try:
                report_data = run_analysis(
                    url, max_pages, use_js, wait_time, use_llm,
                    check_security, check_mobile, check_accessibility
                )
                if report_data:
                    st.session_state.report_data = report_data
                    save_analysis_to_history(report_data)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
            finally:
                st.session_state.analysis_in_progress = False
    
    # Main content area
    if st.session_state.analysis_in_progress:
        st.info("Analysis in progress... Please wait.")
        st.progress(st.session_state.current_progress)
        st.text(st.session_state.current_status)
    elif st.session_state.report_data:
        display_report(st.session_state.report_data)
    
    # Display analysis history
    st.sidebar.markdown("---")
    st.sidebar.header("Analysis History")
    display_analysis_history()

# Import the rest of your functions from web_interface_v2.py
from web_interface_v2 import (
    save_analysis_to_history,
    load_analysis_history,
    create_performance_chart,
    create_score_radar_chart,
    get_nested_value,
    display_analysis_history,
    run_analysis,
    display_report
)

if __name__ == "__main__":
    main() 