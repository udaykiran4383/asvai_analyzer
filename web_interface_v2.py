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

from web_crawler_v2 import WebsiteCrawler
from search_readiness_analyser import AIReadinessAnalyzer
from llm_analyser import AISearchOptimizer
from seo_auditor import SEOAuditor

st.set_page_config(
    page_title="AI Search Readiness Analyzer", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def save_analysis_to_history(report_data):
    """Save analysis results to history."""
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'domain': get_nested_value(report_data, ['site_info', 'domain'], 'Unknown'),
        'overall_score': get_nested_value(report_data, ['ai_readiness_scores', 'overall'], 0),
        'report': report_data
    }
    st.session_state.analysis_history.append(history_entry)
    
    # Save history to file
    with open('analysis_history.json', 'w') as f:
        json.dump(st.session_state.analysis_history, f, indent=2)

def load_analysis_history():
    """Load analysis history from file."""
    try:
        with open('analysis_history.json', 'r') as f:
            st.session_state.analysis_history = json.load(f)
    except FileNotFoundError:
        st.session_state.analysis_history = []

def create_performance_chart(metrics):
    """Create a performance metrics chart."""
    # Explicitly check if metrics is None or not a dictionary
    if not isinstance(metrics, dict) or not metrics:
        return None
        
    # Extract metrics for visualization
    fcp_values = []
    lcp_values = []
    ttfb_values = []
    urls = []
    
    for url, data in metrics.items():
        # Ensure data is a dictionary before accessing keys
        if isinstance(data, dict):
            fcp_values.append(data.get('FCP', 0))
            lcp_values.append(data.get('LCP', 0))
            ttfb_values.append(data.get('TTFB', 0))
            urls.append(url)
    
    if not urls:
        return None # Return None if no valid data was found

    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Bar(
        name='FCP',
        x=urls,
        y=fcp_values,
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        name='LCP',
        x=urls,
        y=lcp_values,
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.add_trace(go.Bar(
        name='TTFB',
        x=urls,
        y=ttfb_values,
        marker_color='rgb(255, 127, 14)'
    ))
    
    # Update layout
    fig.update_layout(
        title='Page Load Performance Metrics',
        xaxis_title='Page URL',
        yaxis_title='Time (ms)',
        barmode='group',
        height=400
    )
    
    return fig

def create_score_radar_chart(scores):
    """Create a radar chart for component scores."""
    # Explicitly check if scores is None or not a dictionary
    if not isinstance(scores, dict) or not scores:
        return None
        
    # Extract component scores
    categories = list(scores.keys())
    values = list(scores.values())
    
    if not categories or not values:
        return None # Return None if no valid scores were found

    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400
    )
    
    return fig

def get_nested_value(data, keys, default=None):
    """Safely get a nested value from a dictionary."""
    if not isinstance(data, dict):
        return default
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default # Cannot get key from non-dict
        # Check if the retrieved value is None before the next iteration
        if value is None and key != keys[-1]: # Allow final value to be None if default is None
             return default
    return value if value is not None else default # Return default if final value is None

def display_analysis_history():
    """Display analysis history in a table."""
    if not st.session_state.analysis_history:
        st.info("No analysis history available.")
        return
        
    # Create DataFrame from history
    history_data = []
    for entry in st.session_state.analysis_history:
        # Ensure entry is a dictionary before processing
        if isinstance(entry, dict):
            history_data.append({
                'Date': get_nested_value(entry, ['timestamp'], ''),
                'Domain': get_nested_value(entry, ['domain'], ''),
                'Score': f"{get_nested_value(entry, ['overall_score'], 0):.1f}" if get_nested_value(entry, ['overall_score']) is not None else 'N/A'
            })
    
    if not history_data:
        st.info("No valid analysis history entries found.")
        return

    df = pd.DataFrame(history_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Add option to view detailed report
    selected_date = st.selectbox(
        "Select a previous analysis to view details:",
        options=df['Date'].tolist(),
        format_func=lambda x: f"{x} - {df[df['Date'] == x]['Domain'].iloc[0]}"
    )
    
    if selected_date:
        selected_entry = next(
            (entry for entry in st.session_state.analysis_history 
             if isinstance(entry, dict) and get_nested_value(entry, ['timestamp'], '') == selected_date),
            None
        )
        
        if selected_entry:
            display_report(selected_entry['report'])

def main():
    try:
        st.title("AI Search Readiness Analyzer")
        st.subheader("Optimize your website for AI assistants like Claude, ChatGPT, and Perplexity")
        
        # Initialize session state
        initialize_session_state()
        load_analysis_history()
        
        # Show initial instructions
        st.info("👈 Enter a website URL in the sidebar to begin analysis")
        
        with st.sidebar:
            st.header("Settings")
            url = st.text_input("Website URL", placeholder="https://example.com")
            max_pages = st.slider("Maximum pages to crawl", min_value=1, max_value=50, value=5)
            
            with st.expander("Advanced Options"):
                use_js = st.checkbox("Enable JavaScript rendering", value=True, 
                                    help="Captures dynamically loaded content but slower")
                wait_time = st.slider("Page load wait time (seconds)", min_value=1, max_value=10, value=3)
                use_llm = st.checkbox("Use AI analysis", value=True)
                check_security = st.checkbox("Check security", value=True)
                check_mobile = st.checkbox("Check mobile optimization", value=True)
                check_accessibility = st.checkbox("Check accessibility", value=True)
            
            analyze_button = st.button("Start Analysis", type="primary")
        
        # Show initial content in main area
        st.write("### How to use this tool:")
        st.write("1. Enter a website URL in the sidebar")
        st.write("2. Adjust settings if needed")
        st.write("3. Click 'Start Analysis'")
        st.write("4. Wait for the analysis to complete")
        
        if analyze_button and url:
            st.info(f"Starting analysis of {url}...")
            run_analysis(url, max_pages, use_js, wait_time, use_llm, 
                        check_security, check_mobile, check_accessibility)
        elif analyze_button:
            st.error("Please enter a website URL first!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try again or contact support if the problem persists.")

def run_analysis(url, max_pages, use_js, wait_time, use_llm, 
                check_security, check_mobile, check_accessibility):
    try:
        # Initialize progress
        st.session_state.analysis_in_progress = True
        st.session_state.current_progress = 0
        st.session_state.current_status = "Initializing analysis..."
        
        # Create progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Clean up URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        url = url.strip()  # Remove any extra whitespace
        
        # Step 1: Crawl website
        status_text.text("Crawling website...")
        try:
            crawler = WebsiteCrawler(start_url=url, max_pages=max_pages, use_selenium=use_js, wait_time=wait_time)
            crawler.crawl()
        except Exception as e:
            st.error(f"Error during website crawling: {str(e)}")
            st.session_state.analysis_in_progress = False
            progress_bar.empty()
            status_text.empty()
            return
            
        st.session_state.current_progress = 25
        progress_bar.progress(25)
        
        if not crawler or not crawler.pages_data:
            st.error("No pages were crawled. Please check the URL and try again.")
            st.session_state.analysis_in_progress = False
            progress_bar.empty()
            status_text.empty()
            return
        
        # Sort pages_data for consistent input to AI analyzer
        sorted_pages_data = sorted(crawler.pages_data, key=lambda x: x.get('url', ''))
        
        # Ensure we have valid metadata
        site_metadata = crawler.site_metadata if crawler and hasattr(crawler, 'site_metadata') else {}
        structured_data = crawler.structured_data if crawler and hasattr(crawler, 'structured_data') else []
        
        crawler_results = {
            "metadata": site_metadata,
            "pages": sorted_pages_data,
            "structured_data": structured_data
        }
        
        # Step 2: Analyze content
        status_text.text("Analyzing content...")
        try:
            analyzer = AIReadinessAnalyzer(crawler_results)
            content_analysis = analyzer.generate_report()
        except Exception as e:
            st.error(f"Error during content analysis: {str(e)}")
            content_analysis = {}
            
        st.session_state.current_progress = 50
        progress_bar.progress(50)
        
        # Step 3: SEO Audit
        status_text.text("Performing SEO audit...")
        try:
            seo_auditor = SEOAuditor(start_url=url, use_selenium=use_js)
            seo_audit = seo_auditor.run_audit()
        except Exception as e:
            st.error(f"Error during SEO audit: {str(e)}")
            seo_audit = {}
            
        st.session_state.current_progress = 75
        progress_bar.progress(75)
        
        # Step 4: LLM Analysis (if enabled)
        llm_insights = None
        if use_llm:
            status_text.text("Running AI analysis...")
            try:
                optimizer = AISearchOptimizer()
                llm_insights = optimizer.evaluate_website(crawler_results)
            except Exception as e:
                st.error(f"Error during AI analysis: {str(e)}")
                llm_insights = None
        
        st.session_state.current_progress = 100
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Calculate content statistics with safe defaults
        content_stats = {
            "total_pages": len(crawler.pages_data) if crawler and crawler.pages_data else 0,
            "avg_word_count": sum(len(page.get('full_text', '').split()) for page in crawler.pages_data) / len(crawler.pages_data) if crawler and crawler.pages_data else 0,
            "pages_with_thin_content": sum(1 for page in crawler.pages_data if len(page.get('full_text', '').split()) < 300) if crawler and crawler.pages_data else 0,
            "total_qa_pairs": sum(len(page.get('qa_pairs', [])) for page in crawler.pages_data) if crawler and crawler.pages_data else 0,
            "has_structured_data": bool(crawler.structured_data) if crawler else False,
            "js_rendered_pages": sum(1 for page in crawler.pages_data if page.get('js_rendered', False)) if crawler and crawler.pages_data else 0
        }
        
        # Default AI readiness scores when LLM analysis is not enabled
        default_scores = {
            'overall': 0,
            'components': {
                'content_quality': 0,
                'technical_optimization': 0,
                'authority_signals': 0,
                'question_answering': 0
            }
        }
        
        # Prepare report data with safe defaults
        report_data = {
            'site_info': {
                'domain': url,
                'pages_analyzed': len(crawler.pages_data) if crawler and crawler.pages_data else 0,
                'title': site_metadata.get('title', '') if site_metadata else '',
                'description': site_metadata.get('description', '') if site_metadata else ''
            },
            'content_analysis': content_analysis,
            'seo_audit': seo_audit,
            'llm_insights': llm_insights,
            'content_stats': content_stats,
            'ai_readiness_scores': {
                'overall': get_nested_value(llm_insights, ['analysis', 'overall_score'], default_scores['overall']),
                'components': {
                    'content_quality': get_nested_value(llm_insights, ['analysis', 'dimensions', 'content_quality', 'score'], default_scores['components']['content_quality']),
                    'technical_optimization': get_nested_value(llm_insights, ['analysis', 'dimensions', 'content_structure', 'score'], default_scores['components']['technical_optimization']),
                    'authority_signals': get_nested_value(llm_insights, ['analysis', 'dimensions', 'entity_clarity', 'score'], default_scores['components']['authority_signals']),
                    'question_answering': get_nested_value(llm_insights, ['analysis', 'dimensions', 'question_answer_coverage', 'score'], default_scores['components']['question_answering'])
                }
            },
            'top_recommendations': get_nested_value(llm_insights, ['recommendations', 'priority_actions'], [])
        }
        
        # Display the report
        display_report(report_data)
        
        # Save to history
        save_analysis_to_history(report_data)
        
        # Reset state
        st.session_state.analysis_in_progress = False
        st.session_state.report_data = report_data
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.session_state.analysis_in_progress = False
        progress_bar.empty()
        status_text.empty()

def display_report(report_data):
    """Display the analysis report with enhanced visualizations."""
    scores = report_data.get("ai_readiness_scores", {})
    site_info = report_data.get("site_info", {})
    content_stats = report_data.get("content_stats", {})
    # Ensure top_recommendations is a list
    recommendations = report_data.get("top_recommendations", [])
    llm_insights = report_data.get("llm_insights")
    seo_audit = report_data.get("seo_audit")
    
    # Create tabs for different sections of the report
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Detailed Analysis", "LLM Insights", "SEO Audit", "Performance"
    ])
    
    with tab1:
        st.header(f"AI Search Readiness Report: {site_info.get('domain', 'Unknown')}")
        
        # Overall score and component scores
        st.subheader("Readiness Scores")
        
        # Create radar chart for component scores
        # Pass scores dictionary directly
        radar_chart = create_score_radar_chart(scores.get("components", {}))
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Overall Score", f"{scores.get('overall', 0):.1f}", delta=None)
        with col2:
            st.metric("Content Quality", f"{scores.get('components', {}).get('content_quality', 0):.1f}", delta=None)
        with col3:
            st.metric("Technical", f"{scores.get('components', {}).get('technical_optimization', 0):.1f}", delta=None)
        with col4:
            st.metric("Authority", f"{scores.get('components', {}).get('authority_signals', 0):.1f}", delta=None)
        with col5:
            st.metric("Q&A Capability", f"{scores.get('components', {}).get('question_answering', 0):.1f}", delta=None)
        
        # Top Recommendations Summary
        st.subheader("Top Recommendations")
        
        if recommendations and isinstance(recommendations, list):
            for i, rec in enumerate(recommendations[:5], 1):
                # Ensure rec is a dictionary before accessing keys
                if isinstance(rec, dict):
                    impact = rec.get("impact", "Medium")
                    impact_color = {
                        "High": "red",
                        "Medium": "orange",
                        "Low": "blue"
                    }.get(impact, "gray")
                    
                    with st.expander(f"{i}. {rec.get('action', 'Unnamed action')}"):
                        st.markdown(f"**Impact:** :{impact_color}[{impact}]")
                        st.markdown(f"**Importance:** {rec.get('importance', 'No details provided')}")
                        st.markdown(f"**Difficulty:** {rec.get('difficulty', 'Unknown')}")
        else:
            st.info("No recommendations available.")
    
    with tab2:
        # Site information
        st.subheader("Website Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Content Statistics")
            # Create DataFrame with proper data types
            stats_data = {
                "Metric": [
                    "Pages Analyzed", 
                    "Average Word Count", 
                    "Pages with Thin Content",
                    "Q&A Pairs Found",
                    "Has Structured Data",
                    "Pages with JS Content"
                ],
                "Value": [
                    str(content_stats.get("total_pages", 0)),
                    f"{content_stats.get('avg_word_count', 0):.1f}",
                    str(content_stats.get("pages_with_thin_content", 0)),
                    str(content_stats.get("total_qa_pairs", 0)),
                    "Yes" if content_stats.get("has_structured_data", False) else "No",
                    str(content_stats.get("js_rendered_pages", 0))
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
        with col2:
            st.markdown("#### Site Details")
            site_df = pd.DataFrame({
                "Property": ["Site Title", "Site Description"],
                "Value": [
                    site_info.get("title", "Not found"),
                    site_info.get("description", "Not found")
                ]
            })
            st.dataframe(site_df, use_container_width=True)
        
        # All Recommendations
        st.subheader("All Recommendations")
        
        if recommendations and isinstance(recommendations, list):
            for i, rec in enumerate(recommendations, 1):
                # Ensure rec is a dictionary before accessing keys
                if isinstance(rec, dict):
                    impact = rec.get("impact", "Medium")
                    impact_color = {
                        "High": "red",
                        "Medium": "orange",
                        "Low": "blue"
                    }.get(impact, "gray")
                    
                    with st.expander(f"{i}. {rec.get('action', 'Unnamed action')}"):
                        st.markdown(f"**Impact:** :{impact_color}[{impact}]")
                        st.markdown(f"**Importance:** {rec.get('importance', 'No details provided')}")
                        st.markdown(f"**Difficulty:** {rec.get('difficulty', 'Unknown')}")
        else:
            st.info("No recommendations available.")
    
    with tab3:
        st.header("AI-Powered Insights")
        
        if llm_insights and isinstance(llm_insights, dict):
            # LLM Overall Score
            st.subheader("LLM Evaluation")
            analysis_data = llm_insights.get("analysis", {})
            llm_overall_score = analysis_data.get("overall_score", 0)
            
            st.metric("AI Search Optimization Score", f"{llm_overall_score}/10", 
                    delta=None, help="Score based on in-depth AI analysis of content")
            
            # Dimension Scores from LLM
            st.subheader("Dimension Scores")
            
            dimensions = analysis_data.get("dimensions", {})
            if dimensions and isinstance(dimensions, dict):
                # Create data for horizontal bar chart
                dimension_data = []
                for dim_name, dim_data in dimensions.items():
                    # Ensure dim_data is a dictionary before accessing keys
                    if isinstance(dim_data, dict):
                        dimension_data.append({
                            "Dimension": dim_name.replace("_", " ").title(),
                            "Score": dim_data.get("score", 0)
                        })
                
                if dimension_data:
                    df = pd.DataFrame(dimension_data)
                    st.bar_chart(df.set_index("Dimension"))
                
                # Display dimension details
                for dim_name, dim_data in dimensions.items():
                     # Ensure dim_data is a dictionary before accessing keys
                    if isinstance(dim_data, dict):
                        with st.expander(f"{dim_name.replace('_', ' ').title()} ({dim_data.get('score', 0)}/10)"):
                            st.markdown(f"**Analysis:** {dim_data.get('explanation', 'No explanation provided')}")
                            
                            st.markdown("**Issues:**")
                            issues = dim_data.get("issues", [])
                            if issues and isinstance(issues, list):
                                for issue in issues:
                                    st.markdown(f"- {issue}")
                            else:
                                st.markdown("No specific issues identified.")
                            
            # Key Insights
            st.subheader("Key Insights")
            key_insights = analysis_data.get("key_insights", [])
            if key_insights and isinstance(key_insights, list):
                for i, insight in enumerate(key_insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            else:
                st.info("No key insights provided by the LLM.")
                
            # Priority Actions from LLM
            st.subheader("Priority Actions")
            recommendations_data = llm_insights.get("recommendations", {})
            priority_actions = recommendations_data.get("priority_actions", [])
            
            if priority_actions and isinstance(priority_actions, list):
                for i, action in enumerate(priority_actions, 1):
                    # Ensure action is a dictionary before accessing keys
                    if isinstance(action, dict):
                        impact = action.get("impact", "Medium")
                        impact_color = {"High": "red", "Medium": "blue", "Low": "green"}.get(impact, "gray")
                        
                        with st.expander(f"{i}. {action.get('action', 'Unnamed action')}"):
                            st.markdown(f"**Impact:** :{impact_color}[{impact}]")
                            st.markdown(f"**Importance:** {action.get('importance', 'No details provided')}")
                            st.markdown(f"**Difficulty:** {action.get('difficulty', 'Unknown')}")
            else:
                st.info("No priority actions provided by the LLM.")

            # Relevant User Queries
            st.subheader("Relevant User Queries")
            relevant_queries = analysis_data.get("relevant_user_queries", [])
            
            if relevant_queries and isinstance(relevant_queries, list):
                st.write("Based on the website content, users might ask AI assistants the following queries:")
                for query in relevant_queries:
                    st.markdown(f"- {query}")
            else:
                st.info("No relevant user queries generated by the LLM. Ensure LLM Analysis is enabled.")

        else:
            st.info("LLM Analysis was not enabled or failed to complete. Enable LLM Analysis in the sidebar settings to see AI-powered insights.")
            st.button("Re-run with LLM Analysis", on_click=lambda: None)
    
    with tab4:
        st.header("SEO Audit Results")
        
        # Explicitly check if seo_audit is a dictionary before accessing its contents
        if seo_audit and isinstance(seo_audit, dict):
            # Performance Metrics
            st.subheader("Performance Metrics")
            # Safely access page_load_times
            performance_metrics = seo_audit.get("performance_metrics", {})
            load_times = performance_metrics.get("page_load_times", {})

            if load_times and isinstance(load_times, dict):
                # Create performance chart
                perf_chart = create_performance_chart(load_times)
                if perf_chart:
                    st.plotly_chart(perf_chart, use_container_width=True)
                
                for url, metrics in load_times.items():
                    # Ensure metrics is a dictionary before accessing keys
                    if isinstance(metrics, dict):
                        with st.expander(f"Page: {url}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("First Contentful Paint (FCP)", f"{metrics.get('FCP', 0):.0f}ms")
                            with col2:
                                st.metric("Largest Contentful Paint (LCP)", f"{metrics.get('LCP', 0):.0f}ms")
                            with col3:
                                st.metric("Time to First Byte (TTFB)", f"{metrics.get('TTFB', 0):.0f}ms")
            
            # Issues Summary
            st.subheader("SEO Issues Summary")
            issues = seo_audit.get("issues", {})
            
            # Create issue summary chart
            # Ensure issues is a dictionary before accessing keys for issue_counts
            if issues and isinstance(issues, dict):
                issue_counts = {
                    "Broken Links": len(issues.get("broken_links", [])),
                    "Missing H1": len(issues.get("missing_h1_pages", [])),
                    "Image Issues": len(issues.get("image_issues", [])),
                    "Security Issues": len(issues.get("security_issues", [])),
                    "Mobile Issues": len(issues.get("mobile_issues", [])),
                    "Accessibility Issues": len(issues.get("accessibility_issues", []))
                }
                
                fig = px.bar(
                    x=list(issue_counts.keys()),
                    y=list(issue_counts.values()),
                    title="SEO Issues by Category",
                    labels={"x": "Issue Type", "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("No SEO issues data available.")

            # Broken Links
            if issues.get("broken_links"):
                broken_links = issues.get("broken_links", [])
                # Ensure broken_links is a list before iterating
                if broken_links and isinstance(broken_links, list):
                    st.error(f"Found {len(broken_links)} broken links")
                    with st.expander("View Broken Links"):
                        for link in broken_links:
                            # Ensure link is a dictionary before accessing keys
                            if isinstance(link, dict):
                                st.markdown(f"- From: {link.get('source_page', 'N/A')}")
                                st.markdown(f"  To: {link.get('broken_url', 'N/A')}")
                                st.markdown(f"  Status: {link.get('status_code', 'Error')}")
            
            # Missing H1 Tags
            if issues.get("missing_h1_pages"):
                missing_h1_pages = issues.get("missing_h1_pages", [])
                # Ensure missing_h1_pages is a list before iterating
                if missing_h1_pages and isinstance(missing_h1_pages, list):
                    st.warning(f"Found {len(missing_h1_pages)} pages missing H1 tags")
                    with st.expander("View Pages Missing H1"):
                        for page in missing_h1_pages:
                            st.markdown(f"- {page}") # Assuming page is a string
            
            # Image Issues
            if issues.get("image_issues"):
                image_issues = issues.get("image_issues", [])
                 # Ensure image_issues is a list before iterating
                if image_issues and isinstance(image_issues, list):
                    st.warning(f"Found {len(image_issues)} image optimization issues")
                    with st.expander("View Image Issues"):
                        for issue_item in image_issues:
                            # Ensure issue_item is a dictionary before accessing keys
                            if isinstance(issue_item, dict):
                                st.markdown(f"- Page: {issue_item.get('page', 'N/A')}")
                                st.markdown(f"  Image: {issue_item.get('url', 'N/A')}")
                                st.markdown(f"  Issue: {issue_item.get('issue', 'No details')}")
            
            # Security Issues
            if issues.get("security_issues"):
                security_issues = issues.get("security_issues", [])
                 # Ensure security_issues is a list before iterating
                if security_issues and isinstance(security_issues, list):
                    st.error(f"Found {len(security_issues)} security issues")
                    with st.expander("View Security Issues"):
                        for issue_item in security_issues:
                             # Ensure issue_item is a dictionary before accessing keys
                            if isinstance(issue_item, dict):
                                st.markdown(f"- {issue_item.get('issue', 'No details')}")
                                st.markdown(f"  Details: {issue_item.get('details', 'No details provided')}")
            
            # Mobile Issues
            if issues.get("mobile_issues"):
                mobile_issues = issues.get("mobile_issues", [])
                 # Ensure mobile_issues is a list before iterating
                if mobile_issues and isinstance(mobile_issues, list):
                    st.warning(f"Found {len(mobile_issues)} mobile optimization issues")
                    with st.expander("View Mobile Issues"):
                        for issue_item in mobile_issues:
                             # Ensure issue_item is a dictionary before accessing keys
                            if isinstance(issue_item, dict):
                                st.markdown(f"- {issue_item.get('issue', 'No details')}")
                                st.markdown(f"  Details: {issue_item.get('details', 'No details provided')}")
            
            # Accessibility Issues
            if issues.get("accessibility_issues"):
                accessibility_issues = issues.get("accessibility_issues", [])
                 # Ensure accessibility_issues is a list before iterating
                if accessibility_issues and isinstance(accessibility_issues, list):
                    st.warning(f"Found {len(accessibility_issues)} accessibility issues")
                    with st.expander("View Accessibility Issues"):
                        for issue_item in accessibility_issues:
                             # Ensure issue_item is a dictionary before accessing keys
                            if isinstance(issue_item, dict):
                                st.markdown(f"- {issue_item.get('issue', 'No details')}")
                                st.markdown(f"  Details: {issue_item.get('details', 'No details provided')}")
            
            # Recommendations
            st.subheader("SEO Recommendations")
            seo_recommendations = seo_audit.get("recommendations", [])
            # Ensure seo_recommendations is a list before iterating
            if seo_recommendations and isinstance(seo_recommendations, list):
                for rec in seo_recommendations:
                    # Ensure rec is a dictionary before accessing keys
                    if isinstance(rec, dict):
                        with st.expander(f"{rec.get('category', 'N/A')}: {rec.get('issue', 'No details')}"):
                            st.markdown(rec.get('recommendation', 'No recommendation provided'))
            else:
                st.info("No specific SEO recommendations available.")
        else:
            st.info("SEO audit was not performed or failed to complete. Please try again.")
    
    with tab5:
        st.header("Performance Analysis")
        
        # Explicitly check if seo_audit is a dictionary before accessing performance metrics
        if seo_audit and isinstance(seo_audit, dict):
            # Resource Usage
            st.subheader("Resource Usage")
            # Safely access page_load_times for resource usage
            performance_metrics = seo_audit.get("performance_metrics", {})
            load_times_resource = performance_metrics.get("page_load_times", {})

            if load_times_resource and isinstance(load_times_resource, dict):
                for url, data in load_times_resource.items():
                    # Ensure data is a dictionary before accessing keys
                    if isinstance(data, dict):
                        resource_types = data.get("Resource_Types", {})
                        # Ensure resource_types is a dictionary before iterating
                        if isinstance(resource_types, dict):
                            # Create resource type chart
                            resource_data = []
                            for r_type, stats in resource_types.items():
                                # Ensure stats is a dictionary before accessing keys
                                if isinstance(stats, dict):
                                    resource_data.append({
                                        "Type": r_type,
                                        "Count": stats.get("count", 0),
                                        "Size (KB)": round(stats.get("size", 0) / 1024, 2)
                                    })
                            
                            if resource_data:
                                df = pd.DataFrame(resource_data)
                                # Create separate charts for count and size
                                fig_count = px.bar(
                                    df,
                                    x="Type",
                                    y="Count",
                                    title=f"Resource Count for {url}",
                                    color="Type",
                                    text="Count"
                                )
                                fig_count.update_traces(textposition='outside')
                                st.plotly_chart(fig_count, use_container_width=True)
                                
                                fig_size = px.bar(
                                    df,
                                    x="Type",
                                    y="Size (KB)",
                                    title=f"Resource Size for {url}",
                                    color="Type",
                                    text="Size (KB)"
                                )
                                fig_size.update_traces(textposition='outside')
                                st.plotly_chart(fig_size, use_container_width=True)
            
            # Performance Trends
            st.subheader("Performance Trends")
            # Safely access page_load_times for performance trends
            load_times_trend = performance_metrics.get("page_load_times", {})

            if load_times_trend and isinstance(load_times_trend, dict):
                trend_data = []
                for url, data in load_times_trend.items():
                    # Ensure data is a dictionary before accessing keys
                    if isinstance(data, dict):
                        trend_data.append({
                            "URL": url,
                            "FCP": data.get("FCP", 0),
                            "LCP": data.get("LCP", 0),
                            "TTFB": data.get("TTFB", 0),
                            "Total Load": data.get("Total_Load", 0)
                        })
                
                if trend_data:
                    df = pd.DataFrame(trend_data)
                    # Create a line chart for performance metrics
                    fig = go.Figure()
                    
                    # Add traces for each metric
                    metrics = ["FCP", "LCP", "TTFB", "Total Load"]
                    colors = ['rgb(55, 83, 109)', 'rgb(26, 118, 255)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
                    
                    for metric, color in zip(metrics, colors):
                        fig.add_trace(go.Scatter(
                            x=df["URL"],
                            y=df[metric],
                            name=metric,
                            line=dict(color=color, width=2),
                            mode='lines+markers'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Performance Metrics by Page",
                        xaxis_title="Page URL",
                        yaxis_title="Time (ms)",
                        height=400,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a table view of the metrics
                    st.subheader("Detailed Metrics")
                    st.dataframe(df, use_container_width=True)
        
        else:
            st.info("No performance metrics available.")
    
    # Raw data at the bottom
    with st.expander("View Raw JSON Data"):
        # Safely display raw JSON data
        if report_data and isinstance(report_data, dict):
            st.json(report_data)
        else:
            st.info("No report data available.")

if __name__ == "__main__":
    main()