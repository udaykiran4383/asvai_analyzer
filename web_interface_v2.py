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

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_crawler_v2 import WebsiteCrawler
from serach_readiness_analyser import AIReadinessAnalyzer
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
        'domain': report_data['site_info']['domain'],
        'overall_score': report_data['ai_readiness_scores']['overall'],
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
    if not metrics:
        return None
        
    # Extract metrics for visualization
    fcp_values = []
    lcp_values = []
    ttfb_values = []
    urls = []
    
    for url, data in metrics.items():
        if isinstance(data, dict):
            fcp_values.append(data.get('FCP', 0))
            lcp_values.append(data.get('LCP', 0))
            ttfb_values.append(data.get('TTFB', 0))
            urls.append(url)
    
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
    if not scores:
        return None
        
    # Extract component scores
    categories = list(scores.keys())
    values = list(scores.values())
    
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

def display_analysis_history():
    """Display analysis history in a table."""
    if not st.session_state.analysis_history:
        st.info("No analysis history available.")
        return
        
    # Create DataFrame from history
    history_data = []
    for entry in st.session_state.analysis_history:
        history_data.append({
            'Date': datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M'),
            'Domain': entry['domain'],
            'Score': f"{entry['overall_score']:.1f}"
        })
    
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
             if datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M') == selected_date),
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
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
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
        
        # Step 1: Crawl website
        status_text.text("Crawling website...")
        crawler = WebsiteCrawler(start_url=url, max_pages=max_pages, use_selenium=use_js, wait_time=wait_time)
        crawler.crawl()
        st.session_state.current_progress = 25
        progress_bar.progress(25)
        
        if not crawler.pages_data:
            raise Exception("No pages were crawled. Please check the URL and try again.")
        
        # Sort pages_data for consistent input to AI analyzer
        sorted_pages_data = sorted(crawler.pages_data, key=lambda x: x.get('url', ''))
        
        crawler_results = {
            "metadata": crawler.site_metadata,
            "pages": sorted_pages_data,
            "structured_data": crawler.structured_data
        }
        
        # Step 2: Analyze content
        status_text.text("Analyzing content...")
        analyzer = AIReadinessAnalyzer(crawler_results)
        content_analysis = analyzer.generate_report()
        st.session_state.current_progress = 50
        progress_bar.progress(50)
        
        # Step 3: SEO Audit
        status_text.text("Performing SEO audit...")
        seo_auditor = SEOAuditor(start_url=url, use_selenium=use_js)
        seo_audit = seo_auditor.run_audit()
        st.session_state.current_progress = 75
        progress_bar.progress(75)
        
        # Step 4: LLM Analysis (if enabled)
        llm_insights = None
        if use_llm:
            status_text.text("Running AI analysis...")
            optimizer = AISearchOptimizer()
            llm_insights = optimizer.evaluate_website(crawler_results)
        
        st.session_state.current_progress = 100
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Calculate content statistics
        content_stats = {
            "total_pages": len(crawler.pages_data),
            "avg_word_count": sum(len(page.get('content', '').split()) for page in crawler.pages_data) / len(crawler.pages_data) if crawler.pages_data else 0,
            "pages_with_thin_content": sum(1 for page in crawler.pages_data if len(page.get('content', '').split()) < 300),
            "total_qa_pairs": sum(len(page.get('qa_pairs', [])) for page in crawler.pages_data),
            "has_structured_data": bool(crawler.structured_data),
            "js_rendered_pages": sum(1 for page in crawler.pages_data if page.get('js_rendered', False))
        }
        
        # Prepare report data
        report_data = {
            'site_info': {
                'domain': url,
                'pages_analyzed': len(crawler.pages_data),
                'title': crawler.site_metadata.get('title', ''),
                'description': crawler.site_metadata.get('description', '')
            },
            'content_analysis': content_analysis,
            'seo_audit': seo_audit,
            'llm_insights': llm_insights,
            'content_stats': content_stats,
            'ai_readiness_scores': {
                'overall': llm_insights.get('overall_score', 0) if llm_insights else 0,
                'components': {
                    'content_quality': llm_insights.get('analysis', {}).get('dimensions', {}).get('content_quality', {}).get('score', 0) if llm_insights else 0,
                    'technical_optimization': llm_insights.get('analysis', {}).get('dimensions', {}).get('content_structure', {}).get('score', 0) if llm_insights else 0,
                    'authority_signals': llm_insights.get('analysis', {}).get('dimensions', {}).get('entity_clarity', {}).get('score', 0) if llm_insights else 0,
                    'question_answering': llm_insights.get('analysis', {}).get('dimensions', {}).get('question_answer_coverage', {}).get('score', 0) if llm_insights else 0
                }
            },
            'top_recommendations': llm_insights.get('recommendations', {}).get('priority_actions', []) if llm_insights else []
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
        component_scores = scores.get("components", {})
        radar_chart = create_score_radar_chart(component_scores)
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Overall Score", f"{scores.get('overall', 0):.1f}", delta=None)
        with col2:
            st.metric("Content Quality", f"{component_scores.get('content_quality', 0):.1f}", delta=None)
        with col3:
            st.metric("Technical", f"{component_scores.get('technical_optimization', 0):.1f}", delta=None)
        with col4:
            st.metric("Authority", f"{component_scores.get('authority_signals', 0):.1f}", delta=None)
        with col5:
            st.metric("Q&A Capability", f"{component_scores.get('question_answering', 0):.1f}", delta=None)
        
        # Top Recommendations Summary
        st.subheader("Top Recommendations")
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
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
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
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
        
        if llm_insights:
            # LLM Overall Score
            st.subheader("LLM Evaluation")
            llm_overall_score = llm_insights.get("analysis", {}).get("overall_score", 0)
            
            st.metric("AI Search Optimization Score", f"{llm_overall_score}/10", 
                    delta=None, help="Score based on in-depth AI analysis of content")
            
            # Dimension Scores from LLM
            st.subheader("Dimension Scores")
            
            dimensions = llm_insights.get("analysis", {}).get("dimensions", {})
            if dimensions:
                # Create data for horizontal bar chart
                dimension_data = []
                for dim_name, dim_data in dimensions.items():
                    dimension_data.append({
                        "Dimension": dim_name.replace("_", " ").title(),
                        "Score": dim_data.get("score", 0)
                    })
                
                # Create a DataFrame for the scores
                df = pd.DataFrame(dimension_data)
                
                # Display as a horizontal bar chart
                st.bar_chart(df.set_index("Dimension"))
                
                # Display dimension details
                for dim_name, dim_data in dimensions.items():
                    with st.expander(f"{dim_name.replace('_', ' ').title()} ({dim_data.get('score', 0)}/10)"):
                        st.markdown(f"**Analysis:** {dim_data.get('explanation', 'No explanation provided')}")
                        
                        st.markdown("**Issues:**")
                        issues = dim_data.get("issues", [])
                        if issues:
                            for issue in issues:
                                st.markdown(f"- {issue}")
                        else:
                            st.markdown("No specific issues identified.")
                            
            # Key Insights
            st.subheader("Key Insights")
            key_insights = llm_insights.get("analysis", {}).get("key_insights", [])
            if key_insights:
                for i, insight in enumerate(key_insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            else:
                st.info("No key insights provided by the LLM.")
                
            # Priority Actions from LLM
            st.subheader("Priority Actions")
            priority_actions = llm_insights.get("recommendations", {}).get("priority_actions", [])
            
            if priority_actions:
                for i, action in enumerate(priority_actions, 1):
                    impact = action.get("impact", "Medium")
                    impact_color = {"High": "red", "Medium": "blue", "Low": "green"}.get(impact, "gray")
                    
                    with st.expander(f"{i}. {action.get('action', 'Unnamed action')}"):
                        st.markdown(f"**Impact:** :{impact_color}[{impact}]")
                        st.markdown(f"**Importance:** {action.get('importance', 'No details provided')}")
                        st.markdown(f"**Difficulty:** {action.get('difficulty', 'Unknown')}")
            else:
                st.info("No priority actions provided by the LLM.")
        else:
            st.info("LLM Analysis was not enabled or failed to complete. Enable LLM Analysis in the sidebar settings to see AI-powered insights.")
            st.button("Re-run with LLM Analysis", on_click=lambda: None)
    
    with tab4:
        st.header("SEO Audit Results")
        
        if seo_audit:
            # Performance Metrics
            st.subheader("Performance Metrics")
            if seo_audit.get("performance_metrics", {}).get("page_load_times"):
                load_times = seo_audit["performance_metrics"]["page_load_times"]
                
                # Create performance chart
                perf_chart = create_performance_chart(load_times)
                if perf_chart:
                    st.plotly_chart(perf_chart, use_container_width=True)
                
                for url, metrics in load_times.items():
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
            
            # Broken Links
            if issues.get("broken_links"):
                st.error(f"Found {len(issues['broken_links'])} broken links")
                with st.expander("View Broken Links"):
                    for link in issues["broken_links"]:
                        st.markdown(f"- From: {link['source_page']}")
                        st.markdown(f"  To: {link['broken_url']}")
                        st.markdown(f"  Status: {link.get('status_code', 'Error')}")
            
            # Missing H1 Tags
            if issues.get("missing_h1_pages"):
                st.warning(f"Found {len(issues['missing_h1_pages'])} pages missing H1 tags")
                with st.expander("View Pages Missing H1"):
                    for page in issues["missing_h1_pages"]:
                        st.markdown(f"- {page}")
            
            # Image Issues
            if issues.get("image_issues"):
                st.warning(f"Found {len(issues['image_issues'])} image optimization issues")
                with st.expander("View Image Issues"):
                    for issue in issues["image_issues"]:
                        st.markdown(f"- Page: {issue['page']}")
                        st.markdown(f"  Image: {issue['url']}")
                        st.markdown(f"  Issue: {issue['issue']}")
            
            # Security Issues
            if issues.get("security_issues"):
                st.error(f"Found {len(issues['security_issues'])} security issues")
                with st.expander("View Security Issues"):
                    for issue in issues["security_issues"]:
                        st.markdown(f"- {issue['issue']}")
                        st.markdown(f"  Details: {issue.get('details', 'No details provided')}")
            
            # Mobile Issues
            if issues.get("mobile_issues"):
                st.warning(f"Found {len(issues['mobile_issues'])} mobile optimization issues")
                with st.expander("View Mobile Issues"):
                    for issue in issues["mobile_issues"]:
                        st.markdown(f"- {issue['issue']}")
                        st.markdown(f"  Details: {issue.get('details', 'No details provided')}")
            
            # Accessibility Issues
            if issues.get("accessibility_issues"):
                st.warning(f"Found {len(issues['accessibility_issues'])} accessibility issues")
                with st.expander("View Accessibility Issues"):
                    for issue in issues["accessibility_issues"]:
                        st.markdown(f"- {issue['issue']}")
                        st.markdown(f"  Details: {issue.get('details', 'No details provided')}")
            
            # Recommendations
            st.subheader("SEO Recommendations")
            recommendations = seo_audit.get("recommendations", [])
            if recommendations:
                for rec in recommendations:
                    with st.expander(f"{rec['category']}: {rec['issue']}"):
                        st.markdown(rec['recommendation'])
            else:
                st.info("No specific SEO recommendations available.")
        else:
            st.info("SEO audit was not performed or failed to complete. Please try again.")
    
    with tab5:
        st.header("Performance Analysis")
        
        if seo_audit and seo_audit.get("performance_metrics"):
            metrics = seo_audit["performance_metrics"]
            
            # Resource Usage
            st.subheader("Resource Usage")
            if metrics.get("page_load_times"):
                for url, data in metrics["page_load_times"].items():
                    if isinstance(data, dict) and "Resource_Types" in data:
                        resource_types = data["Resource_Types"]
                        
                        # Create resource type chart
                        resource_data = []
                        for r_type, stats in resource_types.items():
                            resource_data.append({
                                "Type": r_type,
                                "Count": stats["count"],
                                "Size (KB)": stats["size"] / 1024
                            })
                        
                        if resource_data:
                            df = pd.DataFrame(resource_data)
                            fig = px.bar(
                                df,
                                x="Type",
                                y=["Count", "Size (KB)"],
                                title=f"Resource Usage for {url}",
                                barmode="group"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # Performance Trends
            st.subheader("Performance Trends")
            if metrics.get("page_load_times"):
                trend_data = []
                for url, data in metrics["page_load_times"].items():
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
                    fig = px.line(
                        df,
                        x="URL",
                        y=["FCP", "LCP", "TTFB", "Total Load"],
                        title="Performance Metrics by Page"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No performance metrics available.")
    
    # Raw data at the bottom
    with st.expander("View Raw JSON Data"):
        st.json(report_data)

if __name__ == "__main__":
    main()