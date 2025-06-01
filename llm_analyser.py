import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

class AISearchOptimizer:
    """
    A simple AI search optimization evaluator that sends website data to OpenAI
    and gets back analysis and recommendations.
    """
    
    def __init__(self):
        # Load configuration from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def evaluate_website(self, website_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a website for AI search optimization.
        
        Args:
            website_data: Dictionary containing website metadata and page content
            
        Returns:
            Dictionary with evaluation results and recommendations
        """
        # Perform analysis
        analysis_result = self._analyze_website(website_data)
        
        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations(website_data, analysis_result)
        
        # Combine into final report
        report = {
            "website": website_data.get("metadata", {}).get("domain", "Unknown website"),
            "analysis": analysis_result,
            "recommendations": recommendations,
            "overall_score": analysis_result.get("overall_score", 0)
        }
        
        return report
    
    def _analyze_website(self, website_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze website for AI search optimization factors"""
        
        # Construct prompt for website analysis
        system_prompt, user_prompt = self._build_analysis_prompt(website_data)
        
        # Make API call to OpenAI
        response_data = self._call_openai(system_prompt, user_prompt)
        
        # Parse and return the analysis results
        try:
            # Extract JSON from the response
            analysis_result = self._extract_json_from_text(response_data)
            return analysis_result
        except Exception as e:
            print(f"Error parsing analysis results: {e}")
            return {"error": "Failed to parse analysis results"}
    
    def _generate_recommendations(self, website_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis"""
        
        # Construct prompt for recommendations
        system_prompt, user_prompt = self._build_recommendations_prompt(website_data, analysis_result)
        
        # Make API call to OpenAI
        response_data = self._call_openai(system_prompt, user_prompt)
        
        # Parse and return the recommendations
        try:
            # Extract JSON from the response
            recommendations = self._extract_json_from_text(response_data)
            return recommendations
        except Exception as e:
            print(f"Error parsing recommendations: {e}")
            return {"error": "Failed to parse recommendations"}
    
    def _build_analysis_prompt(self, website_data: Dict[str, Any]) -> str:
        """Build prompt for website analysis"""
        
        metadata = website_data.get("metadata", {})
        # Get a sample of pages (first 3 for brevity in prompt)
        sample_pages = website_data.get("pages", [])[:3]
        
        system_prompt = """
        You are an expert in evaluating websites for AI search optimization. Analyze the following website data for AI search optimization. Evaluate how well this content would be discovered and surfaced by AI assistants like ChatGPT, Claude, or Perplexity when users ask relevant questions.

        Please evaluate the website on these dimensions important for AI search:

        1. Content Quality & Relevance: How informative, accurate, and valuable is the content?
        2. Content Structure: How well-structured is the content with clear headings and logical flow?
        3. Question-Answer Coverage: Does the content include or address common questions in its domain?
        4. Comprehensive Information: How thoroughly does the content cover its topic area?
        5. Entity Clarity: Does the content clearly identify key entities, products, services?
        6. Factual Content: Does the content provide specific facts, data points, or statistics?
        7. Contextual Completeness: Is the content self-contained and understandable without additional context?
        8. Semantic Relevance: How well does the content use relevant terminology and concepts?

        For each dimension:
        - Provide a score from 1-10
        - Brief explanation for the score
        - Specific issues identified
        
        Also provide an overall score from 1-10 and 3-5 key insights about why this website might or might not appear in AI search results.

        Return your analysis in this JSON format:
        ```json
        {
          "overall_score": 7.5,
          "dimensions": {
            "content_quality": {"score": 8, "explanation": "...", "issues": ["issue1", "issue2"]},
            "content_structure": {"score": 7, "explanation": "...", "issues": ["issue1", "issue2"]},
            ...continue for all dimensions...
          },
          "key_insights": [
            "Insight 1 about why this website might not appear in AI search",
            "Insight 2...",
            ...
          ]
        }
        ```
        """

        user_prompt = f"""
        WEBSITE METADATA:
        ```json
        {json.dumps(metadata, indent=2)}
        ```

        SAMPLE PAGES (3 of {len(website_data.get("pages", []))}):
        ```json
        {json.dumps(sample_pages, indent=2)}
        ```
        """
        
        return system_prompt, user_prompt
    
    def _build_recommendations_prompt(self, website_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        """Build prompt for generating recommendations"""
        
        metadata = website_data.get("metadata", {})
        

        system_prompt = """
        You are an expert in optimizing websites for AI search visibility. Based on the analysis provided, generate specific, actionable recommendations to improve the website's visibility in AI search results (ChatGPT, Claude, Perplexity, etc.).

        For each dimension with a score below 8:
        - Provide 2-3 specific, actionable recommendations.

        Additionally:
        - Identify the 5 highest priority actions that would have the biggest impact on improving AI search visibility.
        - Explain why each action is important, its expected impact, and its difficulty level.

        Return your recommendations in this JSON format:
        ```json
        {
          "dimension_recommendations": {
            "content_quality": [
              "Recommendation 1: Detailed explanation of what to change and how",
              "Recommendation 2: ...",
              ...
            ],
            ...continue for all dimensions with scores below 8...
          },
          "priority_actions": [
            {
              "action": "Clear action description",
              "importance": "Why this matters for AI search",
              "impact": "High/Medium/Low",
              "difficulty": "Easy/Moderate/Difficult"
            },
            ...4 more priority actions...
          ]
        }
        ```
        """

        user_prompt = f"""
        WEBSITE METADATA:
        ```json
        {json.dumps(metadata, indent=2)}
        ```

        ANALYSIS RESULTS:
        ```json
        {json.dumps(analysis_result, indent=2)}
        ```
        """

        return system_prompt, user_prompt
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Make an API call to OpenAI"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            response_data = response.json()
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        except requests.exceptions.RequestException as e:
            print(f"API call error: {e}")
            # Fallback to gpt-3.5-turbo
            payload["model"] = "gpt-3.5-turbo"
            try:
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
                return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except requests.exceptions.RequestException as e:
                print(f"Fallback API call error: {e}")
                return ""
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from text that might contain markdown and other content"""
        
        # Find JSON block in the text (typically enclosed in ```json and ```)
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in the response")
    
    def generate_report(self, website_url: str, analysis: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive, user-friendly report"""
        
        # Create system and user prompts for generating a user-friendly report
        system_prompt = """
        You are an expert in creating clear, actionable reports on website optimization. Create a comprehensive, user-friendly report based on the provided AI search optimization analysis and recommendations.

        The report should include:
        1. An executive summary (2-3 paragraphs)
        2. Overall score and what it means
        3. Strengths and weaknesses for AI search
        4. Top priority actions with clear implementation steps
        5. Expected benefits from implementing recommendations

        Format the report as a structured JSON with clear sections.
        """

        user_prompt = f"""
        WEBSITE: {website_url}

        ANALYSIS:
        ```json
        {json.dumps(analysis, indent=2)}
        ```

        RECOMMENDATIONS:
        ```json
        {json.dumps(recommendations, indent=2)}
        ```
        """
        
        # Get report content from the LLM
        report_content = self._call_openai(system_prompt, user_prompt)
        
        try:
            report_data = self._extract_json_from_text(report_content)
            return report_data
        except Exception as e:
            print(f"Error parsing report: {e}")
            # Return a simplified report if parsing fails
            return {
                "website": website_url,
                "overall_score": analysis.get("overall_score", 0),
                "key_insights": analysis.get("key_insights", []),
                "priority_actions": recommendations.get("priority_actions", [])
            }

# Example usage
if __name__ == "__main__":
    # Load website data from file
    with open("juspay_io_crawl_results_v2.json", "r") as f:
        website_data = json.load(f)
    
    # Create and run the evaluator
    evaluator = AISearchOptimizer()
    analysis = evaluator._analyze_website(website_data)
    
    # Print analysis results
    print(json.dumps(analysis, indent=2))
    
    # Generate recommendations
    recommendations = evaluator._generate_recommendations(website_data, analysis)
    
    # Print recommendations
    print(json.dumps(recommendations, indent=2))
    
    # Generate full report
    website_url = website_data.get("metadata", {}).get("domain", "Unknown website")
    report = evaluator.generate_report(website_url, analysis, recommendations)
    
    # Save results to files
    with open("ai_search_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    with open("ai_search_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    
    with open("ai_search_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis completed for {website_url}")
    print(f"Overall AI search optimization score: {analysis.get('overall_score', 0)}/10")