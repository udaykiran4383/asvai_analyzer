import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List
import hashlib
import time

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
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        # Cache for storing results
        self.cache = {}
        self.cache_duration = 3600  # 1 hour in seconds
        
        # Default analysis result in case of API failure
        self.default_analysis = {
            "overall_score": 0,
            "dimensions": {
                "content_quality": {"score": 0, "explanation": "Analysis failed", "issues": ["API error"]},
                "content_structure": {"score": 0, "explanation": "Analysis failed", "issues": ["API error"]},
                "question_answer_coverage": {"score": 0, "explanation": "Analysis failed", "issues": ["API error"]},
                "entity_clarity": {"score": 0, "explanation": "Analysis failed", "issues": ["API error"]}
            },
            "key_insights": ["Analysis failed due to API error"],
            "relevant_user_queries": []
        }
        
        # Default recommendations in case of API failure
        self.default_recommendations = {
            "dimension_recommendations": {},
            "priority_actions": [
                {
                    "action": "Enable API access",
                    "importance": "API access is required for analysis",
                    "impact": "High",
                    "difficulty": "Easy"
                }
            ]
        }
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate a cache key from the website data."""
        # Create a stable string representation of the data
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Dict[str, Any]:
        """Get cached result if it exists and is not expired."""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['result']
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache a result with timestamp."""
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _validate_scores(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize scores to ensure consistency."""
        if not analysis_result or not isinstance(analysis_result, dict):
            return self.default_analysis
            
        # Ensure overall score is present and valid
        overall_score = analysis_result.get('overall_score', 0)
        if not isinstance(overall_score, (int, float)) or overall_score < 0 or overall_score > 10:
            overall_score = 0
            
        # Ensure dimensions are present and valid
        dimensions = analysis_result.get('dimensions', {})
        if not isinstance(dimensions, dict):
            dimensions = {}
            
        # Validate each dimension score
        for dim_name in ['content_quality', 'content_structure', 'question_answer_coverage', 'entity_clarity']:
            dim_data = dimensions.get(dim_name, {})
            if not isinstance(dim_data, dict):
                dim_data = {}
                
            score = dim_data.get('score', 0)
            if not isinstance(score, (int, float)) or score < 0 or score > 10:
                score = 0
                
            dim_data['score'] = score
            dimensions[dim_name] = dim_data
            
        # Ensure other required fields
        key_insights = analysis_result.get('key_insights', [])
        if not isinstance(key_insights, list):
            key_insights = []
            
        relevant_queries = analysis_result.get('relevant_user_queries', [])
        if not isinstance(relevant_queries, list):
            relevant_queries = []
            
        return {
            'overall_score': overall_score,
            'dimensions': dimensions,
            'key_insights': key_insights,
            'relevant_user_queries': relevant_queries
        }
    
    def evaluate_website(self, website_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a website for AI search optimization.
        
        Args:
            website_data: Dictionary containing website metadata and page content
            
        Returns:
            Dictionary with evaluation results and recommendations
        """
        if not self.api_key:
            print("Error: No API key available")
            return {
                "website": website_data.get("metadata", {}).get("domain", "Unknown website"),
                "analysis": self.default_analysis,
                "recommendations": self.default_recommendations,
                "overall_score": 0
            }
        
        # Check cache first
        cache_key = self._get_cache_key(website_data)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            print("Returning cached result")
            return cached_result
        
        try:
            # Perform analysis
            analysis_result = self._analyze_website(website_data)
            
            # Validate scores
            analysis_result = self._validate_scores(analysis_result)
            
            # Generate recommendations based on analysis
            recommendations = self._generate_recommendations(website_data, analysis_result)
            
            # Combine into final report
            report = {
                "website": website_data.get("metadata", {}).get("domain", "Unknown website"),
                "analysis": analysis_result,
                "recommendations": recommendations,
                "overall_score": analysis_result.get("overall_score", 0)
            }
            
            # Cache the result
            self._cache_result(cache_key, report)
            
            return report
        except Exception as e:
            print(f"Error in evaluate_website: {e}")
            return {
                "website": website_data.get("metadata", {}).get("domain", "Unknown website"),
                "analysis": self.default_analysis,
                "recommendations": self.default_recommendations,
                "overall_score": 0
            }
    
    def _analyze_website(self, website_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze website for AI search optimization factors"""
        try:
            # Construct prompt for website analysis
            system_prompt, user_prompt = self._build_analysis_prompt(website_data)
            
            # Make API call to OpenAI
            response_data = self._call_openai(system_prompt, user_prompt)
            
            if not response_data:
                print("No response from API")
                return self.default_analysis
            
            # Parse and return the analysis results
            try:
                # Extract JSON from the response
                analysis_result = self._extract_json_from_text(response_data)
                if not analysis_result:
                    print("Failed to parse analysis results")
                    return self.default_analysis
                return analysis_result
            except Exception as e:
                print(f"Error parsing analysis results: {e}")
                return self.default_analysis
        except Exception as e:
            print(f"Error in _analyze_website: {e}")
            return self.default_analysis
    
    def _generate_recommendations(self, website_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis"""
        try:
            # Construct prompt for recommendations
            system_prompt, user_prompt = self._build_recommendations_prompt(website_data, analysis_result)
            
            # Make API call to OpenAI
            response_data = self._call_openai(system_prompt, user_prompt)
            
            if not response_data:
                print("No response from API")
                return self.default_recommendations
            
            # Parse and return the recommendations
            try:
                # Extract JSON from the response
                recommendations = self._extract_json_from_text(response_data)
                if not recommendations:
                    print("Failed to parse recommendations")
                    return self.default_recommendations
                return recommendations
            except Exception as e:
                print(f"Error parsing recommendations: {e}")
                return self.default_recommendations
        except Exception as e:
            print(f"Error in _generate_recommendations: {e}")
            return self.default_recommendations
    
    def _build_analysis_prompt(self, website_data: Dict[str, Any]) -> str:
        """Build prompt for website analysis"""
        
        metadata = website_data.get("metadata", {})
        # Get a sample of pages (first 3 for brevity in prompt)
        sample_pages = website_data.get("pages", [])[:3]
        
        system_prompt = """
        You are an expert in evaluating websites for AI search optimization. Analyze the following website data for AI search optimization. Evaluate how well this content would be discovered and surfaced by AI assistants like ChatGPT, Claude, or Perplexity when users ask relevant questions.

        Please evaluate the website on these dimensions important for AI search:

        1. Content Quality & Relevance (content_quality):
           - Score 9-10: Exceptional content with deep insights, original research, and comprehensive coverage
           - Score 7-8: Good content with valuable information and clear explanations
           - Score 5-6: Basic content with some useful information
           - Score 1-4: Poor quality or irrelevant content

        2. Content Structure (content_structure):
           - Score 9-10: Perfect structure with clear hierarchy, logical flow, and excellent organization
           - Score 7-8: Good structure with clear sections and headings
           - Score 5-6: Basic structure with some organization
           - Score 1-4: Poor structure or disorganized content

        3. Question-Answer Coverage (question_answer_coverage):
           - Score 9-10: Comprehensive coverage of common questions with detailed answers
           - Score 7-8: Good coverage of main questions with clear answers
           - Score 5-6: Basic coverage of some questions
           - Score 1-4: Minimal or no question-answer coverage

        4. Entity Clarity (entity_clarity):
           - Score 9-10: Clear identification of all key entities with detailed information
           - Score 7-8: Good identification of main entities
           - Score 5-6: Basic entity identification
           - Score 1-4: Poor or missing entity identification

        For each dimension:
        - Provide a score from 1-10 following the scoring guidelines above
        - Brief explanation for the score
        - Specific issues identified
        
        Calculate the overall score as the average of all dimension scores, rounded to one decimal place.
        
        Also provide 3-5 key insights about why this website might or might not appear in AI search results.
        
        Additionally, generate a list of 5-10 generic user search queries that a user might ask an AI assistant, and for which this website's content would be a relevant source of information. These queries should *not* mention the company name or specific product names unless they are common terms.

        Return your analysis in this JSON format:
        ```json
        {
          "overall_score": 7.5,
          "dimensions": {
            "content_quality": {"score": 8, "explanation": "...", "issues": ["issue1", "issue2"]},
            "content_structure": {"score": 7, "explanation": "...", "issues": ["issue1", "issue2"]},
            "question_answer_coverage": {"score": 8, "explanation": "...", "issues": ["issue1", "issue2"]},
            "entity_clarity": {"score": 7, "explanation": "...", "issues": ["issue1", "issue2"]}
          },
          "key_insights": [
            "Insight 1 about why this website might not appear in AI search",
            "Insight 2...",
            ...
          ],
          "relevant_user_queries": [
            "Generic user query 1",
            "Generic user query 2",
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
        if not self.api_key:
            print("No API key available")
            return ""
        
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
            "temperature": 0,  # Set to 0 for consistent results
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
        if not text:
            return {}
            
        try:
            # Find JSON block in the text (typically enclosed in ```json)
            json_start = text.find("```json")
            if json_start == -1:
                json_start = text.find("{")
            else:
                json_start = text.find("{", json_start)
                
            if json_start == -1:
                return {}
                
            json_end = text.rfind("}")
            if json_end == -1:
                return {}
                
            json_str = text[json_start:json_end + 1]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            return {}