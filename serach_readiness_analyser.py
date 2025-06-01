import re
import statistics
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import json

# Download NLTK resources (run this once)
# nltk.download('punkt')
# nltk.download('stopwords')

class AIReadinessAnalyzer:
    def __init__(self, crawler_results):
        """
        Initialize the analyzer with data from the crawler.
        
        Args:
            crawler_results: Dict containing website metadata, pages, and structured data
        """
        self.metadata = crawler_results["metadata"]
        self.pages = crawler_results["pages"]
        self.structured_data = crawler_results["structured_data"]
        self.scores = {
            "content_quality": 0,
            "technical_optimization": 0,
            "authority_signals": 0,
            "question_answering": 0,
            "overall": 0
        }
        self.recommendations = []
        
    def analyze_content_quality(self):
        """Analyze the quality of the content."""
        score = 0
        total_points = 0
        
        # Check if there's enough content overall
        avg_word_count = self.metadata.get("avg_word_count", 0)
        if avg_word_count >= 1000:
            score += 20
        elif avg_word_count >= 750:
            score += 15
        elif avg_word_count >= 500:
            score += 10
        elif avg_word_count >= 300:
            score += 5
        else:
            self.recommendations.append({
                "category": "Content Quality",
                "issue": "Thin content",
                "recommendation": "Increase content length to at least 500 words per page on average.",
                "importance": "High"
            })
        total_points += 20
        
        # Analyze readability
        readability_scores = []
        for page in self.pages:
            if page["word_count"] > 100:  # Only analyze pages with substantial content
                sentences = sent_tokenize(page["full_text"])
                if sentences:
                    # Calculate average sentence length
                    words_per_sentence = [len(s.split()) for s in sentences]
                    avg_sentence_length = statistics.mean(words_per_sentence) if words_per_sentence else 0
                    
                    # A simple readability metric
                    if 10 <= avg_sentence_length <= 20:  # Ideal range
                        readability_scores.append(10)
                    elif avg_sentence_length < 10:  # Too short
                        readability_scores.append(5)
                    else:  # Too long
                        readability_scores.append(max(0, 10 - (avg_sentence_length - 20) / 2))
        
        if readability_scores:
            avg_readability = statistics.mean(readability_scores)
            score += avg_readability
            
            if avg_readability < 7:
                self.recommendations.append({
                    "category": "Content Quality",
                    "issue": "Poor readability",
                    "recommendation": "Improve sentence structure. Aim for an average of 15-20 words per sentence.",
                    "importance": "Medium"
                })
        else:
            score += 0
        total_points += 10
        
        # Check for keyword consistency
        all_text = " ".join([page["full_text"] for page in self.pages])
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', all_text.lower())
        
        # Remove common stop words
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        
        # Get most common words
        word_freq = Counter(filtered_words).most_common(10)
        
        if word_freq:
            # Check if top keywords appear in titles and headings
            top_keywords = [word for word, _ in word_freq]
            
            # Check titles
            title_keywords = sum(1 for page in self.pages 
                               for keyword in top_keywords 
                               if keyword in page["title"].lower())
            
            # Check headings
            heading_keywords = sum(1 for page in self.pages 
                                 for heading in page["headings"] 
                                 for keyword in top_keywords 
                                 if keyword in heading["text"].lower())
            
            keyword_alignment_score = min(20, (title_keywords + heading_keywords) * 2)
            score += keyword_alignment_score
            
            if keyword_alignment_score < 10:
                self.recommendations.append({
                    "category": "Content Quality",
                    "issue": "Poor keyword alignment",
                    "recommendation": f"Ensure top keywords ({', '.join(top_keywords[:5])}) appear in titles and headings.",
                    "importance": "High"
                })
        else:
            score += 0
        total_points += 20
        
        # Normalize the final score to 0-100
        if total_points > 0:
            normalized_score = (score / total_points) * 100
        else:
            normalized_score = 0
            
        self.scores["content_quality"] = normalized_score
        return normalized_score
    
    def analyze_technical_optimization(self):
        """Analyze technical optimization factors."""
        score = 0
        total_points = 0
        
        # Check for structured data
        has_schema = self.metadata.get("has_schema_markup", False)
        if has_schema:
            score += 25
            
            # Check for specific schema types that help AI assistants
            schema_types = set()
            for item in self.structured_data:
                if item["type"] == "JSON-LD":
                    if isinstance(item["data"], dict) and "@type" in item["data"]:
                        schema_types.add(item["data"]["@type"])
                    elif isinstance(item["data"], list):
                        for entry in item["data"]:
                            if isinstance(entry, dict) and "@type" in entry:
                                schema_types.add(entry["@type"])
            
            helpful_schemas = {"FAQPage", "HowTo", "Recipe", "Product", "Article", "LocalBusiness"}
            if schema_types.intersection(helpful_schemas):
                score += 15
                
        else:
            self.recommendations.append({
                "category": "Technical Optimization",
                "issue": "Missing structured data",
                "recommendation": "Implement schema.org markup (JSON-LD) for your content type.",
                "importance": "Critical"
            })
        total_points += 40
        
        # Check heading structure
        heading_counts = self.metadata.get("heading_count", {})
        if heading_counts.get("h1", 0) >= self.metadata.get("pages_crawled", 0):
            score += 10  # Good: At least one H1 per page
        else:
            self.recommendations.append({
                "category": "Technical Optimization",
                "issue": "Missing H1 headings",
                "recommendation": "Ensure each page has exactly one H1 heading.",
                "importance": "High"
            })
            
        # Check for proper heading hierarchy
        proper_hierarchy = True
        for page in self.pages:
            headings = sorted(page["headings"], key=lambda h: h["level"])
            levels_used = [h["level"] for h in headings]
            
            # Check if headings are in order (can skip levels but shouldn't go backward)
            for i in range(1, len(levels_used)):
                if levels_used[i] < levels_used[i-1] and levels_used[i] != 1:  # H1 can appear multiple times
                    proper_hierarchy = False
                    break
        
        if proper_hierarchy:
            score += 10
        else:
            self.recommendations.append({
                "category": "Technical Optimization",
                "issue": "Improper heading hierarchy",
                "recommendation": "Use headings in their proper order (H1 → H2 → H3).",
                "importance": "Medium"
            })
        total_points += 20
        
        # Check meta descriptions
        pages_with_meta = sum(1 for page in self.pages if page["meta_description"])
        meta_coverage = pages_with_meta / len(self.pages) if self.pages else 0
        
        meta_score = meta_coverage * 20
        score += meta_score
        
        if meta_score < 15:
            self.recommendations.append({
                "category": "Technical Optimization",
                "issue": "Missing meta descriptions",
                "recommendation": "Add unique meta descriptions to all pages.",
                "importance": "High"
            })
        total_points += 20
        
        # Check for internal linking
        internal_links = self.metadata.get("internal_links", 0)
        pages_crawled = self.metadata.get("pages_crawled", 0)
        
        if pages_crawled > 0:
            links_per_page = internal_links / pages_crawled
            if links_per_page >= 5:
                score += 20
            elif links_per_page >= 3:
                score += 15
            elif links_per_page >= 1:
                score += 10
            else:
                self.recommendations.append({
                    "category": "Technical Optimization",
                    "issue": "Poor internal linking",
                    "recommendation": "Add more internal links between pages (aim for 3-5 per page).",
                    "importance": "Medium"
                })
        total_points += 20
        
        # Normalize the final score
        if total_points > 0:
            normalized_score = (score / total_points) * 100
        else:
            normalized_score = 0
            
        self.scores["technical_optimization"] = normalized_score
        return normalized_score
    
    def analyze_authority_signals(self):
        """Analyze authority signals in the content."""
        score = 0
        total_points = 0
        
        # Check for citation patterns
        citation_patterns = [
            r'\b(according to|cited by|referenced by|source:)\b',
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2023), (2022), etc.
            r'(\(|\s)et al\.(\)|\s|\.)',
            r'\b(survey|study|research|report)\b.{1,30}\b(found|shows|indicates|suggests)\b'
        ]
        
        citations_found = 0
        for page in self.pages:
            for pattern in citation_patterns:
                citations_found += len(re.findall(pattern, page["full_text"], re.IGNORECASE))
        
        if citations_found >= 5:
            score += 25
        elif citations_found >= 2:
            score += 15
        elif citations_found >= 1:
            score += 5
        else:
            self.recommendations.append({
                "category": "Authority Signals",
                "issue": "Lack of citations",
                "recommendation": "Add citations to credible sources to support key claims.",
                "importance": "High"
            })
        total_points += 25
        
        # Check for expert indicators
        expert_patterns = [
            r'\b(expert|specialist|professional|certified|licensed|PhD|doctor|professor)\b',
            r'\b(years of experience|expertise in|specializing in)\b',
            r'\b(credentials|qualifications|certification|degree)\b'
        ]
        
        expertise_signals = 0
        for page in self.pages:
            for pattern in expert_patterns:
                expertise_signals += len(re.findall(pattern, page["full_text"], re.IGNORECASE))
        
        if expertise_signals >= 3:
            score += 25
        elif expertise_signals >= 1:
            score += 15
        else:
            self.recommendations.append({
                "category": "Authority Signals",
                "issue": "Lack of expertise signals",
                "recommendation": "Add author credentials, expertise information, or professional qualifications.",
                "importance": "Medium"
            })
        total_points += 25
        
        # Check for factual statement patterns
        factual_patterns = [
            r'\b(statistics show|data indicates|research confirms|studies demonstrate)\b',
            r'\b(\d+%|\d+ percent)\b',
            r'\b(increase|decrease) of \d+%\b',
            r'\b(in \d{4}|since \d{4}|as of \d{4})\b'
        ]
        
        factual_signals = 0
        for page in self.pages:
            for pattern in factual_patterns:
                factual_signals += len(re.findall(pattern, page["full_text"], re.IGNORECASE))
        
        if factual_signals >= 5:
            score += 25
        elif factual_signals >= 2:
            score += 15
        else:
            self.recommendations.append({
                "category": "Authority Signals",
                "issue": "Lack of factual statements",
                "recommendation": "Include specific facts, statistics, and data points to support content.",
                "importance": "Medium"
            })
        total_points += 25
        
        # Check for trust indicators
        trust_patterns = [
            r'\b(guarantee|warranty|certified|accredited|trusted|secure)\b',
            r'\b(testimonial|review|rating|stars|feedback)\b',
            r'\b(privacy policy|terms of service|terms and conditions)\b'
        ]
        
        trust_signals = 0
        for page in self.pages:
            for pattern in trust_patterns:
                trust_signals += len(re.findall(pattern, page["full_text"], re.IGNORECASE))
        
        if trust_signals >= 3:
            score += 25
        elif trust_signals >= 1:
            score += 15
        else:
            self.recommendations.append({
                "category": "Authority Signals",
                "issue": "Lack of trust signals",
                "recommendation": "Add trust indicators like testimonials, reviews, or certifications.",
                "importance": "Medium"
            })
        total_points += 25
        
        # Normalize the final score
        if total_points > 0:
            normalized_score = (score / total_points) * 100
        else:
            normalized_score = 0
            
        self.scores["authority_signals"] = normalized_score
        return normalized_score
    
    def analyze_question_answering(self):
        """Analyze how well the content answers questions."""
        score = 0
        total_points = 0
        
        # Count explicit QA pairs
        total_qa_pairs = sum(len(page["qa_pairs"]) for page in self.pages)
        
        if total_qa_pairs >= 10:
            score += 40
        elif total_qa_pairs >= 5:
            score += 30
        elif total_qa_pairs >= 2:
            score += 20
        elif total_qa_pairs >= 1:
            score += 10
        else:
            self.recommendations.append({
                "category": "Question Answering",
                "issue": "Lack of explicit Q&A content",
                "recommendation": "Add FAQ sections with common questions and clear answers.",
                "importance": "Critical"
            })
        total_points += 40
        
        # Check for FAQ schema
        has_faq_schema = False
        for item in self.structured_data:
            if item["type"] == "JSON-LD":
                if isinstance(item["data"], dict) and item["data"].get("@type") == "FAQPage":
                    has_faq_schema = True
                    break
                elif isinstance(item["data"], list):
                    for entry in item["data"]:
                        if isinstance(entry, dict) and entry.get("@type") == "FAQPage":
                            has_faq_schema = True
                            break
        
        if has_faq_schema:
            score += 30
        else:
            self.recommendations.append({
                "category": "Question Answering",
                "issue": "Missing FAQ schema",
                "recommendation": "Implement FAQPage schema markup for question-answer content.",
                "importance": "High"
            })
        total_points += 30
        
        # Look for implicit question patterns in content
        question_patterns = [
            r'\b(how to|what is|why does|when should|where can|who should)\b',
            r'\b(benefits of|advantages of|disadvantages of)\b',
            r'\b(difference between|compared to|versus)\b',
            r'\b(steps to|guide to|tutorial for)\b'
        ]
        
        implicit_questions = 0
        for page in self.pages:
            for pattern in question_patterns:
                implicit_questions += len(re.findall(pattern, page["full_text"], re.IGNORECASE))
        
        if implicit_questions >= 10:
            score += 30
        elif implicit_questions >= 5:
            score += 20
        elif implicit_questions >= 2:
            score += 10
        else:
            self.recommendations.append({
                "category": "Question Answering",
                "issue": "Content doesn't address common questions",
                "recommendation": "Include 'how-to' sections, comparisons, and benefit analyses in your content.",
                "importance": "High"
            })
        total_points += 30
        
        # Normalize the final score
        if total_points > 0:
            normalized_score = (score / total_points) * 100
        else:
            normalized_score = 0
            
        self.scores["question_answering"] = normalized_score
        return normalized_score
    
    def calculate_overall_score(self):
        """Calculate the overall AI readiness score."""
        # Weights for each category
        weights = {
            "content_quality": 0.3,
            "technical_optimization": 0.3, 
            "authority_signals": 0.2,
            "question_answering": 0.2
        }
        
        # Ensure all component scores are calculated
        if self.scores["content_quality"] == 0:
            self.analyze_content_quality()
        if self.scores["technical_optimization"] == 0:
            self.analyze_technical_optimization()
        if self.scores["authority_signals"] == 0:
            self.analyze_authority_signals()
        if self.scores["question_answering"] == 0:
            self.analyze_question_answering()
        
        # Calculate weighted average
        overall_score = sum(
            self.scores[category] * weight 
            for category, weight in weights.items()
        )
        
        self.scores["overall"] = overall_score
        return overall_score
    
    def generate_report(self):
        """Generate a comprehensive report of the analysis."""
        # Calculate overall score if not already done
        if self.scores["overall"] == 0:
            self.calculate_overall_score()
        
        # Prioritize recommendations
        critical_recs = [r for r in self.recommendations if r["importance"] == "Critical"]
        high_recs = [r for r in self.recommendations if r["importance"] == "High"]
        medium_recs = [r for r in self.recommendations if r["importance"] == "Medium"]
        
        prioritized_recs = critical_recs + high_recs + medium_recs
        
        # Generate report
        report = {
            "site_info": {
                "domain": self.metadata["domain"],
                "title": self.metadata["title"],
                "description": self.metadata["description"],
                "pages_analyzed": self.metadata["pages_crawled"]
            },
            "ai_readiness_scores": {
                "overall": round(self.scores["overall"], 1),
                "components": {
                    "content_quality": round(self.scores["content_quality"], 1),
                    "technical_optimization": round(self.scores["technical_optimization"], 1),
                    "authority_signals": round(self.scores["authority_signals"], 1),
                    "question_answering": round(self.scores["question_answering"], 1)
                }
            },
            "content_stats": {
                "total_pages": self.metadata["pages_crawled"],
                "avg_word_count": round(self.metadata.get("avg_word_count", 0), 1),
                "pages_with_thin_content": self.metadata["pages_with_thin_content"],
                "total_qa_pairs": sum(len(page["qa_pairs"]) for page in self.pages),
                "has_structured_data": self.metadata["has_schema_markup"],
                "js_rendered_pages": self.metadata.get("js_rendered_pages", 0)
            },
            "top_recommendations": prioritized_recs[:5],
            "all_recommendations": self.recommendations
        }


        
        return report

# Example usage
if __name__ == "__main__":
    # This would normally come from the crawler
    # sample_data = {
    #     "metadata": {"domain": "example.com", "pages_crawled": 5},
    #     "pages": [],
    #     "structured_data": []
    # }
    # Load sample data from a JSON file
    with open('ribin_in_crawl_results.json', 'r') as file:
        sample_data = json.load(file)
    analyzer = AIReadinessAnalyzer(sample_data)
    report = analyzer.generate_report()
    print(report)
    # Save the report to a JSON file
    website_name = sample_data["metadata"].get("domain", "unknown_website").replace(".", "_")
    report_filename = f'ai_readiness_report_{website_name}.json'
    with open(report_filename, 'w') as report_file:
        json.dump(report, report_file, indent=4)
    print(f"Report has been saved to '{report_filename}'")