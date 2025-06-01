import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
import time
from collections import defaultdict
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebsiteCrawler:
    def __init__(self, start_url, max_pages=10, same_domain_only=True, use_selenium=True, headless=True, wait_time=5):
        """
        Initialize the crawler with a starting URL and constraints.
        
        Args:
            start_url: The URL to start crawling from
            max_pages: Maximum number of pages to crawl
            same_domain_only: Whether to restrict crawling to the same domain
            use_selenium: Whether to use Selenium for JavaScript rendering
            headless: Whether to run the browser in headless mode
            wait_time: How many seconds to wait for page to load JS content
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only
        self.use_selenium = use_selenium
        self.headless = headless
        self.wait_time = wait_time
        
        self.visited_urls = set()
        self.to_visit = [start_url]
        self.domain = urlparse(start_url).netloc
        self.pages_data = []
        self.structured_data = []
        self.site_metadata = {
            "title": "",
            "description": "",
            "domain": self.domain,
            "pages_crawled": 0,
            "total_word_count": 0,
            "heading_count": defaultdict(int),
            "has_schema_markup": False,
            "internal_links": 0,
            "external_links": 0,
            "image_count": 0,
            "pages_with_thin_content": 0,
            "js_rendered_pages": 0
        }
        
        # Set up Selenium WebDriver if needed
        if self.use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Set up the Selenium WebDriver for JavaScript rendering."""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-notifications")
            
            # Set up service and driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            self.use_selenium = False
            logger.warning("Falling back to requests-only mode")
    
    def is_valid_url(self, url):
        """Check if a URL should be crawled based on our constraints."""
        if not url or url in self.visited_urls:
            return False
            
        # Skip URLs with fragments or query parameters to avoid duplicates
        url_parts = urlparse(url)
        if url_parts.fragment:
            # Only skip fragment URLs if they refer to the same page
            base_url = url_parts._replace(fragment='').geturl()
            if base_url in self.visited_urls:
                return False
            
        # If we're restricting to the same domain, check the domain
        if self.same_domain_only and url_parts.netloc != self.domain:
            return False
            
        # Skip common non-content file types
        if url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.css', '.js')):
            return False
            
        return True
    
    def extract_links(self, soup, current_url):
        """Extract all links from a page."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()
            if href and not href.startswith(('javascript:', 'mailto:', 'tel:')):
                # Convert relative URLs to absolute
                full_url = urljoin(current_url, href)
                if self.is_valid_url(full_url):
                    links.append(full_url)
                    
                    # Count internal vs external links
                    if urlparse(full_url).netloc == self.domain:
                        self.site_metadata["internal_links"] += 1
                    else:
                        self.site_metadata["external_links"] += 1
        print(f"Found {len(links)} valid links on {current_url}")
        print(f"Internal links: {self.site_metadata['internal_links']}, External links: {self.site_metadata['external_links']}")
        return links
    
    def extract_structured_data(self, soup, url):
        """Extract JSON-LD structured data from the page."""
        structured_data = []
        
        # Look for JSON-LD
        json_ld_count = 0
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.append({
                    "url": url,
                    "type": "JSON-LD",
                    "data": data
                })
                self.site_metadata["has_schema_markup"] = True
                json_ld_count += 1
            except (json.JSONDecodeError, TypeError):
                pass
                
        # Look for microdata
        items = []
        microdata_count = 0
        for element in soup.find_all(itemscope=True):
            item_type = element.get('itemtype', '')
            if item_type:
                items.append({
                    "type": item_type,
                    "url": url
                })
                self.site_metadata["has_schema_markup"] = True
                microdata_count += 1
                
        if items:
            structured_data.append({
                "url": url,
                "type": "Microdata",
                "data": items
            })

        print(f"Structured data on {url}: {json_ld_count} JSON-LD blocks, {microdata_count} microdata items")    
        return structured_data
            
    
    def extract_page_content(self, soup, url, js_rendered=False):
        """Extract and analyze the content of a page."""
        # Basic page information
        title = soup.title.string.strip() if soup.title else ""
        print(f"Page title: {title}")
        
        # If this is the home page, use it for the site title
        if url == self.start_url:
            self.site_metadata["title"] = title
            
        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '')
            
            # If this is the home page, use it for the site description
            if url == self.start_url:
                self.site_metadata["description"] = meta_desc
        else:
            print("No meta description found")
        
        # Extract all text content (excluding scripts, styles, etc.)
        for script in soup(['script', 'style', 'noscript', 'iframe', 'head']):
            script.extract()
            
        text_content = soup.get_text(separator=' ', strip=True)
        cleaned_text = re.sub(r'\s+', ' ', text_content).strip()
        word_count = len(cleaned_text.split())
        
        # Update site metadata
        self.site_metadata["total_word_count"] += word_count
        if word_count < 300:
            self.site_metadata["pages_with_thin_content"] += 1
        
        if js_rendered:
            self.site_metadata["js_rendered_pages"] += 1
            
        # Count headings by level
        for i in range(1, 7):
            heading_count = len(soup.find_all(f'h{i}'))
            self.site_metadata["heading_count"][f"h{i}"] += heading_count
            
        # Count images
        self.site_metadata["image_count"] += len(soup.find_all('img'))
        
        # Extract headings with their text
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    "level": i,
                    "text": heading.get_text(strip=True)
                })
                
        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        
        # Look for question-answer patterns
        qa_pairs = []
        question_elements = soup.find_all(['h2', 'h3', 'h4', 'strong'], string=lambda s: s and '?' in s)
        for question_el in question_elements:
            question = question_el.get_text(strip=True)
            # Look for the next paragraph as a potential answer
            answer_el = question_el.find_next('p')
            if answer_el:
                answer = answer_el.get_text(strip=True)
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
        
        # Look for FAQ sections
        faq_sections = soup.find_all(['div', 'section'], class_=lambda c: c and ('faq' in c.lower() if c else False))
        if not faq_sections:
            # Try to find by ID
            faq_sections = soup.find_all(id=lambda i: i and ('faq' in i.lower() if i else False))
            
        for faq_section in faq_sections:
            # Look for question-answer pairs within this section
            questions = faq_section.find_all(['h2', 'h3', 'h4', 'dt', 'strong'])
            for q in questions:
                question = q.get_text(strip=True)
                # Find the answer - could be in a dd tag, p tag, or div
                answer_el = q.find_next(['dd', 'p', 'div'])
                if answer_el:
                    answer = answer_el.get_text(strip=True)
                    qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })
        
        return {
            "url": url,
            "title": title,
            "meta_description": meta_desc,
            "word_count": word_count,
            "headings": headings,
            "paragraphs": paragraphs,
            "qa_pairs": qa_pairs,
            "full_text": cleaned_text,
            "js_rendered": js_rendered
        }
    
    def fetch_page_with_requests(self, url):
        """Fetch a page using the requests library."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code}")
                return None
                
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logger.warning(f"Skipping non-HTML content at {url}")
                return None
                
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url} with requests: {e}")
            return None
    
    def fetch_page_with_selenium(self, url):
        """Fetch a page using Selenium for JavaScript rendering."""
        if not self.use_selenium:
            return None
            
        try:
            self.driver.get(url)
            # Wait for page to load and JavaScript to execute
            WebDriverWait(self.driver, self.wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for potential async content
            time.sleep(2)
            
            # Get the page source after JavaScript execution
            return self.driver.page_source
        except TimeoutException:
            logger.warning(f"Timeout while loading {url} with Selenium")
            return None
        except WebDriverException as e:
            logger.error(f"Selenium error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with Selenium for {url}: {e}")
            return None
    
    def compare_content(self, requests_html, selenium_html):
        """Compare content from requests vs. Selenium to detect JS-rendered content."""
        if not requests_html or not selenium_html:
            return False
            
        # Parse both HTML documents
        soup_requests = BeautifulSoup(requests_html, 'html.parser')
        soup_selenium = BeautifulSoup(selenium_html, 'html.parser')
        
        # Extract text content
        requests_text = soup_requests.get_text(separator=' ', strip=True)
        selenium_text = soup_selenium.get_text(separator=' ', strip=True)
        
        # Compare word counts
        requests_words = len(requests_text.split())
        selenium_words = len(selenium_text.split())
        
        # If Selenium found significantly more content, it's likely JS-rendered
        if selenium_words > requests_words * 1.2 and selenium_words - requests_words > 100:
            return True
            
        # Count paragraphs
        requests_paragraphs = len(soup_requests.find_all('p'))
        selenium_paragraphs = len(soup_selenium.find_all('p'))
        
        # If Selenium found significantly more paragraphs, it's likely JS-rendered
        if selenium_paragraphs > requests_paragraphs * 1.2 and selenium_paragraphs - requests_paragraphs > 3:
            return True
            
        return False
    
    def crawl(self):
        """Crawl the website and collect data."""
        try:
            while self.to_visit and len(self.visited_urls) < self.max_pages:
                # Get the next URL to visit
                current_url = self.to_visit.pop(0)
                
                # Skip if we've already visited this URL
                if current_url in self.visited_urls:
                    continue
                    
                logger.info(f"Crawling: {current_url}")
                
                # Mark as visited
                self.visited_urls.add(current_url)
                
                # First try with requests (faster)
                requests_html = self.fetch_page_with_requests(current_url)
                
                # Only use Selenium if requests was successful (to avoid double timeouts)
                js_rendered = False
                page_html = requests_html
                
                if requests_html and self.use_selenium:
                    selenium_html = self.fetch_page_with_selenium(current_url)
                    if selenium_html:
                        # Check if Selenium found more content (indicating JS-rendered content)
                        js_rendered = self.compare_content(requests_html, selenium_html)
                        if js_rendered:
                            page_html = selenium_html
                            logger.info(f"Found JS-rendered content on {current_url}")
                
                if not page_html:
                    logger.warning(f"Skipping {current_url}: Failed to fetch content")
                    continue
                
                # Parse the HTML
                soup = BeautifulSoup(page_html, 'html.parser')
                
                # Extract links and add to the to-visit list
                links = self.extract_links(soup, current_url)
                for link in links:
                    if link not in self.visited_urls and link not in self.to_visit:
                        self.to_visit.append(link)
                
                # Extract structured data
                page_structured_data = self.extract_structured_data(soup, current_url)
                if page_structured_data:
                    self.structured_data.extend(page_structured_data)
                
                # Extract and analyze page content
                page_data = self.extract_page_content(soup, current_url, js_rendered)
                self.pages_data.append(page_data)
                
                # Increment pages crawled
                self.site_metadata["pages_crawled"] += 1
                
                # Be respectful with crawl rate
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error during crawl: {e}")
        finally:
            # Clean up Selenium resources
            if self.use_selenium:
                try:
                    self.driver.quit()
                    logger.info("Selenium WebDriver closed")
                except:
                    pass
        
        # Finalize site metadata
        if self.site_metadata["pages_crawled"] > 0:
            self.site_metadata["avg_word_count"] = self.site_metadata["total_word_count"] / self.site_metadata["pages_crawled"]
            self.site_metadata["js_content_percentage"] = (self.site_metadata["js_rendered_pages"] / self.site_metadata["pages_crawled"]) * 100
        
        results = {
             "metadata": self.site_metadata,
            "pages": self.pages_data,
            "structured_data": self.structured_data
        }
        website_name = urlparse(self.start_url).netloc.replace('.', '_')
        output_file = f"{website_name}_crawl_results_v2.json"
        with open(output_file, "w") as f:
            f.write(json.dumps(results, indent=2))
        print(f"Results saved to {output_file}")
        return results



# Example usage
if __name__ == "__main__":
    crawler = WebsiteCrawler("https://juspay.io", max_pages=5, use_selenium=True)
    results = crawler.crawl()
    print(json.dumps(results, indent=2))