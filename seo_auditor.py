import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Tuple
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import ssl
import socket
from datetime import datetime
import hashlib
import concurrent.futures
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEOAuditor:
    def __init__(self, start_url: str, use_selenium: bool = True):
        """
        Initialize the SEO auditor.
        
        Args:
            start_url: The starting URL to audit
            use_selenium: Whether to use Selenium for JavaScript rendering
        """
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.use_selenium = use_selenium
        self.visited_urls = set()
        self.broken_links = []
        self.duplicate_meta_tags = []
        self.missing_h1_pages = []
        self.page_load_times = {}
        self.image_issues = []
        self.security_issues = []
        self.mobile_issues = []
        self.accessibility_issues = []
        self.performance_issues = []
        
        # Set up requests session with retry strategy
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1)
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        if self.use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Set up Selenium WebDriver for performance metrics."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-notifications")
            
            # Add mobile emulation
            mobile_emulation = {
                "deviceMetrics": { "width": 360, "height": 640, "pixelRatio": 3.0 },
                "userAgent": "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36"
            }
            chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            self.use_selenium = False
    
    def check_security(self, url: str) -> List[Dict[str, Any]]:
        """Check security-related issues."""
        issues = []
        
        try:
            # Check SSL/TLS
            hostname = urlparse(url).netloc
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443)) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    expiry_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    if expiry_date < datetime.now():
                        issues.append({
                            'type': 'security',
                            'issue': 'SSL certificate expired',
                            'details': f'Certificate expired on {expiry_date}'
                        })
                    
                    # Check SSL version
                    if ssock.version() in ['SSLv2', 'SSLv3']:
                        issues.append({
                            'type': 'security',
                            'issue': 'Outdated SSL version',
                            'details': f'Using {ssock.version()} which is insecure'
                        })
        
        except Exception as e:
            issues.append({
                'type': 'security',
                'issue': 'SSL/TLS error',
                'details': str(e)
            })
        
        # Check security headers
        try:
            response = self.session.head(url, timeout=5)
            headers = response.headers
            
            security_headers = {
                'Strict-Transport-Security': 'HSTS not enabled',
                'X-Content-Type-Options': 'Content-Type-Options not set',
                'X-Frame-Options': 'Frame-Options not set',
                'X-XSS-Protection': 'XSS Protection not enabled',
                'Content-Security-Policy': 'CSP not configured'
            }
            
            for header, message in security_headers.items():
                if header not in headers:
                    issues.append({
                        'type': 'security',
                        'issue': message,
                        'details': f'Missing {header} header'
                    })
        
        except Exception as e:
            issues.append({
                'type': 'security',
                'issue': 'Failed to check security headers',
                'details': str(e)
            })
        
        return issues
    
    def check_mobile_responsiveness(self, url: str) -> List[Dict[str, Any]]:
        """Check mobile responsiveness issues."""
        issues = []
        
        if not self.use_selenium:
            return issues
        
        try:
            # Set mobile viewport
            self.driver.set_window_size(360, 640)
            self.driver.get(url)
            
            # Check viewport meta tag
            viewport = self.driver.find_elements(By.CSS_SELECTOR, 'meta[name="viewport"]')
            if not viewport:
                issues.append({
                    'type': 'mobile',
                    'issue': 'Missing viewport meta tag',
                    'details': 'Add viewport meta tag for proper mobile rendering'
                })
            
            # Check for fixed width elements
            fixed_width_elements = self.driver.find_elements(By.CSS_SELECTOR, '*[style*="width:"]')
            for element in fixed_width_elements:
                width = element.get_attribute('style')
                if 'px' in width and int(width.split('px')[0]) > 360:
                    issues.append({
                        'type': 'mobile',
                        'issue': 'Fixed width element too wide',
                        'details': f'Element has width {width} which may cause horizontal scrolling'
                    })
            
            # Check text size
            small_text = self.driver.find_elements(By.CSS_SELECTOR, '*[style*="font-size:"]')
            for element in small_text:
                font_size = element.get_attribute('style')
                if 'px' in font_size and int(font_size.split('px')[0]) < 12:
                    issues.append({
                        'type': 'mobile',
                        'issue': 'Text too small on mobile',
                        'details': f'Text size {font_size} may be hard to read on mobile'
                    })
            
            # Check touch targets
            clickable = self.driver.find_elements(By.CSS_SELECTOR, 'a, button, input, select')
            for element in clickable:
                size = element.size
                if size['width'] < 44 or size['height'] < 44:
                    issues.append({
                        'type': 'mobile',
                        'issue': 'Touch target too small',
                        'details': f'Element size {size} is below recommended 44x44px'
                    })
        
        except Exception as e:
            issues.append({
                'type': 'mobile',
                'issue': 'Mobile responsiveness check failed',
                'details': str(e)
            })
        
        return issues
    
    def check_accessibility(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Check accessibility issues."""
        issues = []
        
        # Check ARIA landmarks
        landmarks = soup.find_all(['header', 'nav', 'main', 'aside', 'footer'])
        if not landmarks:
            issues.append({
                'type': 'accessibility',
                'issue': 'Missing ARIA landmarks',
                'details': 'Add semantic HTML5 elements or ARIA landmarks'
            })
        
        # Check form labels
        forms = soup.find_all('form')
        for form in forms:
            inputs = form.find_all(['input', 'select', 'textarea'])
            for input_elem in inputs:
                if not input_elem.get('id'):
                    continue
                label = form.find('label', attrs={'for': input_elem['id']})
                if not label:
                    issues.append({
                        'type': 'accessibility',
                        'issue': 'Form input missing label',
                        'details': f'Input with id {input_elem["id"]} has no associated label'
                    })
        
        # Check color contrast (basic check)
        text_elements = soup.find_all(['p', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for elem in text_elements:
            style = elem.get('style', '')
            if 'color:' in style and 'background-color:' not in style:
                issues.append({
                    'type': 'accessibility',
                    'issue': 'Potential color contrast issue',
                    'details': 'Text color specified without background color'
                })
        
        # Check skip links
        skip_links = soup.find_all('a', href='#main-content')
        if not skip_links:
            issues.append({
                'type': 'accessibility',
                'issue': 'Missing skip link',
                'details': 'Add a skip link for keyboard navigation'
            })
        
        return issues
    
    def measure_page_load_metrics(self, url: str) -> Dict[str, float]:
        """Measure detailed page load metrics using Selenium."""
        if not self.use_selenium:
            return {}
        
        try:
            self.driver.get(url)
            
            # Wait for page load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get performance metrics
            performance = self.driver.execute_script("""
                var performance = window.performance || window.mozPerformance || window.msPerformance || window.webkitPerformance || {};
                var timing = performance.timing || {};
                var navigation = performance.navigation || {};
                var resources = performance.getEntriesByType('resource') || [];
                
                return {
                    'timing': {
                        'navigationStart': timing.navigationStart,
                        'loadEventEnd': timing.loadEventEnd,
                        'domComplete': timing.domComplete,
                        'domInteractive': timing.domInteractive,
                        'responseEnd': timing.responseEnd,
                        'responseStart': timing.responseStart,
                        'requestStart': timing.requestStart,
                        'connectEnd': timing.connectEnd,
                        'connectStart': timing.connectStart,
                        'domainLookupEnd': timing.domainLookupEnd,
                        'domainLookupStart': timing.domainLookupStart
                    },
                    'navigation': {
                        'type': navigation.type,
                        'redirectCount': navigation.redirectCount
                    },
                    'resources': resources.map(function(r) {
                        return {
                            'name': r.name,
                            'type': r.initiatorType,
                            'duration': r.duration,
                            'size': r.transferSize
                        };
                    })
                };
            """)
            
            # Calculate metrics
            timing = performance.get('timing', {})
            metrics = {
                'FCP': timing.get('domInteractive', 0) - timing.get('navigationStart', 0),
                'LCP': timing.get('loadEventEnd', 0) - timing.get('navigationStart', 0),
                'TTFB': timing.get('responseStart', 0) - timing.get('requestStart', 0),
                'DOM_Load': timing.get('domComplete', 0) - timing.get('domInteractive', 0),
                'Total_Load': timing.get('loadEventEnd', 0) - timing.get('navigationStart', 0)
            }
            
            # Calculate resource metrics
            resources = performance.get('resources', [])
            if resources:
                total_size = sum(r.get('size', 0) for r in resources)
                metrics['Total_Size'] = total_size
                metrics['Resource_Count'] = len(resources)
                
                # Calculate by resource type
                resource_types = {}
                for r in resources:
                    r_type = r.get('type', 'other')
                    if r_type not in resource_types:
                        resource_types[r_type] = {'count': 0, 'size': 0}
                    resource_types[r_type]['count'] += 1
                    resource_types[r_type]['size'] += r.get('size', 0)
                
                metrics['Resource_Types'] = resource_types
            
            return metrics
        except Exception as e:
            logger.error(f"Error measuring page load metrics for {url}: {e}")
            return {}
    
    def check_image_optimization(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Check image optimization issues."""
        issues = []
        for img in soup.find_all('img'):
            img_url = urljoin(url, img.get('src', ''))
            
            # Check for alt text
            if not img.get('alt'):
                issues.append({
                    'url': img_url,
                    'issue': 'Missing alt text',
                    'page': url
                })
            
            # Check for lazy loading
            if not img.get('loading') == 'lazy':
                issues.append({
                    'url': img_url,
                    'issue': 'Missing lazy loading',
                    'page': url
                })
            
            # Check for srcset
            if not img.get('srcset'):
                issues.append({
                    'url': img_url,
                    'issue': 'Missing srcset for responsive images',
                    'page': url
                })
            
            # Check image size and format
            try:
                response = self.session.head(img_url, timeout=5)
                if response.headers.get('content-length'):
                    size = int(response.headers['content-length'])
                    if size > 200000:  # 200KB
                        issues.append({
                            'url': img_url,
                            'issue': f'Large image size ({size/1024:.1f}KB)',
                            'page': url
                        })
                    
                    # Check image format
                    content_type = response.headers.get('content-type', '')
                    if 'image/webp' not in content_type and 'image/avif' not in content_type:
                        issues.append({
                            'url': img_url,
                            'issue': 'Not using modern image format (WebP/AVIF)',
                            'page': url
                        })
            except:
                pass
        
        return issues
    
    def check_meta_tags(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Check meta tags and identify duplicates."""
        issues = []
        meta_tags = {}
        
        # Check title
        title = soup.find('title')
        if not title:
            issues.append({
                'page': url,
                'issue': 'Missing title tag'
            })
        else:
            title_text = title.text.strip()
            meta_tags['title'] = title_text
            
            # Check title length
            if len(title_text) < 30 or len(title_text) > 60:
                issues.append({
                    'page': url,
                    'issue': f'Title length ({len(title_text)} chars) outside recommended range (30-60)'
                })
        
        # Check meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if not meta_desc:
            issues.append({
                'page': url,
                'issue': 'Missing meta description'
            })
        else:
            desc_text = meta_desc.get('content', '').strip()
            meta_tags['description'] = desc_text
            
            # Check description length
            if len(desc_text) < 120 or len(desc_text) > 160:
                issues.append({
                    'page': url,
                    'issue': f'Meta description length ({len(desc_text)} chars) outside recommended range (120-160)'
                })
        
        # Check canonical
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        if not canonical:
            issues.append({
                'page': url,
                'issue': 'Missing canonical tag'
            })
        
        # Check Open Graph tags
        og_tags = ['title', 'description', 'image', 'url']
        for tag in og_tags:
            if not soup.find('meta', attrs={'property': f'og:{tag}'}):
                issues.append({
                    'page': url,
                    'issue': f'Missing Open Graph {tag} tag'
                })
        
        # Check Twitter Card tags
        twitter_tags = ['card', 'title', 'description', 'image']
        for tag in twitter_tags:
            if not soup.find('meta', attrs={'name': f'twitter:{tag}'}):
                issues.append({
                    'page': url,
                    'issue': f'Missing Twitter Card {tag} tag'
                })
        
        # Check for duplicate meta tags
        for tag_type, content in meta_tags.items():
            if content in self.duplicate_meta_tags:
                issues.append({
                    'page': url,
                    'issue': f'Duplicate {tag_type} tag'
                })
            else:
                self.duplicate_meta_tags.append(content)
        
        return issues
    
    def check_headings(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Check heading structure and identify missing H1 tags."""
        issues = []
        h1_tags = soup.find_all('h1')
        
        if not h1_tags:
            issues.append({
                'page': url,
                'issue': 'Missing H1 tag'
            })
            self.missing_h1_pages.append(url)
        elif len(h1_tags) > 1:
            issues.append({
                'page': url,
                'issue': f'Multiple H1 tags found ({len(h1_tags)})'
            })
        
        # Check heading hierarchy
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings:
            prev_level = 0
            for heading in headings:
                current_level = int(heading.name[1])
                if current_level - prev_level > 1:
                    issues.append({
                        'page': url,
                        'issue': f'Heading hierarchy skip: {heading.name} after h{prev_level}'
                    })
                prev_level = current_level
        
        return issues
    
    def check_links(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Check for broken links and track internal/external links."""
        issues = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            
            # Skip javascript: and mailto: links
            if href.startswith(('javascript:', 'mailto:')):
                continue
            
            try:
                response = self.session.head(absolute_url, timeout=5, allow_redirects=True)
                if response.status_code >= 400:
                    issues.append({
                        'page': url,
                        'link': absolute_url,
                        'issue': f'Broken link (Status: {response.status_code})'
                    })
                    self.broken_links.append({
                        'source_page': url,
                        'broken_url': absolute_url,
                        'status_code': response.status_code
                    })
                
                # Check for redirects
                if response.history:
                    issues.append({
                        'page': url,
                        'link': absolute_url,
                        'issue': f'Link redirects ({len(response.history)} redirects)'
                    })
                
                # Check for nofollow
                if not link.get('rel') or 'nofollow' not in link.get('rel', []):
                    issues.append({
                        'page': url,
                        'link': absolute_url,
                        'issue': 'External link missing nofollow attribute'
                    })
            
            except requests.RequestException as e:
                issues.append({
                    'page': url,
                    'link': absolute_url,
                    'issue': f'Link error: {str(e)}'
                })
                self.broken_links.append({
                    'source_page': url,
                    'broken_url': absolute_url,
                    'error': str(e)
                })
        
        return issues
    
    def audit_page(self, url: str) -> Dict[str, Any]:
        """Audit a single page for SEO issues."""
        try:
            if self.use_selenium:
                self.driver.get(url)
                html = self.driver.page_source
            else:
                response = self.session.get(url, timeout=10)
                html = response.text
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Run all checks
            load_metrics = self.measure_page_load_metrics(url)
            self.page_load_times[url] = load_metrics
            
            image_issues = self.check_image_optimization(soup, url)
            meta_issues = self.check_meta_tags(soup, url)
            heading_issues = self.check_headings(soup, url)
            link_issues = self.check_links(soup, url)
            security_issues = self.check_security(url)
            mobile_issues = self.check_mobile_responsiveness(url)
            accessibility_issues = self.check_accessibility(soup, url)
            
            # Combine all issues
            all_issues = (
                image_issues + meta_issues + heading_issues + 
                link_issues + security_issues + mobile_issues + 
                accessibility_issues
            )
            
            return {
                'url': url,
                'load_metrics': load_metrics,
                'issues': all_issues
            }
            
        except Exception as e:
            logger.error(f"Error auditing page {url}: {e}")
            return {
                'url': url,
                'error': str(e)
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive SEO audit report."""
        report = {
            'site_info': {
                'domain': self.domain,
                'start_url': self.start_url,
                'audit_date': datetime.now().isoformat()
            },
            'performance_metrics': {
                'page_load_times': self.page_load_times
            },
            'issues': {
                'broken_links': self.broken_links,
                'missing_h1_pages': self.missing_h1_pages,
                'duplicate_meta_tags': self.duplicate_meta_tags,
                'image_issues': self.image_issues,
                'security_issues': self.security_issues,
                'mobile_issues': self.mobile_issues,
                'accessibility_issues': self.accessibility_issues,
                'performance_issues': self.performance_issues
            },
            'recommendations': []
        }
        
        # Generate recommendations based on issues
        if self.broken_links:
            report['recommendations'].append({
                'category': 'Links',
                'issue': 'Broken links found',
                'recommendation': 'Fix or remove broken links to improve user experience and SEO'
            })
        
        if self.missing_h1_pages:
            report['recommendations'].append({
                'category': 'Content Structure',
                'issue': 'Pages missing H1 tags',
                'recommendation': 'Add H1 tags to all pages for better content hierarchy'
            })
        
        if self.duplicate_meta_tags:
            report['recommendations'].append({
                'category': 'Meta Tags',
                'issue': 'Duplicate meta tags found',
                'recommendation': 'Ensure unique meta titles and descriptions for each page'
            })
        
        if self.image_issues:
            report['recommendations'].append({
                'category': 'Images',
                'issue': 'Image optimization issues',
                'recommendation': 'Optimize images by adding alt text, implementing lazy loading, and compressing large images'
            })
        
        if self.security_issues:
            report['recommendations'].append({
                'category': 'Security',
                'issue': 'Security vulnerabilities found',
                'recommendation': 'Implement security headers and ensure SSL/TLS is properly configured'
            })
        
        if self.mobile_issues:
            report['recommendations'].append({
                'category': 'Mobile',
                'issue': 'Mobile responsiveness issues',
                'recommendation': 'Improve mobile optimization by fixing viewport issues and touch targets'
            })
        
        if self.accessibility_issues:
            report['recommendations'].append({
                'category': 'Accessibility',
                'issue': 'Accessibility issues found',
                'recommendation': 'Implement ARIA landmarks and ensure proper heading hierarchy'
            })
        
        if self.performance_issues:
            report['recommendations'].append({
                'category': 'Performance',
                'issue': 'Performance issues found',
                'recommendation': 'Optimize page load times and resource loading'
            })
        
        return report
    
    def run_audit(self) -> Dict[str, Any]:
        """Run the complete SEO audit."""
        try:
            # Start with the main page
            self.audit_page(self.start_url)
            
            # Generate and return the report
            report = self.generate_report()
            
            # Save report to file
            output_file = f"{self.domain.replace('.', '_')}_seo_audit.json"
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
            
        finally:
            if self.use_selenium:
                self.driver.quit()

if __name__ == "__main__":
    # Example usage
    auditor = SEOAuditor("https://example.com")
    report = auditor.run_audit()
    print(json.dumps(report, indent=2)) 