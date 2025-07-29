#!/usr/bin/env python3
"""
Municipal Web Scraper for Swiss Gemeinde Websites
Specialized for scraping and processing municipal information
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from pathlib import Path
import re
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MunicipalDocument:
    """Structure for municipal documents"""
    title: str
    content: str
    url: str
    document_type: str
    language: str
    importance_score: float
    last_updated: Optional[str] = None
    category: Optional[str] = None

class MunicipalWebScraper:
    """Web scraper optimized for Swiss municipal websites"""
    
    def __init__(self, municipality_name: str, base_url: str):
        self.municipality_name = municipality_name
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Municipal-specific content priorities
        self.content_priorities = {
            'services': 1.0,          # Municipal services
            'verwaltung': 1.0,        # Administration
            'dienstleistungen': 1.0,  # Services
            'formulare': 0.9,         # Forms
            'aktuell': 0.8,          # News/Current
            'veranstaltungen': 0.7,   # Events
            'tourismus': 0.6,         # Tourism
            'politik': 0.8,           # Politics
            'gemeinderat': 0.9,       # Municipal council
            'abstimmungen': 0.9,      # Voting
            'steuern': 0.9,          # Taxes
            'bauen': 0.8,            # Building permits
            'schulen': 0.8,          # Schools
            'soziales': 0.8,         # Social services
        }
        
        # Document type mapping
        self.document_types = {
            '.pdf': 'pdf',
            '.doc': 'document',
            '.docx': 'document',
            '.txt': 'text',
            'html': 'webpage'
        }
    
    def scrape_municipality(self, max_pages: int = 100) -> List[MunicipalDocument]:
        """
        Scrape the entire municipality website
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of MunicipalDocument objects
        """
        documents = []
        visited_urls = set()
        urls_to_visit = [self.base_url]
        
        logger.info(f"Starting scrape of {self.municipality_name} website: {self.base_url}")
        
        for page_count in range(max_pages):
            if not urls_to_visit:
                break
                
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            
            try:
                # Scrape current page
                page_documents = self._scrape_page(current_url)
                documents.extend(page_documents)
                
                # Find new URLs to visit
                new_urls = self._extract_links(current_url)
                for url in new_urls:
                    if url not in visited_urls and self._is_relevant_url(url):
                        urls_to_visit.append(url)
                
                logger.info(f"Scraped page {page_count + 1}: {current_url} ({len(page_documents)} documents)")
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {e}")
                continue
        
        logger.info(f"Scraping complete. Found {len(documents)} documents from {self.municipality_name}")
        return documents
    
    def _scrape_page(self, url: str) -> List[MunicipalDocument]:
        """Scrape a single page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            content = self._extract_content(soup)
            if not content.strip():
                return []
            
            # Determine document properties
            title = self._extract_title(soup)
            document_type = self._determine_document_type(url)
            language = self._detect_language(content)
            importance_score = self._calculate_importance_score(url, content, title)
            category = self._categorize_content(url, content)
            
            document = MunicipalDocument(
                title=title,
                content=content,
                url=url,
                document_type=document_type,
                language=language,
                importance_score=importance_score,
                category=category,
                last_updated=datetime.now().isoformat()
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return []
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page"""
        # Remove navigation, footer, sidebar
        for element in soup.find_all(['nav', 'footer', 'aside', 'script', 'style']):
            element.decompose()
        
        # Look for main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            return main_content.get_text(strip=True, separator=' ')
        else:
            return soup.get_text(strip=True, separator=' ')
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        h1_tag = soup.find('h1')
        
        if h1_tag:
            return h1_tag.get_text(strip=True)
        elif title_tag:
            return title_tag.get_text(strip=True)
        else:
            return "Untitled"
    
    def _extract_links(self, url: str) -> List[str]:
        """Extract all links from a page"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                links.append(absolute_url)
            
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {e}")
            return []
    
    def _is_relevant_url(self, url: str) -> bool:
        """Check if URL is relevant for municipal content"""
        parsed = urlparse(url)
        
        # Must be same domain
        if parsed.netloc != urlparse(self.base_url).netloc:
            return False
        
        # Skip common non-content URLs
        skip_patterns = [
            r'/images?/',
            r'/css/',
            r'/js/',
            r'/assets/',
            r'\.css$',
            r'\.js$',
            r'\.jpg$',
            r'\.png$',
            r'\.gif$',
            r'/search',
            r'/login',
            r'/admin'
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    def _determine_document_type(self, url: str) -> str:
        """Determine document type from URL"""
        for ext, doc_type in self.document_types.items():
            if url.lower().endswith(ext):
                return doc_type
        return 'webpage'
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection for German/French"""
        german_words = ['der', 'die', 'das', 'und', 'ist', 'mit', 'für', 'auf', 'gemeinde', 'verwaltung']
        french_words = ['le', 'la', 'les', 'et', 'est', 'avec', 'pour', 'sur', 'commune', 'administration']
        
        content_lower = content.lower()
        
        german_count = sum(1 for word in german_words if word in content_lower)
        french_count = sum(1 for word in french_words if word in content_lower)
        
        if german_count > french_count:
            return 'de'
        elif french_count > german_count:
            return 'fr'
        else:
            return 'de'  # Default to German for Swiss municipalities
    
    def _calculate_importance_score(self, url: str, content: str, title: str) -> float:
        """Calculate importance score based on content and URL"""
        score = 0.5  # Base score
        
        # URL-based scoring
        url_lower = url.lower()
        for keyword, weight in self.content_priorities.items():
            if keyword in url_lower:
                score += weight * 0.3
        
        # Content-based scoring
        content_lower = content.lower()
        for keyword, weight in self.content_priorities.items():
            if keyword in content_lower:
                score += weight * 0.2
        
        # Title-based scoring
        title_lower = title.lower()
        for keyword, weight in self.content_priorities.items():
            if keyword in title_lower:
                score += weight * 0.4
        
        # Boost for official documents
        if any(term in content_lower for term in ['verordnung', 'reglement', 'gesetz', 'beschluss']):
            score += 0.3
        
        # Boost for contact information
        if any(term in content_lower for term in ['kontakt', 'telefon', 'email', 'adresse']):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _categorize_content(self, url: str, content: str) -> str:
        """Categorize content into municipal categories"""
        categories = {
            'services': ['dienstleistung', 'service', 'formular', 'antrag'],
            'administration': ['verwaltung', 'gemeinderat', 'behörde', 'amt'],
            'news': ['aktuell', 'news', 'mitteilung', 'information'],
            'events': ['veranstaltung', 'event', 'termin', 'kalender'],
            'tourism': ['tourismus', 'sehenswürdigkeit', 'kultur', 'freizeit'],
            'politics': ['politik', 'abstimmung', 'wahlen', 'gemeinderat'],
            'infrastructure': ['bauen', 'verkehr', 'infrastruktur', 'planung'],
            'social': ['sozial', 'gesundheit', 'bildung', 'schule'],
            'finance': ['finanzen', 'steuer', 'budget', 'rechnung']
        }
        
        text = (url + ' ' + content).lower()
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'general'
    
    def save_documents(self, documents: List[MunicipalDocument], output_dir: str = "municipal_data"):
        """Save documents to structured format"""
        output_path = Path(output_dir) / self.municipality_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        documents_data = []
        for doc in documents:
            documents_data.append({
                'title': doc.title,
                'content': doc.content,
                'url': doc.url,
                'document_type': doc.document_type,
                'language': doc.language,
                'importance_score': doc.importance_score,
                'category': doc.category,
                'last_updated': doc.last_updated
            })
        
        json_path = output_path / "documents.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, indent=2, ensure_ascii=False)
        
        # Save high-importance documents separately
        high_importance_docs = [doc for doc in documents if doc.importance_score > 0.7]
        high_importance_path = output_path / "high_importance_documents.json"
        
        high_importance_data = []
        for doc in high_importance_docs:
            high_importance_data.append({
                'title': doc.title,
                'content': doc.content,
                'url': doc.url,
                'importance_score': doc.importance_score,
                'category': doc.category
            })
        
        with open(high_importance_path, 'w', encoding='utf-8') as f:
            json.dump(high_importance_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(documents)} documents to {output_path}")
        logger.info(f"High importance documents: {len(high_importance_docs)}")

# Arlesheim-specific configuration
def scrape_arlesheim():
    """Scrape Arlesheim municipality website"""
    scraper = MunicipalWebScraper(
        municipality_name="Arlesheim",
        base_url="https://www.arlesheim.ch"
    )
    
    documents = scraper.scrape_municipality(max_pages=50)
    scraper.save_documents(documents)
    
    return documents

if __name__ == "__main__":
    # Example usage
    documents = scrape_arlesheim()
    print(f"Scraped {len(documents)} documents from Arlesheim")
    
    # Show high-importance documents
    high_importance = [doc for doc in documents if doc.importance_score > 0.7]
    print(f"\nHigh importance documents ({len(high_importance)}):")
    for doc in high_importance[:5]:  # Show first 5
        print(f"- {doc.title} (Score: {doc.importance_score:.2f})")