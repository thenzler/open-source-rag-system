#!/usr/bin/env python3
"""
Municipal RAG Setup Script
Complete setup for municipality-specific RAG system
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.municipal_web_scraper import MunicipalWebScraper
from tools.municipal.municipal_rag import MunicipalRAG
from sentence_transformers import SentenceTransformer
from ollama_client import get_ollama_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Swiss municipality configurations
MUNICIPAL_CONFIGS = {
    'arlesheim': {
        'name': 'Arlesheim',
        'url': 'https://www.arlesheim.ch',
        'canton': 'BL',
        'language': 'de'
    },
    'basel': {
        'name': 'Basel',
        'url': 'https://www.basel.ch',
        'canton': 'BS', 
        'language': 'de'
    },
    'bern': {
        'name': 'Bern',
        'url': 'https://www.bern.ch',
        'canton': 'BE',
        'language': 'de'
    },
    'zurich': {
        'name': 'Zürich',
        'url': 'https://www.stadt-zuerich.ch',
        'canton': 'ZH',
        'language': 'de'
    },
    'geneva': {
        'name': 'Geneva',
        'url': 'https://www.geneve.ch',
        'canton': 'GE',
        'language': 'fr'
    },
    'lausanne': {
        'name': 'Lausanne',
        'url': 'https://www.lausanne.ch',
        'canton': 'VD',
        'language': 'fr'
    }
}

class MunicipalRagSetup:
    """Complete setup for municipal RAG system"""
    
    def __init__(self, municipality_key: str):
        if municipality_key not in MUNICIPAL_CONFIGS:
            raise ValueError(f"Unknown municipality: {municipality_key}")
        
        self.municipality_key = municipality_key
        self.config = MUNICIPAL_CONFIGS[municipality_key]
        self.municipality_name = self.config['name']
        self.base_url = self.config['url']
        
        # Initialize components
        self.embedding_model = None
        self.ollama_client = None
        self.municipal_rag = None
        
        logger.info(f"Initializing RAG setup for {self.municipality_name}")
    
    def setup_complete_system(self, scrape_fresh: bool = False, max_pages: int = 100):
        """Complete setup: scrape, process, and initialize RAG"""
        logger.info(f"Setting up complete RAG system for {self.municipality_name}")
        
        # Step 1: Scrape municipal website if needed
        if scrape_fresh:
            logger.info("Step 1: Scraping municipal website...")
            self.scrape_municipal_website(max_pages)
        else:
            logger.info("Step 1: Using existing scraped data...")
        
        # Step 2: Initialize embedding model
        logger.info("Step 2: Loading embedding model...")
        self.initialize_embedding_model()
        
        # Step 3: Initialize Ollama client
        logger.info("Step 3: Initializing Ollama client...")
        self.initialize_ollama_client()
        
        # Step 4: Setup municipal RAG
        logger.info("Step 4: Setting up municipal RAG system...")
        self.setup_municipal_rag()
        
        # Step 5: Test the system
        logger.info("Step 5: Testing the system...")
        self.test_system()
        
        logger.info(f"Complete RAG system setup finished for {self.municipality_name}")
        return self.municipal_rag
    
    def scrape_municipal_website(self, max_pages: int = 100):
        """Scrape the municipal website"""
        scraper = MunicipalWebScraper(self.municipality_name, self.base_url)
        documents = scraper.scrape_municipality(max_pages)
        scraper.save_documents(documents)
        
        logger.info(f"Scraped {len(documents)} documents from {self.municipality_name}")
        return documents
    
    def initialize_embedding_model(self):
        """Initialize sentence transformer model"""
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    
    def initialize_ollama_client(self):
        """Initialize Ollama client"""
        logger.info("Initializing Ollama client...")
        self.ollama_client = get_ollama_client()
        
        if not self.ollama_client.is_available():
            logger.warning("Ollama is not available. Some features may be limited.")
        else:
            logger.info("Ollama client initialized successfully")
    
    def setup_municipal_rag(self):
        """Setup municipal RAG system"""
        self.municipal_rag = MunicipalRAG(
            self.municipality_name,
            self.embedding_model,
            self.ollama_client
        )
        
        # Load municipal data
        self.municipal_rag.load_municipal_data("municipal_data")
        
        # Get system statistics
        stats = self.municipal_rag.get_municipal_stats()
        logger.info(f"Municipal RAG stats: {stats}")
    
    def test_system(self):
        """Test the municipal RAG system"""
        test_queries = [
            "Was sind die Öffnungszeiten der Gemeindeverwaltung?",
            "Wie kann ich einen Bauantrag stellen?",
            "Wo finde ich Informationen über Steuern?",
            "Welche Dienstleistungen bietet die Gemeinde?",
            "Wie kontaktiere ich die Gemeindeverwaltung?"
        ]
        
        logger.info("Testing municipal RAG system...")
        
        for query in test_queries:
            try:
                result = self.municipal_rag.generate_municipal_answer(query)
                logger.info(f"Query: {query}")
                logger.info(f"Confidence: {result['confidence']:.2f}")
                logger.info(f"Sources: {len(result['sources'])}")
                logger.info("---")
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
    
    def query_municipal_rag(self, query: str, category: str = None) -> dict:
        """Query the municipal RAG system"""
        if not self.municipal_rag:
            raise ValueError("Municipal RAG system not initialized")
        
        return self.municipal_rag.generate_municipal_answer(query, category)
    
    def get_categories(self) -> dict:
        """Get available categories"""
        if not self.municipal_rag:
            raise ValueError("Municipal RAG system not initialized")
        
        return self.municipal_rag.get_municipal_categories()
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        if not self.municipal_rag:
            raise ValueError("Municipal RAG system not initialized")
        
        return self.municipal_rag.get_municipal_stats()

# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Municipal RAG System Setup")
    parser.add_argument("municipality", choices=list(MUNICIPAL_CONFIGS.keys()), 
                       help="Municipality to setup")
    parser.add_argument("--scrape", action="store_true", 
                       help="Scrape fresh data from website")
    parser.add_argument("--max-pages", type=int, default=100,
                       help="Maximum pages to scrape")
    parser.add_argument("--query", type=str, 
                       help="Test query to run")
    parser.add_argument("--stats", action="store_true",
                       help="Show system statistics")
    
    args = parser.parse_args()
    
    # Setup the system
    setup = MunicipalRagSetup(args.municipality)
    setup.setup_complete_system(scrape_fresh=args.scrape, max_pages=args.max_pages)
    
    # Handle additional commands
    if args.query:
        result = setup.query_municipal_rag(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    if args.stats:
        stats = setup.get_stats()
        print(f"\nStatistics for {stats['municipality']}:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Categories: {stats['categories']}")
        print(f"Average importance: {stats['average_importance']:.2f}")
        print(f"High importance chunks: {stats['high_importance_chunks']}")

if __name__ == "__main__":
    main()