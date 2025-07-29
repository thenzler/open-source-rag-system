#!/usr/bin/env python3
"""
Municipal RAG Demo Script
Demonstrates the municipal RAG system with Arlesheim
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.municipal.municipal_setup import MunicipalRagSetup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_municipal_rag():
    """Demonstrate the municipal RAG system"""
    print("=" * 60)
    print("Municipal RAG System Demo - Arlesheim")
    print("=" * 60)
    
    try:
        # Setup the system
        print("\n1. Setting up Municipal RAG for Arlesheim...")
        setup = MunicipalRagSetup('arlesheim')
        
        # Note: Set scrape_fresh=True for first run to scrape the website
        # For this demo, we'll use scrape_fresh=False to avoid hitting the website
        municipal_rag = setup.setup_complete_system(scrape_fresh=False, max_pages=20)
        
        # Get system statistics
        print("\n2. System Statistics:")
        stats = setup.get_stats()
        print(f"   Municipality: {stats['municipality']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Categories: {list(stats['categories'].keys())}")
        print(f"   Average importance: {stats['average_importance']:.2f}")
        print(f"   High importance chunks: {stats['high_importance_chunks']}")
        
        # Test queries
        print("\n3. Testing Municipal Queries:")
        test_queries = [
            {
                "query": "Was sind die Öffnungszeiten der Gemeindeverwaltung?",
                "category": "services"
            },
            {
                "query": "Wie kann ich einen Bauantrag stellen?",
                "category": "services"
            },
            {
                "query": "Wo finde ich Informationen über Steuern?",
                "category": "finance"
            },
            {
                "query": "Welche Veranstaltungen gibt es in Arlesheim?",
                "category": "events"
            },
            {
                "query": "Wie kontaktiere ich die Gemeindeverwaltung?",
                "category": "administration"
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {test['query']}")
            print(f"   Category: {test['category']}")
            
            start_time = time.time()
            result = setup.query_municipal_rag(test['query'], test['category'])
            processing_time = time.time() - start_time
            
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Sources: {len(result['sources'])}")
            
            if result['sources']:
                print(f"   Top source: {result['sources'][0]['title']}")
                print(f"   URL: {result['sources'][0]['url']}")
            
            # Show answer preview
            answer_preview = result['answer'][:200]
            if len(result['answer']) > 200:
                answer_preview += "..."
            print(f"   Answer preview: {answer_preview}")
            print("   " + "-" * 50)
        
        # Show category distribution
        print("\n4. Category Distribution:")
        categories = setup.get_categories()
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {category}: {count} documents")
        
        print("\n5. System Health Check:")
        print(f"   Embedding model: ✓ Loaded")
        print(f"   Municipal data: ✓ {stats['total_chunks']} chunks")
        print(f"   Ollama client: {'✓ Available' if setup.ollama_client.is_available() else '✗ Not available'}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
        return municipal_rag
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        logger.error(f"Demo failed: {e}")
        return None

def interactive_demo():
    """Interactive demo where user can ask questions"""
    print("\n" + "=" * 60)
    print("Interactive Municipal RAG Demo")
    print("=" * 60)
    
    try:
        # Setup system
        setup = MunicipalRagSetup('arlesheim')
        municipal_rag = setup.setup_complete_system(scrape_fresh=False, max_pages=10)
        
        categories = list(setup.get_categories().keys())
        print(f"\nAvailable categories: {', '.join(categories)}")
        print("\nYou can ask questions about Arlesheim municipality.")
        print("Type 'quit' to exit, 'stats' for statistics, 'categories' for available categories.")
        
        while True:
            print("\n" + "-" * 40)
            query = input("Your question: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                stats = setup.get_stats()
                print(f"Total chunks: {stats['total_chunks']}")
                print(f"Categories: {stats['categories']}")
                continue
            elif query.lower() == 'categories':
                cats = setup.get_categories()
                for cat, count in cats.items():
                    print(f"  {cat}: {count} documents")
                continue
            elif not query:
                continue
            
            # Ask for category (optional)
            category = input(f"Category (optional, available: {', '.join(categories)}): ").strip()
            if category and category not in categories:
                category = None
            
            # Process query
            try:
                start_time = time.time()
                result = setup.query_municipal_rag(query, category)
                processing_time = time.time() - start_time
                
                print(f"\nAnswer ({processing_time:.2f}s, confidence: {result['confidence']:.2f}):")
                print(result['answer'])
                
                if result['sources']:
                    print(f"\nSources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['title']}")
                        print(f"     {source['url']}")
                        print(f"     Category: {source['category']}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nThank you for using the Municipal RAG Demo!")
        
    except Exception as e:
        print(f"Error setting up interactive demo: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Municipal RAG Demo")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive demo")
    parser.add_argument("--scrape", action="store_true",
                       help="Scrape fresh data (use with caution)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    else:
        demo_municipal_rag()