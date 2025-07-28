#!/usr/bin/env python3
"""
Test Simple Professional RAG System
Clean, straightforward testing
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_simple_rag():
    """Test the simple RAG system"""
    try:
        logger.info("=== TESTING SIMPLE PROFESSIONAL RAG SYSTEM ===")
        
        # Set configuration via environment variables
        os.environ['RAG_SIMILARITY_THRESHOLD'] = '0.3'
        os.environ['RAG_MAX_RESULTS'] = '5'
        os.environ['RAG_REQUIRE_SOURCES'] = 'true'
        
        # Import after setting env vars
        from core.repositories.factory import RepositoryFactory
        from core.services.simple_rag_service import SimpleRAGService
        from core.ollama_client import OllamaClient
        
        # Initialize components
        logger.info("Initializing RAG system...")
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        llm_client = OllamaClient()
        
        # Create simple RAG service
        rag_service = SimpleRAGService(vector_repo, llm_client, audit_repo)
        
        # Test queries
        test_queries = [
            "Welche Regeln gelten für Gewerbeabfall?",
            "Wie funktioniert die Abfallentsorgung?",
            "Was kostet die Entsorgung?",
            "Nonsense query about space rockets"  # Should return no results
        ]
        
        logger.info(f"\nTesting {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Test {i}: {query} ---")
            
            try:
                # Process query
                response = await rag_service.answer_query(query)
                
                # Log results
                if "error" in response:
                    logger.info(f"❌ Error: {response['error']}")
                elif "answer" in response:
                    logger.info(f"✅ Answer: {response['answer'][:100]}...")
                    logger.info(f"   Sources: {len(response.get('sources', []))}")
                    logger.info(f"   Confidence: {response.get('confidence', 0):.3f}")
                else:
                    logger.info(f"⚠️ Unexpected response format")
                    
            except Exception as e:
                logger.error(f"❌ Query failed: {e}")
        
        # Test configuration
        logger.info("\n--- Configuration Test ---")
        status = rag_service.get_status()
        logger.info(f"Service: {status['service']}")
        logger.info(f"Mode: {status['mode']}")
        logger.info(f"Config: {status['config']}")
        
        # Test different thresholds
        logger.info("\n--- Threshold Test ---")
        
        # High threshold
        os.environ['RAG_SIMILARITY_THRESHOLD'] = '0.8'
        rag_service_strict = SimpleRAGService(vector_repo, llm_client, audit_repo)
        
        response_strict = await rag_service_strict.answer_query("Welche Regeln gelten für Gewerbeabfall?")
        logger.info(f"High threshold (0.8): {len(response_strict.get('sources', []))} sources")
        
        # Low threshold  
        os.environ['RAG_SIMILARITY_THRESHOLD'] = '0.1'
        rag_service_permissive = SimpleRAGService(vector_repo, llm_client, audit_repo)
        
        response_permissive = await rag_service_permissive.answer_query("Welche Regeln gelten für Gewerbeabfall?")
        logger.info(f"Low threshold (0.1): {len(response_permissive.get('sources', []))} sources")
        
        logger.info("\n🎉 SIMPLE RAG SYSTEM TEST COMPLETED!")
        logger.info("\n=== DEPLOYMENT GUIDE ===")
        logger.info("Environment Variables:")
        logger.info("  RAG_SIMILARITY_THRESHOLD=0.3  # Adjust for domain")
        logger.info("  RAG_MAX_RESULTS=5             # Performance tuning")
        logger.info("  RAG_REQUIRE_SOURCES=true      # Zero hallucination")
        logger.info("  RAG_MAX_QUERY_LENGTH=500      # Input validation")
        logger.info("\nAPI Endpoints:")
        logger.info("  POST /api/v1/rag/query        # Ask questions")
        logger.info("  GET  /api/v1/rag/status       # Check config")
        logger.info("  GET  /api/v1/rag/health       # Health check")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_rag())
    sys.exit(0 if success else 1)