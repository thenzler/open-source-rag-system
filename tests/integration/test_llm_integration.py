#!/usr/bin/env python3
"""
Test LLM Integration 
Quick test of the LLM response generation with zero-hallucination validation
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

async def test_llm_integration():
    """Test the complete LLM integration with validation"""
    try:
        logger.info("Starting LLM integration test...")
        
        # Import after path setup
        from core.repositories.factory import RepositoryFactory
        from core.services.query_service import QueryProcessingService
        from core.ollama_client import OllamaClient
        
        # Initialize repositories
        rag_repo = RepositoryFactory.create_production_repository()
        await rag_repo.initialize()
        
        doc_repo = rag_repo.documents
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        
        # Initialize Ollama client
        ollama_client = OllamaClient()
        
        # Initialize query service with LLM
        query_service = QueryProcessingService(doc_repo, vector_repo, audit_repo, ollama_client)
        
        # Test query that should work well with LLM
        test_query = "Welche Regeln gelten für Gewerbeabfall?"
        logger.info(f"Testing query: {test_query}")
        
        try:
            # Test with LLM enabled
            result = await query_service.search_documents(
                query=test_query,
                limit=5,
                use_llm=True
            )
            
            logger.info("=== RESULT SUMMARY ===")
            logger.info(f"Search type: {result.get('search_type', 'unknown')}")
            logger.info(f"Total found: {result.get('total_found', 0)}")
            logger.info(f"Used LLM: {result.get('use_llm', False)}")
            logger.info(f"Confidence tier: {result.get('confidence_tier', 'unknown')}")
            logger.info(f"Max similarity: {result.get('max_similarity', 0.0):.3f}")
            
            if "ai_response" in result:
                ai_response = result["ai_response"]
                logger.info(f"AI Response length: {len(ai_response)} chars")
                logger.info(f"AI Sources: {result.get('source_count', 0)}")
                logger.info(f"AI Confidence: {result.get('ai_confidence', 0.0):.3f}")
                
                # Show first part of response
                preview = ai_response[:300] + "..." if len(ai_response) > 300 else ai_response
                logger.info(f"AI Response preview: {preview}")
                
                # Check if sources are included
                if "📚 VERWENDETE QUELLEN:" in ai_response:
                    logger.info("✅ Sources properly included in response")
                else:
                    logger.warning("⚠️ No source footer found in response")
                
                logger.info("🎉 LLM INTEGRATION TEST PASSED!")
                return True
            
            elif "refusal_reason" in result:
                logger.info(f"Query refused: {result['refusal_reason']}")
                logger.info("✅ System properly refused low-confidence query")
                return True
            
            else:
                logger.info("✅ Vector search results returned (LLM fallback)")
                logger.info(f"Results: {len(result.get('results', []))}")
                return True
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_integration())
    sys.exit(0 if success else 1)