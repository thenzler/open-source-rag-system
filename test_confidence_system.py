#!/usr/bin/env python3
"""
Test Confidence System Only
Fast test of the intelligent multi-tier confidence system
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

async def test_confidence_system():
    """Test the confidence system with different query types"""
    try:
        logger.info("Starting confidence system test...")
        
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
        
        # Initialize query service (without Ollama to speed up)
        query_service = QueryProcessingService(doc_repo, vector_repo, audit_repo, None)
        
        # Test queries with different confidence expectations
        test_queries = [
            {
                "query": "Welche Regeln gelten für Gewerbeabfall?",
                "expected_tier": "medium",  # Municipal topic, should find results
                "description": "Municipal waste query (should get medium+ confidence)"
            },
            {
                "query": "Wie funktioniert die Abfallentsorgung in Arlesheim?", 
                "expected_tier": "medium",  # Municipal + Arlesheim
                "description": "Arlesheim waste query (should get medium+ confidence)"
            },
            {
                "query": "Welche Regeln gelten für Raumfahrt in Arlesheim?",
                "expected_tier": "refused",  # Irrelevant topic
                "description": "Space query (should be refused)"
            },
            {
                "query": "Wie baue ich einen Kernreaktor?",
                "expected_tier": "refused",  # Irrelevant topic
                "description": "Nuclear reactor query (should be refused)"
            }
        ]
        
        logger.info(f"Running {len(test_queries)} confidence tests...")
        
        passed = 0
        failed = 0
        
        for i, test in enumerate(test_queries, 1):
            logger.info(f"\n--- Test {i}: {test['description']} ---")
            logger.info(f"Query: {test['query']}")
            logger.info(f"Expected: {test['expected_tier']}")
            
            try:
                # Test without LLM to focus on confidence system
                result = await query_service.search_documents(
                    query=test["query"],
                    limit=5,
                    use_llm=False  # No LLM to speed up test
                )
                
                # Analyze confidence tier
                confidence_tier = result.get("confidence_tier", "unknown")
                refusal_reason = result.get("refusal_reason", None)
                total_found = result.get("total_found", 0)
                max_similarity = result.get("max_similarity", 0.0)
                
                logger.info(f"Confidence tier: {confidence_tier}")
                logger.info(f"Total found: {total_found}")
                logger.info(f"Max similarity: {max_similarity:.3f}")
                
                if refusal_reason:
                    logger.info(f"Refusal reason: {refusal_reason}")
                
                # Check if result matches expectation
                success = False
                if test["expected_tier"] == "refused":
                    success = refusal_reason is not None or total_found == 0
                elif test["expected_tier"] == "medium":
                    success = confidence_tier in ["medium", "high"] and total_found > 0
                
                if success:
                    logger.info(f"✅ PASS: {test['description']}")
                    passed += 1
                else:
                    logger.error(f"❌ FAIL: {test['description']}")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Test exception: {e}")
                # For expected external knowledge blocks, exceptions are OK
                if test["expected_tier"] == "refused" and "externes Wissen" in str(e):
                    logger.info(f"✅ PASS: {test['description']} (blocked as expected)")
                    passed += 1
                else:
                    logger.error(f"❌ FAIL: {test['description']}")
                    failed += 1
        
        # Summary
        total = len(test_queries)
        logger.info(f"\n=== CONFIDENCE SYSTEM TEST SUMMARY ===")
        logger.info(f"Total: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL CONFIDENCE TESTS PASSED! Intelligent multi-tier system is working correctly.")
        else:
            logger.warning(f"⚠️ {failed} tests failed. Review confidence thresholds.")
        
        return passed == total
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_confidence_system())
    sys.exit(0 if success else 1)