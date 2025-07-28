#!/usr/bin/env python3
"""
Test Zero-Hallucination System
Test the implementation with real data from processed documents
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

async def test_zero_hallucination():
    """Test the zero-hallucination system with various queries"""
    try:
        logger.info("Starting zero-hallucination system test...")
        
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
        
        # Initialize query service
        query_service = QueryProcessingService(doc_repo, vector_repo, audit_repo, ollama_client)
        
        # Test queries
        test_queries = [
            # Should find answers in documents
            {
                "query": "Welche Regeln gelten für Gewerbeabfall?",
                "expected": "should_find_answer",
                "description": "Commercial waste rules (should be in Brosch_Gewerbekehricht.pdf)"
            },
            {
                "query": "Wie funktioniert die Abfallentsorgung in Arlesheim?",
                "expected": "should_find_answer", 
                "description": "Waste disposal in Arlesheim (should be in multiple PDFs)"
            },
            {
                "query": "Was gehört in den Bioabfall?",
                "expected": "should_find_answer",
                "description": "Bio waste sorting (should be in Fly_240523_Was-gehoert-in-den-Bioabfall_bh.pdf)"
            },
            
            # Should be blocked for external knowledge
            {
                "query": "Wie ist das Wetter heute?",
                "expected": "external_knowledge_blocked",
                "description": "Weather query (requires external knowledge)"
            },
            {
                "query": "Wer ist der Präsident der Schweiz?",
                "expected": "external_knowledge_blocked", 
                "description": "Current politics (requires external knowledge)"
            },
            {
                "query": "Was kostet Bitcoin heute?",
                "expected": "external_knowledge_blocked",
                "description": "Financial data (requires external knowledge)"
            },
            
            # Should be refused for low confidence
            {
                "query": "Welche Regeln gelten für Raumfahrt in Arlesheim?",
                "expected": "low_confidence_refused",
                "description": "Irrelevant topic (should have low similarity)"
            },
            {
                "query": "Wie baue ich einen Kernreaktor?",
                "expected": "low_confidence_refused", 
                "description": "Completely unrelated topic"
            }
        ]
        
        logger.info(f"Running {len(test_queries)} test queries...")
        
        passed = 0
        failed = 0
        
        for i, test in enumerate(test_queries, 1):
            logger.info(f"\n--- Test {i}: {test['description']} ---")
            logger.info(f"Query: {test['query']}")
            logger.info(f"Expected: {test['expected']}")
            
            test_error = None
            result = None
            
            try:
                # Test query
                result = await query_service.search_documents(
                    query=test["query"],
                    limit=5,
                    use_llm=True
                )
                
            except Exception as e:
                test_error = str(e)
                logger.info(f"Query exception (may be expected): {e}")
                result = {}  # Empty result for analysis
            
            # Analyze result (pass error info for external knowledge tests)
            success = analyze_test_result(result, test["expected"], test_error)
            
            if success:
                logger.info(f"✅ PASS: {test['description']}")
                passed += 1
            else:
                logger.error(f"❌ FAIL: {test['description']}")
                failed += 1
                
            # Show result details
            if result:
                log_result_details(result)
            elif test_error:
                logger.info(f"Error details: {test_error}")
        
        # Summary
        total = len(test_queries)
        logger.info(f"\n=== TEST SUMMARY ===")
        logger.info(f"Total: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Zero-hallucination system is working correctly.")
        else:
            logger.warning(f"⚠️ {failed} tests failed. Review implementation.")
        
        return passed == total
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False

def analyze_test_result(result: dict, expected: str, test_error: str = None) -> bool:
    """Analyze if the test result matches expectations"""
    try:
        # Check for external knowledge blocking (can be caught at validation or error level)
        if expected == "external_knowledge_blocked":
            # Can be blocked at validation (refusal_reason) or as an exception
            if "refusal_reason" in result and "externes Wissen" in result["refusal_reason"]:
                return True
            if test_error and "externes Wissen" in test_error:
                return True
            return False
        
        # Check for low confidence refusal
        elif expected == "low_confidence_refused":
            if "refusal_reason" in result:
                return "Zuverlässigkeit" in result["refusal_reason"] or "keine" in result["refusal_reason"].lower()
            return False
        
        # Check for successful answer (more flexible with new multi-tier system)
        elif expected == "should_find_answer":
            # Should have results at medium+ confidence
            has_results = result.get("total_found", 0) > 0 or len(result.get("results", [])) > 0
            confidence_tier = result.get("confidence_tier", "")
            has_decent_confidence = confidence_tier in ["high", "medium"] or any(
                item.get("similarity", 0) >= 0.25  # Lower threshold for municipal docs
                for item in result.get("results", [])
            )
            return has_results and has_decent_confidence
        
        return False
        
    except Exception as e:
        logger.error(f"Error analyzing result: {e}")
        return False

def log_result_details(result: dict):
    """Log detailed result information"""
    try:
        # Basic info
        logger.info(f"Search type: {result.get('search_type', 'unknown')}")
        logger.info(f"Total found: {result.get('total_found', 0)}")
        logger.info(f"Use LLM: {result.get('use_llm', False)}")
        
        # Refusal info
        if "refusal_reason" in result:
            logger.info(f"Refusal reason: {result['refusal_reason']}")
            if "max_similarity" in result:
                logger.info(f"Max similarity: {result['max_similarity']:.3f}")
                logger.info(f"Threshold: {result.get('confidence_threshold', 0.8):.3f}")
        
        # Results info
        results = result.get("results", [])
        if results:
            logger.info(f"Top results:")
            for i, item in enumerate(results[:3]):
                similarity = item.get("similarity", 0)
                doc_id = item.get("document_id", "unknown")
                content_preview = item.get("content", "")[:100] + "..." if len(item.get("content", "")) > 100 else item.get("content", "")
                logger.info(f"  {i+1}. Doc {doc_id}, Sim: {similarity:.3f}, Content: {content_preview}")
        
        # AI response info
        if "ai_response" in result:
            ai_response = result["ai_response"][:200] + "..." if len(result.get("ai_response", "")) > 200 else result.get("ai_response", "")
            logger.info(f"AI Response: {ai_response}")
            logger.info(f"AI Sources: {result.get('source_count', 0)}")
            logger.info(f"AI Confidence: {result.get('ai_confidence', 0):.3f}")
        
    except Exception as e:
        logger.warning(f"Error logging result details: {e}")

if __name__ == "__main__":
    success = asyncio.run(test_zero_hallucination())
    sys.exit(0 if success else 1)