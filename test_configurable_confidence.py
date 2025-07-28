#!/usr/bin/env python3
"""
Test Configurable Confidence System
Demonstrates the flexible, configurable confidence system
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

async def test_configurable_system():
    """Test the configurable confidence system with different settings"""
    try:
        logger.info("=== TESTING CONFIGURABLE CONFIDENCE SYSTEM ===")
        
        # Import after path setup
        from core.services.confidence_manager import ConfidenceManager
        from core.repositories.factory import RepositoryFactory
        from core.services.query_service import QueryProcessingService
        
        # Test 1: Default Configuration
        logger.info("\n--- TEST 1: Default Configuration ---")
        confidence_manager = ConfidenceManager()
        
        test_queries = [
            "Welche Regeln gelten für Gewerbeabfall?",  # Municipal (should get lower thresholds)
            "Welche Regeln gelten für Raumfahrt in Arlesheim?",  # Irrelevant (should get higher thresholds)
            "Wie ist das Wetter heute?",  # External knowledge (should be blocked)
            "Was ist ein Vertrag?"  # General query
        ]
        
        for query in test_queries:
            logger.info(f"Query: {query}")
            
            # Test confidence tiers
            tiers = confidence_manager.determine_confidence_tiers(query)
            logger.info(f"  Tiers: {tiers}")
            
            # Test external knowledge detection
            is_external, reason = confidence_manager.detect_external_knowledge(query)
            if is_external:
                logger.info(f"  🚫 Blocked: {reason}")
            else:
                logger.info(f"  ✅ Allowed")
            
            logger.info("")
        
        # Test 2: Strict Profile
        logger.info("--- TEST 2: Strict Profile ---")
        confidence_manager.set_profile("strict")
        logger.info("Applied strict profile")
        
        query = "Welche Regeln gelten für Gewerbeabfall?"
        tiers = confidence_manager.determine_confidence_tiers(query)
        logger.info(f"Strict mode tiers: {tiers}")
        
        # Test 3: Permissive Profile  
        logger.info("\n--- TEST 3: Permissive Profile ---")
        confidence_manager.set_profile("permissive")
        logger.info("Applied permissive profile")
        
        tiers = confidence_manager.determine_confidence_tiers(query)
        logger.info(f"Permissive mode tiers: {tiers}")
        
        # Check if external knowledge is disabled
        is_external, reason = confidence_manager.detect_external_knowledge("Wie ist das Wetter heute?")
        logger.info(f"Weather query in permissive mode - Blocked: {is_external}")
        
        # Test 4: Simple Mode
        logger.info("\n--- TEST 4: Simple Mode ---")  
        confidence_manager.set_profile("simple")
        logger.info("Applied simple profile")
        
        logger.info(f"Simple mode enabled: {not confidence_manager.is_enabled()}")
        logger.info(f"Simple threshold: {confidence_manager.get_simple_threshold()}")
        
        tiers = confidence_manager.determine_confidence_tiers(query)
        logger.info(f"Simple mode tiers: {tiers}")
        
        # Test 5: Custom Domain
        logger.info("\n--- TEST 5: Custom Domain ---")
        confidence_manager = ConfidenceManager()  # Reset to default
        
        # Add a custom domain
        success = confidence_manager.add_custom_domain(
            domain_name="legal",
            terms=["vertrag", "klage", "recht", "gesetz"],
            adjustment=-0.06,
            priority=2
        )
        logger.info(f"Added legal domain: {success}")
        
        # Enable legal domain
        confidence_manager.enable_domain("legal")
        
        # Test legal query
        legal_query = "Was ist ein Vertrag?"
        tiers = confidence_manager.determine_confidence_tiers(legal_query)
        logger.info(f"Legal query '{legal_query}' tiers: {tiers}")
        
        # Test 6: Status Summary
        logger.info("\n--- TEST 6: System Status ---")
        status = confidence_manager.get_status_summary()
        
        logger.info(f"Intelligent system: {status['intelligent_system_enabled']}")
        logger.info(f"Active profile: {status['active_profile']}")
        logger.info(f"Enabled domains: {status['enabled_domains']}")
        logger.info(f"External knowledge enabled: {status['external_knowledge_enabled']}")
        
        # Test 7: Integration with Query Service
        logger.info("\n--- TEST 7: Query Service Integration ---")
        try:
            # Initialize repositories
            rag_repo = RepositoryFactory.create_production_repository()
            await rag_repo.initialize()
            
            # Create query service with custom confidence config
            query_service = QueryProcessingService(
                doc_repo=rag_repo.documents,
                vector_repo=rag_repo.vector_search,
                audit_repo=rag_repo.audit,
                confidence_config_path="config/confidence_config.yaml"
            )
            
            # Test query processing with custom confidence system
            test_query = "Welche Regeln gelten für Gewerbeabfall?"
            logger.info(f"Testing integrated query: {test_query}")
            
            # This would use the configurable confidence system
            result = await query_service.search_documents(
                query=test_query,
                limit=5,
                use_llm=False  # Skip LLM for faster test
            )
            
            logger.info(f"Result: {result.get('search_type', 'unknown')} - {result.get('total_found', 0)} results")
            logger.info(f"Confidence tier: {result.get('confidence_tier', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
        
        logger.info("\n🎉 CONFIGURABLE CONFIDENCE SYSTEM TEST COMPLETED!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_api_usage():
    """Demonstrate how to use the confidence system via API"""
    logger.info("\n=== API USAGE EXAMPLES ===")
    
    examples = [
        "GET /api/v1/confidence/status - Get current configuration",
        "POST /api/v1/confidence/profile {'profile_name': 'strict'} - Set strict mode",
        "POST /api/v1/confidence/domain/enable {'domain_name': 'legal'} - Enable legal domain",
        "POST /api/v1/confidence/domain/add {'domain_name': 'medical', 'terms': ['patient', 'diagnose'], 'adjustment': -0.05} - Add custom domain",
        "GET /api/v1/confidence/test/Welche Regeln gelten für Gewerbeabfall? - Test query analysis",
        "GET /api/v1/confidence/profiles - List available profiles",
        "GET /api/v1/confidence/domains - List configured domains"
    ]
    
    for example in examples:
        logger.info(f"  {example}")

if __name__ == "__main__":
    success = asyncio.run(test_configurable_system())
    asyncio.run(demo_api_usage())
    sys.exit(0 if success else 1)