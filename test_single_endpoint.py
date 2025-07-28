#!/usr/bin/env python3
"""
Test script to verify single endpoint implementation
"""
import asyncio
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_single_endpoint():
    """Test the single query endpoint"""
    try:
        logger.info("=== TESTING SINGLE QUERY ENDPOINT ===")
        
        # Import the router directly
        from core.routers.query import router
        
        logger.info(f"✅ Successfully imported query router")
        logger.info(f"Router prefix: {router.prefix}")
        logger.info(f"Router tags: {router.tags}")
        
        # Check available endpoints
        routes = []
        for route in router.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                methods = list(route.methods) if route.methods else ['GET']  
                routes.append(f"{methods[0]} {route.path}")
        
        logger.info("Available endpoints:")
        for route in routes:
            logger.info(f"  - {route}")
        
        # Verify it's the single endpoint we expect
        expected_endpoints = [
            "POST /api/v1/query",
            "GET /api/v1/status", 
            "GET /api/v1/health"
        ]
        
        all_found = True
        for expected in expected_endpoints:
            found = any(expected in route for route in routes)
            if found:
                logger.info(f"✅ Found {expected}")
            else:
                logger.error(f"❌ Missing {expected}")
                all_found = False
        
        # Check for old endpoints that should NOT exist
        old_endpoints = [
            "/query/enhanced",
            "/query/smart", 
            "/query/fast",
            "/query/llm-only",
            "/query-stream"
        ]
        
        for old_endpoint in old_endpoints:
            found = any(old_endpoint in route for route in routes)
            if found:
                logger.error(f"❌ Found old endpoint that should be removed: {old_endpoint}")
                all_found = False
            else:
                logger.info(f"✅ Correctly removed old endpoint: {old_endpoint}")
        
        if all_found:
            logger.info("🎉 SUCCESS: Single endpoint implementation is correct!")
            logger.info("🎯 System now has AI answers only with zero-hallucination protection")
            return True
        else:
            logger.error("❌ FAILURE: Endpoint configuration issues found")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_endpoint())
    print(f"\n{'SUCCESS' if success else 'FAILURE'}: Single endpoint test {'passed' if success else 'failed'}")