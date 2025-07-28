#!/usr/bin/env python3
"""
Debug imports to identify dependency injection issues
"""
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_individual_components():
    """Test each component individually"""
    print("=== Testing Individual Components ===")
    
    # Test 1: OllamaClient standalone
    print("\n1. Testing OllamaClient standalone...")
    try:
        from core.ollama_client import OllamaClient
        client = OllamaClient()
        print(f"SUCCESS: OllamaClient created: {client.model}")
        
        # Test health check
        health = client.health_check()
        print(f"Health status: {health['available']}")
        if health['error']:
            print(f"Health error: {health['error']}")
        
        # Test basic generation if available
        if health['available']:
            print("Testing generation...")
            result = client.generate_answer("Test question", "Test context")
            print(f"Generation result: {result is not None}")
        
    except Exception as e:
        print(f"FAILED: OllamaClient failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Repository Factory
    print("\n2. Testing Repository Factory...")
    try:
        from core.repositories.factory import RepositoryFactory
        repo = RepositoryFactory.create_production_repository()
        print(f"SUCCESS: Repository created: {repo}")
        print(f"Vector repo: {repo.vector_search}")
        print(f"Audit repo: {repo.audit}")
    except Exception as e:
        print(f"FAILED: Repository Factory failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: SimpleRAGService creation
    print("\n3. Testing SimpleRAGService creation...")
    try:
        from core.repositories.factory import RepositoryFactory
        from core.services.simple_rag_service import SimpleRAGService
        from core.ollama_client import OllamaClient
        
        # Get repositories
        rag_repo = RepositoryFactory.create_production_repository()
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        
        # Get LLM client
        llm_client = OllamaClient()
        
        # Create simple RAG service
        rag_service = SimpleRAGService(vector_repo, llm_client, audit_repo)
        print(f"SUCCESS: SimpleRAGService created: {rag_service}")
        
        # Test status
        status = rag_service.get_status()
        print(f"RAG Service status: {status}")
        
    except Exception as e:
        print(f"FAILED: SimpleRAGService creation failed: {e}")
        import traceback
        traceback.print_exc()

def test_query_dependency():
    """Test the query router dependency function"""
    print("\n=== Testing Query Router Dependency ===")
    
    try:
        from core.routers.query import get_rag_service
        
        print("Testing get_rag_service()...")
        rag_service = get_rag_service()
        print(f"SUCCESS: get_rag_service() successful: {rag_service}")
        
        # Test a simple query
        print("Testing a simple query...")
        import asyncio
        
        async def test_query():
            try:
                result = await rag_service.answer_query("Test question")
                print(f"Query result keys: {result.keys()}")
                if 'error' in result:
                    print(f"Query error: {result['error']}")
                else:
                    print(f"Query answer: {result.get('answer', 'No answer')[:100]}...")
            except Exception as e:
                print(f"FAILED: Query test failed: {e}")
                import traceback
                traceback.print_exc()
        
        asyncio.run(test_query())
        
    except Exception as e:
        print(f"FAILED: get_rag_service() failed: {e}")
        import traceback
        traceback.print_exc()

def test_llm_client_methods():
    """Test LLM client methods specifically"""
    print("\n=== Testing LLM Client Methods ===")
    
    try:
        from core.ollama_client import OllamaClient
        client = OllamaClient()
        
        print(f"Model: {client.model}")
        print(f"Base URL: {client.base_url}")
        print(f"Available: {client.is_available()}")
        
        # Test the exact method used by SimpleRAGService
        print("\nTesting generate_answer method...")
        try:
            result = client.generate_answer(
                query="Was ist das?",
                context="[Quelle 1]: Test content"
            )
            print(f"generate_answer result: {result}")
            print(f"Result type: {type(result)}")
            print(f"Result length: {len(result) if result else 0}")
        except Exception as e:
            print(f"FAILED: generate_answer failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"FAILED: LLM Client test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Debug: Starting import and dependency tests...")
    
    test_individual_components()
    test_query_dependency()
    test_llm_client_methods()
    
    print("\n=== Debug Complete ===")