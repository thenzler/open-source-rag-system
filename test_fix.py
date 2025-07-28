#!/usr/bin/env python3
"""
Test the SimpleRAGService fix
"""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.abspath('.'))

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def test_rag_service():
    """Test the fixed RAG service"""
    try:
        print("Testing fixed SimpleRAGService...")
        
        from core.routers.query import get_rag_service
        
        # Get the service (should use new timeout and configs)
        rag_service = get_rag_service()
        print(f"RAG Service created: {rag_service}")
        print(f"LLM Client timeout: {rag_service.llm_client.timeout}")
        
        # Test with a simple query
        print("Testing query processing...")
        result = await rag_service.answer_query("Was ist das Rathaus?")
        
        print(f"Result keys: {list(result.keys())}")
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return False
        
        if 'answer' in result:
            answer = result['answer']
            print(f"SUCCESS: Got answer ({len(answer)} chars)")
            print(f"Answer preview: {answer[:200]}...")
            return True
        else:
            print("ERROR: No answer in result")
            return False
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rag_service())
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")