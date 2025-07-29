#!/usr/bin/env python3
"""
Simple test of the RAG service fix - no Unicode printing
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

async def test_simple():
    try:
        from core.routers.query import get_rag_service
        
        rag_service = get_rag_service()
        result = await rag_service.answer_query("Was ist das Rathaus?")
        
        # Check if it worked without printing Unicode
        has_answer = 'answer' in result and len(result['answer']) > 0
        has_sources = 'sources' in result
        has_confidence = 'confidence' in result
        no_error = 'error' not in result
        
        print(f"Has answer: {has_answer}")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print(f"Has sources: {has_sources}")
        print(f"Sources count: {len(result.get('sources', []))}")
        print(f"Has confidence: {has_confidence}")
        print(f"No error: {no_error}")
        print(f"LLM timeout: {rag_service.llm_client.timeout}")
        
        success = has_answer and has_sources and has_confidence and no_error
        print(f"Overall success: {success}")
        
        return success
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple())
    print(f"RESULT: {'PASS' if success else 'FAIL'}")