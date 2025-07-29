#!/usr/bin/env python3
"""
Test the cleaned RAG system with bio waste questions
"""
import asyncio
import sys
sys.path.append('.')

async def test_cleaned_rag():
    """Test the RAG system after cleanup"""
    try:
        from core.services.simple_rag_service import SimpleRAGService
        from core.repositories.factory import get_vector_search_repository
        from core.ollama_client import OllamaClient
        
        print("Testing cleaned RAG system...")
        
        # Initialize components
        vector_repo = get_vector_search_repository()
        llm_client = OllamaClient()
        rag_service = SimpleRAGService(vector_repo, llm_client)
        
        # Test queries
        test_queries = [
            "Was gehört in den Bioabfall?",
            "Wie entsorge ich organische Abfälle?",
            "What goes in the bio waste container?",
            "Welche Früchte kann ich kompostieren?"
        ]
        
        for query in test_queries:
            print(f"\n=== Query: {query} ===")
            
            try:
                response = await rag_service.answer_query(query)
                
                if "error" in response:
                    print(f"ERROR: {response['error']}")
                else:
                    print(f"Answer: {response['answer'][:200]}...")
                    print(f"Sources: {len(response.get('sources', []))}")
                    print(f"Confidence: {response.get('confidence', 0):.2f}")
                    
            except Exception as e:
                print(f"Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_cleaned_rag())
    if success:
        print("\nRAG system test completed!")
        sys.exit(0)
    else:
        print("\nRAG system test failed!")
        sys.exit(1)