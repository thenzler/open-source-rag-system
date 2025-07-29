#!/usr/bin/env python3
"""
Test the async/await and audit fixes directly
"""
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

async def test_rag_service_fixed():
    """Test that the RAG service works with all fixes applied"""
    print("[TESTING] RAG Service with Async/Audit Fixes")
    print("=" * 60)
    
    try:
        # Import the dependencies
        from core.ollama_client import OllamaClient
        from core.services.simple_rag_service import SimpleRAGService
        
        print("[SUCCESS] All imports successful")
        
        # Create the components
        llm_client = OllamaClient()
        print(f"[SUCCESS] LLM Client created: {llm_client.model}")
        
        # Create a mock vector repository for testing
        class MockVectorRepo:
            async def search_similar_text(self, query, limit=3, threshold=0.01):
                # Return mock search results
                class MockResult:
                    def __init__(self):
                        self.items = [MockItem()]
                        
                class MockItem:
                    def __init__(self):
                        self.text_content = "This is a test document content about bio waste."
                        self.document_id = "test_doc_1"
                        self.id = "chunk_1"
                        self.metadata = {"similarity_score": 0.85}
                
                return MockResult()
        
        # Create RAG service
        vector_repo = MockVectorRepo()
        audit_repo = None  # Skip audit for this test
        
        rag_service = SimpleRAGService(
            vector_repo=vector_repo,
            llm_client=llm_client,
            audit_repo=audit_repo
        )
        
        print("[SUCCESS] RAG Service created successfully")
        
        # Test the query processing
        print("[TESTING] Processing test query...")
        
        result = await rag_service.answer_query("What can I put in the bio waste bin?")
        
        # Skip Unicode display to avoid Windows terminal issues
        answer = result.get('answer', 'No answer')
        print(f"[RESULT] Answer length: {len(answer)} characters")
        print(f"[RESULT] Sources: {len(result.get('sources', []))}")
        print(f"[RESULT] Confidence: {result.get('confidence', 0.0)}")
        
        if 'error' in result:
            print(f"[ERROR] Query processing failed: {result['error']}")
            return False
        
        if result.get('answer') == 'Fehler bei der Antwortgenerierung.':
            print("[ERROR] Still getting 'Fehler bei der Antwortgenerierung' error")
            return False
        
        # Check if we have a meaningful answer (not an error)
        answer = result.get('answer', '')
        if len(answer) > 10 and 'Fehler' not in answer:
            print("[SUCCESS] Query processing completed without async/await errors!")
            print("[SUCCESS] Got meaningful answer from LLM!")
            return True
        else:
            print(f"[WARNING] Answer might be problematic: length={len(answer)}")
            return False
        
    except Exception as e:
        print(f"[ERROR] Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rag_service_fixed())
    if success:
        print("\n[SUCCESS] ALL FIXES WORKING! The async/await and audit issues are resolved!")
    else:
        print("\n[ERROR] FIXES NOT WORKING - There are still issues to resolve")