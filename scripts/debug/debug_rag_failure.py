#!/usr/bin/env python3
"""
Debug the exact failure point in SimpleRAGService._generate_answer
"""
import logging
import sys
import os
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_generate_answer():
    """Test the _generate_answer method specifically"""
    print("=== Testing SimpleRAGService._generate_answer ===")
    
    try:
        from core.repositories.factory import RepositoryFactory
        from core.services.simple_rag_service import SimpleRAGService
        from core.ollama_client import OllamaClient
        
        # Create components
        rag_repo = RepositoryFactory.create_production_repository()
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        llm_client = OllamaClient()
        
        # Create RAG service
        rag_service = SimpleRAGService(vector_repo, llm_client, audit_repo)
        
        print("Components created successfully")
        
        # Test with mock search results
        mock_search_results = [
            {
                'text': 'This is test content from document 1',
                'document_id': 'test_doc_1',
                'similarity': 0.8,
                'chunk_id': 'chunk_1'
            },
            {
                'text': 'This is test content from document 2',
                'document_id': 'test_doc_2',
                'similarity': 0.7,
                'chunk_id': 'chunk_2'
            }
        ]
        
        print("Testing _generate_answer with mock data...")
        
        # Call the private method directly
        result = await rag_service._generate_answer("What is this about?", mock_search_results)
        
        print(f"Result keys: {result.keys()}")
        print(f"Result text: {result.get('text', 'NO TEXT')[:200]}...")
        print(f"Result sources: {len(result.get('sources', []))}")
        print(f"Result confidence: {result.get('confidence', 0)}")
        
        if 'error' in str(result.get('text', '')).lower():
            print("ERROR: The result contains an error message!")
            return False
        
        return True
        
    except Exception as e:
        print(f"FAILED: Exception in _generate_answer test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_llm_client_direct():
    """Test LLM client directly with the same prompt format"""
    print("\n=== Testing LLM Client Directly ===")
    
    try:
        from core.ollama_client import OllamaClient
        
        llm_client = OllamaClient()
        
        # Create the exact same prompt as SimpleRAGService
        context = "[Quelle 1]: This is test content from document 1\n\n[Quelle 2]: This is test content from document 2"
        query = "What is this about?"
        
        prompt = f"""Beantworte die Frage basierend NUR auf den gegebenen Quellen. 
Zitiere die Quellen als [Quelle X] im Text.

Kontext:
{context}

Frage: {query}

Antwort:"""
        
        print("Testing direct LLM call with exact prompt...")
        print(f"Prompt length: {len(prompt)}")
        
        # Set shorter timeout for testing
        original_timeout = llm_client.timeout
        llm_client.timeout = 30  # 30 seconds
        
        response = llm_client.generate_answer(
            query=prompt,
            context=""  # Context is in the prompt
        )
        
        # Restore original timeout
        llm_client.timeout = original_timeout
        
        print(f"Direct LLM response: {response}")
        print(f"Response type: {type(response)}")
        print(f"Response length: {len(response) if response else 0}")
        
        return response is not None and response.strip() != ""
        
    except Exception as e:
        print(f"FAILED: Direct LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_step_by_step():
    """Test each step of the _generate_answer method"""
    print("\n=== Testing Step by Step ===")
    
    try:
        from core.repositories.factory import RepositoryFactory
        from core.services.simple_rag_service import SimpleRAGService
        from core.ollama_client import OllamaClient
        
        # Create components
        rag_repo = RepositoryFactory.create_production_repository()
        vector_repo = rag_repo.vector_search
        audit_repo = rag_repo.audit
        llm_client = OllamaClient()
        
        # Create RAG service
        rag_service = SimpleRAGService(vector_repo, llm_client, audit_repo)
        
        search_results = [
            {
                'text': 'This is test content',
                'document_id': 'test_doc',
                'similarity': 0.8,
                'chunk_id': 'chunk_1'
            }
        ]
        
        query = "What is this?"
        
        print("Step 1: Preparing context...")
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Quelle {i}]: {result['text']}")
            sources.append({
                "id": i,
                "document_id": result['document_id'],
                "similarity": result['similarity'],
                "download_url": f"/api/v1/documents/{result['document_id']}/download"
            })
        
        context = "\n\n".join(context_parts)
        print(f"Context prepared: {len(context)} chars")
        
        print("Step 2: Checking cache...")
        cached_response = rag_service.cache.get(query, context)
        if cached_response:
            print("Found cached response")
            return True
        else:
            print("No cached response")
        
        print("Step 3: Preparing prompt...")
        prompt = f"""Beantworte die Frage basierend NUR auf den gegebenen Quellen. 
Zitiere die Quellen als [Quelle X] im Text.

Kontext:
{context}

Frage: {query}

Antwort:"""
        
        print(f"Prompt prepared: {len(prompt)} chars")
        
        print("Step 4: Calling LLM with short timeout...")
        # Test with very short timeout to see if that's the issue
        original_timeout = llm_client.timeout
        llm_client.timeout = 10  # 10 seconds
        
        try:
            response = llm_client.generate_answer(
                query=prompt,
                context=""
            )
            print(f"LLM response received: {response is not None}")
            
        except Exception as llm_e:
            print(f"LLM call failed: {llm_e}")
            return False
        
        finally:
            llm_client.timeout = original_timeout
        
        print("Step 5: Processing response...")
        answer_text = response if response else 'Keine Antwort generiert.'
        
        print("Step 6: Adding source footer...")
        if sources and rag_service.config.require_sources:
            try:
                source_footer = "\n\n📚 Quellen:\n" + "\n".join([
                    f"[Quelle {s['id']}] Dokument {s['document_id']} - {s['download_url']}"
                    for s in sources
                ])
                answer_text += source_footer
                print("Source footer added successfully")
            except Exception as footer_e:
                print(f"Source footer failed: {footer_e}")
                # This might be the Unicode issue
                source_footer = "\n\nQuellen:\n" + "\n".join([
                    f"[Quelle {s['id']}] Dokument {s['document_id']} - {s['download_url']}"
                    for s in sources
                ])
                answer_text += source_footer
                print("Source footer added without emoji")
        
        print("Step 7: Creating result...")
        result = {
            "text": answer_text,
            "sources": sources,
            "confidence": max(r['similarity'] for r in search_results)
        }
        
        print("Step 8: Caching result...")
        rag_service.cache.set(query, context, result)
        
        print("SUCCESS: All steps completed")
        return True
        
    except Exception as e:
        print(f"FAILED: Step-by-step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("Debug: Testing RAG failure points...")
    
    # Test 1: Direct LLM
    success1 = await test_llm_client_direct()
    print(f"Direct LLM test: {'PASS' if success1 else 'FAIL'}")
    
    # Test 2: Generate answer method
    success2 = await test_generate_answer()
    print(f"Generate answer test: {'PASS' if success2 else 'FAIL'}")
    
    # Test 3: Step by step
    success3 = await test_step_by_step()
    print(f"Step-by-step test: {'PASS' if success3 else 'FAIL'}")
    
    print("\n=== Summary ===")
    print(f"Direct LLM: {'✓' if success1 else '✗'}")
    print(f"Generate Answer: {'✓' if success2 else '✗'}")
    print(f"Step by Step: {'✓' if success3 else '✗'}")

if __name__ == "__main__":
    asyncio.run(main())