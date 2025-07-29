#!/usr/bin/env python3
"""
Debug LLM generation issue in RAG service
"""
import logging
logging.basicConfig(level=logging.INFO)

def test_llm_generation():
    """Test LLM generation directly"""
    print("[DEBUG] Testing LLM Generation Issue")
    print("=" * 50)
    
    try:
        # Test 1: Import and create Ollama client
        from core.ollama_client import OllamaClient
        client = OllamaClient()
        
        print(f"[SUCCESS] OllamaClient created: {client.model}")
        print(f"[INFO] Base URL: {client.base_url}")
        
        # Test 2: Check if available
        is_available = client.is_available()
        print(f"[INFO] Client available: {is_available}")
        
        if not is_available:
            print("[ERROR] Ollama client not available - this is the issue")
            return False
        
        # Test 3: Test generate_answer method directly
        print("[TEST] Testing generate_answer method...")
        response = client.generate_answer(
            query="Test question", 
            context="Test context"
        )
        
        if response is None:
            print("[ERROR] generate_answer returned None - this is the issue")
            return False
        else:
            print(f"[SUCCESS] generate_answer returned: {type(response)}")
            print(f"[CONTENT] {response[:100]}...")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Exception during LLM test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_generation()
    if success:
        print("\n[SUCCESS] LLM generation is working correctly!")
        print("[INFO] The issue must be in the RAG service integration")
    else:
        print("\n[ERROR] LLM generation is broken - this is the root cause")