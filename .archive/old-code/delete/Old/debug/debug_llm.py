#!/usr/bin/env python3
"""
Debug LLM functionality
"""

import sys
import os
sys.path.append("C:/Users/THE/open-source-rag-system")

def debug_llm():
    print("=== LLM DEBUG ===")
    
    try:
        from ollama_client import get_ollama_client
        print("OK: Imported ollama_client successfully")
        
        client = get_ollama_client()
        print(f"OK: Got client instance: {client}")
        print(f"OK: Client model: {client.model}")
        
        is_available = client.is_available()
        print(f"OK: Client availability: {is_available}")
        
        if is_available:
            print("Testing answer generation...")
            test_query = "Hello"
            test_context = "This is a test document."
            
            answer = client.generate_answer(test_query, test_context)
            print(f"OK: Generated answer: {answer}")
            
            if answer:
                print("SUCCESS: LLM is working correctly!")
            else:
                print("ERROR: LLM generated empty answer")
        else:
            print("ERROR: LLM is not available")
            
            # Check health status
            health = client.health_check()
            print(f"Health status: {health}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm()