#!/usr/bin/env python3
"""
Direct Ollama speed test - bypass RAG system
"""
import subprocess
import time

def test_ollama_direct():
    """Test Ollama directly without RAG system"""
    print("[TESTING] Direct Ollama Performance")
    print("=" * 50)
    
    # Simple test prompt
    prompt = "Answer in 10 words or less: What is 2+2?"
    
    print(f"Model: tinyllama")
    print(f"Prompt: {prompt}")
    print("Starting direct test...")
    
    start_time = time.time()
    
    try:
        # Direct ollama command
        result = subprocess.run([
            "C:/Users/THE/AppData/Local/Programs/Ollama/ollama.exe",
            "run", 
            "tinyllama",
            prompt
        ], capture_output=True, text=True, timeout=60)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"\n[RESULT] Response time: {response_time:.2f} seconds")
        print(f"Output: {result.stdout}")
        
        if response_time < 10:
            print("[EXCELLENT] Under 10 seconds!")
        elif response_time < 30:
            print("[GOOD] Under 30 seconds")
        else:
            print("[SLOW] Over 30 seconds")
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Command timed out after 60 seconds")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_ollama_direct()