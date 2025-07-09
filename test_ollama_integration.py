#!/usr/bin/env python3
"""
Test script for Ollama LLM integration
"""
import requests
import json
import time
from pathlib import Path

API_BASE = "http://localhost:8001"

def test_system_status():
    """Test system status endpoint"""
    print("🔧 Testing system status...")
    try:
        response = requests.get(f"{API_BASE}/api/v1/status")
        if response.status_code == 200:
            status = response.json()
            print("✓ System status retrieved")
            print(f"  - Service: {status['service']}")
            print(f"  - Vector Search: {'✅' if status['features']['vector_search'] else '❌'}")
            print(f"  - LLM Generation: {'✅' if status['features']['llm_generation'] else '❌'}")
            print(f"  - Document Processing: {'✅' if status['features']['document_processing'] else '❌'}")
            
            if 'ollama' in status:
                print(f"  - Ollama Available: {'✅' if status['ollama']['available'] else '❌'}")
                if status['ollama']['available']:
                    print(f"    Model: {status['ollama'].get('model', 'Unknown')}")
                    print(f"    Available Models: {status['ollama'].get('models', [])}")
                else:
                    print(f"    Error: {status['ollama'].get('error', 'Unknown')}")
            
            return status
        else:
            print(f"✗ System status failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ System status failed: {e}")
        return None

def test_ollama_client():
    """Test Ollama client directly"""
    print("\n🤖 Testing Ollama client directly...")
    try:
        from ollama_client import test_ollama_connection
        result = test_ollama_connection()
        return result
    except ImportError:
        print("✗ Ollama client module not found")
        return False
    except Exception as e:
        print(f"✗ Ollama client test failed: {e}")
        return False

def upload_test_document():
    """Upload a test document for LLM testing"""
    print("\n📄 Uploading test document...")
    
    # Create a comprehensive test document
    test_content = """
    # Machine Learning and AI Guide

    ## What is Machine Learning?
    Machine learning is a subset of artificial intelligence (AI) that enables computer systems to automatically learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data, identify patterns, and make predictions or decisions.

    ## Types of Machine Learning
    
    ### 1. Supervised Learning
    In supervised learning, algorithms learn from labeled training data. Examples include:
    - Classification: Predicting categories (email spam detection)
    - Regression: Predicting continuous values (house prices)
    
    ### 2. Unsupervised Learning
    Unsupervised learning finds patterns in data without labeled examples:
    - Clustering: Grouping similar data points
    - Association: Finding relationships between variables
    
    ### 3. Reinforcement Learning
    This type learns through interaction with an environment, receiving rewards or penalties for actions taken.

    ## Natural Language Processing (NLP)
    Natural Language Processing is a branch of AI that helps computers understand, interpret, and generate human language. Key applications include:
    - Machine translation (Google Translate)
    - Chatbots and virtual assistants
    - Sentiment analysis
    - Text summarization

    ## Vector Databases
    Vector databases are specialized systems designed to store and search high-dimensional vectors efficiently. They are crucial for:
    - Semantic search applications
    - Recommendation systems
    - Image and video similarity search
    - Retrieval-Augmented Generation (RAG) systems

    ## Deep Learning
    Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. It excels at:
    - Image recognition and computer vision
    - Speech recognition and synthesis
    - Natural language understanding
    - Game playing (like AlphaGo)

    ## Applications of AI
    Artificial Intelligence is transforming various industries:
    - Healthcare: Medical diagnosis, drug discovery
    - Finance: Fraud detection, algorithmic trading
    - Transportation: Autonomous vehicles, route optimization
    - Entertainment: Recommendation systems, content generation
    - Manufacturing: Quality control, predictive maintenance
    """
    
    test_file = Path("test_ml_guide.txt")
    test_file.write_text(test_content)
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'text/plain')}
            response = requests.post(f"{API_BASE}/api/v1/documents", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Test document uploaded: {result['filename']}")
            print(f"  Status: {result['status']}")
            test_file.unlink()  # Clean up
            return True
        else:
            print(f"✗ Document upload failed: {response.status_code}")
            print(f"  Error: {response.text}")
            test_file.unlink()  # Clean up
            return False
    except Exception as e:
        print(f"✗ Document upload failed: {e}")
        if test_file.exists():
            test_file.unlink()  # Clean up
        return False

def test_vector_search():
    """Test vector search endpoint"""
    print("\n🔍 Testing vector search...")
    
    test_query = "What is machine learning?"
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/query",
            json={"query": test_query, "top_k": 3},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Vector search successful: {result['total_results']} results")
            for i, res in enumerate(result['results']):
                print(f"  Result {i+1}: Score {res['score']:.3f} from {res['source_document']}")
            return True
        else:
            print(f"✗ Vector search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Vector search failed: {e}")
        return False

def test_enhanced_query(use_llm=True):
    """Test enhanced query with LLM"""
    mode = "LLM" if use_llm else "fallback"
    print(f"\n🤖 Testing enhanced query ({mode})...")
    
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "What are vector databases used for?",
        "How is NLP applied in real-world applications?"
    ]
    
    success_count = 0
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/api/v1/query/enhanced",
                json={"query": query, "top_k": 3, "use_llm": use_llm},
                headers={"Content-Type": "application/json"},
                timeout=60  # Give LLM time to respond
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Query successful ({elapsed:.2f}s)")
                print(f"  Method: {result['method']}")
                print(f"  Sources: {result['total_sources']}")
                
                # Show part of the answer
                answer = result['answer']
                preview = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"  Answer: {preview}")
                
                success_count += 1
            else:
                print(f"✗ Query failed: {response.status_code}")
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"✗ Query failed: {e}")
    
    print(f"\n{mode} queries: {success_count}/{len(test_queries)} successful")
    return success_count == len(test_queries)

def test_ollama_setup_instructions():
    """Provide Ollama setup instructions if not available"""
    print("\n📋 Ollama Setup Instructions")
    print("=" * 50)
    print("If Ollama is not available, follow these steps:")
    print()
    print("1. Download Ollama:")
    print("   Visit: https://ollama.ai/download")
    print("   Install for your operating system")
    print()
    print("2. Start Ollama:")
    print("   Open terminal/command prompt and run: ollama serve")
    print()
    print("3. Pull a model:")
    print("   ollama pull llama3.1:8b")
    print("   (or another model like phi-3:mini for lower RAM)")
    print()
    print("4. Verify installation:")
    print("   ollama list")
    print()
    print("5. Test with this RAG system:")
    print("   python test_ollama_integration.py")

def main():
    print("Ollama LLM Integration Test")
    print("=" * 40)
    
    # Test system status
    status = test_system_status()
    if not status:
        print("\n❌ Cannot connect to API. Make sure the server is running:")
        print("python simple_api.py")
        return
    
    # Test Ollama client directly
    ollama_available = test_ollama_client()
    
    # Upload test document
    if not upload_test_document():
        print("\n❌ Cannot test queries without documents")
        return
    
    # Wait for processing
    time.sleep(2)
    
    # Test vector search
    if not test_vector_search():
        print("\n❌ Vector search failed")
        return
    
    # Test enhanced queries
    if ollama_available:
        print("\n" + "="*50)
        print("Testing LLM-powered queries...")
        llm_success = test_enhanced_query(use_llm=True)
        
        if llm_success:
            print("\n✅ All LLM tests passed!")
            print("\n🎉 Your RAG system with Ollama is working perfectly!")
            print("\nYou can now:")
            print("1. Open http://localhost:8001/simple_frontend.html")
            print("2. Upload your own documents")
            print("3. Ask questions and get AI-generated answers")
        else:
            print("\n⚠️ Some LLM tests failed")
    else:
        print("\n" + "="*50)
        print("Ollama not available - testing fallback mode...")
        fallback_success = test_enhanced_query(use_llm=False)
        
        if fallback_success:
            print("\n✅ Fallback mode works!")
            print("\n⚠️ Install Ollama for AI-generated answers:")
            test_ollama_setup_instructions()
        else:
            print("\n❌ Even fallback mode failed")

if __name__ == "__main__":
    main()