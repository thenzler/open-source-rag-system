#!/usr/bin/env python3
"""
Test Speed Improvements for RAG System
"""
import requests
import time
import json
from typing import Dict, List

class RAGSpeedTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_queries = [
            "What is the main topic of the document?",
            "Explain the key concepts discussed.",
            "What are the important findings?",
            "Summarize the main points.",
            "What methodology was used?"
        ]
    
    def test_server_health(self) -> bool:
        """Test if server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_regular_query(self, query: str) -> Dict:
        """Test regular query endpoint"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/query-enhanced",
                json={"query": query, "use_llm": True, "top_k": 5},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "method": response.json().get("method", "unknown"),
                    "sources": len(response.json().get("sources", []))
                }
            else:
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": response.text
                }
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    def test_streaming_query(self, query: str) -> Dict:
        """Test streaming query endpoint"""
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/query-stream",
                json={"query": query, "use_llm": True, "top_k": 5},
                timeout=30,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            if line.startswith(b"data: "):
                                data = json.loads(line[6:])
                                if data.get("type") == "answer_chunk":
                                    if first_chunk_time is None:
                                        first_chunk_time = time.time()
                                    total_chunks += 1
                                elif data.get("type") == "done":
                                    break
                        except json.JSONDecodeError:
                            continue
                
                end_time = time.time()
                return {
                    "success": True,
                    "total_time": end_time - start_time,
                    "first_chunk_time": first_chunk_time - start_time if first_chunk_time else None,
                    "total_chunks": total_chunks
                }
            else:
                return {
                    "success": False,
                    "total_time": time.time() - start_time,
                    "error": response.text
                }
        except Exception as e:
            return {
                "success": False,
                "total_time": time.time() - start_time,
                "error": str(e)
            }
    
    def test_cache_performance(self, query: str) -> Dict:
        """Test cache performance by running same query multiple times"""
        print(f"Testing cache performance for: '{query}'")
        
        # First run (cold cache)
        print("  First run (cold cache)...")
        cold_result = self.test_regular_query(query)
        
        # Second run (warm cache)
        print("  Second run (warm cache)...")
        warm_result = self.test_regular_query(query)
        
        # Third run (confirm cache)
        print("  Third run (confirm cache)...")
        confirm_result = self.test_regular_query(query)
        
        return {
            "cold_cache": cold_result,
            "warm_cache": warm_result,
            "confirm_cache": confirm_result,
            "speedup": cold_result.get("response_time", 0) / warm_result.get("response_time", 1) if warm_result.get("response_time") else 0
        }
    
    def clear_cache(self):
        """Clear server cache"""
        try:
            requests.post(f"{self.base_url}/api/v1/clear-cache", timeout=5)
            print("✅ Cache cleared")
        except:
            print("❌ Failed to clear cache")
    
    def run_comprehensive_test(self):
        """Run comprehensive speed tests"""
        print("🚀 RAG System Speed Test")
        print("=" * 60)
        
        # Check server health
        if not self.test_server_health():
            print("❌ Server not running at", self.base_url)
            return
        
        print("✅ Server is running")
        
        # Test 1: Regular vs Streaming comparison
        print("\n📊 Test 1: Regular vs Streaming Response Times")
        print("-" * 40)
        
        test_query = self.test_queries[0]
        
        # Regular query
        regular_result = self.test_regular_query(test_query)
        print(f"Regular query: {regular_result.get('response_time', 0):.2f}s")
        
        # Streaming query
        streaming_result = self.test_streaming_query(test_query)
        if streaming_result['success']:
            print(f"Streaming query - First chunk: {streaming_result.get('first_chunk_time', 0):.2f}s")
            print(f"Streaming query - Total time: {streaming_result.get('total_time', 0):.2f}s")
            print(f"Streaming query - Total chunks: {streaming_result.get('total_chunks', 0)}")
        
        # Test 2: Cache performance
        print("\n💾 Test 2: Cache Performance")
        print("-" * 40)
        
        self.clear_cache()
        cache_result = self.test_cache_performance(test_query)
        
        if cache_result['speedup'] > 0:
            print(f"Cache speedup: {cache_result['speedup']:.2f}x faster")
        
        # Test 3: Multiple queries performance
        print("\n🔄 Test 3: Multiple Queries Performance")
        print("-" * 40)
        
        total_time = 0
        successful_queries = 0
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"Query {i}: {query[:30]}...")
            result = self.test_regular_query(query)
            if result['success']:
                print(f"  ✅ {result['response_time']:.2f}s ({result['method']})")
                total_time += result['response_time']
                successful_queries += 1
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        
        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"\nAverage response time: {avg_time:.2f}s")
            print(f"Total successful queries: {successful_queries}/{len(self.test_queries)}")
        
        # Summary
        print("\n📈 Performance Summary")
        print("=" * 60)
        print("Speed Optimizations Implemented:")
        print("✅ Embedding caching (faster repeated queries)")
        print("✅ Vector search optimization (batch computation)")
        print("✅ Query result caching (instant cached responses)")
        print("✅ Response streaming (faster perceived response)")
        print("✅ Early termination for high-similarity matches")
        
        print("\nTo get maximum speed:")
        print("1. Use streaming endpoint for better user experience")
        print("2. Repeated queries will be much faster due to caching")
        print("3. Upload documents once, query many times for best performance")

def main():
    tester = RAGSpeedTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()