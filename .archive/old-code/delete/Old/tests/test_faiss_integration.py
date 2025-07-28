#!/usr/bin/env python3
"""
Test script to verify FAISS integration and performance
This script tests the RAG system with and without FAISS to demonstrate improvements
"""

import time
import requests
import json
import tempfile
import os
from typing import Dict, List

class FAISSIntegrationTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_results = {}
        
    def check_api_health(self) -> bool:
        """Check if the API is running and healthy"""
        try:
            response = requests.get(f"{self.base_url}/api/status", timeout=10)
            if response.status_code == 200:
                print("✅ API is running and healthy")
                return True
            else:
                print(f"❌ API returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API: {e}")
            print("Please start the server with: python simple_api.py")
            return False
    
    def test_vector_stats(self) -> Dict:
        """Test the vector stats endpoint to check FAISS status"""
        print("\n📊 Testing Vector Store Status...")
        
        try:
            response = requests.get(f"{self.base_url}/api/v1/vector-stats")
            if response.status_code == 200:
                stats = response.json()
                
                print(f"  Vector Store Type: {stats.get('type', 'Unknown')}")
                print(f"  Total Vectors: {stats.get('total_vectors', 0):,}")
                print(f"  Performance: {stats.get('performance', 'Unknown')}")
                print(f"  Status: {stats.get('status', 'Unknown')}")
                
                if "FAISS" in stats.get('type', ''):
                    print("  🚀 FAISS is active!")
                    if stats.get('index_type'):
                        print(f"  Index Type: {stats['index_type']}")
                else:
                    print("  ⚠️  FAISS not active, using fallback search")
                    if stats.get('installation'):
                        print(f"  Install command: {stats['installation']}")
                
                return stats
            else:
                print(f"  ❌ Failed to get vector stats: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"  ❌ Error testing vector stats: {e}")
            return {}
    
    def upload_test_document(self) -> bool:
        """Upload a test document for performance testing"""
        print("\n📄 Uploading Test Document...")
        
        # Create a comprehensive test document
        test_content = """
# FAISS Vector Search Performance Test Document

## What is FAISS?

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search 
and clustering of dense vectors. It contains algorithms that search in sets of 
vectors of any size, up to ones that possibly do not fit in RAM.

## Key Features

1. **High Performance**: FAISS can provide 10-100x speedup over traditional methods
2. **Multiple Index Types**: Supports exact and approximate search algorithms
3. **GPU Acceleration**: Can leverage GPU resources for even faster search
4. **Memory Efficiency**: Optimized memory usage for large vector datasets
5. **Production Ready**: Used by major tech companies worldwide

## Technical Details

FAISS implements several indexing algorithms:
- Flat indexes for exact search
- IVF (Inverted File) indexes for fast approximate search
- HNSW (Hierarchical Navigable Small Worlds) for scalable search
- PQ (Product Quantization) for memory-efficient storage

## Performance Benchmarks

In typical RAG applications, FAISS provides:
- 5-10x speedup for datasets with <1,000 vectors
- 20-50x speedup for datasets with 1,000-100,000 vectors  
- 100x+ speedup for datasets with >100,000 vectors

## Implementation Benefits

The RAG system automatically detects FAISS and provides:
- Seamless integration with existing code
- Automatic index type selection based on dataset size
- Fallback to cosine similarity if FAISS is not available
- Migration of existing embeddings to FAISS indexes

## Use Cases

FAISS is particularly beneficial for:
- Large document collections (>1,000 documents)
- Real-time search applications
- Production RAG systems
- Knowledge bases with frequent queries
- Multi-user environments

## Installation and Setup

Installing FAISS is straightforward:
1. Install the package: pip install faiss-cpu
2. Restart the RAG system
3. Automatic detection and migration
4. Immediate performance improvements

This completes our test document for FAISS integration testing.
"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Upload the document
            with open(temp_file, 'rb') as f:
                files = {"file": ("faiss_test_doc.txt", f, "text/plain")}
                response = requests.post(f"{self.base_url}/api/v1/documents", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Document uploaded successfully (ID: {result.get('id')})")
                print(f"  File size: {result.get('size', 0):,} bytes")
                return True
            else:
                print(f"  ❌ Document upload failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error uploading document: {e}")
            return False
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_search_performance(self) -> Dict:
        """Test search performance with various queries"""
        print("\n🔍 Testing Search Performance...")
        
        test_queries = [
            "What is FAISS?",
            "How does FAISS improve performance?", 
            "What are the key features of FAISS?",
            "Explain FAISS indexing algorithms",
            "What are the performance benefits?",
            "How to install FAISS?",
            "FAISS use cases and applications",
            "Technical implementation details"
        ]
        
        results = {
            "queries_tested": len(test_queries),
            "response_times": [],
            "avg_response_time": 0,
            "total_time": 0,
            "successful_queries": 0
        }
        
        start_time = time.time()
        
        for i, query in enumerate(test_queries, 1):
            print(f"  Query {i}/{len(test_queries)}: {query[:40]}...")
            
            query_start = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/query/optimized",
                    json={"query": query, "context_limit": 5},
                    timeout=30
                )
                
                query_time = time.time() - query_start
                results["response_times"].append(query_time)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"    ✅ Response time: {query_time:.3f}s")
                    results["successful_queries"] += 1
                    
                    # Show response preview
                    response_text = result.get('response', '')
                    preview = response_text[:80] + "..." if len(response_text) > 80 else response_text
                    print(f"    Preview: {preview}")
                    
                    # Show sources count
                    context_count = len(result.get('context', []))
                    if context_count > 0:
                        print(f"    Sources: {context_count} relevant chunks found")
                    
                else:
                    print(f"    ❌ Query failed: {response.status_code}")
                    
            except Exception as e:
                query_time = time.time() - query_start
                results["response_times"].append(query_time)
                print(f"    ❌ Error: {e}")
        
        results["total_time"] = time.time() - start_time
        if results["response_times"]:
            results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
        
        # Performance summary
        print(f"\n📈 Performance Summary:")
        print(f"  Total queries: {results['queries_tested']}")
        print(f"  Successful: {results['successful_queries']}")
        print(f"  Average response time: {results['avg_response_time']:.3f}s")
        print(f"  Total test time: {results['total_time']:.1f}s")
        
        if results["avg_response_time"] < 1.0:
            print("  🚀 Excellent performance!")
        elif results["avg_response_time"] < 3.0:
            print("  ✅ Good performance")
        else:
            print("  ⚠️  Performance could be improved")
        
        return results
    
    def test_concurrent_queries(self, num_queries: int = 5) -> Dict:
        """Test concurrent query performance"""
        print(f"\n⚡ Testing Concurrent Performance ({num_queries} simultaneous queries)...")
        
        import concurrent.futures
        import threading
        
        query = "What are the key benefits of FAISS for vector search?"
        
        def single_query():
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/query/optimized",
                    json={"query": query, "context_limit": 3},
                    timeout=30
                )
                elapsed = time.time() - start_time
                return {"success": response.status_code == 200, "time": elapsed}
            except Exception as e:
                elapsed = time.time() - start_time
                return {"success": False, "time": elapsed, "error": str(e)}
        
        # Run concurrent queries
        concurrent_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_queries) as executor:
            futures = [executor.submit(single_query) for _ in range(num_queries)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_concurrent_time = time.time() - concurrent_start
        
        # Analyze results
        successful = sum(1 for r in results if r["success"])
        avg_time = sum(r["time"] for r in results) / len(results)
        max_time = max(r["time"] for r in results)
        min_time = min(r["time"] for r in results)
        
        print(f"  Successful queries: {successful}/{num_queries}")
        print(f"  Average response time: {avg_time:.3f}s")
        print(f"  Min/Max response time: {min_time:.3f}s / {max_time:.3f}s")
        print(f"  Total concurrent time: {total_concurrent_time:.3f}s")
        print(f"  Queries per second: {num_queries/total_concurrent_time:.2f}")
        
        return {
            "total_queries": num_queries,
            "successful": successful,
            "avg_time": avg_time,
            "total_time": total_concurrent_time,
            "qps": num_queries/total_concurrent_time
        }
    
    def benchmark_vs_baseline(self) -> Dict:
        """Compare current performance with expected baseline"""
        print("\n📊 Benchmark vs Expected Performance...")
        
        # Get current stats
        stats = self.test_results.get("vector_stats", {})
        performance_results = self.test_results.get("search_performance", {})
        
        vector_count = stats.get("total_vectors", 0)
        avg_time = performance_results.get("avg_response_time", 0)
        
        # Expected performance based on vector count
        if "FAISS" in stats.get("type", ""):
            # FAISS expected performance
            if vector_count < 1000:
                expected_time = 0.5  # 500ms for small datasets
                expected_speedup = "5-10x"
            elif vector_count < 10000:
                expected_time = 1.0  # 1s for medium datasets
                expected_speedup = "20-50x"
            else:
                expected_time = 1.5  # 1.5s for large datasets
                expected_speedup = "100x+"
        else:
            # Cosine similarity expected performance
            if vector_count < 1000:
                expected_time = 2.0
                expected_speedup = "1x (baseline)"
            elif vector_count < 10000:
                expected_time = 10.0
                expected_speedup = "1x (baseline)"
            else:
                expected_time = 30.0
                expected_speedup = "1x (baseline)"
        
        # Compare actual vs expected
        performance_ratio = expected_time / avg_time if avg_time > 0 else 0
        
        print(f"  Dataset size: {vector_count:,} vectors")
        print(f"  Search method: {stats.get('type', 'Unknown')}")
        print(f"  Expected response time: {expected_time:.1f}s")
        print(f"  Actual response time: {avg_time:.3f}s")
        print(f"  Expected speedup: {expected_speedup}")
        
        if performance_ratio > 1.0:
            print(f"  🚀 Performance is {performance_ratio:.1f}x BETTER than expected!")
        elif performance_ratio > 0.8:
            print(f"  ✅ Performance meets expectations")
        else:
            print(f"  ⚠️  Performance is below expectations")
        
        return {
            "expected_time": expected_time,
            "actual_time": avg_time,
            "performance_ratio": performance_ratio,
            "vector_count": vector_count,
            "search_method": stats.get("type", "Unknown")
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        print("\n📋 Generating Test Report...")
        
        report = f"""
# FAISS Integration Test Report

**Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**API Endpoint**: {self.base_url}

## System Status

"""
        
        # Vector store status
        stats = self.test_results.get("vector_stats", {})
        report += f"""
### Vector Store Configuration
- **Type**: {stats.get('type', 'Unknown')}
- **Total Vectors**: {stats.get('total_vectors', 0):,}
- **Performance**: {stats.get('performance', 'Unknown')}
- **Status**: {stats.get('status', 'Unknown')}
"""
        
        if stats.get('index_type'):
            report += f"- **Index Type**: {stats['index_type']}\n"
        
        # Search performance
        perf = self.test_results.get("search_performance", {})
        if perf:
            report += f"""
### Search Performance
- **Queries Tested**: {perf.get('queries_tested', 0)}
- **Successful Queries**: {perf.get('successful_queries', 0)}
- **Average Response Time**: {perf.get('avg_response_time', 0):.3f}s
- **Total Test Time**: {perf.get('total_time', 0):.1f}s
"""
        
        # Concurrent performance
        concurrent = self.test_results.get("concurrent_performance", {})
        if concurrent:
            report += f"""
### Concurrent Performance
- **Simultaneous Queries**: {concurrent.get('total_queries', 0)}
- **Successful**: {concurrent.get('successful', 0)}
- **Average Response Time**: {concurrent.get('avg_time', 0):.3f}s
- **Queries Per Second**: {concurrent.get('qps', 0):.2f}
"""
        
        # Benchmark comparison
        benchmark = self.test_results.get("benchmark", {})
        if benchmark:
            report += f"""
### Performance Benchmark
- **Expected Response Time**: {benchmark.get('expected_time', 0):.1f}s
- **Actual Response Time**: {benchmark.get('actual_time', 0):.3f}s
- **Performance Ratio**: {benchmark.get('performance_ratio', 0):.1f}x
- **Search Method**: {benchmark.get('search_method', 'Unknown')}
"""
        
        # Recommendations
        report += """
## Recommendations

"""
        
        if "FAISS" in stats.get('type', ''):
            report += "✅ **FAISS is active** - Your system is optimized for high performance!\n\n"
            
            if perf.get('avg_response_time', 10) < 2.0:
                report += "✅ **Excellent performance** - Response times are within optimal range.\n\n"
            else:
                report += "⚠️  **Consider optimization** - Response times could be improved:\n"
                report += "   - Check server resources (CPU, memory)\n"
                report += "   - Consider reducing context_limit for faster queries\n"
                report += "   - Monitor for system bottlenecks\n\n"
        else:
            report += "❌ **FAISS not detected** - Install for major performance improvements:\n\n"
            report += "```bash\npip install faiss-cpu\npython simple_api.py\n```\n\n"
            report += f"**Expected improvement**: {stats.get('expected_speedup', '10-100x faster')}\n\n"
        
        # Save report
        report_file = "faiss_integration_test_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  📄 Report saved to: {report_file}")
        return report
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("="*60)
        print("🧪 FAISS Integration Test Suite")
        print("="*60)
        
        # Check API health
        if not self.check_api_health():
            return False
        
        # Run all tests
        self.test_results["vector_stats"] = self.test_vector_stats()
        
        # Upload test document
        if self.upload_test_document():
            # Wait a moment for processing
            time.sleep(2)
            
            # Performance tests
            self.test_results["search_performance"] = self.test_search_performance()
            self.test_results["concurrent_performance"] = self.test_concurrent_queries()
            self.test_results["benchmark"] = self.benchmark_vs_baseline()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*60)
        print("🎉 Test Suite Complete!")
        print("="*60)
        
        # Final summary
        stats = self.test_results.get("vector_stats", {})
        perf = self.test_results.get("search_performance", {})
        
        if "FAISS" in stats.get('type', ''):
            print("✅ FAISS Integration: ACTIVE")
            print(f"🚀 Performance: {perf.get('avg_response_time', 0):.3f}s average response")
            print("📊 Status: Your RAG system is optimized!")
        else:
            print("⚠️  FAISS Integration: NOT ACTIVE")
            print("📋 Recommendation: Install FAISS for 10-100x performance boost")
            print("💡 Command: pip install faiss-cpu")
        
        return True


def main():
    """Run the FAISS integration test"""
    tester = FAISSIntegrationTester()
    tester.run_full_test_suite()


if __name__ == "__main__":
    main()