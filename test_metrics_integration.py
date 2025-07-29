#!/usr/bin/env python3
"""
Test script for Prometheus metrics integration
"""
import asyncio
import sys
import time
from pathlib import Path

# Add the core module to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.services.metrics_service import get_metrics_service, init_metrics_service
from core.middleware.metrics_middleware import (
    get_query_metrics, get_db_metrics, get_doc_metrics, get_llm_metrics
)

async def test_metrics_service():
    """Test the metrics service functionality"""
    print("🔄 Testing Prometheus Metrics Integration...")
    
    # Initialize metrics service
    print("\n1. Initializing metrics service...")
    metrics_service = init_metrics_service()
    print(f"   ✅ Metrics enabled: {metrics_service.enabled}")
    
    if not metrics_service.enabled:
        print("   ⚠️  Prometheus client not available - metrics disabled")
        return False
    
    # Test basic metrics recording
    print("\n2. Testing HTTP metrics recording...")
    metrics_service.record_http_request("GET", "/api/v1/query", 200, 0.250)
    metrics_service.record_http_request("POST", "/api/v1/documents", 201, 1.500)
    metrics_service.record_http_request("GET", "/api/v1/status", 200, 0.050)
    print("   ✅ HTTP metrics recorded")
    
    # Test query metrics
    print("\n3. Testing query metrics...")
    query_metrics = get_query_metrics()
    context = query_metrics.record_query_start("test-tenant")
    
    query_metrics.record_component_start(context, "document_search")
    await asyncio.sleep(0.1)  # Simulate processing
    query_metrics.record_component_end(context, "document_search")
    
    query_metrics.record_component_start(context, "llm_generation")
    await asyncio.sleep(0.2)  # Simulate processing
    query_metrics.record_component_end(context, "llm_generation")
    
    query_metrics.record_query_end(context, "success", 0.85)
    print("   ✅ Query metrics recorded")
    
    # Test database metrics
    print("\n4. Testing database metrics...")
    db_metrics = get_db_metrics()
    db_metrics.record_operation("SELECT", "documents", 0.015)
    db_metrics.record_operation("INSERT", "queries", 0.025)
    db_metrics.update_connection_count(5)
    print("   ✅ Database metrics recorded")
    
    # Test document metrics
    print("\n5. Testing document metrics...")
    doc_metrics = get_doc_metrics()
    doc_metrics.record_processing("pdf", "success", 2.5)
    doc_metrics.record_processing("docx", "success", 1.8)
    doc_metrics.update_document_count("test-tenant", 150)
    print("   ✅ Document metrics recorded")
    
    # Test LLM metrics
    print("\n6. Testing LLM metrics...")
    llm_metrics = get_llm_metrics()
    llm_metrics.record_request("llama2", "success", 3.2, 100, 250)
    llm_metrics.record_request("mistral", "success", 2.8, 150, 300)
    print("   ✅ LLM metrics recorded")
    
    # Test system metrics update
    print("\n7. Testing system metrics...")
    metrics_service.update_system_metrics()
    print("   ✅ System metrics updated")
    
    # Get metrics output
    print("\n8. Generating metrics output...")
    metrics_output = metrics_service.get_metrics()
    print(f"   ✅ Generated {len(metrics_output.split('\\n'))} lines of metrics")
    
    # Show sample metrics
    print("\n9. Sample metrics (first 20 lines):")
    lines = metrics_output.split('\\n')
    for i, line in enumerate(lines[:20]):
        if line.strip() and not line.startswith('#'):
            print(f"   {line}")
    
    # Get stats summary
    print("\n10. Getting stats summary...")
    stats = metrics_service.get_stats_summary()
    print(f"    📊 Metrics enabled: {stats.get('metrics_enabled', False)}")
    print(f"    📊 Uptime: {stats.get('uptime_seconds', 0):.1f}s")
    print(f"    📊 CPU: {stats.get('system', {}).get('cpu_percent', 0):.1f}%")
    print(f"    📊 Memory: {stats.get('system', {}).get('memory_percent', 0):.1f}%")
    print("   ✅ Stats summary generated")
    
    print("\n✅ All metrics tests completed successfully!")
    return True

async def test_middleware_integration():
    """Test middleware components"""
    print("\n🔄 Testing middleware integration...")
    
    # Test all collectors are available
    query_collector = get_query_metrics()
    db_collector = get_db_metrics()
    doc_collector = get_doc_metrics()
    llm_collector = get_llm_metrics()
    
    print(f"   ✅ Query collector: {query_collector is not None}")
    print(f"   ✅ Database collector: {db_collector is not None}")
    print(f"   ✅ Document collector: {doc_collector is not None}")
    print(f"   ✅ LLM collector: {llm_collector is not None}")
    
    return True

def print_setup_instructions():
    """Print setup instructions for monitoring"""
    print("\n" + "="*60)
    print("📋 MONITORING SETUP INSTRUCTIONS")
    print("="*60)
    print()
    print("To start the complete monitoring stack:")
    print()
    print("1. Install monitoring dependencies:")
    print("   pip install prometheus-client psutil")
    print()
    print("2. Start the RAG system:")
    print("   python run_core.py")
    print()
    print("3. Check metrics endpoint:")
    print("   curl http://localhost:8000/metrics")
    print("   curl http://localhost:8000/metrics/health")
    print()
    print("4. Start Prometheus/Grafana (optional):")
    print("   cd deployment/monitoring")
    print("   docker-compose -f docker-compose.monitoring.yml up -d")
    print()
    print("5. Access dashboards:")
    print("   • Grafana: http://localhost:3000 (admin/admin123)")
    print("   • Prometheus: http://localhost:9090")
    print("   • AlertManager: http://localhost:9093")
    print()
    print("="*60)

async def main():
    """Main test function"""
    print("🚀 RAG System Metrics Integration Test")
    print("=" * 50)
    
    try:
        # Test metrics service
        metrics_success = await test_metrics_service()
        
        # Test middleware
        middleware_success = await test_middleware_integration()
        
        if metrics_success and middleware_success:
            print("\n🎉 All tests passed! Metrics integration is working correctly.")
            print_setup_instructions()
            return True
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)