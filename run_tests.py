#!/usr/bin/env python3
"""
Simple test runner for the RAG system
Runs tests without requiring pytest installation
"""

import sys
import os
import importlib.util
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_module_tests(module_path, test_name):
    """Run tests from a module"""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    try:
        # Load module
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Run main function if it exists
        if hasattr(test_module, 'main'):
            test_module.main()
        elif hasattr(test_module, 'run_simple_tests'):
            test_module.run_simple_tests()
        else:
            print("No main() or run_simple_tests() function found")
            
    except Exception as e:
        print(f"Error running {test_name}: {e}")
        import traceback
        traceback.print_exc()

def check_api_running():
    """Check if API server is running"""
    try:
        import requests
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def check_ollama_available():
    """Check if Ollama is available"""
    try:
        from ollama_client import get_ollama_client
        client = get_ollama_client()
        return client.is_available()
    except Exception:
        return False

def main():
    """Main test runner"""
    print("RAG System Test Runner")
    print("=" * 60)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    api_running = check_api_running()
    ollama_available = check_ollama_available()
    
    print(f"API Server: {'OK' if api_running else 'NOT RUNNING'}")
    print(f"Ollama: {'OK' if ollama_available else 'NOT AVAILABLE'}")
    
    if not api_running:
        print("\nX API server not running!")
        print("Please start the API server first:")
        print("  python simple_api.py")
        print("\nSome tests will be skipped.")
        time.sleep(2)
    
    # Define test modules
    tests = [
        {
            "path": "tests/test_simple_rag.py",
            "name": "Simple RAG Tests",
            "requires_api": True
        },
        {
            "path": "tests/test_api_fixed.py", 
            "name": "API Tests",
            "requires_api": True
        },
        {
            "path": "tests/test_ollama_integration.py",
            "name": "Ollama Integration Tests", 
            "requires_api": True,
            "requires_ollama": True
        }
    ]
    
    # Run tests
    results = {}
    
    for test in tests:
        if test.get("requires_api") and not api_running:
            print(f"\nSkipping {test['name']} (API not running)")
            results[test['name']] = "SKIPPED"
            continue
            
        if test.get("requires_ollama") and not ollama_available:
            print(f"\nSkipping {test['name']} (Ollama not available)")
            results[test['name']] = "SKIPPED"
            continue
        
        if os.path.exists(test["path"]):
            try:
                run_module_tests(test["path"], test["name"])
                results[test['name']] = "PASSED"
            except Exception as e:
                print(f"FAILED: {e}")
                results[test['name']] = "FAILED"
        else:
            print(f"\nTest file not found: {test['path']}")
            results[test['name']] = "NOT FOUND"
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Results Summary")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        status_symbol = {
            "PASSED": "+",
            "FAILED": "X", 
            "SKIPPED": "-",
            "NOT FOUND": "?"
        }.get(result, "?")
        
        print(f"{status_symbol} {test_name}: {result}")
    
    # Overall status
    failed_count = sum(1 for r in results.values() if r == "FAILED")
    passed_count = sum(1 for r in results.values() if r == "PASSED")
    
    print(f"\nOverall: {passed_count} passed, {failed_count} failed")
    
    if failed_count == 0:
        print("\n+ All tests passed!")
    else:
        print(f"\nX {failed_count} tests failed")
    
    return failed_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)