#!/usr/bin/env python3
"""
Patch script to add optimized RAG to existing simple_api.py
Run this to add the optimized endpoints to your existing API
"""
import os
import sys
from pathlib import Path

def patch_simple_api():
    """Add import and integration code to simple_api.py"""
    
    simple_api_path = Path("simple_api.py")
    
    if not simple_api_path.exists():
        print("Error: simple_api.py not found in current directory")
        return False
    
    # Read the current file
    with open(simple_api_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if "integrate_optimized_rag" in content:
        print("simple_api.py appears to be already patched")
        return True
    
    # Find the line after the last import
    import_insert_point = content.rfind("logger = logging.getLogger(__name__)")
    if import_insert_point == -1:
        print("Could not find appropriate insertion point for import")
        return False
    
    # Find the line before if __name__ == "__main__":
    main_insert_point = content.find('if __name__ == "__main__":')
    if main_insert_point == -1:
        print("Could not find main block")
        return False
    
    # Prepare the import statement
    import_statement = '''

# Import optimized RAG integration
try:
    from integrate_optimized_rag import integrate_optimized_rag
    OPTIMIZED_RAG_SUPPORT = True
except ImportError as e:
    print(f"Optimized RAG not available: {e}")
    OPTIMIZED_RAG_SUPPORT = False
'''
    
    # Prepare the integration code
    integration_code = '''
# Integrate optimized RAG endpoints
if OPTIMIZED_RAG_SUPPORT:
    try:
        optimized_rag_system = integrate_optimized_rag(app, document_chunks, document_embeddings)
        logger.info("Optimized RAG endpoints added successfully")
    except Exception as e:
        logger.error(f"Failed to integrate optimized RAG: {e}")

'''
    
    # Insert the import after the logger line
    end_of_logger_line = content.find('\n', import_insert_point)
    new_content = (
        content[:end_of_logger_line] + 
        import_statement + 
        content[end_of_logger_line:main_insert_point] + 
        integration_code +
        content[main_insert_point:]
    )
    
    # Create backup
    backup_path = simple_api_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created backup: {backup_path}")
    
    # Write the patched content
    with open(simple_api_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully patched simple_api.py")
    print("\nNew endpoints added:")
    print("  - POST /api/v1/query/optimized - Fast query with concise responses")
    print("  - POST /api/v1/query/compare - Compare optimized vs original")
    print("  - POST /api/v1/cache/clear - Clear response cache")
    print("  - POST /api/v1/model/set - Set preferred model")
    print("  - GET  /api/v1/model/list - List available models")
    
    return True

if __name__ == "__main__":
    success = patch_simple_api()
    sys.exit(0 if success else 1)