#!/usr/bin/env python3
"""
Test admin system after fixing container issues
"""
import time

def test_admin_system():
    """Test that admin system is working"""
    print("[TESTING] Admin System After Fixes")
    print("=" * 50)
    
    # Test 1: Import admin router
    try:
        from core.routers import admin
        print("[SUCCESS] Admin router imports without errors")
    except Exception as e:
        print(f"[ERROR] Admin router import failed: {e}")
        return False
    
    # Test 2: Test Ollama client creation
    try:
        from core.ollama_client import OllamaClient
        client = OllamaClient()
        print(f"[SUCCESS] Ollama client created: {client.model}")
    except Exception as e:
        print(f"[ERROR] Ollama client creation failed: {e}")
        return False
    
    # Test 3: Test config loading
    try:
        import yaml
        from pathlib import Path
        config_path = Path("config/llm_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        models_count = len(config.get('models', {}))
        current_model = config.get('default_model')
        print(f"[SUCCESS] Config loaded: {models_count} models, current: {current_model}")
    except Exception as e:
        print(f"[ERROR] Config loading failed: {e}")
        return False
    
    # Test 4: Test template directory
    try:
        from pathlib import Path
        template_dir = Path("core/templates")
        templates = list(template_dir.glob("*.html"))
        print(f"[SUCCESS] Templates found: {len(templates)} files")
    except Exception as e:
        print(f"[ERROR] Template check failed: {e}")
        return False
    
    print(f"\n[COMPLETED] All admin system tests passed!")
    print(f"[READY] Start server: python -m uvicorn core.main:app --host 0.0.0.0 --port 8000")
    print(f"[ACCESS] Main UI: http://localhost:8000/ui (click Settings)")
    print(f"[ACCESS] Admin: http://localhost:8000/admin")
    
    return True

if __name__ == "__main__":
    success = test_admin_system()
    if success:
        print(f"\n[SUCCESS] Admin system is ready to use!")
    else:
        print(f"\n[ERROR] Admin system has issues - check errors above")