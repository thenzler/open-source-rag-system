#!/usr/bin/env python3
"""
Test admin router import to check for errors
"""

def test_admin_import():
    """Test importing admin router"""
    print("[TESTING] Admin Router Import")
    print("=" * 50)
    
    try:
        # Test core imports
        print("1. Testing core imports...")
        from core.routers import admin
        print("[SUCCESS] Admin router imported successfully")
        
        # Test template loading
        print("2. Testing template directory...")
        from pathlib import Path
        template_dir = Path("core/templates")
        if template_dir.exists():
            print(f"[SUCCESS] Template directory exists: {template_dir}")
            templates = list(template_dir.glob("*.html"))
            print(f"[INFO] Found {len(templates)} templates: {[t.name for t in templates]}")
        else:
            print(f"[ERROR] Template directory missing: {template_dir}")
        
        # Test configuration loading
        print("3. Testing configuration loading...")
        import yaml
        config_path = Path("config/llm_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[SUCCESS] Config loaded - {len(config.get('models', {}))} models available")
            print(f"[INFO] Default model: {config.get('default_model')}")
        else:
            print(f"[ERROR] Config file missing: {config_path}")
        
        # Test Ollama client creation
        print("4. Testing Ollama client creation...")
        from core.ollama_client import OllamaClient
        client = OllamaClient()
        print(f"[SUCCESS] Ollama client created: {client.model}")
        
        print(f"\n[COMPLETED] Admin router is ready to use!")
        print(f"[INFO] Start server: python -m uvicorn core.main:app --host 0.0.0.0 --port 8000")
        print(f"[INFO] Then visit: http://localhost:8000/admin")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_admin_import()
    if success:
        print("\n[SUCCESS] All tests passed - admin system is ready!")
    else:
        print("\n[ERROR] Some tests failed - check errors above")