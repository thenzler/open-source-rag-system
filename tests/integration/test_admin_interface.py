#!/usr/bin/env python3
"""
Test script for admin interface functionality
This is a manual integration test that requires a running server
"""
import pytest

@pytest.mark.integration
@pytest.mark.skip(reason="Requires running server - manual test only")
def test_admin_interface():
    """Test the admin interface endpoints - manual test only"""
    import requests
    
    base_url = "http://localhost:8000"
    
    # Test admin dashboard
    response = requests.get(f"{base_url}/admin/", timeout=10)
    assert response.status_code == 200
    
    # Test models API
    response = requests.get(f"{base_url}/admin/models", timeout=10)
    assert response.status_code == 200
    
    models_data = response.json()
    assert "total_count" in models_data
    assert "current_model" in models_data

if __name__ == "__main__":
    # Run as script for manual testing
    test_admin_interface()