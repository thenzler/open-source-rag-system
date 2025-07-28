import pytest
from app.core.config import get_settings


def test_settings_creation():
    """Test that settings can be created successfully."""
    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, 'database_url')
    assert hasattr(settings, 'api_host')
    assert hasattr(settings, 'api_port')


def test_settings_singleton():
    """Test that settings is a singleton."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2
