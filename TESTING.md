# Testing Guide

This document describes the testing setup for the RAG System.

## Test Structure

### Test Files
- `tests/test_simple_rag.py` - Basic functionality tests
- `tests/test_api_fixed.py` - API endpoint tests 
- `tests/test_ollama_integration.py` - Ollama LLM integration tests
- `tests/test_services.py` - Service layer tests
- `tests/conftest_fixed.py` - Test configuration and fixtures

### Test Runner
- `run_tests.py` - Simple test runner (no pytest required)
- `pytest.ini` - Pytest configuration (if pytest is available)

## Running Tests

### Option 1: Simple Test Runner (Recommended)
```bash
# Run all tests
python run_tests.py

# Run individual test files
python tests/test_simple_rag.py
python tests/test_api_fixed.py
python tests/test_ollama_integration.py
```

### Option 2: With pytest (if installed)
```bash
# Install pytest
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run specific tests
pytest tests/test_simple_rag.py -v
pytest tests/test_api_fixed.py -v
```

## Test Prerequisites

### API Server
Most tests require the API server to be running:
```bash
python simple_api.py
```

### Ollama (Optional)
For LLM integration tests, install Ollama:
```bash
# Install Ollama
# Visit: https://ollama.ai/download

# Start Ollama
ollama serve

# Pull a model
ollama pull mistral:latest
```

## Test Categories

### Unit Tests
- Test individual functions and classes
- No external dependencies required
- Fast execution

### Integration Tests
- Test API endpoints and workflows
- Require API server to be running
- Test document upload, query, and retrieval

### LLM Tests
- Test Ollama integration
- Require both API server and Ollama
- Test AI-powered query responses

## Test Output

Tests use simple symbols for Windows compatibility:
- `+` = Test passed
- `X` = Test failed
- `-` = Test skipped
- `?` = Test not found

## Common Issues

### Unicode Errors
All Unicode characters have been replaced with ASCII equivalents for Windows compatibility.

### API Server Not Running
```
X API health check failed: [Errno 61] Connection refused
```
**Solution**: Start the API server with `python simple_api.py`

### Ollama Not Available
```
X Ollama client test failed: Connection refused
```
**Solution**: Start Ollama with `ollama serve`

### Document Upload Failures
```
X Document upload failed: 500
```
**Solution**: Check server logs and ensure storage directory exists

## GitHub Actions

Automated testing runs on:
- Push to main/develop branches
- Pull requests to main branch
- Multiple Python versions (3.9, 3.10, 3.11)

The workflow includes:
- Linting (flake8, black, isort)
- Security scanning (bandit, safety)
- Unit and integration tests
- Test result artifacts

## Adding New Tests

### For API Endpoints
1. Add test class to `tests/test_api_fixed.py`
2. Use `@requires_api` decorator
3. Test both success and failure cases
4. Include proper assertions

### For Ollama Integration
1. Add test function to `tests/test_ollama_integration.py`
2. Use `@requires_ollama` decorator
3. Test with and without Ollama available
4. Include timeout handling

### For Services
1. Add test class to `tests/test_services.py`
2. Use mocking for external dependencies
3. Test error handling and edge cases
4. Include proper setup and teardown

## Test Best Practices

1. **Make tests independent** - Each test should be able to run alone
2. **Clean up after tests** - Remove test files and data
3. **Use descriptive names** - Test names should explain what they test
4. **Test error conditions** - Not just happy paths
5. **Mock external dependencies** - For reliable unit tests
6. **Include assertions** - Every test should verify something
7. **Handle timeouts** - For LLM and API calls
8. **Use fixtures** - For common test data and setup

## Performance Testing

For load testing, use the included locust file:
```bash
# Install locust
pip install locust

# Run load test
locust -f tests/performance/locustfile.py
```

## Troubleshooting

### Tests Hang
- Check for infinite loops in test code
- Verify API server is responsive
- Check for unhandled exceptions

### Import Errors
- Ensure project root is in Python path
- Check for missing dependencies
- Verify file paths are correct

### Flaky Tests
- Add retry logic for network calls
- Use proper waiting mechanisms
- Mock unreliable external services

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Add tests for edge cases
4. Update this documentation
5. Run the full test suite before submitting PR