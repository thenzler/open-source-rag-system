# Contributing to Open Source RAG System

🎉 Thank you for your interest in contributing to the Open Source RAG System! We welcome contributions from everyone and appreciate your help in making this project better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Security](#security)
- [Community](#community)

## Code of Conduct

This project adheres to a [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- **Docker & Docker Compose**: For containerized development
- **Python 3.11+**: For local development
- **Git**: For version control
- **Make**: For build automation (optional but recommended)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/open-source-rag-system.git
   cd open-source-rag-system
   ```

2. **Set Up Environment**
   ```bash
   make setup-env
   # Edit .env file with your configuration
   ```

3. **Start Development Environment**
   ```bash
   make dev
   ```

4. **Verify Installation**
   ```bash
   make health-check
   make test
   ```

## How to Contribute

### Ways to Contribute

- 🐛 **Bug Reports**: Help us identify and fix issues
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve docs, tutorials, or examples
- 🔧 **Code Contributions**: Fix bugs or implement features
- 🧪 **Testing**: Add or improve test coverage
- 🎨 **UI/UX**: Enhance user interface and experience
- 🔒 **Security**: Identify and fix security vulnerabilities
- 🌐 **Translation**: Help localize the project

### Good First Issues

Look for issues labeled `good first issue` or `help wanted`. These are perfect for newcomers and typically include:

- Documentation improvements
- Simple bug fixes
- Adding tests
- Code cleanup
- Small feature additions

## Contribution Guidelines

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for significant changes to discuss the approach
3. **Fork the repository** to your account
4. **Create a feature branch** from `develop`

### Contribution Types

#### 🐛 Bug Fixes

1. Create an issue describing the bug
2. Reference the issue in your commit messages
3. Include tests that reproduce the bug
4. Ensure fix doesn't break existing functionality

#### ✨ New Features

1. Discuss the feature in an issue first
2. Design the feature with maintainers
3. Implement with comprehensive tests
4. Update documentation
5. Consider backward compatibility

#### 📚 Documentation

1. Check for outdated information
2. Follow the documentation style guide
3. Test code examples
4. Update table of contents if needed

## Pull Request Process

### Creating a Pull Request

1. **Create a Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation
   - Run local tests

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

### PR Requirements

- ✅ **Descriptive title and description**
- ✅ **Reference related issues**
- ✅ **Pass all tests**
- ✅ **Meet code quality standards**
- ✅ **Include documentation updates**
- ✅ **No conflicts with target branch**

### PR Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Fixes #123
```

## Issue Guidelines

### Bug Reports

Use the bug report template and include:

- **Environment details** (OS, Python version, Docker version)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages** and logs
- **Screenshots** if applicable

```markdown
**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- Docker: [e.g., 24.0.6]

**Steps to Reproduce:**
1. Start the system with `make dev`
2. Upload a PDF document
3. Query the document
4. See error

**Expected Behavior:**
The query should return relevant results.

**Actual Behavior:**
Error: "Vector search failed"

**Logs:**
```
[Include relevant log output]
```
```

### Feature Requests

Use the feature request template and include:

- **Problem description** the feature would solve
- **Proposed solution** or implementation approach
- **Alternative solutions** considered
- **Use cases** and examples

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Import order**: Standard library, third-party, local imports
- **Docstrings**: Google style docstrings
- **Type hints**: Required for all public functions

#### Code Formatting

We use automated tools for consistent formatting:

```bash
# Format code
make format

# Check formatting
make lint

# Type checking
make type-check
```

#### Tools Used

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning

### Example Code Style

```python
"""
Module docstring describing the module's purpose.
"""

from typing import List, Dict, Optional, Any
import asyncio
import logging

from third_party_library import SomeClass
from local_module import LocalClass

logger = logging.getLogger(__name__)


class ExampleService:
    """
    Example service class demonstrating our coding standards.
    
    Args:
        config: Configuration dictionary
        timeout: Request timeout in seconds
    """
    
    def __init__(self, config: Dict[str, Any], timeout: int = 30):
        self.config = config
        self.timeout = timeout
        self._client: Optional[SomeClass] = None
    
    async def process_data(
        self, 
        data: List[str], 
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Process data in batches.
        
        Args:
            data: List of data items to process
            batch_size: Number of items per batch
            
        Returns:
            Dictionary with processing results
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            results = {}
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_result = await self._process_batch(batch)
                results.update(batch_result)
            
            logger.info(f"Processed {len(data)} items successfully")
            return results
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Failed to process data: {e}")
    
    async def _process_batch(self, batch: List[str]) -> Dict[str, int]:
        """Process a single batch of data."""
        # Implementation details...
        pass
```

### API Design Standards

- **RESTful endpoints**: Follow REST conventions
- **Consistent naming**: Use snake_case for JSON fields
- **Error handling**: Return structured error responses
- **Validation**: Use Pydantic models for request/response validation
- **Documentation**: Auto-generated with FastAPI/OpenAPI

### Database Standards

- **Migration scripts**: Use Alembic for schema changes
- **Naming conventions**: snake_case for tables and columns
- **Indexes**: Add appropriate indexes for performance
- **Constraints**: Use database constraints for data integrity

## Testing

### Test Requirements

All contributions must include appropriate tests:

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test service interactions
- **API tests**: Test HTTP endpoints
- **End-to-end tests**: Test complete workflows

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-api

# Run with coverage
make test-coverage

# Run performance tests
make test-load
```

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_services/
│   ├── test_models/
│   └── test_utils/
├── integration/          # Integration tests
│   ├── test_database/
│   ├── test_vector_db/
│   └── test_llm/
├── api/                  # API tests
│   ├── test_documents/
│   ├── test_queries/
│   └── test_auth/
├── e2e/                  # End-to-end tests
├── performance/          # Load tests
└── fixtures/             # Test data
```

### Writing Good Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestDocumentService:
    """Test suite for DocumentService."""
    
    @pytest.fixture
    async def document_service(self):
        """Create a document service for testing."""
        service = DocumentService()
        await service.initialize()
        return service
    
    @pytest.mark.asyncio
    async def test_upload_document_success(self, document_service):
        """Test successful document upload."""
        # Arrange
        mock_file = AsyncMock()
        mock_file.filename = "test.pdf"
        mock_file.content_type = "application/pdf"
        mock_file.size = 1024
        
        # Act
        result = await document_service.upload_document(
            file=mock_file,
            user_id="test-user",
            metadata={"category": "test"}
        )
        
        # Assert
        assert result.filename == "test.pdf"
        assert result.status == ProcessingStatus.PENDING
        assert result.metadata["category"] == "test"
    
    @pytest.mark.asyncio
    async def test_upload_document_invalid_file(self, document_service):
        """Test upload with invalid file type."""
        # Arrange
        mock_file = AsyncMock()
        mock_file.filename = "test.exe"
        mock_file.content_type = "application/octet-stream"
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Unsupported file type"):
            await document_service.upload_document(
                file=mock_file,
                user_id="test-user"
            )
```

## Documentation

### Documentation Types

1. **API Documentation**: Auto-generated from code
2. **User Guides**: Step-by-step instructions
3. **Developer Docs**: Technical implementation details
4. **Deployment Guides**: Installation and configuration
5. **Troubleshooting**: Common issues and solutions

### Documentation Standards

- **Clear and concise**: Easy to understand
- **Up-to-date**: Reflect current functionality
- **Examples**: Include practical examples
- **Screenshots**: Visual aids where helpful
- **Links**: Reference related documentation

### Writing Documentation

```markdown
# Title

Brief description of what this document covers.

## Prerequisites

List any requirements or assumptions.

## Step-by-Step Instructions

### Step 1: Initial Setup

Detailed instructions with code examples.

```bash
# Example command
make setup-env
```

### Step 2: Configuration

More detailed steps...

## Examples

Practical examples that users can copy and modify.

## Troubleshooting

Common issues and their solutions.

## See Also

Links to related documentation.
```

## Security

### Security Guidelines

- **Never commit secrets**: Use environment variables
- **Input validation**: Validate all user inputs
- **SQL injection**: Use parameterized queries
- **XSS protection**: Sanitize output
- **Authentication**: Secure token handling
- **Authorization**: Proper access controls

### Reporting Security Issues

🔒 **Do not open public issues for security vulnerabilities.**

Instead, please email security@yourragdomain.com with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work with you to resolve the issue.

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord/Slack**: Real-time chat (if available)
- **Email**: Contact maintainers directly

### Getting Help

1. **Check documentation** first
2. **Search existing issues** for similar problems
3. **Ask in discussions** for general questions
4. **Create an issue** for bugs or feature requests

### Recognition

Contributors are recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Release notes
- **GitHub**: Contributor graphs and statistics

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes (backward compatible)

### Release Workflow

1. **Feature freeze** on `develop` branch
2. **Release candidate** testing
3. **Merge to main** and tag release
4. **Deploy to production**
5. **Update documentation**

## Questions?

If you have any questions about contributing, please:

1. Check this guide and other documentation
2. Search existing GitHub issues and discussions
3. Ask in GitHub Discussions
4. Contact the maintainers

Thank you for contributing to the Open Source RAG System! 🚀
