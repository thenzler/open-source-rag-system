# Legacy Files Notice

## Municipal-Specific Files (Legacy)

The following files were part of the original municipal-focused implementation and are kept for reference:

### Training Data
- `data/training_data/arlesheim*/` - Arlesheim-specific training data
- `training_data/arlesheim*/` - Additional training materials

### Tools
- `tools/municipal*/` - Municipal web scraping and RAG tools
- `tools/municipal_*.py` - Municipal-specific utilities

### Documentation
- `docs/business/strategy/MUNICIPAL_USE_CASES.md` - Original municipal use cases
- `docs/business/training/ARLESHEIM_TRAINING_GUIDE.md` - Training guide for Arlesheim

## Current System

The system has been redesigned to be **domain-agnostic** with:

- Configurable keyword filtering via admin interface
- Document management and analysis tools
- Support for any domain through the admin interface
- Generic RAG capabilities without municipal-specific code

## Recommendation

These legacy files can be:
1. **Kept as examples** of domain-specific configuration
2. **Archived** to reduce repository size
3. **Removed** if municipal functionality is no longer needed

The current admin interface provides all the functionality needed for domain-specific configuration without hardcoded municipal logic.

---

**Last Updated**: January 2025
**System Version**: 2.0 (Domain-Agnostic)