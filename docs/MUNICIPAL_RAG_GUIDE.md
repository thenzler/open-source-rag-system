# Municipal RAG System Guide

This guide explains how to set up and use the Municipal RAG system for Swiss municipalities like Arlesheim.

## Overview

The Municipal RAG system is specifically designed for Swiss municipalities (Gemeinden) and provides:

- **Specialized web scraping** for municipal websites
- **Weighted importance scoring** for official documents
- **Municipal-specific prompts** for different categories
- **Multi-language support** (German/French)
- **Category-based filtering** (services, administration, events, etc.)

## Quick Start for Arlesheim

### 1. Scrape Municipal Website

```bash
# Scrape Arlesheim website
python municipal_setup.py arlesheim --scrape --max-pages 50

# Test with a query
python municipal_setup.py arlesheim --query "Was sind die Öffnungszeiten der Gemeindeverwaltung?"
```

### 2. Python API Usage

```python
from municipal_setup import MunicipalRagSetup

# Setup for Arlesheim
setup = MunicipalRagSetup('arlesheim')
municipal_rag = setup.setup_complete_system(scrape_fresh=True)

# Query the system
result = municipal_rag.generate_municipal_answer(
    "Wie kann ich einen Bauantrag stellen?"
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
print(f"Confidence: {result['confidence']}")
```

### 3. REST API Usage

```bash
# Query municipal RAG
curl -X POST "http://localhost:8001/api/municipal/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Öffnungszeiten Gemeindeverwaltung",
    "municipality": "Arlesheim",
    "category": "services"
  }'

# Get municipal statistics
curl "http://localhost:8001/api/municipal/stats/Arlesheim"

# Get available categories
curl "http://localhost:8001/api/municipal/categories/Arlesheim"
```

## System Architecture

### Components

1. **Municipal Web Scraper** (`municipal_web_scraper.py`)
   - Scrapes Swiss municipal websites
   - Assigns importance scores based on content type
   - Categorizes content (services, administration, events, etc.)

2. **Municipal RAG** (`municipal_rag.py`)
   - Weighted vector search with importance scoring
   - Municipal-specific prompt templates
   - Category-based filtering

3. **API Integration** (`municipal_api_integration.py`)
   - REST endpoints for municipal queries
   - Statistics and health monitoring
   - Multi-municipality support

### Data Flow

```
Municipal Website → Web Scraper → JSON Documents → Municipal RAG → Embeddings + Database → API Endpoints
```

## Configuration

### Supported Municipalities

The system comes pre-configured for these Swiss municipalities:

- **Arlesheim** (BL) - `arlesheim`
- **Basel** (BS) - `basel`
- **Bern** (BE) - `bern`
- **Zürich** (ZH) - `zurich`
- **Geneva** (GE) - `geneva`
- **Lausanne** (VD) - `lausanne`

### Adding New Municipalities

Edit `municipal_setup.py` and add your municipality:

```python
MUNICIPAL_CONFIGS = {
    'your_municipality': {
        'name': 'Your Municipality',
        'url': 'https://www.your-municipality.ch',
        'canton': 'XX',
        'language': 'de'  # or 'fr'
    }
}
```

## Content Prioritization

The system automatically assigns importance scores based on:

### High Priority (1.0)
- Municipal services (`dienstleistungen`)
- Administration (`verwaltung`)
- Official forms (`formulare`)

### Medium Priority (0.8-0.9)
- Municipal council (`gemeinderat`)
- Politics (`politik`)
- Building permits (`bauen`)
- Schools (`schulen`)
- Tax information (`steuern`)

### Lower Priority (0.6-0.7)
- Tourism (`tourismus`)
- Events (`veranstaltungen`)

## Categories

Documents are automatically categorized into:

- **services** - Municipal services and forms
- **administration** - Administrative information
- **news** - Current news and announcements
- **events** - Local events and calendar
- **politics** - Political processes and decisions
- **infrastructure** - Building and planning
- **social** - Social services and healthcare
- **finance** - Financial information and taxes
- **tourism** - Tourism and culture

## Prompt Templates

The system uses specialized prompts for different categories:

### Services Prompt
```
Du bist ein Experte für die Dienstleistungen der Gemeinde {municipality}.
Beantworte Fragen über Gemeindeverwaltung, Formulare, Öffnungszeiten und Dienstleistungen.
```

### Politics Prompt
```
Du bist ein Experte für das politische System der Gemeinde {municipality}.
Beantworte Fragen über Gemeinderat, Abstimmungen, Wahlen und politische Prozesse.
```

### General Prompt
```
Du bist ein Assistent für die Gemeinde {municipality} in der Schweiz.
Du hilfst Bürgern mit Fragen zur Gemeindeverwaltung, Dienstleistungen und lokalen Informationen.
```

## API Endpoints

### Municipal Query
```
POST /api/municipal/query
```
Query the municipal RAG system with specialized knowledge.

**Request:**
```json
{
  "query": "Öffnungszeiten Gemeindeverwaltung",
  "municipality": "Arlesheim",
  "category": "services"
}
```

**Response:**
```json
{
  "answer": "Die Gemeindeverwaltung Arlesheim ist...",
  "sources": [{"title": "...", "url": "...", "category": "services"}],
  "confidence": 0.95,
  "municipality": "Arlesheim",
  "processing_time": 1.2
}
```

### Municipal Statistics
```
GET /api/municipal/stats/{municipality}
```

### Search Content
```
GET /api/municipal/search/{municipality}?query=...&category=...
```

### Available Municipalities
```
GET /api/municipal/available
```

## Business Use Cases

### 1. Municipal Chatbot
Deploy as a chatbot for citizens to ask questions about municipal services.

### 2. Municipal Website Integration
Embed search functionality directly into municipal websites.

### 3. Multi-Municipality Platform
Create a platform serving multiple municipalities with shared infrastructure.

### 4. Municipal Service Directory
Provide intelligent search across all municipal services and forms.

## Performance Optimization

### Weighted Scoring
The system uses weighted similarity scoring:
```python
weighted_score = similarity * (0.7 + 0.3 * importance_score)
```

### Caching
- Document embeddings are cached in SQLite database
- Municipal RAG instances are cached in memory
- Results can be cached for repeated queries

### Batch Processing
- Documents are processed in batches during scraping
- Embeddings are computed efficiently using sentence transformers

## Security Considerations

### Rate Limiting
- Respectful scraping with delays between requests
- API rate limiting on municipal endpoints

### Data Privacy
- Only public municipal information is scraped
- No personal data is collected or stored

### Content Validation
- URLs are validated before scraping
- Content is filtered for relevance

## Monitoring and Maintenance

### Health Checks
```bash
curl "http://localhost:8001/api/municipal/health"
```

### System Statistics
```bash
curl "http://localhost:8001/api/municipal/stats/Arlesheim"
```

### Data Refresh
```bash
# Reload municipal data
curl -X POST "http://localhost:8001/api/municipal/reload/Arlesheim"
```

## Troubleshooting

### Common Issues

1. **Scraping Fails**
   - Check if municipality website is accessible
   - Verify URL in configuration
   - Check for anti-bot measures

2. **No Results**
   - Verify data was scraped successfully
   - Check if documents.json exists in municipal_data/
   - Ensure embedding model is loaded

3. **Poor Answer Quality**
   - Increase scraping depth (max_pages)
   - Check importance scoring
   - Verify prompt templates

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system statistics
setup = MunicipalRagSetup('arlesheim')
stats = setup.get_stats()
print(stats)
```

## Future Enhancements

### Planned Features
- Real-time website monitoring for updates
- Multi-language query support
- Advanced document categorization
- Integration with municipal databases
- Automated content validation

### Business Model
This system is perfect for:
- Municipal IT contractors
- Government software providers
- Digital transformation consultants
- Public service platforms

The AGPL license ensures open-source availability while creating commercial licensing opportunities for proprietary deployments.

## Support

For questions or issues:
- Check the troubleshooting section
- Review system logs
- Test with simple queries first
- Verify all dependencies are installed

---

**Note**: This system is designed for Swiss municipalities but can be adapted for other countries by modifying the scraping logic and prompt templates.