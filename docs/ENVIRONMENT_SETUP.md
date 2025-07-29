# Environment Configuration Guide

This guide explains how to configure the RAG System using environment variables for different deployment scenarios.

## 🚀 Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your specific values

3. **Source the environment** (if needed):
   ```bash
   source .env  # On Linux/macOS
   # or load in your deployment system
   ```

## 📋 Configuration Categories

### 🔧 Core System Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SYSTEM_BASE_DIR` | `C:/Users/THE/open-source-rag-system` | Base directory for the system |
| `RAG_DATA_DIR` | `./data` | Data storage directory |
| `RAG_CONFIG_DIR` | `./config` | Configuration files directory |
| `ENVIRONMENT` | `development` | Environment type (development/production) |

### 🗄️ Database Configuration

#### SQLite (Development)
```env
DATABASE_TYPE=sqlite
DATABASE_PATH=data/rag_database.db
AUDIT_DATABASE_PATH=data/audit.db
```

#### PostgreSQL (Production)
```env
DATABASE_TYPE=postgresql
USE_POSTGRESQL=true
DATABASE_URL=postgresql://user:password@host:5432/dbname
DB_POOL_SIZE=10
DB_POOL_MIN_SIZE=2
```

#### Connection Pooling
- `DB_POOL_SIZE`: Maximum pool size (default: 10)
- `DB_POOL_MIN_SIZE`: Minimum pool size (default: 2)
- `DB_POOL_TIMEOUT`: Connection timeout in seconds
- `DB_POOL_RECYCLE`: Connection recycle time in seconds

### 🔒 Security Configuration

#### Core Security
```env
SECRET_KEY=your-256-bit-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
BCRYPT_ROUNDS=12
```

#### Encryption at Rest
```env
ENCRYPTION_ENABLED=true
ENCRYPTION_KEY_FILE=data/encryption.key
ENCRYPTION_ALGORITHM=Fernet
```

#### ID Obfuscation
```env
ID_OBFUSCATION_ENABLED=true
ID_OBFUSCATION_SALT=your-obfuscation-salt
```

#### CSRF Protection
```env
CSRF_PROTECTION_ENABLED=true
CSRF_TOKEN_EXPIRY_HOURS=24
```

### 🏢 Multi-Tenancy Configuration

#### Basic Multi-Tenancy
```env
ENABLE_MULTI_TENANCY=true
DEFAULT_TENANT=default
TENANT_ISOLATION_LEVEL=strict
```

#### Tenant Resolution
```env
# Resolve tenants from subdomain (tenant.yourdomain.com)
TENANT_RESOLUTION_STRATEGY=subdomain

# Or from custom domain (tenant-domain.com)
TENANT_RESOLUTION_STRATEGY=domain

# Or from HTTP header
TENANT_RESOLUTION_STRATEGY=header
TENANT_HEADER_NAME=X-Tenant-ID

# Or from URL path (/tenant/tenant-name/...)
TENANT_RESOLUTION_STRATEGY=path
TENANT_PATH_PREFIX=/tenant
```

#### Tenant Features
```env
TENANT_SPECIFIC_ENCRYPTION=true
TENANT_RESOURCE_QUOTAS=true
DEFAULT_DOCUMENT_LIMIT=1000
DEFAULT_QUERY_LIMIT_PER_HOUR=1000
```

### 🤖 LLM Configuration

#### Ollama (Default)
```env
OLLAMA_HOST=http://localhost:11434
LLM_MODEL_NAME=llama3.1:8b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000
LLM_TIMEOUT_SECONDS=60
```

#### Alternative LLM Providers
```env
# vLLM
LLM_PROVIDER=vllm
VLLM_MODEL_PATH=/models/llama-3.1-8b-instruct

# OpenAI (for testing)
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key
```

### 🔍 Embedding Configuration

```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DEVICE=cpu  # or cuda
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_LENGTH=512
```

#### Model Options
- **English**: `sentence-transformers/all-mpnet-base-v2`
- **Multilingual**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Code**: `microsoft/codebert-base`

### 📊 Monitoring Configuration

#### Prometheus Metrics
```env
ENABLE_METRICS=true
METRICS_PORT=8001
TRACK_QUERY_METRICS=true
TRACK_DOCUMENT_METRICS=true
TRACK_LLM_METRICS=true
```

#### System Monitoring
```env
ENABLE_SYSTEM_METRICS=true
CPU_MONITORING_INTERVAL=5
MEMORY_MONITORING_INTERVAL=5
DISK_MONITORING_INTERVAL=30
```

#### Grafana
```env
GRAFANA_ADMIN_PASSWORD=admin123
GRAFANA_PORT=3000
GRAFANA_AUTO_PROVISION_DASHBOARDS=true
```

### 💾 Backup Configuration

```env
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
```

#### Backup Components
```env
BACKUP_DATABASES=true
BACKUP_DOCUMENTS=true
BACKUP_CONFIGURATION=true
BACKUP_VECTOR_INDICES=true
```

### 🇨🇭 Swiss Data Protection Compliance

```env
ENABLE_DATA_PROTECTION_MODE=true
DATA_RESIDENCY_REGION=CH
ENABLE_RIGHT_TO_DELETION=true
ENABLE_DATA_EXPORT=true
```

#### Data Retention
```env
PERSONAL_DATA_RETENTION_DAYS=2555  # 7 years
QUERY_LOG_RETENTION_DAYS=365
AUDIT_LOG_RETENTION_DAYS=2555
```

#### Privacy Settings
```env
ANONYMIZE_IP_ADDRESSES=true
HASH_USER_IDENTIFIERS=true
ENABLE_PRIVACY_BY_DESIGN=true
```

## 🏗️ Deployment Scenarios

### Development Environment
```env
ENVIRONMENT=development
DEBUG=true
DATABASE_TYPE=sqlite
ENABLE_MULTI_TENANCY=false
ENCRYPTION_ENABLED=false
LOG_LEVEL=DEBUG
```

### Staging Environment
```env
ENVIRONMENT=staging
DEBUG=false
DATABASE_TYPE=postgresql
ENABLE_MULTI_TENANCY=true
ENCRYPTION_ENABLED=true
LOG_LEVEL=INFO
```

### Production Environment
```env
ENVIRONMENT=production
DEBUG=false
DATABASE_TYPE=postgresql
USE_POSTGRESQL=true
ENABLE_MULTI_TENANCY=true
ENCRYPTION_ENABLED=true
ENABLE_DATA_PROTECTION_MODE=true
LOG_LEVEL=WARNING
BACKUP_ENABLED=true
```

## 🔐 Security Best Practices

### 1. Generate Strong Keys
```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 2. Environment-Specific Settings
- **Development**: Use SQLite, disable encryption for faster iteration
- **Production**: Use PostgreSQL, enable all security features
- **Testing**: Use separate test database, enable debug features

### 3. Secrets Management
- Never commit `.env` files to version control
- Use secret management services (AWS Secrets Manager, Azure Key Vault)
- Rotate keys regularly
- Use different keys for each environment

## 🚨 Common Issues

### Database Connection Issues
```env
# Check these settings
DATABASE_URL=postgresql://user:pass@host:port/db
DB_POOL_SIZE=10  # Reduce if connection errors
DB_POOL_TIMEOUT=30  # Increase if slow connections
```

### Performance Issues
```env
# Increase pool sizes
DB_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=100

# Optimize processing
WORKER_CONCURRENCY=8
EMBEDDING_BATCH_SIZE=64
```

### Memory Issues
```env
# Reduce memory usage
EMBEDDING_BATCH_SIZE=16
CHUNK_SIZE=256
DB_POOL_SIZE=5
```

## 📝 Environment Validation

The system automatically validates critical environment variables on startup:

- **Required**: `SECRET_KEY`, `DATABASE_TYPE`
- **Security**: Warns if using default values in production
- **Compatibility**: Checks for conflicting settings
- **Resources**: Validates file paths and network endpoints

## 🔄 Environment Updates

When updating environment variables:

1. **Test in development** first
2. **Backup current configuration**
3. **Update gradually** (one service at a time)
4. **Monitor logs** for errors
5. **Have rollback plan** ready

## 📞 Support

For environment configuration issues:
1. Check logs at the configured `LOG_FILE` location
2. Verify all required variables are set
3. Test database connectivity
4. Validate file permissions
5. Check network access to external services

---

**Security Note**: Always review the `.env.example` file for the latest configuration options and security requirements before deploying to production.