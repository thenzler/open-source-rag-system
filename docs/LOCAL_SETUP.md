# Local Setup Guide (Without Docker)

This guide helps you run the RAG system directly on your machine without Docker.

## Prerequisites

1. **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
2. **Node.js 16+** - [Download Node.js](https://nodejs.org/)
3. **PostgreSQL 15+** - [Download PostgreSQL](https://www.postgresql.org/download/)
4. **Redis** - [Download Redis for Windows](https://github.com/tporadowski/redis/releases)
5. **Git** - [Download Git](https://git-scm.com/downloads)

## Step-by-Step Setup

### 1. Install PostgreSQL

1. Download and install PostgreSQL from the official website
2. During installation, remember the password for the 'postgres' user
3. After installation, create the database:

```sql
-- Open psql or pgAdmin
CREATE DATABASE ragdb;
CREATE USER raguser WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;
```

### 2. Install Redis

**Option A: Redis for Windows**
```bash
# Download from: https://github.com/tporadowski/redis/releases
# Extract and run redis-server.exe
```

**Option B: Memurai (Redis-compatible)**
```bash
# Download from: https://www.memurai.com/
# Install and it runs as a Windows service
```

### 3. Install Qdrant

```bash
# Download Qdrant binary
mkdir qdrant
cd qdrant
curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-pc-windows-msvc.zip -o qdrant.zip
# Extract the zip file
# Run: qdrant.exe
```

### 4. Install Python Dependencies

```bash
# API Gateway
cd services/api-gateway
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
deactivate

# Document Processor
cd ../document-processor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# For OCR support:
pip install pytesseract
deactivate

# Vector Engine
cd ../vector-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
deactivate
```

### 5. Install Node.js Dependencies

```bash
cd services/web-interface
npm install
```

### 6. Configure Environment

```bash
# Copy the local environment file
copy .env.local .env

# Edit .env and update:
# - DATABASE_URL with your PostgreSQL credentials
# - File paths for your system
```

### 7. Initialize Database

```bash
cd services/api-gateway
venv\Scripts\activate
python -m alembic upgrade head
```

### 8. Install Ollama (for LLM)

1. Download Ollama from: https://ollama.ai/download
2. Install and run Ollama
3. Pull the model:
```bash
ollama pull llama3.1:8b
```

## Running the System

### Option 1: Use the Batch Script

```bash
# First time setup
setup-local.bat

# Start all services
start-local-services.bat
```

### Option 2: Manual Start

Start each service in a separate terminal:

```bash
# Terminal 1: Qdrant
cd qdrant
qdrant.exe

# Terminal 2: Redis
redis-server

# Terminal 3: API Gateway
cd services/api-gateway
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 4: Document Processor
cd services/document-processor
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 5: Celery Worker
cd services/document-processor
venv\Scripts\activate
celery -A app.processor worker --loglevel=info

# Terminal 6: Vector Engine
cd services/vector-engine
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

# Terminal 7: Web Interface
cd services/web-interface
npm start
```

## Verify Installation

1. Check API Gateway: http://localhost:8000/docs
2. Check Qdrant: http://localhost:6333/dashboard
3. Check Web Interface: http://localhost:3000

## Troubleshooting

### Python Module Not Found
```bash
# Make sure you're in the virtual environment
venv\Scripts\activate
pip install -r requirements.txt
```

### PostgreSQL Connection Error
- Check PostgreSQL service is running
- Verify credentials in .env file
- Check firewall settings

### Port Already in Use
```bash
# Find what's using the port (e.g., 8000)
netstat -ano | findstr :8000
# Kill the process using the PID
taskkill /PID <PID> /F
```

### Redis Connection Error
- Make sure Redis server is running
- Check if Redis is listening on port 6379

## Performance Tips

1. **Use SSD**: Store PostgreSQL and Qdrant data on SSD for better performance
2. **Increase RAM**: Allocate more memory to PostgreSQL and Qdrant
3. **GPU Support**: If you have a GPU, change EMBEDDING_DEVICE=cuda in .env
4. **Parallel Processing**: Increase Celery workers for faster document processing

## Development Tips

- Use `--reload` flag with uvicorn for auto-reload during development
- Check logs in each terminal for debugging
- Use PostgreSQL pgAdmin for database management
- Monitor Redis with Redis Commander: `npm install -g redis-commander`