# Quick Start Guide

## Prerequisites

1. **Docker Desktop** must be installed and running
2. **Git** (to clone the repository)
3. At least 8GB of RAM available
4. 10GB of free disk space

## Step 1: Start Docker Desktop

**Windows/Mac:**
- Open Docker Desktop application
- Wait for the Docker icon to show "Docker Desktop is running"

**Linux:**
```bash
sudo systemctl start docker
```

## Step 2: Configure Environment

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` file and set these required values:
```
POSTGRES_PASSWORD=your_secure_password_here
SECRET_KEY=generate_a_random_32_char_string
JWT_SECRET_KEY=another_random_32_char_string
```

## Step 3: Start the System

**Windows:**
```cmd
start-rag-system.bat
```

**Mac/Linux:**
```bash
chmod +x start-rag-system.sh
./start-rag-system.sh
```

**Or manually:**
```bash
docker-compose up -d
```

## Step 4: Verify Services

Check that all services are running:
```bash
docker-compose ps
```

All services should show as "Up".

## Step 5: Download LLM Model

The system needs to download the LLM model on first run:

```bash
# Connect to Ollama container
docker exec -it open-source-rag-system-ollama-1 bash

# Pull the model (this may take 10-30 minutes)
ollama pull llama3.1:8b
```

## Step 6: Access the System

- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Default Login**: admin / admin123

## Troubleshooting

### Docker not running
- Make sure Docker Desktop is fully started
- On Windows, check the system tray for Docker icon
- Try restarting Docker Desktop

### Port conflicts
If you get port already in use errors:
```bash
# Stop all containers
docker-compose down

# Check what's using the ports
netstat -ano | findstr :8000
netstat -ano | findstr :3000
```

### Services failing to start
Check the logs:
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api-gateway
docker-compose logs document-processor
```

### Reset everything
```bash
# Stop and remove all containers
docker-compose down -v

# Start fresh
docker-compose up -d
```

## Next Steps

1. Upload your first document:
   - Go to http://localhost:3000
   - Click "Upload Document"
   - Select a PDF, Word, or text file

2. Test the search:
   - Enter a query about your document
   - View the results with source attribution

3. Explore the API:
   - Visit http://localhost:8000/docs
   - Try the interactive API documentation