#!/bin/bash
echo "Starting Open Source RAG System..."
echo

# Check if Docker is running
if ! docker version &> /dev/null; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker first."
    echo
    echo "On Windows: Start Docker Desktop"
    echo "On Linux: sudo systemctl start docker"
    echo "On Mac: Start Docker Desktop"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo
    echo "IMPORTANT: Please edit .env file and set:"
    echo "- POSTGRES_PASSWORD"
    echo "- SECRET_KEY" 
    echo "- JWT_SECRET_KEY"
    echo
    read -p "Press enter to continue after editing .env..."
fi

# Start the services
echo "Starting all services with Docker Compose..."
docker-compose up -d

echo
echo "Waiting for services to start..."
sleep 10

# Check service health
echo
echo "Checking service status..."
docker-compose ps

echo
echo "RAG System should be starting up!"
echo
echo "Access points:"
echo "- API Gateway: http://localhost:8000"
echo "- Web Interface: http://localhost:3000"
echo "- Qdrant UI: http://localhost:6333/dashboard"
echo "- Grafana: http://localhost:3001 (admin/admin)"
echo
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"