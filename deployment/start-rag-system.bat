@echo off
echo Starting Open Source RAG System...
echo.

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop first.
    echo.
    echo Steps:
    echo 1. Open Docker Desktop
    echo 2. Wait for it to fully start
    echo 3. Run this script again
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env file and set:
    echo - POSTGRES_PASSWORD
    echo - SECRET_KEY
    echo - JWT_SECRET_KEY
    echo.
    pause
)

REM Start the services
echo Starting all services with Docker Compose...
docker-compose up -d

echo.
echo Waiting for services to start...
timeout /t 10

REM Check service health
echo.
echo Checking service status...
docker-compose ps

echo.
echo RAG System should be starting up!
echo.
echo Access points:
echo - API Gateway: http://localhost:8000
echo - Web Interface: http://localhost:3000
echo - Qdrant UI: http://localhost:6333/dashboard
echo - Grafana: http://localhost:3001 (admin/admin)
echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down
echo.
pause