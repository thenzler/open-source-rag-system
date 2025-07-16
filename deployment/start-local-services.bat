@echo off
echo Starting RAG System Services Locally...
echo.

REM Start Qdrant
echo Starting Qdrant Vector Database...
start "Qdrant" cmd /k "cd qdrant && qdrant.exe"
timeout /t 5

REM Start Redis (assuming Redis is installed)
echo Starting Redis...
start "Redis" cmd /k "redis-server"
timeout /t 3

REM Start API Gateway
echo Starting API Gateway...
start "API Gateway" cmd /k "cd services\api-gateway && venv\Scripts\activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 5

REM Start Document Processor
echo Starting Document Processor...
start "Document Processor" cmd /k "cd services\document-processor && venv\Scripts\activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload"
timeout /t 3

REM Start Celery Worker
echo Starting Celery Worker...
start "Celery Worker" cmd /k "cd services\document-processor && venv\Scripts\activate && celery -A app.processor worker --loglevel=info"
timeout /t 3

REM Start Vector Engine
echo Starting Vector Engine...
start "Vector Engine" cmd /k "cd services\vector-engine && venv\Scripts\activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload"
timeout /t 3

REM Start Web Interface
echo Starting Web Interface...
start "Web Interface" cmd /k "cd services\web-interface && npm start"

echo.
echo All services starting...
echo.
echo Services will be available at:
echo - API Gateway: http://localhost:8000
echo - Document Processor: http://localhost:8001
echo - Vector Engine: http://localhost:8002
echo - Web Interface: http://localhost:3000
echo - Qdrant: http://localhost:6333
echo.
echo Press Ctrl+C in each window to stop services
pause