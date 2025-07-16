@echo off
echo Setting up RAG System locally (without Docker)...
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

echo === Step 1: Installing PostgreSQL ===
echo Please install PostgreSQL manually from: https://www.postgresql.org/download/windows/
echo After installation, create a database called 'ragdb' and user 'raguser'
echo.
pause

echo === Step 2: Installing Qdrant ===
echo Downloading Qdrant...
if not exist "qdrant" mkdir qdrant
cd qdrant
curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-pc-windows-msvc.zip -o qdrant.zip
tar -xf qdrant.zip
cd ..
echo Qdrant downloaded to ./qdrant directory
echo.

echo === Step 3: Installing Redis ===
echo For Windows, download Redis from: https://github.com/tporadowski/redis/releases
echo Or use Memurai: https://www.memurai.com/get-memurai
echo.
pause

echo === Step 4: Setting up Python environments ===
echo.

REM API Gateway
echo Setting up API Gateway...
cd services\api-gateway
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
deactivate
cd ..\..

REM Document Processor
echo Setting up Document Processor...
cd services\document-processor
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
pip install pytesseract
deactivate
cd ..\..

REM Vector Engine
echo Setting up Vector Engine...
cd services\vector-engine
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
deactivate
cd ..\..

REM Web Interface
echo Setting up Web Interface...
cd services\web-interface
call npm install
cd ..\..

echo.
echo === Setup Complete! ===
echo.
echo Next steps:
echo 1. Start PostgreSQL service
echo 2. Start Redis service
echo 3. Run start-local-services.bat to start all services
echo.
pause