@echo off
echo Starting Simple RAG System...
echo ===============================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements
echo Installing requirements...
pip install -r simple_requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

REM Create directories
if not exist "storage" mkdir storage
if not exist "storage\uploads" mkdir storage\uploads
if not exist "storage\processed" mkdir storage\processed

REM Start the API
echo Starting Simple RAG API...
echo Access the web interface at: http://localhost:8001/simple_frontend.html
echo API documentation at: http://localhost:8001/docs
echo Press Ctrl+C to stop the server
echo.

python simple_api.py

pause