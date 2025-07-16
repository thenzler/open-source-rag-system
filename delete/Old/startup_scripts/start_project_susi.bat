@echo off
echo.
echo   Starting Project SUSI - Smart Universal Search Intelligence
echo   ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ Python not found! Please install Python 3.8+ first.
    echo   📥 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo   🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo   ⚠️  No virtual environment found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo   📦 Installing requirements...
    pip install -r simple_requirements.txt
    pip install PyYAML rank-bm25
)

echo   🚀 Starting Project SUSI server...
echo   📱 Interface will open automatically when ready
echo.

REM Start the server and open interface
start /min python simple_api.py
timeout /t 3 /nobreak >nul

REM Open the beautiful interface
start "" project_susi_frontend.html

echo   ✅ Project SUSI is now running!
echo   🌐 Server: http://localhost:8001
echo   💻 Interface: project_susi_frontend.html
echo.
echo   📚 Available commands:
echo   • python manage_llm.py list      - List AI models
echo   • python manage_llm.py status    - Check system status
echo   • python manage_llm.py switch    - Switch AI models
echo.
echo   Press any key to view server logs...
pause >nul

echo   📊 Server Status:
curl -s http://localhost:8001/api/v1/status || echo Server is starting...
echo.
echo   Press Ctrl+C to stop the server.