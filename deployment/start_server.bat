@echo off
echo Starting RAG System Server...
echo.
echo Dependencies check...
python startup_checks.py
if %errorlevel% neq 0 (
    echo.
    echo [FAIL] Dependencies check failed!
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies check passed!
echo.
echo Starting server at http://localhost:8001
echo Press Ctrl+C to stop the server
echo.

cd /d "C:\Users\THE\open-source-rag-system"
python simple_api.py

pause