@echo off
echo Searching for Ollama installation...
echo.

echo Checking common installation paths...
if exist "C:\Program Files\Ollama\ollama.exe" (
    echo Found Ollama at: C:\Program Files\Ollama\ollama.exe
    set OLLAMA_PATH=C:\Program Files\Ollama\ollama.exe
    goto :found
)

if exist "C:\Program Files (x86)\Ollama\ollama.exe" (
    echo Found Ollama at: C:\Program Files (x86)\Ollama\ollama.exe
    set OLLAMA_PATH=C:\Program Files (x86)\Ollama\ollama.exe
    goto :found
)

if exist "%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe" (
    echo Found Ollama at: %USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe
    set OLLAMA_PATH=%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe
    goto :found
)

echo Ollama not found in common locations.
echo Please install Ollama from: https://ollama.com/download
echo.
pause
exit

:found
echo.
echo Testing Ollama...
"%OLLAMA_PATH%" --version
echo.
echo Installing fast models...
"%OLLAMA_PATH%" pull phi3-mini
"%OLLAMA_PATH%" pull llama3.2:1b
"%OLLAMA_PATH%" pull llama3.2:3b
echo.
echo Installation complete!
pause