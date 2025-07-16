@echo off
echo Installing faster models for better RAG performance...
echo.

echo [1/3] Installing Phi3-Mini (very fast, small model)...
ollama pull phi3-mini

echo.
echo [2/3] Installing Llama3.2 1B (ultra fast)...
ollama pull llama3.2:1b

echo.
echo [3/3] Installing Llama3.2 3B (fast, better quality)...
ollama pull llama3.2:3b

echo.
echo Installation complete! Your RAG system will now use faster models.
echo Restart your server with: python simple_api.py
echo.
pause