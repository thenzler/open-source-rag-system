@echo off
echo ===================================
echo Kanboard Setup für Swiss RAG Project
echo ===================================

:: Check if PHP is installed
php --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ PHP ist nicht installiert!
    echo Bitte PHP 7.4+ installieren von: https://www.php.net/downloads
    pause
    exit /b 1
)

echo ✅ PHP gefunden!

:: Download Kanboard
echo.
echo Downloading Kanboard...
curl -L https://github.com/kanboard/kanboard/archive/v1.2.32.zip -o kanboard.zip

:: Extract
echo Extracting...
tar -xf kanboard.zip

:: Move to project directory
move kanboard-1.2.32 kanboard
cd kanboard

:: Create initial database
echo.
echo Setting up database...
if not exist data mkdir data

:: Create config file
echo ^<?php > config.php
echo. >> config.php
echo // Kanboard Configuration >> config.php
echo define('DB_DRIVER', 'sqlite'); >> config.php
echo define('ENABLE_URL_REWRITE', true); >> config.php
echo define('DEFAULT_LOCALE', 'de_DE'); >> config.php
echo. >> config.php

:: Start server
echo.
echo ===================================
echo ✅ Kanboard ist bereit!
echo ===================================
echo.
echo Öffne: http://localhost:8000
echo.
echo Standard Login:
echo Username: admin
echo Password: admin
echo.
echo Drücke Ctrl+C zum Beenden
echo ===================================
echo.

:: Start PHP server
php -S localhost:8000

pause