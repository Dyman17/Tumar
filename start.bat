@echo off
echo Starting Pedestrian Tracking System...
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Проверка и установка зависимостей
echo Checking dependencies...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating packages...
pip install -r requirements.txt

echo.
echo Starting web server...
echo Web interface will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

REM Запуск веб-сервера
python web_server.py

pause
