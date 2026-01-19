@echo off
REM YouTube Automation Scheduler Startup Script
REM This script is called by Windows Task Scheduler

cd /d "C:\Users\fkozi\youtube-automation"

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Load environment variables
if exist "config\.env" (
    for /f "tokens=*" %%a in (config\.env) do (
        set "%%a" 2>nul
    )
)

REM Start the scheduler
echo Starting YouTube Automation Scheduler...
echo Started at: %date% %time%
python run.py daily-all

REM Keep window open if error
if errorlevel 1 (
    echo.
    echo ERROR: Scheduler failed to start
    pause
)
