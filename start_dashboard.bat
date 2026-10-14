@echo off
REM Starts the Verge backend API and frontend dashboard together,
REM then opens the dashboard in your default browser.
REM Double-click this file, or run it from a terminal: start_dashboard.bat

cd /d "%~dp0"

echo Installing/checking dependencies...
python -m pip install -r requirements.txt --quiet
REM Optional GPU acceleration (DirectML). Failure here is harmless: the CPU path is used.
python -m pip install -r requirements-gpu.txt --quiet >nul 2>&1

if not exist "data\simulated_traffic.mp4" (
    echo Generating synthetic demo video...
    python scripts\generate_sample_video.py
)

if not exist "data\demo_clips\crossing_near_miss.mp4" (
    echo Generating synthetic demo clips for the "Analyze a video" tab...
    python scripts\generate_demo_clips.py
)

if not exist "outputs\hotspots.geojson" (
    echo Running the pipeline on demo data...
    python scripts\run_pipeline.py
)

echo Starting API server on http://localhost:8000 ...
start "Verge API" cmd /k python -m uvicorn backend.app.main:app --port 8000

echo Starting dashboard server on http://localhost:5500 ...
start "Verge Dashboard" cmd /k python -m http.server 5500 --directory frontend

timeout /t 2 /nobreak >nul
start http://localhost:5500

echo.
echo Two windows just opened (API + dashboard servers) - leave them running.
echo Your browser should open to http://localhost:5500 automatically.
echo Close those windows, or press Ctrl+C in each, to stop the servers.
pause
