@echo off
REM Starts the Verge backend API and frontend dashboard together,
REM then opens the dashboard in your default browser.
REM Double-click this file, or run it from a terminal: start_dashboard.bat

cd /d "%~dp0"

echo Installing/checking dependencies...
python -m pip install -r requirements.txt --quiet

if not exist "data\simulated_traffic.mp4" (
    echo Generating synthetic demo video...
    python scripts\generate_sample_video.py
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
