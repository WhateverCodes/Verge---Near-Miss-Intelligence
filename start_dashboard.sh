#!/usr/bin/env bash
# Starts the Verge backend API and frontend dashboard together, then
# opens the dashboard in your default browser.
# Run: ./start_dashboard.sh   (chmod +x start_dashboard.sh first if needed)
set -e
cd "$(dirname "$0")"

echo "Installing/checking dependencies..."
python3 -m pip install -q -r requirements.txt

if [ ! -f "data/simulated_traffic.mp4" ]; then
    echo "Generating synthetic demo video..."
    python3 scripts/generate_sample_video.py
fi

if [ ! -f "outputs/hotspots.geojson" ]; then
    echo "Running the pipeline on demo data..."
    python3 scripts/run_pipeline.py
fi

cleanup() {
    echo "Stopping servers..."
    kill "$API_PID" "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting API server on http://localhost:8000 ..."
python3 -m uvicorn backend.app.main:app --port 8000 &
API_PID=$!

echo "Starting dashboard server on http://localhost:5500 ..."
python3 -m http.server 5500 --directory frontend &
WEB_PID=$!

sleep 2

URL="http://localhost:5500"
if command -v open >/dev/null 2>&1; then
    open "$URL"            # macOS
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"        # Linux
else
    echo "Open $URL in your browser."
fi

echo ""
echo "Servers running. Press Ctrl+C to stop both."
wait
