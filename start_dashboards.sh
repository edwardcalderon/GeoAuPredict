#!/bin/bash

# GeoAuPredict Dashboard Startup Script
# This script starts both Streamlit and Dash dashboards in the background

echo "🚀 Starting GeoAuPredict Dashboards..."
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Check and install web requirements
echo "📦 Checking web dependencies..."
if [ -f "web_requirements.txt" ]; then
    echo "   Installing/updating from web_requirements.txt..."
    pip install -q -r web_requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install web requirements"
        echo "   Trying verbose installation..."
        pip install -r web_requirements.txt
        if [ $? -ne 0 ]; then
            echo "❌ Critical: Could not install requirements"
            exit 1
        fi
    fi
else
    echo "⚠️  web_requirements.txt not found, installing core packages..."
    pip install -q streamlit plotly pandas numpy dash
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install core packages"
        exit 1
    fi
fi

# Verify critical packages are installed
echo "🔍 Verifying critical packages..."
python -c "import streamlit, plotly, pandas, numpy, dash" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Critical packages missing. Installing..."
    pip install streamlit plotly pandas numpy dash
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install critical packages"
        exit 1
    fi
fi
echo "✅ All dependencies verified"

# Kill any existing dashboard processes
echo "🧹 Cleaning up existing processes..."
pkill -f "streamlit run.*spatial_validation_dashboard.py" 2>/dev/null
pkill -f "python.*3d_visualization_dashboard.py" 2>/dev/null

# Wait a moment for processes to clean up
sleep 2

# Start Streamlit dashboard in background
echo "🌐 Starting Streamlit Dashboard (port 8501)..."
nohup streamlit run src/app/spatial_validation_dashboard.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Wait a moment
sleep 2

# Start Dash dashboard in background
echo "🎮 Starting Dash 3D Dashboard (port 8050)..."
nohup python src/app/3d_visualization_dashboard.py \
    > logs/dash.log 2>&1 &
DASH_PID=$!

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 5

# Check if processes are running
if ps -p $STREAMLIT_PID > /dev/null 2>&1; then
    echo "✅ Streamlit Dashboard running (PID: $STREAMLIT_PID)"
    echo "   📍 URL: http://localhost:8501"
else
    echo "❌ Failed to start Streamlit Dashboard"
    echo "   📋 Check logs: logs/streamlit.log"
fi

if ps -p $DASH_PID > /dev/null 2>&1; then
    echo "✅ Dash 3D Dashboard running (PID: $DASH_PID)"
    echo "   📍 URL: http://localhost:8050"
else
    echo "❌ Failed to start Dash Dashboard"
    echo "   📋 Check logs: logs/dash.log"
fi

echo ""
echo "======================================"
echo "🎉 Dashboards are running!"
echo "======================================"
echo ""
echo "📊 Access dashboards at:"
echo "   • Streamlit: http://localhost:8501"
echo "   • Dash 3D:   http://localhost:8050"
echo "   • Next.js:   http://localhost:3000/dashboards"
echo ""
echo "📋 View logs:"
echo "   • Streamlit: tail -f logs/streamlit.log"
echo "   • Dash:      tail -f logs/dash.log"
echo ""
echo "🛑 To stop dashboards:"
echo "   ./stop_dashboards.sh"
echo ""
echo "Process IDs saved to .dashboard_pids"
echo "$STREAMLIT_PID" > .dashboard_pids
echo "$DASH_PID" >> .dashboard_pids

