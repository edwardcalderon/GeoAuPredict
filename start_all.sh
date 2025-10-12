#!/bin/bash

# Start all services for GeoAuPredict development
echo "🚀 Starting GeoAuPredict services..."

# Activate virtual environment
source venv/bin/activate

# Start Streamlit in background
echo "📊 Starting Streamlit dashboard on port 8501..."
streamlit run src/app/spatial_validation_dashboard.py --server.port 8501 --server.headless true > /dev/null 2>&1 &
STREAMLIT_PID=$!

# Start Dash in background
echo "🎮 Starting Dash 3D visualization on port 8050..."
python src/app/3d_visualization_dashboard.py > /dev/null 2>&1 &
DASH_PID=$!

# Wait for dashboards to start
echo "⏳ Waiting for Python dashboards to initialize..."
sleep 3

# Start Next.js
echo "🌐 Starting Next.js frontend on port 3000..."
npm run dev

# Cleanup on exit
echo "🛑 Stopping all services..."
kill $STREAMLIT_PID $DASH_PID 2>/dev/null

