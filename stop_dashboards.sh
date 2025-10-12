#!/bin/bash

# GeoAuPredict Dashboard Stop Script
# This script stops both Streamlit and Dash dashboards

echo "🛑 Stopping GeoAuPredict Dashboards..."
echo "======================================"

# Try to stop using saved PIDs first
if [ -f .dashboard_pids ]; then
    echo "📋 Reading saved process IDs..."
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "   Stopping process $pid..."
            kill $pid
        fi
    done < .dashboard_pids
    rm .dashboard_pids
    sleep 2
fi

# Kill any remaining dashboard processes
echo "🧹 Cleaning up dashboard processes..."
pkill -f "streamlit run.*spatial_validation_dashboard.py"
pkill -f "python.*3d_visualization_dashboard.py"

sleep 2

# Verify all processes are stopped
REMAINING=$(ps aux | grep -E '(streamlit|dash).*dashboard' | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✅ All dashboard processes stopped"
else
    echo "⚠️  Some processes may still be running"
    echo "   Run: ps aux | grep dashboard"
fi

echo ""
echo "======================================"
echo "✅ Dashboards stopped"
echo "======================================"

