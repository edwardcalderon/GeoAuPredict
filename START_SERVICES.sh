#!/bin/bash
# GeoAuPredict - Start All Services Script
# This script starts all services using Docker Compose V2

echo "🚀 Starting GeoAuPredict Services..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Build and start services
echo "📦 Building and starting services..."
docker compose up -d --build

echo ""
echo "✅ Services starting! Checking status..."
echo ""

# Wait a moment for services to start
sleep 5

# Show status
docker compose ps

echo ""
echo "🌐 Available Services:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📡 REST API:           http://localhost:8000"
echo "  📊 API Docs:           http://localhost:8000/docs"
echo "  🗺️  Spatial Dashboard:  http://localhost:8501"
echo "  📦 3D Visualization:   http://localhost:8502"
echo "  📈 MLflow UI:          http://localhost:5000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 To view logs: docker compose logs -f"
echo "🛑 To stop all:  docker compose down"
echo ""

