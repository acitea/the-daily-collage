#!/bin/bash
# Quick start script for The Daily Collage

set -e

echo "================================"
echo "The Daily Collage - Quick Start"
echo "================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the project root"
    exit 1
fi

echo "✓ Project directory verified"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python: $PYTHON_VERSION"

if [ ! -d "backend/server/.venv" ]; then
    echo ""
    echo "📦 Setting up virtual environment..."
    cd backend/server
    uv sync
    cd ../..
    echo "✓ Virtual environment created"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API server, run:"
echo ""
echo "  cd backend/server"
echo "  source .venv/bin/activate"
echo "  uv run python -m uvicorn main:app --reload"
echo ""
echo "Then open your browser to:"
echo "  - http://localhost:8000/docs (API documentation)"
echo "  - http://localhost:8000 (Health check)"
echo ""
echo "To test the full pipeline, run:"
echo ""
echo "  source backend/server/.venv/bin/activate"
echo "  python test_pipeline.py"
echo ""
echo "To fetch real GDELT data, use:"
echo ""
echo "  curl http://localhost:8000/api/visualization/gdelt/sweden"
echo ""
