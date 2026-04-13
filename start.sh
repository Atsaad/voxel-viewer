#!/bin/bash
# Voxel Viewer — Startup Script
# Creates a virtual environment if needed, installs dependencies, and launches the GUI.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🎲 Starting Voxel Viewer..."
echo ""

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -e ".[gui]"
else
    source venv/bin/activate
fi

echo ""
python -m voxel_viewer --gui
