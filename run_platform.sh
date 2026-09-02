#!/usr/bin/env bash
# Scanvidence 1-Click Platform Launcher

set -e

PORT=8000
echo "==============================================================="
echo "   SCANVIDENCE — HYBRID QUANTUM-CLASSICAL CLINICAL PLATFORM   "
echo "==============================================================="
echo "Starting FastAPI Server on http://localhost:$PORT ..."
echo ""

# Check python environment
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

# Run the server
$PYTHON_BIN server.py
