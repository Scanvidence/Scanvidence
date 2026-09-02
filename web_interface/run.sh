#!/usr/bin/env bash
set -e

PORT=${1:-8000}
echo "=========================================================="
echo " Starting Scanvidence Web Interface (Standalone Server)"
echo " Local URL: http://localhost:${PORT}"
echo " Workstation: http://localhost:${PORT}/workstation"
echo " Benchmarks: http://localhost:${PORT}/benchmarks"
echo "=========================================================="

python3 server.py "$PORT"
