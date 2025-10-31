#!/bin/bash
# Linux/Mac script to build executable

set -e

echo "============================================================"
echo "Building PID Tuner executable"
echo "============================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

python3 --version

# Install dependencies
echo
echo "Installing dependencies..."
pip3 install -r requirements.txt
pip3 install pyinstaller

# Build
echo
echo "Building executable..."
python3 build_exe.py

echo
echo "============================================================"
echo "Done! Check the 'dist' folder for pid_tuner"
echo "============================================================"
