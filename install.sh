#!/bin/bash

# Copyright (c) 2025 sylcrala | sylcrala.xyz
# Licensed under the MIT License - see LICENSE.md for details

# eBay CrossLister - Installation Wrapper for Unix/Linux/macOS
# Cross-platform installer that works on Unix/Linux/macOS (via Git Bash)

set -e

echo "  eBay CrossLister v1.0.0 - Universal Installer"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo " Python not found. Please install Python 3.8+ first."
        echo "   Download from: https://www.python.org/downloads/"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo " Using Python: $PYTHON_CMD"
echo " Launching cross-platform installer..."
echo ""

# Pass all arguments to the Python installer
exec $PYTHON_CMD install.py "$@"