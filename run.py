#!/usr/bin/env python3
"""
Entry point for Real-Time Workout Form Analyzer

This script adds the src/ directory to the Python path and runs main.py
"""

import sys
import os
from pathlib import Path

# Change to project root for relative path references
os.chdir(Path(__file__).parent)

# Add src/ to Python path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

# Now import and run main
from main import main

if __name__ == "__main__":
    sys.exit(main())
