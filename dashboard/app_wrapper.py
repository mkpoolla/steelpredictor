#!/usr/bin/env python3
"""
Wrapper script for Streamlit app to handle imghdr module issue in Python 3.13+
"""
import sys
import os
from pathlib import Path

# Add the dashboard directory to Python path to make our custom imghdr accessible
DASHBOARD_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(DASHBOARD_DIR))

# Import the Streamlit app
from app import main

if __name__ == "__main__":
    main()
