#!/usr/bin/env python3
"""
Main entry point for Streamlit Cloud deployment
This file is automatically detected by Streamlit Cloud and run as the entry point
"""
import sys
import os
from pathlib import Path

# First, make sure our project root is in the Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our custom patch and install it
from streamlit_patch import ImghdrFinder

# Add our custom imghdr finder to sys.meta_path
if any(isinstance(finder, ImghdrFinder) for finder in sys.meta_path) == False:
    sys.meta_path.insert(0, ImghdrFinder())

# Import and run the main application
from dashboard.app import main

# This is what Streamlit Cloud will run
main()
