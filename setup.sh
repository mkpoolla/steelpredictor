#!/bin/bash
# Setup script for Streamlit Cloud deployment

# Create necessary directories for data and models
mkdir -p data
mkdir -p models
mkdir -p alerts

# Create empty placeholder files to ensure directories are committed to git
touch data/.gitkeep
touch models/.gitkeep
touch alerts/.gitkeep

# Add execute permissions to scripts
chmod +x scripts/*.py

# Make setup.sh executable
chmod +x setup.sh

# Initial setup is complete
echo "Setup completed successfully!"

