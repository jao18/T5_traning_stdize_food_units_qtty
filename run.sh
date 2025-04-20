#!/bin/bash
# Run script for the ingredient standardization application

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment and run the application
source venv/bin/activate
python app/main.py
