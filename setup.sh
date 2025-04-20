#!/bin/bash
# Setup script for the ingredient standardization application

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 could not be found. Please install Python 3.8 or newer."
    exit 1
fi

# Check for python3-venv or python3-full
if ! python3 -m venv --help &> /dev/null; then
    echo "Python venv module not found. Installing python3-full..."
    sudo apt-get update
    sudo apt-get install -y python3-full
fi

# Set up the virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create the resources directory if it doesn't exist
mkdir -p resources

# Extract resources from the notebook if available
NOTEBOOK_PATH="/home/recipes/ingred_cleaning_python/ingredient-standardization-via-machine-translation.ipynb"
if [ -f "$NOTEBOOK_PATH" ]; then
    echo "Extracting resources from notebook..."
    python app/data_utils.py --notebook "$NOTEBOOK_PATH" --output resources
fi

# Print usage instructions
echo ""
echo "Setup complete! To run the application:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the API server:"
echo "   python app/main.py"
echo ""
echo "The API will be available at http://localhost:8000"
echo "API documentation will be available at http://localhost:8000/docs"
echo ""
echo "To deactivate the virtual environment when finished:"
echo "   deactivate"
