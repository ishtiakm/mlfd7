#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, please install Anaconda or Miniconda first."
    exit
fi

# Create or update the conda environment
if [ -f environment.yml ]; then
    echo "Creating or updating the conda environment from environment.yml..."
    conda env update -f environment.yml --prune
else
    echo "No environment.yml found, exporting the current environment..."
    conda env export --no-builds > environment.yml
    conda env create -f environment.yml
fi

# Activate the environment
echo "Activating the environment 'mlfd7'..."
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate mlfd7; then
    echo "Environment 'mlfd7' activated successfully."
else
    echo "Failed to activate the 'mlfd7' environment."
    exit 1
fi

# Install pip packages (if requirements.txt is provided)
if [ -f requirements.txt ]; then
    echo "Installing additional pip packages from requirements.txt..."
    pip install -r requirements.txt
fi

# Run the Python script
echo "Environment Setup is complete"
