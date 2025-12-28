#!/bin/bash
# TC-DrugRAG Setup Script
# Run this after cloning the repository

set -e

echo "=============================================="
echo "TC-DrugRAG Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ required. Found: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

echo "✓ Virtual environment created"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/{raw,processed,external/{string,disgenet,drugbank,pubmed,pharmgkb,clinicaltrials,ttd}}
mkdir -p experiments/results
mkdir -p checkpoints

echo "✓ Directories created"

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file (please add your API keys)"
else
    echo "✓ .env file already exists"
fi

# Download freely available data
echo ""
echo "Downloading freely available data..."
echo "(This may take a few minutes)"

# STRING database
echo ""
echo "Downloading STRING database..."
python scripts/download_data.py --source string

# TTD for evaluation
echo ""
echo "Downloading TTD database..."
python scripts/download_data.py --source ttd

# Process STRING
echo ""
echo "Processing STRING data..."
python scripts/download_data.py --source string --process

echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add your API keys to .env:"
echo "   - OPENAI_API_KEY (for GPT-4)"
echo "   - ANTHROPIC_API_KEY (for Claude)"
echo ""
echo "3. Download additional data (optional):"
echo "   python scripts/download_data.py --source pubmed --pubmed-query 'drug repurposing'"
echo ""
echo "4. For full dataset, register at:"
echo "   - DisGeNET: https://www.disgenet.org/signup/"
echo "   - DrugBank: https://go.drugbank.com/public_users/sign_up"
echo ""
echo "5. Run the demo:"
echo "   jupyter notebook notebooks/demo.ipynb"
echo ""
echo "=============================================="
