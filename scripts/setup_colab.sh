#!/bin/bash
set -e  

echo "🚀 Setting up RALFS on Google Colab..."

# Install Poetry
echo "📦 Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
echo "📚 Installing dependencies with Poetry..."
poetry install

# Install Spacy model
echo "🧠 Downloading Spacy model..."
poetry run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

# Install ColBERT
echo "🔍 Installing ColBERT..."
poetry run pip install git+https://github.com/stanford-futuredata/ColBERT.git

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed data/index
mkdir -p checkpoints results

echo "✅ Setup complete! Run: poetry run ralfs --help"