#!/bin/bash
set -e  

echo "🚀 Setting up RALFS on Google Colab..."

# Install Poetry
echo "📦 Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify Poetry installation before proceeding
# This ensures we fail fast if Poetry installation didn't work correctly
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry installation failed. Please try installing Poetry manually:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    echo "   Or visit: https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "✓ Poetry installed successfully"

# Install dependencies using Poetry
# This installs runtime dependencies by default (not dev dependencies)
# Now that Poetry is verified to be available, we can safely use it
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