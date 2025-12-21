# ============================================================================
# Dockerfile for RALFS - Retrieval-Augmented Long-Form Summarization
# ============================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Install Spacy model
RUN poetry run pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

# Install ColBERT (optional, may be skipped if causing issues)
RUN poetry run pip install git+https://github.com/stanford-futuredata/ColBERT.git || echo "ColBERT installation skipped"

# Copy application code
COPY . .

# Install the package
RUN poetry install --no-interaction --no-ansi

# Create necessary directories
RUN mkdir -p data/raw data/processed data/index results outputs checkpoints

# Expose port for potential API server (if implemented)
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["poetry", "run", "ralfs"]

# Default command (show help)
CMD ["--help"]
