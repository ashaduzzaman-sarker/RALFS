#!/bin/bash
# ============================================================================
# External Dependencies Configuration
# This file contains versions for external dependencies used across
# GitHub Actions, Dockerfile, and other installation scripts.
# ============================================================================

# Spacy model configuration
export SPACY_MODEL_VERSION="3.7.1"
export SPACY_MODEL_NAME="en_core_web_sm"
export SPACY_MODEL_URL="https://github.com/explosion/spacy-models/releases/download/${SPACY_MODEL_NAME}-${SPACY_MODEL_VERSION}/${SPACY_MODEL_NAME}-${SPACY_MODEL_VERSION}.tar.gz"

# ColBERT repository
export COLBERT_REPO_URL="git+https://github.com/stanford-futuredata/ColBERT.git"

# Function to install Spacy model
install_spacy_model() {
    echo "Installing Spacy model: ${SPACY_MODEL_NAME} v${SPACY_MODEL_VERSION}"
    poetry run pip install "${SPACY_MODEL_URL}"
}

# Function to install ColBERT
install_colbert() {
    echo "Installing ColBERT from: ${COLBERT_REPO_URL}"
    poetry run pip install "${COLBERT_REPO_URL}" || echo "ColBERT installation skipped"
}
