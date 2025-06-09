#!/bin/bash

# Run the preprocessing pipeline with model-based translation
# Usage: ./run_translation.sh [corpus]
#   corpus: "wikimedia", "tatoeba", or "all" (default)

# Set default corpus to "all" if not provided
CORPUS=${1:-all}

echo "Running preprocessing pipeline for corpus: $CORPUS"
echo "Using model-based Russian to English translation"

# Create directories if they don't exist
mkdir -p data/processed/parallel
mkdir -p data/processed/analysis

# Run the preprocessing script
python3 project/src/preprocessing/run_preprocessing.py \
  --process-parallel \
  --translate \
  --parallel-corpus $CORPUS \
  --src-lang sah \
  --tgt-lang en \
  --skip-analysis

echo "Translation completed successfully!"
echo "Translated files are in data/processed/parallel/" 