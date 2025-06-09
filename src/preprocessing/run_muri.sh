#!/bin/bash
# Script to run MURI data processing for Yakut and Russian languages

PARALLEL_DIR="project/data/processed/parallel"
OUTPUT_DIR="project/data/murin_instructions"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if .env file exists
ENV_FILE="project/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please create a .env file and set up Google Cloud authentication:"
    echo "GOOGLE_CLOUD_PROJECT=your_project_id_here"
    echo "MODEL_NAME=google/flan-t5-large"
    echo ""
    echo "Also set the environment variable:"
    echo "export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json"
    exit 1
fi

# Check if Google Cloud credentials are set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
    echo "Make sure to set it to point to your service account JSON file:"
    echo "export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json"
fi

echo "Starting MURI processing for Wikimedia data..."
python project/scripts/create_muri_instructions.py \
    --parallel_dir "$PARALLEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --languages sah ru

echo "Starting MURI processing for Tatoeba data..."
python project/scripts/process_tatoeba.py \
    --parallel_dir "$PARALLEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --languages sah ru

echo "MURI processing complete. Results saved to $OUTPUT_DIR"

# Setup Instructions:
# 1. Create a Google Cloud project: https://console.cloud.google.com/
# 2. Enable the Cloud Translation API for your project
# 3. Create a service account and download the JSON key file
# 4. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to point to the JSON file
# 5. Set GOOGLE_CLOUD_PROJECT environment variable to your project ID 