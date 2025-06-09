# Data Directory

This directory contains all datasets used in the Yakut NLP project.

## Directory Structure

- `raw/`: Original unprocessed data
  - `opus/`: OPUS Sakha-Russian parallel corpus (Wikimedia and Tatoeba)
  - `oscar/`: OSCAR monolingual Yakut text
- `processed/`: Cleaned and preprocessed data
  - `parallel/`: Processed parallel data (Yakut-Russian-English)
  - `monolingual/`: Processed monolingual Yakut data
  - `analysis/`: Data analysis results and statistics
- `muri_instructions/`: MURI-based instruction data
  - Contains JSON files with generated instruction-tuning data for different corpora

## Data Sources

### OPUS Sakha-Russian Corpus
- Source: https://opus.nlpl.eu/results/sah&ru/corpus-result-table
- Description: Parallel corpus containing Sakha-Russian text pairs
- Download instructions: Follow the link and download available corpus files

### OSCAR Corpus
- Source: https://oscar-project.github.io/documentation/versions/oscar-2019/#downloading-oscar
- Description: Monolingual Yakut text extracted from Common Crawl
- Download instructions: Find "Yakut" or "Sakha" language in the table and download

### Data Processing Workflow

1. Download raw data to `raw/` directory
2. Run preprocessing scripts from `src/preprocessing/`
3. Store processed data in `processed/` directory
4. Generate synthetic data and store in `augmented/` directory

## Data Format Guidelines

- Parallel data: Tab-separated or JSON format with source and target text
- Monolingual data: One sentence per line, cleaned text
- Instruction data: JSON format with instruction, input, and response fields

## Data Versioning

- Include version information in directory names when processing data
- Document data processing steps in README files within each data subdirectory 