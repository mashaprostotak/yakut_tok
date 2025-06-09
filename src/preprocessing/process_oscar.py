#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to process the OSCAR monolingual corpus for Yakut language
# Extracts, cleans, and normalizes the data

import os
import gzip
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Fix import path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing.text_cleaner import TextCleaner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def extract_oscar_data(gz_file_path, output_file_path):
    # Extract data from gzipped OSCAR file
    logger.info(f"Extracting {gz_file_path} to {output_file_path}")
    
    with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f_in:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc="Extracting"):
                f_out.write(line)
    
    logger.info("Extraction completed")

def process_oscar_data(input_file_path, output_file_path, cleaner=None):
    # Process and clean the OSCAR data
    if cleaner is None:
        cleaner = TextCleaner(
            remove_urls=True,
            remove_html=True,
            normalize_unicode=True,
            remove_extra_spaces=True,
            min_chars=2,
            min_words=2  # Require at least 2 words to filter out very short content
        )
    
    logger.info(f"Processing {input_file_path}")
    total_lines, kept_lines = cleaner.clean_file(input_file_path, output_file_path)
    
    logger.info(f"Processing completed: {kept_lines}/{total_lines} lines kept ({kept_lines/total_lines:.2%})")
    return total_lines, kept_lines

def main():
    parser = argparse.ArgumentParser(description="Process OSCAR monolingual corpus for Yakut")
    parser.add_argument("--input-dir", type=str, default="data/raw/oscar",
                        help="Directory containing the OSCAR data")
    parser.add_argument("--output-dir", type=str, default="data/processed/monolingual",
                        help="Directory to save processed data")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract data without cleaning")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only clean extracted data without extraction")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    gz_file_path = Path(args.input_dir) / "oscar.gz"
    extracted_file_path = Path(args.input_dir) / "oscar"
    processed_file_path = Path(args.output_dir) / "yakut_clean.txt"
    
    if not args.clean_only and (not extracted_file_path.exists() or args.extract_only):
        extract_oscar_data(gz_file_path, extracted_file_path)
    
    if not args.extract_only:
        process_oscar_data(extracted_file_path, processed_file_path)
    
    logger.info("OSCAR data processing completed")

if __name__ == "__main__":
    main() 