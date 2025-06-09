#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to run the full pipeline for Russian to English translation and analysis
# This script:
# 1. Translates the Russian-Yakut parallel data to English-Yakut
# 2. Creates tokenizer samples
# 3. Runs the analysis on the English data

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def run_preprocessing_with_translation(base_dir, corpora=None):
    # Run the preprocessing pipeline with Russian to English translation
    logger.info("Starting preprocessing with Russian to English translation")
    
    # Construct command for preprocessing with translation
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "run_preprocessing.py"),
        "--parallel-only",
        "--translate-to-english"
    ]
    
    # Add specific corpora if provided
    if corpora:
        cmd.extend(["--corpora"] + corpora)
    
    logger.info("Running command: " + " ".join(cmd))
    
    # Run the preprocessing script
    try:
        subprocess.run(cmd, check=True)
        logger.info("Preprocessing with translation completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("Error during preprocessing: " + str(e))
        return False
    
    return True

def run_english_analysis(base_dir):
    # Run the analysis on the translated English data
    logger.info("Starting analysis of English translated data")
    
    # Construct command for English analysis
    cmd = [
        sys.executable,
        str(base_dir / "src" / "preprocessing" / "analysis" / "analyze_english_data.py")
    ]
    
    logger.info("Running command: " + " ".join(cmd))
    
    # Run the analysis script
    try:
        subprocess.run(cmd, check=True)
        logger.info("English data analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("Error during analysis: " + str(e))
        return False
    
    return True

def compare_english_russian_analysis(base_dir):
    # Compare the English and Russian analysis results
    logger.info("Starting comparison of English and Russian analysis results")
    
    # Here you would add code to load the analysis results from both languages
    # and create comparative visualizations and statistics
    
    # For demonstration purposes, we'll just log a message
    logger.info("English-Russian analysis comparison would be implemented here")
    
    # Example of what this function could do:
    # 1. Load the statistics from both analyses
    # 2. Create side-by-side plots comparing key metrics
    # 3. Generate a report with statistical tests of differences
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Russian to English translation and analysis pipeline")
    parser.add_argument("--base-dir", type=str, default=str(Path(__file__).parent.parent),
                      help="Base project directory")
    parser.add_argument("--corpora", type=str, nargs="+", choices=["wikimedia", "tatoeba"],
                      help="Specify which parallel corpora to process")
    parser.add_argument("--translation-only", action="store_true",
                      help="Only run the translation step without analysis")
    parser.add_argument("--analysis-only", action="store_true",
                      help="Only run the analysis on already translated data")
    parser.add_argument("--compare", action="store_true",
                      help="Compare English and Russian analysis results")
    
    args = parser.parse_args()
    
    # Convert base directory to Path
    base_dir = Path(args.base_dir)
    
    # Run steps based on arguments
    success = True
    
    if not args.analysis_only:
        # Run preprocessing with translation
        success = run_preprocessing_with_translation(base_dir, args.corpora)
        if not success:
            logger.error("Translation step failed, stopping pipeline")
            return 1
    
    if not args.translation_only and success:
        # Run analysis on translated data
        success = run_english_analysis(base_dir)
        if not success:
            logger.error("Analysis step failed")
            return 1
    
    if args.compare and success:
        # Compare English and Russian analysis results
        success = compare_english_russian_analysis(base_dir)
        if not success:
            logger.error("Comparison step failed")
            return 1
    
    if success:
        logger.info("English translation and analysis pipeline completed successfully")
        return 0
    else:
        logger.error("Pipeline failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 