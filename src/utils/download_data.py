#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to download Yakut (Sakha) language resources from OPUS and OSCAR

import argparse
import os
import sys
import logging
import requests
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Define constants
OPUS_BASE_URL = "https://opus.nlpl.eu/download.php"
OSCAR_BASE_URL = "https://oscar-public.huma-num.fr/shuff-orig"

def download_file(url, destination, chunk_size=8192):
    # Download a file from URL to destination with progress bar
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {os.path.basename(destination)} successfully")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def download_opus_data(output_dir):
    # Download Yakut-Russian parallel data from OPUS
    output_dir = Path(output_dir) / "opus"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define corpus IDs and names to download
    # These need to be updated with actual corpus IDs available for Sakha-Russian
    corpus_list = [
        {"name": "bible-uedin", "params": {"s": "sah", "t": "ru"}},
        {"name": "QED", "params": {"s": "sah", "t": "ru"}},
        # Add more corpus sources as they become available
    ]
    
    for corpus in corpus_list:
        corpus_name = corpus["name"]
        params = corpus["params"]
        params["d"] = corpus_name
        
        logger.info(f"Downloading {corpus_name} corpus")
        
        # Constructing download URL
        url = f"{OPUS_BASE_URL}?{corpus_name}/v1/moses/sah-ru.txt.zip"
        output_path = output_dir / f"{corpus_name}_sah-ru.zip"
        
        download_file(url, output_path)

def download_oscar_data(output_dir, language_code="sah"):
    # Download Yakut monolingual data from OSCAR
    output_dir = Path(output_dir) / "oscar"
    os.makedirs(output_dir, exist_ok=True)
    
    # Note: This URL needs to be verified, as OSCAR might require authentication
    url = f"{OSCAR_BASE_URL}/{language_code}"
    output_path = output_dir / f"oscar_{language_code}.txt"
    
    logger.info(f"Downloading OSCAR corpus for language {language_code}")
    logger.warning("Note: OSCAR might require authentication through HumanID")
    logger.info(f"Please visit {url} in your browser and download manually if this fails")
    
    download_file(url, output_path)

def main():
    parser = argparse.ArgumentParser(description="Download Yakut language resources")
    parser.add_argument("--output-dir", type=str, default="./data/raw", 
                        help="Output directory for downloaded data")
    parser.add_argument("--opus", action="store_true", help="Download OPUS data")
    parser.add_argument("--oscar", action="store_true", help="Download OSCAR data")
    parser.add_argument("--all", action="store_true", help="Download all available data")
    
    args = parser.parse_args()
    
    if not (args.opus or args.oscar or args.all):
        parser.print_help()
        sys.exit(1)
    
    if args.all or args.opus:
        logger.info("Downloading OPUS Yakut-Russian parallel data")
        download_opus_data(args.output_dir)
    
    if args.all or args.oscar:
        logger.info("Downloading OSCAR Yakut monolingual data")
        download_oscar_data(args.output_dir)
    
    logger.info("Download completed")

if __name__ == "__main__":
    main() 