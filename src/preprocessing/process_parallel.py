#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to process Yakut-English parallel corpora from OPUS
# Translates the original Russian-Yakut corpus to English-Yakut for analysis

import os
import logging
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Fix import path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing.text_cleaner import TextCleaner, clean_parallel_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def process_corpus(corpus_name, src_file, tgt_file, output_dir, 
                   src_lang="sah", tgt_lang="en", min_ratio=0.3, max_ratio=3.0,
                   translate_russian_to_english=True, max_lines=None, batch_size=8,
                   output_prefix=None):
    # Process a parallel corpus
    logger.info(f"Processing {corpus_name} parallel corpus")
    
    if translate_russian_to_english:
        logger.info(f"Will translate Russian text to English")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cleaners for source and target languages
    src_cleaner = TextCleaner(
        remove_urls=True,
        remove_html=True,
        normalize_unicode=True,
        remove_extra_spaces=True,
        min_chars=2,
        min_words=1
    )
    
    tgt_cleaner = TextCleaner(
        remove_urls=True,
        remove_html=True,
        normalize_unicode=True,
        remove_extra_spaces=True,
        min_chars=2,
        min_words=1
    )
    
    # Define output file paths
    prefix = output_prefix or corpus_name
    out_src_file = Path(output_dir) / f"{prefix}.{src_lang}"
    out_tgt_file = Path(output_dir) / f"{prefix}.{tgt_lang}"
    
    # Clean parallel data
    total_pairs, kept_pairs = clean_parallel_data(
        src_file=src_file,
        tgt_file=tgt_file,
        out_src_file=out_src_file,
        out_tgt_file=out_tgt_file,
        src_cleaner=src_cleaner,
        tgt_cleaner=tgt_cleaner,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        translate_russian_to_english=translate_russian_to_english
    )
    
    logger.info(f"Processed {total_pairs} sentence pairs, kept {kept_pairs} pairs ({kept_pairs/total_pairs:.2%})")
    return total_pairs, kept_pairs

def parse_xml_corpus(xml_file, src_lang="sah", tgt_lang="en"):
    # Parse XML corpus file to extract parallel sentences
    logger.info(f"Parsing XML file: {xml_file}")
    
    src_sentences = []
    tgt_sentences = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # This is a simplified parser, actual implementation may need to be adjusted
        # based on the specific XML structure of OPUS corpora
        for sentence_pair in tqdm(root.findall(".//s"), desc="Parsing XML"):
            src_text = sentence_pair.find(f".//{src_lang}").text
            # Original Russian text that will be translated to English
            tgt_text = sentence_pair.find(".//ru").text
            
            if src_text and tgt_text:
                src_sentences.append(src_text)
                tgt_sentences.append(tgt_text)
    
    except Exception as e:
        logger.error(f"Error parsing XML: {e}")
        
    logger.info(f"Extracted {len(src_sentences)} sentence pairs from XML")
    return src_sentences, tgt_sentences

def main():
    parser = argparse.ArgumentParser(description="Process Yakut parallel corpora with translation from Russian to English")
    parser.add_argument("--input-dir", type=str, default="data/raw/opus",
                        help="Directory containing the OPUS data")
    parser.add_argument("--output-dir", type=str, default="data/processed/parallel",
                        help="Directory to save processed data")
    parser.add_argument("--corpus", type=str, choices=["wikimedia", "tatoeba", "sample", "all"], default="all",
                        help="Which corpus to process")
    parser.add_argument("--src-lang", type=str, default="sah", help="Source language code")
    parser.add_argument("--tgt-lang", type=str, default="en", help="Target language code (English for translated Russian)")
    parser.add_argument("--translate", action="store_true", help="Translate Russian text to English")
    parser.add_argument("--max-lines", type=int, default=None, help="Maximum number of lines to translate (for testing)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for translation")
    
    # Custom file options for testing
    parser.add_argument("--src-file", type=str, default=None, help="Custom source file path (for testing)")
    parser.add_argument("--tgt-file", type=str, default=None, help="Custom target file path (for testing)")
    parser.add_argument("--output-prefix", type=str, default=None, help="Custom output file prefix (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Custom file processing (for testing)
    if args.src_file and args.tgt_file:
        logger.info(f"Processing custom files: {args.src_file} and {args.tgt_file}")
        
        corpus_name = args.output_prefix or "custom"
        src_file = Path(args.src_file)
        tgt_file = Path(args.tgt_file)
        
        if not src_file.exists() or not tgt_file.exists():
            logger.error("Source or target file not found")
            return
        
        total, kept = process_corpus(
            corpus_name=corpus_name,
            src_file=src_file,
            tgt_file=tgt_file,
            output_dir=args.output_dir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            translate_russian_to_english=args.translate,
            max_lines=args.max_lines,
            batch_size=args.batch_size,
            output_prefix=args.output_prefix
        )
        
        logger.info(f"Custom file processing completed: Processed {total} pairs, kept {kept} pairs")
        return
    
    # Regular corpus processing
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    corpora_to_process = []
    if args.corpus == "sample":
        corpora_to_process.append({
            "name": "sample",
            "src_file": input_dir / f"sample.{args.src_lang}",
            "tgt_file": input_dir / "sample.ru"  # Russian file to be translated to English
        })
    elif args.corpus == "all" or args.corpus == "wikimedia":
        corpora_to_process.append({
            "name": "wikimedia",
            "src_file": input_dir / f"wikimedia.ru-{args.src_lang}.{args.src_lang}",
            "tgt_file": input_dir / f"wikimedia.ru-{args.src_lang}.ru"
        })
    
    if args.corpus == "all" or args.corpus == "tatoeba":
        corpora_to_process.append({
            "name": "tatoeba",
            "src_file": input_dir / f"Tatoeba.ru-{args.src_lang}.{args.src_lang}",
            "tgt_file": input_dir / f"Tatoeba.ru-{args.src_lang}.ru"
        })
    
    total_stats = {"total": 0, "kept": 0}
    
    for corpus in corpora_to_process:
        corpus_name = corpus["name"]
        src_file = corpus["src_file"]
        tgt_file = corpus["tgt_file"]
        
        print(f"Checking for file: {src_file}") # DEBUG
        if not src_file.exists() or not tgt_file.exists():
            logger.warning(f"Source or target file not found for {corpus_name}: {src_file}")
            continue
        
        total, kept = process_corpus(
            corpus_name=corpus_name,
            src_file=src_file,
            tgt_file=tgt_file,
            output_dir=output_dir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            translate_russian_to_english=args.translate,
            max_lines=args.max_lines,
            batch_size=args.batch_size
        )
        
        total_stats["total"] += total
        total_stats["kept"] += kept
    
    if total_stats["total"] > 0:
        logger.info(f"Total statistics: Processed {total_stats['total']} sentence pairs, "
                   f"kept {total_stats['kept']} pairs ({total_stats['kept']/total_stats['total']:.2%})")
    
    logger.info("Parallel corpus processing completed")

if __name__ == "__main__":
    main() 