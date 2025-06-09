#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Text cleaning utilities for Yakut NLP project

import re
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import time

# Import the model-based translator
from src.preprocessing.translate_using_model import TranslationModel

logger = logging.getLogger(__name__)

class TextCleaner:
    # Class for cleaning and normalizing text data
    # Specifically designed for Yakut (Sakha) language text
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_html: bool = True,
                 normalize_unicode: bool = True,
                 remove_extra_spaces: bool = True,
                 min_chars: int = 2,
                 min_words: int = 1):
        # Initialize the text cleaner with configurable options
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode
        self.remove_extra_spaces = remove_extra_spaces
        self.min_chars = min_chars
        self.min_words = min_words
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.extra_spaces_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        # Apply all cleaning steps to a single text string
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        if self.remove_html:
            text = self.html_pattern.sub(' ', text)
            
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
            
        if self.remove_extra_spaces:
            text = self.extra_spaces_pattern.sub(' ', text)
            text = text.strip()
            
        return text
    
    def is_valid_line(self, line: str) -> bool:
        # Check if a line meets the minimum quality criteria
        if len(line) < self.min_chars:
            return False
            
        if len(line.split()) < self.min_words:
            return False
            
        return True
    
    def clean_file(self, 
                   input_file: Union[str, Path], 
                   output_file: Union[str, Path], 
                   encoding: str = 'utf-8') -> Tuple[int, int]:
        # Clean an entire text file line by line
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_lines = 0
        kept_lines = 0
        
        with open(input_path, 'r', encoding=encoding) as in_f:
            with open(output_path, 'w', encoding=encoding) as out_f:
                for line in in_f:
                    total_lines += 1
                    clean_line = self.clean_text(line)
                    
                    if self.is_valid_line(clean_line):
                        out_f.write(clean_line + '\n')
                        kept_lines += 1
        
        logger.info(f"Processed {total_lines} lines, kept {kept_lines} lines ({kept_lines/total_lines:.2%})")
        return total_lines, kept_lines

# Initialize the translator model (will be lazy-loaded only when needed)
_translator = None

def translate_russian_to_english(text: str) -> str:
    # Translate Russian text to English using the Hugging Face model
    global _translator
    
    # Initialize the translator if it's not already loaded
    if _translator is None:
        logger.info("Initializing translation model for the first time")
        _translator = TranslationModel()
    
    # Translate the text
    return _translator.translate(text)

def clean_parallel_data(src_file: Union[str, Path], 
                        tgt_file: Union[str, Path],
                        out_src_file: Union[str, Path],
                        out_tgt_file: Union[str, Path],
                        src_cleaner: TextCleaner,
                        tgt_cleaner: TextCleaner,
                        min_ratio: float = 0.3,
                        max_ratio: float = 3.0,
                        translate_russian_to_english: bool = False) -> Tuple[int, int]:
    # Clean parallel data files, filtering out sentence pairs that don't meet criteria
    src_path = Path(src_file)
    tgt_path = Path(tgt_file)
    out_src_path = Path(out_src_file)
    out_tgt_path = Path(out_tgt_file)
    
    # Create output directories if they don't exist
    out_src_path.parent.mkdir(parents=True, exist_ok=True)
    out_tgt_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_pairs = 0
    kept_pairs = 0
    
    # Initialize translator if needed
    if translate_russian_to_english:
        global _translator
        if _translator is None:
            logger.info("Initializing translation model")
            _translator = TranslationModel()
    
    with open(src_path, 'r', encoding='utf-8') as src_f, \
         open(tgt_path, 'r', encoding='utf-8') as tgt_f, \
         open(out_src_path, 'w', encoding='utf-8') as out_src_f, \
         open(out_tgt_path, 'w', encoding='utf-8') as out_tgt_f:
        
        # Collect lines for batch processing if translating
        src_lines = []
        tgt_lines = []
        batch_size = 8
        
        if translate_russian_to_english:
            # Read all lines for batch processing
            logger.info("Reading all lines for batch translation")
            src_lines = [src_cleaner.clean_text(line) for line in src_f]
            tgt_lines = [tgt_cleaner.clean_text(line) for line in tgt_f]
            total_pairs = len(src_lines)
            
            # Filter invalid lines
            valid_pairs = [(i, s, t) for i, (s, t) in enumerate(zip(src_lines, tgt_lines)) 
                        if src_cleaner.is_valid_line(s) and tgt_cleaner.is_valid_line(t)]
            
            # Translate in batches
            logger.info(f"Translating {len(valid_pairs)} valid pairs")
            translated_pairs = []
            
            for i in range(0, len(valid_pairs), batch_size):
                batch = valid_pairs[i:i+batch_size]
                batch_indices = [idx for idx, _, _ in batch]
                batch_tgt_texts = [t for _, _, t in batch]
                
                # Translate the batch
                translated_batch = _translator.translate_batch(batch_tgt_texts)
                
                # Store the translations with their indices
                for j, (idx, src, _) in enumerate(batch):
                    translated_text = translated_batch[j] if j < len(translated_batch) else ""
                    translated_pairs.append((idx, src, translated_text))
                
                # Log progress
                if (i + batch_size) >= len(valid_pairs) or (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Translated {min(i+batch_size, len(valid_pairs))}/{len(valid_pairs)} pairs")
            
            # Process the translated pairs
            for idx, src, tgt in translated_pairs:
                # Check length ratio
                src_len = len(src.split())
                tgt_len = len(tgt.split())
                
                if src_len == 0 or tgt_len == 0:
                    continue
                    
                ratio = src_len / tgt_len
                if ratio < min_ratio or ratio > max_ratio:
                    continue
                    
                # Write to output files
                out_src_f.write(src + '\n')
                out_tgt_f.write(tgt + '\n')
                kept_pairs += 1
        
        else:
            # Process line by line without translation
            for src_line, tgt_line in zip(src_f, tgt_f):
                total_pairs += 1
                
                clean_src = src_cleaner.clean_text(src_line)
                clean_tgt = tgt_cleaner.clean_text(tgt_line)
                
                if not src_cleaner.is_valid_line(clean_src) or not tgt_cleaner.is_valid_line(clean_tgt):
                    continue
                
                # Check length ratio
                src_len = len(clean_src.split())
                tgt_len = len(clean_tgt.split())
                
                if src_len == 0 or tgt_len == 0:
                    continue
                    
                ratio = src_len / tgt_len
                if ratio < min_ratio or ratio > max_ratio:
                    continue
                    
                # Write to output files
                out_src_f.write(clean_src + '\n')
                out_tgt_f.write(clean_tgt + '\n')
                kept_pairs += 1
                
                # Log progress occasionally
                if kept_pairs % 1000 == 0:
                    logger.info(f"Processed {total_pairs} pairs, kept {kept_pairs} pairs so far")
    
    logger.info(f"Processed {total_pairs} sentence pairs, kept {kept_pairs} pairs ({kept_pairs/total_pairs:.2%})")
    return total_pairs, kept_pairs


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a text cleaner for Yakut
    yakut_cleaner = TextCleaner(
        remove_urls=True,
        remove_html=True,
        normalize_unicode=True,
        remove_extra_spaces=True,
        min_chars=2,
        min_words=1
    )
    
    # Example monolingual cleaning
    # yakut_cleaner.clean_file("data/raw/oscar/yakut.txt", "data/processed/monolingual/yakut_clean.txt")
    
    # Example parallel cleaning
    # russian_cleaner = TextCleaner(min_chars=2, min_words=1)
    # clean_parallel_data(
    #     "data/raw/opus/parallel.sah", 
    #     "data/raw/opus/parallel.ru",
    #     "data/processed/parallel/parallel.sah",
    #     "data/processed/parallel/parallel.ru",
    #     yakut_cleaner,
    #     russian_cleaner
    # ) 