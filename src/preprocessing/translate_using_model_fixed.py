#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Translation script using Hugging Face's pre-trained models for Russian to English translation
# Fixed version that handles batches correctly and prevents infinite loops

import argparse
import logging
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class TranslationModel:
    # Class for handling Russian to English translation using a pre-trained model
    
    def __init__(self):
        # Initialize the translation model and tokenizer
        self.model_name = "Helsinki-NLP/opus-mt-ru-en"
        logger.info(f"Loading translation model {self.model_name}...")
        
        # Load model and tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"Model loaded successfully. Using device: {self.device}")
    
    def translate_single(self, text):
        # Translate a single text from Russian to English
        # Handle empty or whitespace-only strings
        if not text or text.isspace():
            return ""
        
        try:
            # Tokenize the text
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate translation
            with torch.no_grad():
                output = self.model.generate(**encoded, max_length=512, num_beams=4, early_stopping=True)
            
            # Decode the generated tokens
            translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return translated_text.strip()
        except Exception as e:
            logger.warning(f"Error translating text: {e}")
            return ""

def translate_file(input_file, output_file, batch_size=8, max_lines=None):
    # Translate each line in a file from Russian to English
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        logger.error(f"Input file {input_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize translation model
    translator = TranslationModel()
    
    # Read all lines
    logger.info(f"Reading lines from {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines()]
    
    # Limit lines if specified
    if max_lines is not None:
        all_lines = all_lines[:max_lines]
        logger.info(f"Limited to {max_lines} lines for testing")
    
    total_lines = len(all_lines)
    logger.info(f"Translating {total_lines} lines")
    
    if total_lines == 0:
        logger.warning("No lines to translate")
        return
    
    # Process line by line to avoid batch complications
    translated_lines = []
    start_time = time.time()
    
    for i, line in enumerate(all_lines):
        translation = translator.translate_single(line)
        translated_lines.append(translation)
        
        # Log progress every 100 lines or at the end
        if (i + 1) % 100 == 0 or (i + 1) == total_lines:
            elapsed = time.time() - start_time
            progress = (i + 1) / total_lines
            remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            logger.info(f"Translated {i+1}/{total_lines} lines "
                       f"({progress:.1%}) - Time elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
    
    # Write translations to output file
    logger.info(f"Writing translations to {output_file}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')
    
    total_time = time.time() - start_time
    logger.info(f"Translation completed: {total_lines} lines in {total_time:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Translate Russian to English (Fixed version)")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to Russian input file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Path to English output file")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                       help="Batch size (not used - processes line by line)")
    parser.add_argument("--max-lines", "-m", type=int, default=None,
                       help="Maximum number of lines to translate (for testing)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting translation from {args.input} to {args.output}")
    
    # Translate the file
    translate_file(args.input, args.output, args.batch_size, args.max_lines)

if __name__ == "__main__":
    main() 