#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Translation script using Hugging Face's pre-trained models for Russian to English translation
# This avoids API usage and works offline, which prevents timeouts

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
    
    def translate(self, text):
        # Translate a piece of text from Russian to English
        # Skip empty strings
        if not text or text.isspace():
            return text
        
        # Tokenize the text
        encoded = self.tokenizer(text, return_tensors="pt", padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate translation
        with torch.no_grad():
            output = self.model.generate(**encoded)
        
        # Decode the generated tokens
        translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return translated_text
    
    def translate_batch(self, texts, batch_size=8):
        # Translate a batch of texts more efficiently
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Skip empty strings
            batch = [text for text in batch if text and not text.isspace()]
            
            if not batch:
                continue
            
            # Tokenize the batch
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate translations
            with torch.no_grad():
                output = self.model.generate(**encoded)
            
            # Decode the generated tokens
            translations = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
            results.extend(translations)
        
        return results

def translate_file(input_file, output_file, batch_size=8, max_lines=None):
    # Translate each line in a file from Russian to English
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize translation model
    translator = TranslationModel()
    
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines()]
    
    # Limit lines if specified
    if max_lines is not None:
        all_lines = all_lines[:max_lines]
    
    total_lines = len(all_lines)
    logger.info(f"Translating {total_lines} lines from {input_file}")
    
    # Process in batches
    translated_lines = []
    start_time = time.time()
    
    for i in range(0, total_lines, batch_size):
        batch = all_lines[i:i+batch_size]
        translations = translator.translate_batch(batch, batch_size)
        translated_lines.extend(translations)
        
        # Log progress
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= total_lines:
            elapsed = time.time() - start_time
            progress = min((i + batch_size), total_lines) / total_lines
            remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            logger.info(f"Translated {min(i+batch_size, total_lines)}/{total_lines} lines "
                       f"({progress:.1%}) - Time elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
    
    # Write translations to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')
    
    total_time = time.time() - start_time
    logger.info(f"Translation completed: {total_lines} lines in {total_time:.1f}s "
               f"({total_lines/total_time:.1f} lines/sec)")

def main():
    parser = argparse.ArgumentParser(description="Translate Russian to English using a pre-trained model")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to Russian input file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Path to English output file")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                       help="Number of sentences to translate at once")
    parser.add_argument("--max-lines", "-m", type=int, default=None,
                       help="Maximum number of lines to translate (for testing)")
    
    args = parser.parse_args()
    
    # Translate the file
    translate_file(args.input, args.output, args.batch_size, args.max_lines)

if __name__ == "__main__":
    main() 