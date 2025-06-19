import argparse
import logging
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer
import time


class TranslationModel:
    # class for handling Russian to English translation using a pre-trained model
    
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-ru-en"
        
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    
    def translate_single(self, text):
        # translate a single text from Russian to English
        if not text or text.isspace():
            return ""
        
        try:
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                output = self.model.generate(**encoded, max_length=512, num_beams=4, early_stopping=True)
            
            translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return translated_text.strip()
        except Exception as e:
            return ""

def translate_file(input_file, output_file, batch_size=8, max_lines=None):
    # translate each line in a file from Russian to English
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    translator = TranslationModel()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines()]
    
    if max_lines is not None:
        all_lines = all_lines[:max_lines]
    
    total_lines = len(all_lines)
    
    if total_lines == 0:
        return
    
    translated_lines = []
    start_time = time.time()
    
    for i, line in enumerate(all_lines):
        translation = translator.translate_single(line)
        translated_lines.append(translation)
        
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in translated_lines:
            f.write(line + '\n')
    
    total_time = time.time() - start_time

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
    
    
    translate_file(args.input, args.output, args.batch_size, args.max_lines)

if __name__ == "__main__":
    main() 