

import os
import gzip
import argparse
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing.text_cleaner import TextCleaner

def extract_oscar_data(gz_file_path, output_file_path):
    with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f_in:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc="Extracting"):
                f_out.write(line)

def process_oscar_data(input_file_path, output_file_path, cleaner=None):
    if cleaner is None:
        cleaner = TextCleaner(
            remove_urls=True,
            remove_html=True,
            normalize_unicode=True,
            remove_extra_spaces=True,
            min_chars=2,
            min_words=2
        )
    
    total_lines, kept_lines = cleaner.clean_file(input_file_path, output_file_path)
    return total_lines, kept_lines

def main():
    parser = argparse.ArgumentParser(description="Process OSCAR monolingual corpus")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--lang", type=str, default="sah")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    gz_file = input_dir / "oscar"
    if not gz_file.exists():
        gz_file = input_dir / "oscar.gz"
    
    if not gz_file.exists():
        return
    
    raw_file = output_dir / f"{args.lang}_raw.txt"
    clean_file = output_dir / f"{args.lang}_clean.txt"
    
    if gz_file.suffix == '.gz':
        extract_oscar_data(gz_file, raw_file)
        input_file = raw_file
    else:
        input_file = gz_file
    
    total, kept = process_oscar_data(input_file, clean_file)

if __name__ == "__main__":
    main() 