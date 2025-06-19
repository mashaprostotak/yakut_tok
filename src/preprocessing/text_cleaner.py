

import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import time

from src.preprocessing.translate_using_model import TranslationModel

class TextCleaner:
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_html: bool = True,
                 normalize_unicode: bool = True,
                 remove_extra_spaces: bool = True,
                 min_chars: int = 2,
                 min_words: int = 1):
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode
        self.remove_extra_spaces = remove_extra_spaces
        self.min_chars = min_chars
        self.min_words = min_words
        
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.extra_spaces_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
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
        if len(line) < self.min_chars:
            return False
            
        if len(line.split()) < self.min_words:
            return False
            
        return True
    
    def clean_file(self, 
                   input_file: Union[str, Path], 
                   output_file: Union[str, Path], 
                   encoding: str = 'utf-8') -> Tuple[int, int]:
        in_path = Path(input_file)
        out_path = Path(output_file)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        total = 0
        kept = 0
        
        with open(in_path, 'r', encoding=encoding) as f_in:
            with open(out_path, 'w', encoding=encoding) as f_out:
                for line in f_in:
                    total += 1
                    clean_line = self.clean_text(line)
                    
                    if self.is_valid_line(clean_line):
                        f_out.write(clean_line + '\n')
                        kept += 1
        
        return total, kept

translator_model = None

def translate_russian_to_english(text: str) -> str:
    global translator_model
    
    if translator_model is None:
        translator_model = TranslationModel()
    
    return translator_model.translate(text)

def clean_parallel_data(src_file: Union[str, Path], 
                        tgt_file: Union[str, Path],
                        out_src_file: Union[str, Path],
                        out_tgt_file: Union[str, Path],
                        src_cleaner: TextCleaner,
                        tgt_cleaner: TextCleaner,
                        min_ratio: float = 0.3,
                        max_ratio: float = 3.0,
                        translate_russian_to_english: bool = False) -> Tuple[int, int]:
    src_path = Path(src_file)
    tgt_path = Path(tgt_file)
    out_src_path = Path(out_src_file)
    out_tgt_path = Path(out_tgt_file)
    
    out_src_path.parent.mkdir(parents=True, exist_ok=True)
    out_tgt_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    kept = 0
    
    if translate_russian_to_english:
        global translator_model
        if translator_model is None:
            translator_model = TranslationModel()
    
    with open(src_path, 'r', encoding='utf-8') as f_src, \
         open(tgt_path, 'r', encoding='utf-8') as f_tgt, \
         open(out_src_path, 'w', encoding='utf-8') as f_out_src, \
         open(out_tgt_path, 'w', encoding='utf-8') as f_out_tgt:
        
        src_lines = []
        tgt_lines = []
        batch_size = 8
        
        if translate_russian_to_english:
            src_lines = [src_cleaner.clean_text(line) for line in f_src]
            tgt_lines = [tgt_cleaner.clean_text(line) for line in f_tgt]
            total = len(src_lines)
            
            good_pairs = [(i, s, t) for i, (s, t) in enumerate(zip(src_lines, tgt_lines)) 
                        if src_cleaner.is_valid_line(s) and tgt_cleaner.is_valid_line(t)]
            
            translated_stuff = []
            
            for i in range(0, len(good_pairs), batch_size):
                batch = good_pairs[i:i+batch_size]
                batch_indices = [idx for idx, _, _ in batch]
                batch_texts = [t for _, _, t in batch]
                
                translated_batch = translator_model.translate_batch(batch_texts)
                
                for j, (idx, src, _) in enumerate(batch):
                    translated_text = translated_batch[j] if j < len(translated_batch) else ""
                    translated_stuff.append((idx, src, translated_text))
            
            for idx, src, tgt in translated_stuff:
                src_len = len(src.split())
                tgt_len = len(tgt.split())
                
                if src_len == 0 or tgt_len == 0:
                    continue
                    
                ratio = src_len / tgt_len
                if ratio < min_ratio or ratio > max_ratio:
                    continue
                    
                f_out_src.write(src + '\n')
                f_out_tgt.write(tgt + '\n')
                kept += 1
        
        else:
            for src_line, tgt_line in zip(f_src, f_tgt):
                total += 1
                
                clean_src = src_cleaner.clean_text(src_line)
                clean_tgt = tgt_cleaner.clean_text(tgt_line)
                
                if not src_cleaner.is_valid_line(clean_src) or not tgt_cleaner.is_valid_line(clean_tgt):
                    continue
                
                src_len = len(clean_src.split())
                tgt_len = len(clean_tgt.split())
                
                if src_len == 0 or tgt_len == 0:
                    continue
                    
                ratio = src_len / tgt_len
                if ratio < min_ratio or ratio > max_ratio:
                    continue
                    
                f_out_src.write(clean_src + '\n')
                f_out_tgt.write(clean_tgt + '\n')
                kept += 1
    
    return total, kept


if __name__ == "__main__":
    yakut_cleaner = TextCleaner(
        remove_urls=True,
        remove_html=True,
        normalize_unicode=True,
        remove_extra_spaces=True,
        min_chars=2,
        min_words=1
    )
    
