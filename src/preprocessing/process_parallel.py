import os
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing.text_cleaner import TextCleaner, clean_parallel_data

def process_corpus(corpus_name, src_file, tgt_file, output_dir, 
                   src_lang="sah", tgt_lang="en", min_ratio=0.3, max_ratio=3.0,
                   translate_russian_to_english=True, max_lines=None, batch_size=8,
                   output_prefix=None):
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    prefix = output_prefix or corpus_name
    out_src_file = Path(output_dir) / f"{prefix}.{src_lang}"
    out_tgt_file = Path(output_dir) / f"{prefix}.{tgt_lang}"
    
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
    
    return total_pairs, kept_pairs

def parse_xml_corpus(xml_file, src_lang="sah", tgt_lang="en"):
    src_sentences = []
    tgt_sentences = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for sentence_pair in tqdm(root.findall(".//s"), desc="Parsing XML"):
            src_text = sentence_pair.find(f".//{src_lang}").text
            tgt_text = sentence_pair.find(".//ru").text
            
            if src_text and tgt_text:
                src_sentences.append(src_text)
                tgt_sentences.append(tgt_text)
    
    except Exception as e:
        print(f"Error parsing XML: {e}")
        
    print(f"Extracted {len(src_sentences)} sentence pairs from XML")
    return src_sentences, tgt_sentences

def main():
    parser = argparse.ArgumentParser(description="Process parallel corpus data")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--corpus", type=str, choices=["tatoeba", "wikimedia", "all", "sample"], default="all")
    parser.add_argument("--src-lang", type=str, default="sah")
    parser.add_argument("--tgt-lang", type=str, default="en")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--src-file", type=str, default=None)
    parser.add_argument("--tgt-file", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.src_file and args.tgt_file:
        corpus_name = args.output_prefix or "custom"
        src_file = Path(args.src_file)
        tgt_file = Path(args.tgt_file)
        
        if not src_file.exists() or not tgt_file.exists():
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
        return
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    corpora_to_process = []
    if args.corpus == "sample":
        corpora_to_process.append({
            "name": "sample",
            "src_file": input_dir / f"sample.{args.src_lang}",
            "tgt_file": input_dir / "sample.ru"
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
    
    for corpus in corpora_to_process:
        if corpus["src_file"].exists() and corpus["tgt_file"].exists():
            total, kept = process_corpus(
                corpus_name=corpus["name"],
                src_file=corpus["src_file"],
                tgt_file=corpus["tgt_file"],
                output_dir=output_dir,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                translate_russian_to_english=args.translate,
                max_lines=args.max_lines,
                batch_size=args.batch_size
            )

if __name__ == "__main__":
    main() 