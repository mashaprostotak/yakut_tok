#!/usr/bin/env python3
"""
Analyze parallel data for the Yakut NLP project.
This script compares the source (Yakut) and target (English) languages
in the parallel corpus.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, output_dir):
        self.terminal = sys.stdout
        self.log_file = open(os.path.join(output_dir, 'parallel_analysis_report.txt'), 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def load_parallel_data(src_file, tgt_file, max_lines=None):
    """Load parallel data from source and target files."""
    src_lines = []
    tgt_lines = []
    
    with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
        for i, (src_line, tgt_line) in enumerate(zip(src_f, tgt_f)):
            if max_lines is not None and i >= max_lines:
                break
            src_lines.append(src_line.strip())
            tgt_lines.append(tgt_line.strip())
    
    logger.info(f"Loaded {len(src_lines)} parallel sentences")
    
    return pd.DataFrame({
        'src_text': src_lines,
        'tgt_text': tgt_lines
    })

def analyze_length_ratios(df, src_lang, tgt_lang):
    """Analyze length ratios between source and target languages."""
    # Calculate lengths
    df['src_word_len'] = df['src_text'].apply(lambda x: len(x.split()))
    df['tgt_word_len'] = df['tgt_text'].apply(lambda x: len(x.split()))
    df['src_char_len'] = df['src_text'].apply(len)
    df['tgt_char_len'] = df['tgt_text'].apply(len)
    
    # Calculate ratios
    df['word_ratio'] = df['tgt_word_len'] / df['src_word_len']
    df['char_ratio'] = df['tgt_char_len'] / df['src_char_len']
    
    # Replace infinity and NaN with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Print statistics
    print(f"\nLength ratio statistics ({tgt_lang}/{src_lang}):")
    print("\nWord count ratio:")
    print(f"Mean: {df['word_ratio'].mean():.2f}")
    print(f"Median: {df['word_ratio'].median():.2f}")
    print(f"Min: {df['word_ratio'].min():.2f}")
    print(f"Max: {df['word_ratio'].max():.2f}")
    
    print("\nCharacter count ratio:")
    print(f"Mean: {df['char_ratio'].mean():.2f}")
    print(f"Median: {df['char_ratio'].median():.2f}")
    print(f"Min: {df['char_ratio'].min():.2f}")
    print(f"Max: {df['char_ratio'].max():.2f}")
    
    # Print example sentences
    print("\nExample sentences with length information:")
    for i in range(min(5, len(df))):
        print(f"\nExample {i+1}:")
        print(f"{src_lang}: {df['src_text'].iloc[i]}")
        print(f"Word count: {df['src_word_len'].iloc[i]}")
        print(f"Char count: {df['src_char_len'].iloc[i]}")
        print(f"{tgt_lang}: {df['tgt_text'].iloc[i]}")
        print(f"Word count: {df['tgt_word_len'].iloc[i]}")
        print(f"Char count: {df['tgt_char_len'].iloc[i]}")
        print(f"Word ratio: {df['word_ratio'].iloc[i]:.2f}")
        print(f"Char ratio: {df['char_ratio'].iloc[i]:.2f}")
    
    return df

def create_ratio_plots(df, output_dir, src_lang, tgt_lang):
    """Create plots for length ratios."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create histograms for word and character ratios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Word ratio histogram
    sns.histplot(df['word_ratio'].clip(0, 5), bins=30, kde=True, ax=ax1)
    ax1.set_title(f'Word Count Ratio ({tgt_lang}/{src_lang})')
    ax1.set_xlabel('Ratio')
    ax1.set_ylabel('Frequency')
    
    # Character ratio histogram
    sns.histplot(df['char_ratio'].clip(0, 5), bins=30, kde=True, ax=ax2)
    ax2.set_title(f'Character Count Ratio ({tgt_lang}/{src_lang})')
    ax2.set_xlabel('Ratio')
    ax2.set_ylabel('Frequency')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parallel_ratio_histogram.png'), dpi=300)
    plt.close()
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['src_word_len'], df['tgt_word_len'], alpha=0.5)
    
    # Add a line of best fit
    z = np.polyfit(df['src_word_len'], df['tgt_word_len'], 1)
    p = np.poly1d(z)
    plt.plot(df['src_word_len'], p(df['src_word_len']), 'r--', 
             label=f"Best fit: y = {z[0]:.2f}x + {z[1]:.2f}")
    
    plt.title(f'Word Count Correlation: {src_lang} vs {tgt_lang}')
    plt.xlabel(f'{src_lang} Word Count')
    plt.ylabel(f'{tgt_lang} Word Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'parallel_word_correlation.png'), dpi=300)
    plt.close()
    
    logger.info(f"Created ratio plots in {output_dir}")

def analyze_vocabulary_overlap(df, output_dir, src_lang, tgt_lang):
    """Analyze vocabulary overlap between languages."""
    # Extract all words
    src_words = []
    tgt_words = []
    
    for text in df['src_text']:
        src_words.extend(text.lower().split())
    
    for text in df['tgt_text']:
        tgt_words.extend(text.lower().split())
    
    # Count unique words
    src_vocab = set(src_words)
    tgt_vocab = set(tgt_words)
    
    # Calculate statistics
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    src_total_words = len(src_words)
    tgt_total_words = len(tgt_words)
    
    # Print vocabulary statistics
    print(f"\nVocabulary statistics:")
    print(f"{src_lang} vocabulary size: {src_vocab_size} unique words")
    print(f"{tgt_lang} vocabulary size: {tgt_vocab_size} unique words")
    print(f"{src_lang} vocabulary diversity: {src_vocab_size/src_total_words:.4f}")
    print(f"{tgt_lang} vocabulary diversity: {tgt_vocab_size/tgt_total_words:.4f}")
    
    # Most common words
    src_word_counts = pd.Series(src_words).value_counts()
    tgt_word_counts = pd.Series(tgt_words).value_counts()
    
    print(f"\nMost common {src_lang} words:")
    for word, count in src_word_counts.head(10).items():
        print(f"  {word}: {count}")
    
    print(f"\nMost common {tgt_lang} words:")
    for word, count in tgt_word_counts.head(10).items():
        print(f"  {word}: {count}")
    
    # Save vocabulary statistics
    vocab_stats = {
        'language': [src_lang, tgt_lang],
        'vocab_size': [src_vocab_size, tgt_vocab_size],
        'total_words': [src_total_words, tgt_total_words],
        'diversity': [src_vocab_size/src_total_words, tgt_vocab_size/tgt_total_words]
    }
    
    vocab_df = pd.DataFrame(vocab_stats)
    vocab_df.to_csv(os.path.join(output_dir, 'parallel_vocab_stats.csv'), index=False)
    
    # Save top words
    src_top_df = pd.DataFrame({
        'word': src_word_counts.head(1000).index,
        'frequency': src_word_counts.head(1000).values
    })
    src_top_df.to_csv(os.path.join(output_dir, f'top_words_{src_lang}.csv'), index=False)
    
    tgt_top_df = pd.DataFrame({
        'word': tgt_word_counts.head(1000).index,
        'frequency': tgt_word_counts.head(1000).values
    })
    tgt_top_df.to_csv(os.path.join(output_dir, f'top_words_{tgt_lang}.csv'), index=False)
    
    logger.info(f"Analyzed vocabulary overlap and saved statistics")

def main():
    parser = argparse.ArgumentParser(description="Analyze parallel corpus data")
    parser.add_argument("--input-dir", type=str, required=True, 
                        help="Directory containing parallel corpus files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save analysis results")
    parser.add_argument("--src-lang", type=str, default="sah", 
                        help="Source language code")
    parser.add_argument("--tgt-lang", type=str, default="en",
                        help="Target language code")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Maximum number of lines to analyze")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a subdirectory for parallel analysis
    parallel_dir = os.path.join(args.output_dir, "parallel")
    os.makedirs(parallel_dir, exist_ok=True)
    
    # Redirect stdout to both terminal and file
    sys.stdout = Logger(parallel_dir)
    
    print(f"=== Parallel Corpus Analysis ({args.src_lang}-{args.tgt_lang}) ===")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {parallel_dir}")
    
    # Find parallel corpus files
    input_dir = Path(args.input_dir)
    corpus_files = list(input_dir.glob(f"*.{args.src_lang}"))
    
    if not corpus_files:
        logger.error(f"No parallel corpus files found in {args.input_dir} with extension .{args.src_lang}")
        return
    
    logger.info(f"Found {len(corpus_files)} parallel corpus files")
    
    # Process each corpus
    for src_file in corpus_files:
        corpus_name = src_file.stem
        tgt_file = src_file.with_suffix(f".{args.tgt_lang}")
        
        if not tgt_file.exists():
            logger.warning(f"Target file {tgt_file} not found, skipping")
            continue
        
        logger.info(f"Processing corpus: {corpus_name}")
        
        # Create a subdirectory for this corpus
        corpus_output_dir = os.path.join(parallel_dir, corpus_name)
        os.makedirs(corpus_output_dir, exist_ok=True)
        
        # Load and analyze the data
        df = load_parallel_data(src_file, tgt_file, args.max_lines)
        
        # Analyze length ratios
        df = analyze_length_ratios(df, args.src_lang, args.tgt_lang)
        
        # Create plots
        create_ratio_plots(df, corpus_output_dir, args.src_lang, args.tgt_lang)
        
        # Analyze vocabulary
        analyze_vocabulary_overlap(df, corpus_output_dir, args.src_lang, args.tgt_lang)
        
        # Save processed data
        df.to_csv(os.path.join(corpus_output_dir, f"{corpus_name}_parallel_data.csv"), index=False)
        
        logger.info(f"Completed analysis for corpus: {corpus_name}")
    
    print("\n=== Analysis Complete ===")
    logger.info("Parallel corpus analysis completed successfully")

if __name__ == "__main__":
    main() 