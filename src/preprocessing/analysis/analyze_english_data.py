#!/usr/bin/env python3
"""
Analyze English-Yakut parallel data for the Yakut NLP project.
This script compares the source (Yakut) and target (English) languages
in the parallel corpus, mirroring the comprehensive analysis done for Russian.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import datetime
import random
import nltk
from nltk.tokenize import word_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, output_dir):
        self.terminal = sys.stdout
        self.log_file = open(os.path.join(output_dir, 'english_parallel_analysis_report.txt'), 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Set paths
PARALLEL_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/data/processed/parallel"
OUTPUT_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/src/preprocessing/analysis/english"

def load_parallel_data(src_path, tgt_path, max_lines=None, sample_size=10000, random_seed=42):
    """Load parallel data from source (Yakut) and target (English) files."""
    random.seed(random_seed)
    
    with open(src_path, 'r', encoding='utf-8') as src_file, \
         open(tgt_path, 'r', encoding='utf-8') as tgt_file:
        src_lines = [line.strip() for line in src_file.readlines()]
        tgt_lines = [line.strip() for line in tgt_file.readlines()]
    
    # Ensure equal lengths
    min_length = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:min_length]
    tgt_lines = tgt_lines[:min_length]
    
    # Apply max_lines limit if specified
    if max_lines:
        src_lines = src_lines[:max_lines]
        tgt_lines = tgt_lines[:max_lines]
    
    # Sample if dataset is too large
    total_lines = len(src_lines)
    if total_lines > sample_size:
        # Generate random indices for sampling
        indices = sorted(random.sample(range(total_lines), sample_size))
        src_lines = [src_lines[i] for i in indices]
        tgt_lines = [tgt_lines[i] for i in indices]
    
    logger.info(f"Loaded {len(src_lines)} parallel sentences from {total_lines} total lines")
    
    return pd.DataFrame({
        'source': src_lines,  # Yakut
        'target': tgt_lines   # English
    }), total_lines

def analyze_sentence_lengths(df, dataset_name):
    """Analyze sentence lengths and save statistics for both languages."""
    # Tokenization functions
    def count_sah_words(text): 
        return len(text.split())  # Basic split for Yakut
    
    def count_en_words(text): 
        try:
            return len(word_tokenize(text.lower()))  # NLTK tokenization for English
        except:
            # Fallback to regex tokenization
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            return len(words)
    
    def count_chars(text): 
        return len(text)
    
    # Calculate lengths
    df['src_word_len'] = df['source'].apply(count_sah_words)
    df['tgt_word_len'] = df['target'].apply(count_en_words)
    df['src_char_len'] = df['source'].apply(count_chars)
    df['tgt_char_len'] = df['target'].apply(count_chars)
    df['char_ratio'] = df['tgt_char_len'] / df['src_char_len'].replace(0, 1)  # English/Yakut ratio
    df['word_ratio'] = df['tgt_word_len'] / df['src_word_len'].replace(0, 1)  # Avoid division by zero
    
    # Print example tokenizations
    print(f"\nExample tokenizations for {dataset_name} dataset:")
    for i in range(min(3, len(df))):
        yakut_text = df['source'].iloc[i]
        english_text = df['target'].iloc[i]
        
        print(f"\nYakut (basic split)  : {yakut_text}")
        print(f"Tokens: {yakut_text.split()}")
        print(f"English (NLTK)       : {english_text}")
        try:
            en_tokens = word_tokenize(english_text.lower())
            print(f"Tokens: {en_tokens}")
        except:
            import re
            en_tokens = re.findall(r'\b\w+\b', english_text.lower())
            print(f"Tokens: {en_tokens} (fallback)")
    
    # Print summary statistics
    print(f"\nSummary Statistics for {dataset_name} dataset:")
    print("\nCharacter lengths:")
    print(f"Yakut  : mean={df['src_char_len'].mean():.1f}, std={df['src_char_len'].std():.1f}")
    print(f"English: mean={df['tgt_char_len'].mean():.1f}, std={df['tgt_char_len'].std():.1f}")
    
    print("\nWord lengths:")
    print(f"Yakut  : mean={df['src_word_len'].mean():.1f}, std={df['src_word_len'].std():.1f}")
    print(f"English: mean={df['tgt_word_len'].mean():.1f}, std={df['tgt_word_len'].std():.1f}")
    
    print("\nLength ratios (English/Yakut):")
    print(f"Character ratio: mean={df['char_ratio'].mean():.2f}, std={df['char_ratio'].std():.2f}")
    print(f"Word ratio     : mean={df['word_ratio'].mean():.2f}, std={df['word_ratio'].std():.2f}")
    
    # Save detailed statistics to CSV
    stats_data = [
        # Character length statistics
        {
            'metric': 'char_len',
            'type': 'Yakut',
            'mean': df['src_char_len'].mean(),
            'std': df['src_char_len'].std(),
            'min': df['src_char_len'].min(),
            'max': df['src_char_len'].max(),
            'median': df['src_char_len'].median()
        },
        {
            'metric': 'char_len',
            'type': 'English',
            'mean': df['tgt_char_len'].mean(),
            'std': df['tgt_char_len'].std(),
            'min': df['tgt_char_len'].min(),
            'max': df['tgt_char_len'].max(),
            'median': df['tgt_char_len'].median()
        },
        # Word length statistics
        {
            'metric': 'word_len',
            'type': 'Yakut',
            'mean': df['src_word_len'].mean(),
            'std': df['src_word_len'].std(),
            'min': df['src_word_len'].min(),
            'max': df['src_word_len'].max(),
            'median': df['src_word_len'].median()
        },
        {
            'metric': 'word_len',
            'type': 'English',
            'mean': df['tgt_word_len'].mean(),
            'std': df['tgt_word_len'].std(),
            'min': df['tgt_word_len'].min(),
            'max': df['tgt_word_len'].max(),
            'median': df['tgt_word_len'].median()
        },
        # Ratio statistics
        {
            'metric': 'ratio',
            'type': 'char',
            'mean': df['char_ratio'].mean(),
            'std': df['char_ratio'].std(),
            'min': df['char_ratio'].min(),
            'max': df['char_ratio'].max(),
            'median': df['char_ratio'].median()
        },
        {
            'metric': 'ratio',
            'type': 'word',
            'mean': df['word_ratio'].mean(),
            'std': df['word_ratio'].std(),
            'min': df['word_ratio'].min(),
            'max': df['word_ratio'].max(),
            'median': df['word_ratio'].median()
        }
    ]
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_parallel_statistics.csv'), index=False)
    
    # Also save raw data with length information
    df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_parallel_full_data.csv'), index=False)
    
    return df, stats_df

def analyze_vocabulary_overlap(df, dataset_name):
    """Analyze vocabulary statistics and cross-language characteristics."""
    # Extract all words
    yakut_words = []
    english_words = []
    
    for text in df['source']:
        yakut_words.extend(text.lower().split())
    
    for text in df['target']:
        try:
            english_words.extend([w.lower() for w in word_tokenize(text) if w.isalpha()])
        except:
            english_words.extend([w.lower() for w in text.split() if w.isalpha()])
    
    # Count unique words
    yakut_vocab = set(yakut_words)
    english_vocab = set(english_words)
    
    # Calculate statistics
    yakut_vocab_size = len(yakut_vocab)
    english_vocab_size = len(english_vocab)
    yakut_total_words = len(yakut_words)
    english_total_words = len(english_words)
    
    # Print vocabulary statistics
    print(f"\nVocabulary statistics for {dataset_name}:")
    print(f"Yakut vocabulary size: {yakut_vocab_size} unique words")
    print(f"English vocabulary size: {english_vocab_size} unique words")
    print(f"Yakut vocabulary diversity: {yakut_vocab_size/yakut_total_words:.4f}")
    print(f"English vocabulary diversity: {english_vocab_size/english_total_words:.4f}")
    
    # Most common words
    yakut_word_counts = pd.Series(yakut_words).value_counts()
    english_word_counts = pd.Series(english_words).value_counts()
    
    print(f"\nMost common Yakut words:")
    for word, count in yakut_word_counts.head(10).items():
        print(f"  {word}: {count}")
    
    print(f"\nMost common English words:")
    for word, count in english_word_counts.head(10).items():
        print(f"  {word}: {count}")
    
    # Word length distribution
    yakut_word_lengths = pd.Series([len(word) for word in yakut_vocab])
    english_word_lengths = pd.Series([len(word) for word in english_vocab])
    
    print(f"\nWord length distribution:")
    print(f"Yakut - Mean: {yakut_word_lengths.mean():.2f}, Median: {yakut_word_lengths.median()}")
    print(f"English - Mean: {english_word_lengths.mean():.2f}, Median: {english_word_lengths.median()}")
    
    # Save vocabulary statistics
    vocab_stats = {
        'language': ['Yakut', 'English'],
        'vocab_size': [yakut_vocab_size, english_vocab_size],
        'total_words': [yakut_total_words, english_total_words],
        'diversity': [yakut_vocab_size/yakut_total_words, english_vocab_size/english_total_words],
        'avg_word_length': [yakut_word_lengths.mean(), english_word_lengths.mean()]
    }
    
    vocab_df = pd.DataFrame(vocab_stats)
    vocab_df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_vocab_comparison.csv'), index=False)
    
    # Save top words for both languages
    yakut_top_df = pd.DataFrame({
        'word': yakut_word_counts.head(1000).index,
        'frequency': yakut_word_counts.head(1000).values
    })
    yakut_top_df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_top_words_yakut.csv'), index=False)
    
    english_top_df = pd.DataFrame({
        'word': english_word_counts.head(1000).index,
        'frequency': english_word_counts.head(1000).values
    })
    english_top_df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_top_words_english.csv'), index=False)
    
    return yakut_word_counts, english_word_counts

def create_parallel_distribution_plots(tatoeba_data, wikimedia_data):
    """Create and save distribution plots comparing both languages."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    def add_stats_box(ax, data, label):
        """Add a statistics box to the plot"""
        stats = f'{label} stats:\n'
        stats += f'mean = {data.mean():.1f}\n'
        stats += f'std  = {data.std():.1f}\n'
        stats += f'min  = {data.min():.1f}\n'
        stats += f'max  = {data.max():.1f}\n'
        stats += f'median = {data.median():.1f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, stats,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=props)
    
    # Create word length distribution plot
    plt.figure(figsize=(15, 6))
    
    # Plot word lengths - Tatoeba
    plt.subplot(1, 2, 1)
    sns.histplot(data=tatoeba_data, x='src_word_len', alpha=0.5, label='Yakut (Tatoeba)', color='blue')
    sns.histplot(data=tatoeba_data, x='tgt_word_len', alpha=0.5, label='English (Tatoeba)', color='green')
    plt.title('Word Length Distribution (Tatoeba)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats_box(plt.gca(), tatoeba_data['src_word_len'], 'Yakut')
    add_stats_box(plt.gca(), tatoeba_data['tgt_word_len'], 'English')
    
    # Plot word lengths - Wikimedia
    plt.subplot(1, 2, 2)
    sns.histplot(data=wikimedia_data, x='src_word_len', alpha=0.5, label='Yakut (Wikimedia)', color='blue')
    sns.histplot(data=wikimedia_data, x='tgt_word_len', alpha=0.5, label='English (Wikimedia)', color='green')
    plt.title('Word Length Distribution (Wikimedia)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats_box(plt.gca(), wikimedia_data['src_word_len'], 'Yakut')
    add_stats_box(plt.gca(), wikimedia_data['tgt_word_len'], 'English')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'english_parallel_word_length_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create character length distribution plot
    plt.figure(figsize=(15, 6))
    
    # Plot character lengths - Tatoeba
    plt.subplot(1, 2, 1)
    sns.histplot(data=tatoeba_data, x='src_char_len', alpha=0.5, label='Yakut (Tatoeba)', color='blue')
    sns.histplot(data=tatoeba_data, x='tgt_char_len', alpha=0.5, label='English (Tatoeba)', color='green')
    plt.title('Character Length Distribution (Tatoeba)')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats_box(plt.gca(), tatoeba_data['src_char_len'], 'Yakut')
    add_stats_box(plt.gca(), tatoeba_data['tgt_char_len'], 'English')
    
    # Plot character lengths - Wikimedia
    plt.subplot(1, 2, 2)
    sns.histplot(data=wikimedia_data, x='src_char_len', alpha=0.5, label='Yakut (Wikimedia)', color='blue')
    sns.histplot(data=wikimedia_data, x='tgt_char_len', alpha=0.5, label='English (Wikimedia)', color='green')
    plt.title('Character Length Distribution (Wikimedia)')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats_box(plt.gca(), wikimedia_data['src_char_len'], 'Yakut')
    add_stats_box(plt.gca(), wikimedia_data['tgt_char_len'], 'English')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'english_parallel_char_length_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create ratio distribution plots
    plt.figure(figsize=(15, 6))
    
    # Plot character ratio
    plt.subplot(1, 2, 1)
    combined_data = pd.concat([
        tatoeba_data.assign(dataset='Tatoeba'),
        wikimedia_data.assign(dataset='Wikimedia')
    ])
    sns.histplot(data=combined_data, x='char_ratio', hue='dataset', alpha=0.5, multiple="layer")
    plt.title('Character Length Ratio Distribution\n(English/Yakut)')
    plt.xlabel('Character Ratio')
    plt.ylabel('Frequency')
    add_stats_box(plt.gca(), tatoeba_data['char_ratio'], 'Tatoeba')
    add_stats_box(plt.gca(), wikimedia_data['char_ratio'], 'Wikimedia')
    
    # Plot word ratio
    plt.subplot(1, 2, 2)
    sns.histplot(data=combined_data, x='word_ratio', hue='dataset', alpha=0.5, multiple="layer")
    plt.title('Word Length Ratio Distribution\n(English/Yakut)')
    plt.xlabel('Word Ratio')
    plt.ylabel('Frequency')
    add_stats_box(plt.gca(), tatoeba_data['word_ratio'], 'Tatoeba')
    add_stats_box(plt.gca(), wikimedia_data['word_ratio'], 'Wikimedia')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'english_parallel_ratio_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(tatoeba_stats, wikimedia_stats):
    """Create comparison plots between datasets."""
    plt.figure(figsize=(15, 10))
    
    # Combine statistics for comparison
    all_stats = pd.concat([
        tatoeba_stats.assign(dataset='Tatoeba'),
        wikimedia_stats.assign(dataset='Wikimedia')
    ])
    
    # Character length comparison
    plt.subplot(2, 2, 1)
    char_stats = all_stats[all_stats['metric'] == 'char_len']
    sns.barplot(data=char_stats, x='type', y='mean', hue='dataset')
    plt.title('Mean Character Length Comparison')
    plt.ylabel('Mean Character Length')
    
    # Word length comparison
    plt.subplot(2, 2, 2)
    word_stats = all_stats[all_stats['metric'] == 'word_len']
    sns.barplot(data=word_stats, x='type', y='mean', hue='dataset')
    plt.title('Mean Word Length Comparison')
    plt.ylabel('Mean Word Length')
    
    # Ratio comparison
    plt.subplot(2, 2, 3)
    ratio_stats = all_stats[all_stats['metric'] == 'ratio']
    sns.barplot(data=ratio_stats, x='type', y='mean', hue='dataset')
    plt.title('Mean Ratio Comparison (English/Yakut)')
    plt.ylabel('Mean Ratio')
    
    # Standard deviation comparison
    plt.subplot(2, 2, 4)
    sns.barplot(data=all_stats[all_stats['metric'].isin(['char_len', 'word_len'])], 
                x='metric', y='std', hue='dataset')
    plt.title('Standard Deviation Comparison')
    plt.ylabel('Standard Deviation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'english_parallel_dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up logging to both terminal and file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout = Logger(OUTPUT_DIR)
    
    print(f"English-Yakut Parallel Corpus Analysis Report")
    print(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=================================================================")
    
    # Debug: Print system information
    print("\nSystem Information:")
    print("-" * 30)
    print(f"Current working directory: {os.getcwd()}")
    print(f"PARALLEL_DIR path: {PARALLEL_DIR}")
    print(f"PARALLEL_DIR exists: {os.path.exists(PARALLEL_DIR)}")
    print(f"OUTPUT_DIR exists: {os.path.exists(OUTPUT_DIR)}")
    
    # Define corpus files to analyze (Yakut-English pairs)
    corpus_files = [
        {
            'name': 'tatoeba',
            'src_file': os.path.join(PARALLEL_DIR, "tatoeba.sah"),    # Yakut
            'tgt_file': os.path.join(PARALLEL_DIR, "tatoeba.en")     # English
        },
        {
            'name': 'wikimedia', 
            'src_file': os.path.join(PARALLEL_DIR, "wikimedia.sah"), # Yakut
            'tgt_file': os.path.join(PARALLEL_DIR, "wikimedia.en")   # English
        }
    ]
    
    # Store results for combined analysis
    all_data = []
    all_stats = []
    
    # Analyze each corpus
    for corpus in corpus_files:
        corpus_name = corpus['name']
        src_file = corpus['src_file']
        tgt_file = corpus['tgt_file']
        
        print(f"\n\n{corpus_name.title()} Corpus Analysis")
        print(f"=" * 40)
        
        # Check if files exist
        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            print(f"Files not found:")
            print(f"  Yakut : {src_file} (exists: {os.path.exists(src_file)})")
            print(f"  English: {tgt_file} (exists: {os.path.exists(tgt_file)})")
            continue
        
        # Print file information
        print(f"\nInput Files:")
        print(f"  Yakut : {src_file}")
        print(f"         Size: {os.path.getsize(src_file)/1024:.1f}KB")
        print(f"  English: {tgt_file}")
        print(f"         Size: {os.path.getsize(tgt_file)/1024:.1f}KB")
        
        # Load and analyze data
        df, total_lines = load_parallel_data(src_file, tgt_file)
        print(f"\nLoaded {len(df)} sample sentences from {total_lines} total parallel pairs")
        
        # Analyze sentence lengths and relationships
        df, stats_df = analyze_sentence_lengths(df, corpus_name)
        
        # Analyze vocabulary characteristics
        yakut_words, english_words = analyze_vocabulary_overlap(df, corpus_name)
        
        # Store for combined analysis
        df['dataset'] = corpus_name
        all_data.append(df)
        all_stats.append(stats_df)
    
    # Combined analysis if we have data
    if all_data:
        print(f"\n\nCombined English-Yakut Parallel Analysis")
        print(f"=" * 50)
        
        # Separate data by corpus for plotting
        tatoeba_data = all_data[0] if len(all_data) > 0 and all_data[0]['dataset'].iloc[0] == 'tatoeba' else None
        wikimedia_data = all_data[1] if len(all_data) > 1 and all_data[1]['dataset'].iloc[0] == 'wikimedia' else None
        
        # Handle case where datasets might be in different order
        if tatoeba_data is None:
            tatoeba_data = next((df for df in all_data if df['dataset'].iloc[0] == 'tatoeba'), None)
        if wikimedia_data is None:
            wikimedia_data = next((df for df in all_data if df['dataset'].iloc[0] == 'wikimedia'), None)
        
        # Create plots if we have both datasets
        if tatoeba_data is not None and wikimedia_data is not None:
            print(f"\nCreating parallel corpus visualizations...")
            create_parallel_distribution_plots(tatoeba_data, wikimedia_data)
            
            # Create comparison plots
            tatoeba_stats = all_stats[0] if len(all_stats) > 0 else None
            wikimedia_stats = all_stats[1] if len(all_stats) > 1 else None
            
            if tatoeba_stats is not None and wikimedia_stats is not None:
                create_comparison_plots(tatoeba_stats, wikimedia_stats)
            
            # Combined statistics
            combined_df = pd.concat(all_data)
            print(f"\nOverall Statistics:")
            print(f"Total parallel pairs analyzed: {len(combined_df)}")
            print(f"Mean character ratio (English/Yakut): {combined_df['char_ratio'].mean():.2f}")
            print(f"Mean word ratio (English/Yakut): {combined_df['word_ratio'].mean():.2f}")
            
            # Save combined data
            combined_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_english_yakut_parallel_data.csv'), index=False)
            
        else:
            print("Warning: Not enough data for combined analysis")
    
    print(f"\n\nEnglish-Yakut parallel analysis completed.")
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Analysis includes:")
    print(f"  - Parallel corpus statistics and ratios")
    print(f"  - Cross-language vocabulary analysis") 
    print(f"  - Advanced tokenization (NLTK for English)")
    print(f"  - Comparative visualizations")
    print(f"  - Dataset comparison metrics")

if __name__ == "__main__":
    main() 