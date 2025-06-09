#!/usr/bin/env python3
"""
Analyze trilingual parallel data (English-Russian-Yakut)
"""
import os
import pandas as pd
import numpy as np
from razdel import tokenize as ru_tokenizer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import datetime
from collections import defaultdict

class Logger:
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log_file = open(output_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

# Set paths
PARALLEL_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/data/processed/parallel"
OUTPUT_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/src/preprocessing/analysis/parallel"

def load_parallel_data(files_dict, max_lines=None):
    """
    Load parallel data from multiple files into a pandas DataFrame.
    
    Args:
        files_dict (dict): Dictionary mapping language codes to file paths
        max_lines (int, optional): Maximum number of lines to read
        
    Returns:
        pd.DataFrame: DataFrame with columns for each language
    """
    # Read all files
    data = {}
    min_lines = float('inf')
    
    for lang, path in files_dict.items():
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            data[lang] = lines
            min_lines = min(min_lines, len(lines))
    
    # Ensure all files have the same number of lines
    for lang in data:
        data[lang] = data[lang][:min_lines]
    
    # Apply max_lines limit if specified
    if max_lines and max_lines < min_lines:
        for lang in data:
            data[lang] = data[lang][:max_lines]
    
    # Create DataFrame
    return pd.DataFrame(data)

def tokenize_text(text, lang):
    """
    Tokenize text based on language.
    
    Args:
        text (str): Text to tokenize
        lang (str): Language code
        
    Returns:
        list: List of tokens
    """
    if lang == 'ru':
        return list(ru_tokenizer(text))
    else:
        # Simple whitespace tokenization for other languages
        return text.split()

def analyze_sentence_lengths(df, dataset_name):
    """
    Analyze sentence lengths and save statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with parallel data
        dataset_name (str): Name of the dataset
        
    Returns:
        pd.DataFrame: DataFrame with length information added
    """
    print(f"\nAnalyzing {dataset_name} dataset:")
    
    # Add dataset column
    df['dataset'] = dataset_name
    
    # Calculate lengths for each language
    for lang in df.columns:
        if lang in ['dataset'] or not isinstance(df[lang].iloc[0], str):
            continue
            
        # Calculate character and word lengths
        df[f'{lang}_char_len'] = df[lang].apply(len)
        df[f'{lang}_word_len'] = df[lang].apply(lambda x: len(tokenize_text(x, lang)))
    
    # Calculate ratios between languages
    lang_pairs = []
    languages = [col for col in df.columns if col not in ['dataset'] and isinstance(df[col].iloc[0], str)]
    
    for i, lang1 in enumerate(languages):
        for lang2 in languages[i+1:]:
            lang_pairs.append((lang1, lang2))
            
            # Character ratio
            ratio_col = f'{lang1}_{lang2}_char_ratio'
            df[ratio_col] = df[f'{lang2}_char_len'] / df[f'{lang1}_char_len'].replace(0, 1)
            
            # Word ratio
            ratio_col = f'{lang1}_{lang2}_word_ratio'
            df[ratio_col] = df[f'{lang2}_word_len'] / df[f'{lang1}_word_len'].replace(0, 1)
    
    # Print example tokenizations
    print(f"\nExample tokenizations for {dataset_name} dataset:")
    for i in range(min(3, len(df))):
        for lang in languages:
            print(f"\n{lang.upper()}: {df[lang].iloc[i]}")
            tokens = tokenize_text(df[lang].iloc[i], lang)
            print(f"Tokens: {tokens}")
    
    # Print summary statistics
    print(f"\nSummary Statistics for {dataset_name} dataset:")
    
    print("\nCharacter lengths:")
    for lang in languages:
        char_len_col = f'{lang}_char_len'
        print(f"{lang.upper()}: mean={df[char_len_col].mean():.1f}, std={df[char_len_col].std():.1f}, " +
              f"min={df[char_len_col].min()}, max={df[char_len_col].max()}")
    
    print("\nWord lengths:")
    for lang in languages:
        word_len_col = f'{lang}_word_len'
        print(f"{lang.upper()}: mean={df[word_len_col].mean():.1f}, std={df[word_len_col].std():.1f}, " +
              f"min={df[word_len_col].min()}, max={df[word_len_col].max()}")
    
    print("\nLength ratios:")
    for lang1, lang2 in lang_pairs:
        char_ratio_col = f'{lang1}_{lang2}_char_ratio'
        word_ratio_col = f'{lang1}_{lang2}_word_ratio'
        print(f"{lang2.upper()}/{lang1.upper()} character ratio: mean={df[char_ratio_col].mean():.2f}, " +
              f"std={df[char_ratio_col].std():.2f}")
        print(f"{lang2.upper()}/{lang1.upper()} word ratio: mean={df[word_ratio_col].mean():.2f}, " +
              f"std={df[word_ratio_col].std():.2f}")
    
    # Save full data with length information
    df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_trilingual_data.csv'), index=False)
    
    # Create and save statistics
    stats_data = []
    
    # Character and word length statistics
    for lang in languages:
        for metric in ['char_len', 'word_len']:
            col = f'{lang}_{metric}'
            stats_data.append({
                'dataset': dataset_name,
                'metric': metric,
                'language': lang,
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            })
    
    # Ratio statistics
    for lang1, lang2 in lang_pairs:
        for metric in ['char_ratio', 'word_ratio']:
            col = f'{lang1}_{lang2}_{metric}'
            stats_data.append({
                'dataset': dataset_name,
                'metric': metric,
                'language_pair': f'{lang2}/{lang1}',
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            })
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_trilingual_stats.csv'), index=False)
    
    return df

def create_length_distribution_plots(datasets, output_dir):
    """
    Create length distribution plots for all datasets.
    
    Args:
        datasets (dict): Dictionary mapping dataset names to dataframes
        output_dir (str): Directory to save plots
    """
    # Get all languages across all datasets
    languages = set()
    for df in datasets.values():
        for col in df.columns:
            if col.endswith('_char_len'):
                lang = col.split('_')[0]
                languages.add(lang)
    
    # Character length distributions
    plt.figure(figsize=(12, 8))
    
    for i, lang in enumerate(sorted(languages)):
        plt.subplot(len(languages), 1, i+1)
        
        for name, df in datasets.items():
            col = f'{lang}_char_len'
            if col in df.columns:
                sns.histplot(data=df, x=col, alpha=0.5, label=f'{name}', bins=30)
        
        plt.title(f'Character Length Distribution - {lang.upper()}')
        plt.xlabel('Number of Characters')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trilingual_char_length_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Word length distributions
    plt.figure(figsize=(12, 8))
    
    for i, lang in enumerate(sorted(languages)):
        plt.subplot(len(languages), 1, i+1)
        
        for name, df in datasets.items():
            col = f'{lang}_word_len'
            if col in df.columns:
                sns.histplot(data=df, x=col, alpha=0.5, label=f'{name}', bins=30)
        
        plt.title(f'Word Length Distribution - {lang.upper()}')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trilingual_word_length_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create ratio plots
    language_pairs = set()
    for df in datasets.values():
        for col in df.columns:
            if col.endswith('_char_ratio'):
                lang_pair = '_'.join(col.split('_')[:2])
                language_pairs.add(lang_pair)
    
    # Ratio distributions
    plt.figure(figsize=(12, len(language_pairs) * 4))
    
    for i, lang_pair in enumerate(sorted(language_pairs)):
        lang1, lang2 = lang_pair.split('_')
        
        # Character ratio
        plt.subplot(len(language_pairs), 2, 2*i+1)
        for name, df in datasets.items():
            col = f'{lang_pair}_char_ratio'
            if col in df.columns:
                sns.histplot(data=df, x=col, alpha=0.5, label=f'{name}', bins=30)
        
        plt.title(f'Character Ratio Distribution - {lang2.upper()}/{lang1.upper()}')
        plt.xlabel('Character Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Word ratio
        plt.subplot(len(language_pairs), 2, 2*i+2)
        for name, df in datasets.items():
            col = f'{lang_pair}_word_ratio'
            if col in df.columns:
                sns.histplot(data=df, x=col, alpha=0.5, label=f'{name}', bins=30)
        
        plt.title(f'Word Ratio Distribution - {lang2.upper()}/{lang1.upper()}')
        plt.xlabel('Word Ratio')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trilingual_ratio_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_statistics_table(datasets, output_dir):
    """
    Create statistics table for all datasets.
    
    Args:
        datasets (dict): Dictionary mapping dataset names to dataframes
        output_dir (str): Directory to save table
    """
    # Combine all statistics
    all_stats = []
    for name, df in datasets.items():
        stats_file = os.path.join(output_dir, f'{name}_trilingual_stats.csv')
        if os.path.exists(stats_file):
            stats = pd.read_csv(stats_file)
            all_stats.append(stats)
    
    if not all_stats:
        print("No statistics files found")
        return
    
    combined_stats = pd.concat(all_stats)
    combined_stats.to_csv(os.path.join(output_dir, 'trilingual_combined_stats.csv'), index=False)
    
    # Create pivot tables for easier comparison
    pivot_tables = {}
    
    # Character length statistics by language and dataset
    char_len_stats = combined_stats[combined_stats['metric'] == 'char_len'].pivot_table(
        values=['mean', 'std', 'min', 'max', 'median'],
        index='language',
        columns='dataset'
    )
    pivot_tables['char_len'] = char_len_stats
    
    # Word length statistics by language and dataset
    word_len_stats = combined_stats[combined_stats['metric'] == 'word_len'].pivot_table(
        values=['mean', 'std', 'min', 'max', 'median'],
        index='language',
        columns='dataset'
    )
    pivot_tables['word_len'] = word_len_stats
    
    # Ratio statistics by language pair and dataset
    char_ratio_stats = combined_stats[combined_stats['metric'] == 'char_ratio'].pivot_table(
        values=['mean', 'std', 'min', 'max', 'median'],
        index='language_pair',
        columns='dataset'
    )
    pivot_tables['char_ratio'] = char_ratio_stats
    
    word_ratio_stats = combined_stats[combined_stats['metric'] == 'word_ratio'].pivot_table(
        values=['mean', 'std', 'min', 'max', 'median'],
        index='language_pair',
        columns='dataset'
    )
    pivot_tables['word_ratio'] = word_ratio_stats
    
    # Save pivot tables
    for name, table in pivot_tables.items():
        table.to_csv(os.path.join(output_dir, f'trilingual_{name}_pivot.csv'))
    
    # Create HTML table for visualization
    html_output = '<html><head><style>'
    html_output += 'table { border-collapse: collapse; margin: 20px; }'
    html_output += 'th, td { border: 1px solid black; padding: 8px; text-align: right; }'
    html_output += 'th { background-color: #f2f2f2; }'
    html_output += 'h2 { margin-top: 40px; }'
    html_output += '</style></head><body>'
    
    html_output += '<h1>Trilingual Parallel Data Statistics</h1>'
    
    for name, table in pivot_tables.items():
        if name == 'char_len':
            title = 'Character Length Statistics'
        elif name == 'word_len':
            title = 'Word Length Statistics'
        elif name == 'char_ratio':
            title = 'Character Ratio Statistics'
        elif name == 'word_ratio':
            title = 'Word Ratio Statistics'
        
        html_output += f'<h2>{title}</h2>'
        html_output += table.to_html()
    
    html_output += '</body></html>'
    
    with open(os.path.join(output_dir, 'trilingual_statistics.html'), 'w', encoding='utf-8') as f:
        f.write(html_output)

def main():
    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, 'trilingual_analysis.log')
    sys.stdout = Logger(log_file)
    
    # Print header
    print("Trilingual Parallel Text Analysis Report")
    print("=" * 50)
    print(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print("\n")
    
    # Define datasets to analyze
    datasets = {
        'wikimedia': {
            'en': os.path.join(PARALLEL_DIR, 'wikimedia.en'),
            'ru': os.path.join(PARALLEL_DIR, 'wikimedia.ru'),
            'sah': os.path.join(PARALLEL_DIR, 'wikimedia.sah')
        },
        'tatoeba': {
            'en': os.path.join(PARALLEL_DIR, 'tatoeba.en'),
            'ru': os.path.join(PARALLEL_DIR, 'tatoeba.ru'),
            'sah': os.path.join(PARALLEL_DIR, 'tatoeba.sah')
        }
    }
    
    # Load and analyze datasets
    analyzed_data = {}
    for name, files in datasets.items():
        print(f"\nLoading {name} dataset...")
        df = load_parallel_data(files)
        print(f"Loaded {len(df)} parallel sentences from {name} dataset")
        
        analyzed_df = analyze_sentence_lengths(df, name)
        analyzed_data[name] = analyzed_df
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_length_distribution_plots(analyzed_data, OUTPUT_DIR)
    
    # Create statistics table
    print("\nCreating statistics table...")
    create_statistics_table(analyzed_data, OUTPUT_DIR)
    
    print("\nAnalysis complete. Results saved to:")
    print(f"- {OUTPUT_DIR}/trilingual_analysis.log")
    print(f"- {OUTPUT_DIR}/trilingual_char_length_distributions.png")
    print(f"- {OUTPUT_DIR}/trilingual_word_length_distributions.png")
    print(f"- {OUTPUT_DIR}/trilingual_ratio_distributions.png")
    print(f"- {OUTPUT_DIR}/trilingual_statistics.html")
    
    # Close log file
    sys.stdout.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()