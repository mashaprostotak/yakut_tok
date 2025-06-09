#!/usr/bin/env python3
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

class Logger:
    def __init__(self, output_dir):
        self.terminal = sys.stdout
        self.log_file = open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Set paths
PARALLEL_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/data/processed/parallel"
OUTPUT_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/src/preprocessing/analysis"

def load_parallel_data(src_path, tgt_path, max_lines=None):
    """Load parallel data from two files into a pandas DataFrame."""
    with open(src_path, 'r', encoding='utf-8') as src_file, \
         open(tgt_path, 'r', encoding='utf-8') as tgt_file:
        src_lines = src_file.readlines()
        tgt_lines = tgt_file.readlines()
        
    if max_lines:
        src_lines = src_lines[:max_lines]
        tgt_lines = tgt_lines[:max_lines]
        
    return pd.DataFrame({
        'source': [line.strip() for line in src_lines],
        'target': [line.strip() for line in tgt_lines]
    })

def analyze_sentence_lengths(df, dataset_name):
    """Analyze sentence lengths and save statistics."""
    # Count functions
    def count_sah_words(text): return len(text.split())
    def count_ru_words(text): return len(list(ru_tokenizer(text)))
    def count_chars(text): return len(text)
    
    # Calculate lengths
    df['src_word_len'] = df['source'].apply(count_sah_words)
    df['tgt_word_len'] = df['target'].apply(count_ru_words)
    df['src_char_len'] = df['source'].apply(count_chars)
    df['tgt_char_len'] = df['target'].apply(count_chars)
    df['char_ratio'] = df['tgt_char_len'] / df['src_char_len']
    df['word_ratio'] = df['tgt_word_len'] / df['src_word_len'].replace(0, 1)  # Avoid division by zero
    
    # Print example tokenizations
    print(f"\nExample tokenizations for {dataset_name} dataset:")
    for i in range(min(3, len(df))):
        print(f"\nSakha (basic split)   : {df['source'].iloc[i]}")
        print(f"Tokens: {df['source'].iloc[i].split()}")
        print(f"Russian (razdel)      : {df['target'].iloc[i]}")
        print(f"Tokens: {list(ru_tokenizer(df['target'].iloc[i]))}")
    
    # Print summary statistics
    print(f"\nSummary Statistics for {dataset_name} dataset:")
    print("\nCharacter lengths:")
    print(f"Sakha  : mean={df['src_char_len'].mean():.1f}, std={df['src_char_len'].std():.1f}")
    print(f"Russian: mean={df['tgt_char_len'].mean():.1f}, std={df['tgt_char_len'].std():.1f}")
    
    print("\nWord lengths:")
    print(f"Sakha  : mean={df['src_word_len'].mean():.1f}, std={df['src_word_len'].std():.1f}")
    print(f"Russian: mean={df['tgt_word_len'].mean():.1f}, std={df['tgt_word_len'].std():.1f}")
    
    print("\nLength ratios (Russian/Sakha):")
    print(f"Character ratio: mean={df['char_ratio'].mean():.2f}, std={df['char_ratio'].std():.2f}")
    print(f"Word ratio    : mean={df['word_ratio'].mean():.2f}, std={df['word_ratio'].std():.2f}")
    
    # Save detailed statistics to CSV
    stats_data = [
        # Character length statistics
        {
            'metric': 'char_len',
            'type': 'Sakha',
            'mean': df['src_char_len'].mean(),
            'std': df['src_char_len'].std(),
            'min': df['src_char_len'].min(),
            'max': df['src_char_len'].max()
        },
        {
            'metric': 'char_len',
            'type': 'Russian',
            'mean': df['tgt_char_len'].mean(),
            'std': df['tgt_char_len'].std(),
            'min': df['tgt_char_len'].min(),
            'max': df['tgt_char_len'].max()
        },
        # Word length statistics
        {
            'metric': 'word_len',
            'type': 'Sakha',
            'mean': df['src_word_len'].mean(),
            'std': df['src_word_len'].std(),
            'min': df['src_word_len'].min(),
            'max': df['src_word_len'].max()
        },
        {
            'metric': 'word_len',
            'type': 'Russian',
            'mean': df['tgt_word_len'].mean(),
            'std': df['tgt_word_len'].std(),
            'min': df['tgt_word_len'].min(),
            'max': df['tgt_word_len'].max()
        },
        # Ratio statistics
        {
            'metric': 'ratio',
            'type': 'char',
            'mean': df['char_ratio'].mean(),
            'std': df['char_ratio'].std(),
            'min': df['char_ratio'].min(),
            'max': df['char_ratio'].max()
        },
        {
            'metric': 'ratio',
            'type': 'word',
            'mean': df['word_ratio'].mean(),
            'std': df['word_ratio'].std(),
            'min': df['word_ratio'].min(),
            'max': df['word_ratio'].max()
        }
    ]
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_statistics.csv'), index=False)
    
    # Also save raw data with length information
    df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_full_data.csv'), index=False)
    
    return df

def create_distribution_plots(tatoeba_data, wikimedia_data):
    """Create and save distribution plots."""
    # Set style
    plt.style.use('seaborn-v0_8')  # Use a specific seaborn style version
    
    def add_stats_box(ax, data, label):
        """Add a statistics box to the plot"""
        stats = f'{label} stats:\n'
        stats += f'mean = {data.mean():.1f}\n'
        stats += f'std  = {data.std():.1f}\n'
        stats += f'min  = {data.min():.1f}\n'
        stats += f'max  = {data.max():.1f}\n'
        stats += f'median = {data.median():.1f}'
        
        # Position the text box in figure coords
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
    sns.histplot(data=tatoeba_data, x='src_word_len', alpha=0.5, label='Sakha (Tatoeba)', color='blue')
    sns.histplot(data=tatoeba_data, x='tgt_word_len', alpha=0.5, label='Russian (Tatoeba)', color='red')
    plt.title('Word Length Distribution (Tatoeba)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    # Add stats boxes
    add_stats_box(plt.gca(), tatoeba_data['src_word_len'], 'Sakha')
    add_stats_box(plt.gca(), tatoeba_data['tgt_word_len'], 'Russian')
    
    # Plot word lengths - Wikimedia
    plt.subplot(1, 2, 2)
    sns.histplot(data=wikimedia_data, x='src_word_len', alpha=0.5, label='Sakha (Wikimedia)', color='blue')
    sns.histplot(data=wikimedia_data, x='tgt_word_len', alpha=0.5, label='Russian (Wikimedia)', color='red')
    plt.title('Word Length Distribution (Wikimedia)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    # Add stats boxes
    add_stats_box(plt.gca(), wikimedia_data['src_word_len'], 'Sakha')
    add_stats_box(plt.gca(), wikimedia_data['tgt_word_len'], 'Russian')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'word_length_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create character length distribution plot
    plt.figure(figsize=(15, 6))
    
    # Plot character lengths - Tatoeba
    plt.subplot(1, 2, 1)
    sns.histplot(data=tatoeba_data, x='src_char_len', alpha=0.5, label='Sakha (Tatoeba)', color='blue')
    sns.histplot(data=tatoeba_data, x='tgt_char_len', alpha=0.5, label='Russian (Tatoeba)', color='red')
    plt.title('Character Length Distribution (Tatoeba)')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.legend()
    # Add stats boxes
    add_stats_box(plt.gca(), tatoeba_data['src_char_len'], 'Sakha')
    add_stats_box(plt.gca(), tatoeba_data['tgt_char_len'], 'Russian')
    
    # Plot character lengths - Wikimedia
    plt.subplot(1, 2, 2)
    sns.histplot(data=wikimedia_data, x='src_char_len', alpha=0.5, label='Sakha (Wikimedia)', color='blue')
    sns.histplot(data=wikimedia_data, x='tgt_char_len', alpha=0.5, label='Russian (Wikimedia)', color='red')
    plt.title('Character Length Distribution (Wikimedia)')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.legend()
    # Add stats boxes
    add_stats_box(plt.gca(), wikimedia_data['src_char_len'], 'Sakha')
    add_stats_box(plt.gca(), wikimedia_data['tgt_char_len'], 'Russian')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'char_length_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create ratio distribution plots
    plt.figure(figsize=(15, 6))
    
    # Plot character ratio
    plt.subplot(1, 2, 1)
    combined_data = pd.concat([tatoeba_data, wikimedia_data])
    sns.histplot(data=combined_data, x='char_ratio', hue='dataset', alpha=0.5, multiple="layer")
    plt.title('Character Length Ratio Distribution\n(Russian/Sakha)')
    plt.xlabel('Character Ratio')
    plt.ylabel('Frequency')
    # Add stats boxes
    add_stats_box(plt.gca(), tatoeba_data['char_ratio'], 'Tatoeba')
    add_stats_box(plt.gca(), wikimedia_data['char_ratio'], 'Wikimedia')
    
    # Plot word ratio
    plt.subplot(1, 2, 2)
    sns.histplot(data=combined_data, x='word_ratio', hue='dataset', alpha=0.5, multiple="layer")
    plt.title('Word Length Ratio Distribution\n(Russian/Sakha)')
    plt.xlabel('Word Ratio')
    plt.ylabel('Frequency')
    # Add stats boxes
    add_stats_box(plt.gca(), tatoeba_data['word_ratio'], 'Tatoeba')
    add_stats_box(plt.gca(), wikimedia_data['word_ratio'], 'Wikimedia')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ratio_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_statistics_table_plot(stats_df, dataset_name):
    """Create a visual table plot from statistics DataFrame."""
    # Prepare the data
    metrics = stats_df['metric'].unique()
    types = stats_df['type'].unique()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(metrics) * 1.2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    header = ['Metric', 'Type', 'Mean', 'Std', 'Min', 'Max']
    
    for metric in metrics:
        metric_data = stats_df[stats_df['metric'] == metric]
        for _, row in metric_data.iterrows():
            table_data.append([
                metric,
                row['type'],
                f"{row['mean']:.2f}",
                f"{row['std']:.2f}",
                f"{row['min']:.2f}",
                f"{row['max']:.2f}"
            ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=header,
                    loc='center',
                    cellLoc='center',
                    colColours=['#f2f2f2'] * len(header),
                    cellColours=[['#ffffff'] * len(header)] * len(table_data))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Set title
    plt.title(f'Statistics for {dataset_name.capitalize()} Dataset', pad=20)
    
    # Save plot
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_name}_statistics_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_table_plot(tatoeba_stats, wikimedia_stats):
    """Create a visual comparison table between Tatoeba and Wikimedia datasets."""
    # Prepare the data
    metrics = tatoeba_stats['metric'].unique()
    types = tatoeba_stats['type'].unique()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, len(metrics) * 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    header = ['Metric', 'Type', 'Tatoeba Mean', 'Wikimedia Mean', 'Tatoeba Std', 'Wikimedia Std']
    
    for metric in metrics:
        for type_ in types:
            tat_row = tatoeba_stats[(tatoeba_stats['metric'] == metric) & 
                                  (tatoeba_stats['type'] == type_)]
            wiki_row = wikimedia_stats[(wikimedia_stats['metric'] == metric) & 
                                     (wikimedia_stats['type'] == type_)]
            
            if not tat_row.empty and not wiki_row.empty:
                table_data.append([
                    metric,
                    type_,
                    f"{tat_row['mean'].iloc[0]:.2f}",
                    f"{wiki_row['mean'].iloc[0]:.2f}",
                    f"{tat_row['std'].iloc[0]:.2f}",
                    f"{wiki_row['std'].iloc[0]:.2f}"
                ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=header,
                    loc='center',
                    cellLoc='center',
                    colColours=['#f2f2f2'] * len(header),
                    cellColours=[['#ffffff'] * len(header)] * len(table_data))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Set title
    plt.title('Comparison between Tatoeba and Wikimedia Datasets', pad=20)
    
    # Save plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_comparison_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up logging to both terminal and file
    sys.stdout = Logger(OUTPUT_DIR)
    
    print("Parallel Text Analysis Report")
    print("=" * 50)
    print(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Debug: Print current working directory and paths
    print("\nSystem Information:")
    print("-" * 30)
    print(f"Current working directory: {os.getcwd()}")
    print(f"PARALLEL_DIR path: {PARALLEL_DIR}")
    print(f"PARALLEL_DIR exists: {os.path.exists(PARALLEL_DIR)}")
    print(f"OUTPUT_DIR exists: {os.path.exists(OUTPUT_DIR)}")
    
    # Load data
    print("\nData Loading:")
    print("-" * 30)
    tatoeba_sah = os.path.join(PARALLEL_DIR, "tatoeba.sah")
    tatoeba_ru = os.path.join(PARALLEL_DIR, "tatoeba.ru")
    wikimedia_sah = os.path.join(PARALLEL_DIR, "wikimedia.sah")
    wikimedia_ru = os.path.join(PARALLEL_DIR, "wikimedia.ru")
    
    # Debug: Print file paths and existence
    print("\nInput Files Status:")
    print("Tatoeba files:")
    print(f"  Sakha : {tatoeba_sah}")
    print(f"         Size: {os.path.getsize(tatoeba_sah)/1024:.1f}KB")
    print(f"         Exists: {os.path.exists(tatoeba_sah)}")
    print(f"  Russian: {tatoeba_ru}")
    print(f"         Size: {os.path.getsize(tatoeba_ru)/1024:.1f}KB")
    print(f"         Exists: {os.path.exists(tatoeba_ru)}")
    print("\nWikimedia files:")
    print(f"  Sakha : {wikimedia_sah}")
    print(f"         Size: {os.path.getsize(wikimedia_sah)/1024:.1f}KB")
    print(f"         Exists: {os.path.exists(wikimedia_sah)}")
    print(f"  Russian: {wikimedia_ru}")
    print(f"         Size: {os.path.getsize(wikimedia_ru)/1024:.1f}KB")
    print(f"         Exists: {os.path.exists(wikimedia_ru)}")
    
    tatoeba_data = load_parallel_data(tatoeba_sah, tatoeba_ru)
    wikimedia_data = load_parallel_data(wikimedia_sah, wikimedia_ru)
    
    print(f"\nDataset Sizes:")
    print(f"Tatoeba  : {len(tatoeba_data)} sentence pairs")
    print(f"Wikimedia: {len(wikimedia_data)} sentence pairs")
    
    # Add dataset identifier
    tatoeba_data['dataset'] = 'Tatoeba'
    wikimedia_data['dataset'] = 'Wikimedia'
    
    # Analyze data
    print("\nAnalyzing Datasets:")
    print("-" * 30)
    tatoeba_data = analyze_sentence_lengths(tatoeba_data, 'tatoeba')
    wikimedia_data = analyze_sentence_lengths(wikimedia_data, 'wikimedia')
    
    # Create plots
    print("\nGenerating Visualizations:")
    print("-" * 30)
    print("Creating distribution plots...")
    create_distribution_plots(tatoeba_data, wikimedia_data)
    
    # Create statistics tables as plots
    print("Creating statistics tables...")
    # Load statistics from CSV
    tatoeba_stats = pd.read_csv(os.path.join(OUTPUT_DIR, 'tatoeba_statistics.csv'))
    wikimedia_stats = pd.read_csv(os.path.join(OUTPUT_DIR, 'wikimedia_statistics.csv'))
    
    create_statistics_table_plot(tatoeba_stats, 'tatoeba')
    create_statistics_table_plot(wikimedia_stats, 'wikimedia')
    create_comparison_table_plot(tatoeba_stats, wikimedia_stats)
    
    print("\nOutput Files Generated:")
    print("-" * 30)
    output_files = os.listdir(OUTPUT_DIR)
    for file in sorted(output_files):
        file_path = os.path.join(OUTPUT_DIR, file)
        file_size = os.path.getsize(file_path) / 1024  # Convert to KB
        print(f"{file:<30} {file_size:>8.1f} KB")
    
    print("\nAnalysis complete. Results saved in:", OUTPUT_DIR)
    
    # Close the log file
    sys.stdout.log_file.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()