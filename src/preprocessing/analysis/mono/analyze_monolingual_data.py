#!/usr/bin/env python3
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

class Logger:
    def __init__(self, output_dir):
        self.terminal = sys.stdout
        self.log_file = open(os.path.join(output_dir, 'monolingual_analysis_report.txt'), 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Set paths
MONOLINGUAL_DIR = "../../../../data/processed/monolingual"
OUTPUT_DIR = "../"

def load_monolingual_data(file_path, sample_size=10000, random_seed=42):
    """Load monolingual data from a file into a pandas DataFrame.
    For large files, take a random sample of specified size."""
    random.seed(random_seed)
    
    # Count total lines first (this is faster than reading all lines at once)
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    # If file is too large, take a random sample
    if total_lines > sample_size:
        # Generate random line indices
        line_indices = sorted(random.sample(range(total_lines), sample_size))
        
        # Read only selected lines
        lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            current_line = 0
            next_index = 0
            for line in f:
                if next_index < len(line_indices) and current_line == line_indices[next_index]:
                    lines.append(line.strip())
                    next_index += 1
                current_line += 1
    else:
        # Read all lines if file is small enough
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
    
    return pd.DataFrame({
        'text': lines
    }), total_lines

def analyze_sentence_lengths(df):
    """Analyze sentence lengths and save statistics."""
    # Count functions
    def count_words(text): return len(text.split())
    def count_chars(text): return len(text)
    
    # Calculate lengths
    df['word_len'] = df['text'].apply(count_words)
    df['char_len'] = df['text'].apply(count_chars)
    
    # Print example sentences
    print("\nExample sentences:")
    for i in range(min(5, len(df))):
        print(f"\nSentence {i+1}: {df['text'].iloc[i]}")
        print(f"Word count: {df['word_len'].iloc[i]}")
        print(f"Character count: {df['char_len'].iloc[i]}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nCharacter lengths:")
    print(f"Mean: {df['char_len'].mean():.1f}")
    print(f"Standard deviation: {df['char_len'].std():.1f}")
    print(f"Minimum: {df['char_len'].min()}")
    print(f"Maximum: {df['char_len'].max()}")
    print(f"Median: {df['char_len'].median()}")
    
    print("\nWord lengths:")
    print(f"Mean: {df['word_len'].mean():.1f}")
    print(f"Standard deviation: {df['word_len'].std():.1f}")
    print(f"Minimum: {df['word_len'].min()}")
    print(f"Maximum: {df['word_len'].max()}")
    print(f"Median: {df['word_len'].median()}")
    
    # Save detailed statistics to CSV
    stats_data = [
        # Character length statistics
        {
            'metric': 'char_len',
            'mean': df['char_len'].mean(),
            'std': df['char_len'].std(),
            'min': df['char_len'].min(),
            'max': df['char_len'].max(),
            'median': df['char_len'].median(),
            'q1': df['char_len'].quantile(0.25),
            'q3': df['char_len'].quantile(0.75)
        },
        # Word length statistics
        {
            'metric': 'word_len',
            'mean': df['word_len'].mean(),
            'std': df['word_len'].std(),
            'min': df['word_len'].min(),
            'max': df['word_len'].max(),
            'median': df['word_len'].median(),
            'q1': df['word_len'].quantile(0.25),
            'q3': df['word_len'].quantile(0.75)
        }
    ]
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'monolingual_statistics.csv'), index=False)
    
    # Also save raw data with length information (for the sample)
    df.to_csv(os.path.join(OUTPUT_DIR, 'monolingual_sample_data.csv'), index=False)
    
    return df, stats_df

def analyze_vocabulary(df):
    """Analyze vocabulary statistics."""
    # Extract all words and count frequencies
    all_words = []
    for text in df['text']:
        all_words.extend(text.split())
    
    # Count unique words and frequencies
    word_counts = pd.Series(all_words).value_counts()
    
    # Print vocabulary statistics
    print("\nVocabulary Statistics:")
    print(f"Total words in sample: {len(all_words)}")
    print(f"Unique words in sample: {len(word_counts)}")
    print(f"Vocabulary diversity (unique/total): {len(word_counts) / len(all_words):.4f}")
    
    # Top and bottom frequency words
    print("\nMost common words:")
    for word, count in word_counts.head(20).items():
        print(f"  {word}: {count}")
    
    print("\nLeast common words (occurring once):")
    hapaxes = word_counts[word_counts == 1]
    print(f"  Number of words occurring once: {len(hapaxes)}")
    print(f"  Percentage of vocabulary occurring once: {len(hapaxes) / len(word_counts):.2%}")
    
    # Word length distribution
    word_lengths = pd.Series([len(word) for word in word_counts.index])
    print("\nWord length distribution in vocabulary:")
    print(f"  Mean word length: {word_lengths.mean():.2f} characters")
    print(f"  Median word length: {word_lengths.median()} characters")
    print(f"  Most common word length: {word_lengths.value_counts().index[0]} characters")
    
    # Save top words to CSV
    top_words_df = pd.DataFrame({
        'word': word_counts.head(1000).index,
        'frequency': word_counts.head(1000).values
    })
    top_words_df.to_csv(os.path.join(OUTPUT_DIR, 'monolingual_top_words.csv'), index=False)
    
    return word_counts, word_lengths

def create_distribution_plots(df, stats_df):
    """Create and save distribution plots."""
    # Set style
    plt.style.use('seaborn-v0_8')  # Use a specific seaborn style version
    
    def add_stats_box(ax, data, label=None):
        """Add a statistics box to the plot"""
        stats = ''
        if label:
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
    
    # Create sentence length distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot word lengths
    sns.histplot(data=df, x='word_len', ax=ax1, kde=True, color='blue')
    ax1.set_title('Word Length Distribution')
    ax1.set_xlabel('Number of Words per Sentence')
    ax1.set_ylabel('Frequency')
    add_stats_box(ax1, df['word_len'])
    
    # Plot character lengths
    sns.histplot(data=df, x='char_len', ax=ax2, kde=True, color='green')
    ax2.set_title('Character Length Distribution')
    ax2.set_xlabel('Number of Characters per Sentence')
    ax2.set_ylabel('Frequency')
    add_stats_box(ax2, df['char_len'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monolingual_length_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create word length vs frequency scatterplot
    plt.figure(figsize=(12, 6))
    word_counts, word_lengths = analyze_vocabulary(df)
    
    # Create bins for word lengths and calculate average frequency for each length
    word_length_freq = pd.DataFrame({
        'length': [len(word) for word in word_counts.index],
        'frequency': word_counts.values
    })
    length_avg_freq = word_length_freq.groupby('length')['frequency'].mean().reset_index()
    
    plt.scatter(length_avg_freq['length'], length_avg_freq['frequency'], alpha=0.7)
    plt.title('Average Word Frequency by Word Length')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Average Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monolingual_word_length_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create cumulative vocabulary plot (vocabulary size vs corpus size)
    plt.figure(figsize=(10, 6))
    
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate cumulative vocabulary size
    vocab_size = []
    current_size = 0
    for _, _ in sorted_words:
        current_size += 1
        vocab_size.append(current_size)
    
    # Calculate coverage percentage
    total_tokens = sum(word_counts)
    cumulative_tokens = 0
    coverage = []
    
    for _, freq in sorted_words:
        cumulative_tokens += freq
        coverage.append(cumulative_tokens / total_tokens * 100)
    
    # Plot cumulative vocabulary size
    plt.plot(range(1, len(vocab_size) + 1), vocab_size, color='blue')
    plt.title('Vocabulary Growth Curve')
    plt.xlabel('Rank of Word (by frequency)')
    plt.ylabel('Cumulative Vocabulary Size')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monolingual_vocabulary_growth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot vocabulary coverage
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(coverage) + 1), coverage, color='green')
    plt.title('Vocabulary Coverage')
    plt.xlabel('Number of Most Frequent Words')
    plt.ylabel('Percentage of Corpus Covered')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for key coverage points
    coverage_points = [50, 80, 90, 95, 99]
    for p in coverage_points:
        # Find the first index where coverage exceeds p%
        idx = next((i for i, cov in enumerate(coverage) if cov >= p), len(coverage) - 1)
        plt.scatter(idx + 1, coverage[idx], color='red', zorder=5)
        plt.annotate(f"{p}% coverage: {idx + 1} words", 
                     (idx + 1, coverage[idx]),
                     xytext=(20, 10), 
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monolingual_coverage.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_statistics_table_plot(stats_df):
    """Create a visual table plot from statistics DataFrame."""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    header = ['Metric', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Q1', 'Q3']
    
    for _, row in stats_df.iterrows():
        table_data.append([
            row['metric'],
            f"{row['mean']:.2f}",
            f"{row['std']:.2f}",
            f"{row['min']:.0f}",
            f"{row['max']:.0f}",
            f"{row['median']:.0f}",
            f"{row['q1']:.0f}",
            f"{row['q3']:.0f}"
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
    plt.title('Statistics for Monolingual Sakha Dataset', pad=20)
    
    # Save plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'monolingual_statistics_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up logging to both terminal and file
    sys.stdout = Logger(OUTPUT_DIR)
    
    print("Monolingual Sakha Text Analysis Report")
    print("=" * 50)
    print(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Load data
    print("\nData Loading:")
    print("-" * 30)
    sakha_file = os.path.join(MONOLINGUAL_DIR, "yakut_clean.txt")
    
    # Debug: Print file path and existence
    print(f"Sakha file: {sakha_file}")
    print(f"File exists: {os.path.exists(sakha_file)}")
    print(f"File size: {os.path.getsize(sakha_file)/1024/1024:.1f} MB")
    
    # Load data with sampling for large files
    df, total_lines = load_monolingual_data(sakha_file, sample_size=10000)
    
    print(f"\nDataset Information:")
    print(f"Total lines in file: {total_lines}")
    print(f"Sample size analyzed: {len(df)} sentences")
    
    # Analyze data
    print("\nAnalyzing Dataset:")
    print("-" * 30)
    df, stats_df = analyze_sentence_lengths(df)
    
    # Create plots
    print("\nGenerating Visualizations:")
    print("-" * 30)
    print("Creating distribution plots and vocabulary analysis...")
    create_distribution_plots(df, stats_df)
    
    # Create statistics table as plot
    print("Creating statistics table...")
    create_statistics_table_plot(stats_df)
    
    print("\nOutput Files Generated:")
    print("-" * 30)
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('monolingual_')]
    for file in sorted(output_files):
        file_path = os.path.join(OUTPUT_DIR, file)
        file_size = os.path.getsize(file_path) / 1024  # Convert to KB
        print(f"{file:<40} {file_size:>8.1f} KB")
    
    print("\nAnalysis complete. Results saved in:", OUTPUT_DIR)
    
    # Close the log file
    sys.stdout.log_file.close()
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main() 