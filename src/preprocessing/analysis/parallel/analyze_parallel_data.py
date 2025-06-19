#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from razdel import tokenize as ru_tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

PARALLEL_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/data/processed/parallel"
OUTPUT_DIR = "/Users/SvetlaMaria/Desktop/WORK/ETH/CSNLP/Project/project/src/preprocessing/analysis"

def load_data(src_path, tgt_path):
    with open(src_path, 'r', encoding='utf-8') as f1, open(tgt_path, 'r', encoding='utf-8') as f2:
        src = [line.strip() for line in f1.readlines()]
        tgt = [line.strip() for line in f2.readlines()]
    return pd.DataFrame({'source': src, 'target': tgt})

def analyze_lengths(df, name):
    df['src_word_len'] = df['source'].apply(lambda x: len(x.split()))
    df['tgt_word_len'] = df['target'].apply(lambda x: len(list(ru_tokenizer(x))))
    df['src_char_len'] = df['source'].apply(len)
    df['tgt_char_len'] = df['target'].apply(len)
    df['char_ratio'] = df['tgt_char_len'] / df['src_char_len']
    df['word_ratio'] = df['tgt_word_len'] / df['src_word_len'].replace(0, 1)
    
    print(f"\n{name} examples:")
    for i in range(min(3, len(df))):
        print(f"Sakha: {df['source'].iloc[i]}")
        print(f"Tokens: {df['source'].iloc[i].split()}")
        print(f"Russian: {df['target'].iloc[i]}")
        print(f"Tokens: {list(ru_tokenizer(df['target'].iloc[i]))}")
    
    print(f"\n{name} stats:")
    print(f"Char - Sakha: {df['src_char_len'].mean():.1f}±{df['src_char_len'].std():.1f}")
    print(f"Char - Russian: {df['tgt_char_len'].mean():.1f}±{df['tgt_char_len'].std():.1f}")
    print(f"Word - Sakha: {df['src_word_len'].mean():.1f}±{df['src_word_len'].std():.1f}")
    print(f"Word - Russian: {df['tgt_word_len'].mean():.1f}±{df['tgt_word_len'].std():.1f}")
    print(f"Char ratio: {df['char_ratio'].mean():.2f}±{df['char_ratio'].std():.2f}")
    print(f"Word ratio: {df['word_ratio'].mean():.2f}±{df['word_ratio'].std():.2f}")
    
    stats = []
    for metric, col in [('char_len', 'src_char_len'), ('char_len', 'tgt_char_len'), 
                       ('word_len', 'src_word_len'), ('word_len', 'tgt_word_len')]:
        lang = 'Sakha' if 'src' in col else 'Russian'
        stats.append({
            'metric': metric, 'type': lang,
            'mean': df[col].mean(), 'std': df[col].std(),
            'min': df[col].min(), 'max': df[col].max()
        })
    
    for ratio_type, col in [('char', 'char_ratio'), ('word', 'word_ratio')]:
        stats.append({
            'metric': 'ratio', 'type': ratio_type,
            'mean': df[col].mean(), 'std': df[col].std(),
            'min': df[col].min(), 'max': df[col].max()
        })
    
    pd.DataFrame(stats).to_csv(f"{OUTPUT_DIR}/{name}_statistics.csv", index=False)
    df.to_csv(f"{OUTPUT_DIR}/{name}_full_data.csv", index=False)
    return df

def plot_distributions(tat_data, wiki_data):
    plt.style.use('seaborn-v0_8')
    
    def add_stats(ax, data, label):
        stats = f'{label}:\nmean={data.mean():.1f}\nstd={data.std():.1f}\nmin={data.min():.1f}\nmax={data.max():.1f}\nmedian={data.median():.1f}'
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=tat_data, x='src_word_len', alpha=0.5, label='Sakha', color='blue')
    sns.histplot(data=tat_data, x='tgt_word_len', alpha=0.5, label='Russian', color='red')
    plt.title('Word Length (Tatoeba)')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats(plt.gca(), tat_data['src_word_len'], 'Sakha')
    add_stats(plt.gca(), tat_data['tgt_word_len'], 'Russian')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=wiki_data, x='src_word_len', alpha=0.5, label='Sakha', color='blue')
    sns.histplot(data=wiki_data, x='tgt_word_len', alpha=0.5, label='Russian', color='red')
    plt.title('Word Length (Wikimedia)')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats(plt.gca(), wiki_data['src_word_len'], 'Sakha')
    add_stats(plt.gca(), wiki_data['tgt_word_len'], 'Russian')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/word_length_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=tat_data, x='src_char_len', alpha=0.5, label='Sakha', color='blue')
    sns.histplot(data=tat_data, x='tgt_char_len', alpha=0.5, label='Russian', color='red')
    plt.title('Char Length (Tatoeba)')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats(plt.gca(), tat_data['src_char_len'], 'Sakha')
    add_stats(plt.gca(), tat_data['tgt_char_len'], 'Russian')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=wiki_data, x='src_char_len', alpha=0.5, label='Sakha', color='blue')
    sns.histplot(data=wiki_data, x='tgt_char_len', alpha=0.5, label='Russian', color='red')
    plt.title('Char Length (Wikimedia)')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.legend()
    add_stats(plt.gca(), wiki_data['src_char_len'], 'Sakha')
    add_stats(plt.gca(), wiki_data['tgt_char_len'], 'Russian')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/char_length_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    combined = pd.concat([tat_data, wiki_data])
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=combined, x='char_ratio', hue='dataset', alpha=0.5, multiple="layer")
    plt.title('Char Ratio (Russian/Sakha)')
    plt.xlabel('Char Ratio')
    plt.ylabel('Frequency')
    add_stats(plt.gca(), tat_data['char_ratio'], 'Tatoeba')
    add_stats(plt.gca(), wiki_data['char_ratio'], 'Wikimedia')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=combined, x='word_ratio', hue='dataset', alpha=0.5, multiple="layer")
    plt.title('Word Ratio (Russian/Sakha)')
    plt.xlabel('Word Ratio')
    plt.ylabel('Frequency')
    add_stats(plt.gca(), tat_data['word_ratio'], 'Tatoeba')
    add_stats(plt.gca(), wiki_data['word_ratio'], 'Wikimedia')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ratio_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

def make_table(stats_df, name):
    fig, ax = plt.subplots(figsize=(12, len(stats_df['metric'].unique()) * 1.2))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for _, row in stats_df.iterrows():
        table_data.append([row['metric'], row['type'], f"{row['mean']:.2f}", 
                          f"{row['std']:.2f}", f"{row['min']:.2f}", f"{row['max']:.2f}"])
    
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Type', 'Mean', 'Std', 'Min', 'Max'],
                    loc='center', cellLoc='center', colColours=['#f2f2f2'] * 6,
                    cellColours=[['#ffffff'] * 6] * len(table_data))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    plt.title(f'{name.capitalize()} Stats')
    plt.savefig(f"{OUTPUT_DIR}/{name}_statistics_table.png", dpi=300, bbox_inches='tight')
    plt.close()

def compare_tables(tat_stats, wiki_stats):
    fig, ax = plt.subplots(figsize=(15, len(tat_stats['metric'].unique()) * 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for metric in tat_stats['metric'].unique():
        for type_ in tat_stats['type'].unique():
            tat_row = tat_stats[(tat_stats['metric'] == metric) & (tat_stats['type'] == type_)]
            wiki_row = wiki_stats[(wiki_stats['metric'] == metric) & (wiki_stats['type'] == type_)]
            if not tat_row.empty and not wiki_row.empty:
                table_data.append([metric, type_, f"{tat_row['mean'].iloc[0]:.2f}",
                                 f"{wiki_row['mean'].iloc[0]:.2f}", f"{tat_row['std'].iloc[0]:.2f}",
                                 f"{wiki_row['std'].iloc[0]:.2f}"])
    
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Type', 'Tatoeba Mean', 'Wikimedia Mean', 'Tatoeba Std', 'Wikimedia Std'],
                    loc='center', cellLoc='center', colColours=['#f2f2f2'] * 6,
                    cellColours=[['#ffffff'] * 6] * len(table_data))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    plt.title('Tatoeba vs Wikimedia')
    plt.savefig(f"{OUTPUT_DIR}/dataset_comparison_table.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Parallel Text Analysis")
    print("=" * 50)
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nCurrent dir: {os.getcwd()}")
    print(f"Data dir: {PARALLEL_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    tat_sah = f"{PARALLEL_DIR}/tatoeba.sah"
    tat_ru = f"{PARALLEL_DIR}/tatoeba.ru"
    wiki_sah = f"{PARALLEL_DIR}/wikimedia.sah"
    wiki_ru = f"{PARALLEL_DIR}/wikimedia.ru"
    
    print(f"\nFiles:")
    for f in [tat_sah, tat_ru, wiki_sah, wiki_ru]:
        print(f"{f}: {os.path.getsize(f)/1024:.1f}KB")
    
    tat_data = load_data(tat_sah, tat_ru)
    wiki_data = load_data(wiki_sah, wiki_ru)
    
    print(f"\nSizes: Tatoeba={len(tat_data)}, Wikimedia={len(wiki_data)}")
    
    tat_data['dataset'] = 'Tatoeba'
    wiki_data['dataset'] = 'Wikimedia'
    
    tat_data = analyze_lengths(tat_data, 'tatoeba')
    wiki_data = analyze_lengths(wiki_data, 'wikimedia')
    
    print("\nMaking plots...")
    plot_distributions(tat_data, wiki_data)
    
    tat_stats = pd.read_csv(f"{OUTPUT_DIR}/tatoeba_statistics.csv")
    wiki_stats = pd.read_csv(f"{OUTPUT_DIR}/wikimedia_statistics.csv")
    
    make_table(tat_stats, 'tatoeba')
    make_table(wiki_stats, 'wikimedia')
    compare_tables(tat_stats, wiki_stats)
    
    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024
        print(f"{f}: {size:.1f}KB")
    
    print(f"\nDone! Results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()