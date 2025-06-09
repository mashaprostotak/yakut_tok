## Yakut NLP Project - Preprocessing Pipeline

This directory contains scripts for preprocessing Yakut language data, including parallel corpora and monolingual data.

### Model-based Translation

The preprocessing pipeline now uses a pre-trained neural machine translation model for translating Russian to English, which provides significantly better quality than dictionary-based approaches. The model uses the Helsinki-NLP/opus-mt-ru-en model from Hugging Face.

### Main Components

- `run_preprocessing.py`: Main entry point for the preprocessing pipeline
- `process_parallel.py`: Process parallel corpus data (Yakut-Russian/English)
- `process_oscar.py`: Process monolingual Yakut data from OSCAR
- `text_cleaner.py`: Text cleaning utilities
- `translate_using_model.py`: Neural machine translation from Russian to English
- `analysis/`: Tools for analyzing processed data

### Usage

The simplest way to run the preprocessing pipeline is to use the shell script:

```bash
./project/src/utils/run_translation.sh [corpus]
```

where `[corpus]` is optional and can be "wikimedia", "tatoeba", or "all" (default).

For more control, you can run the preprocessing script directly:

```bash
python -m src.preprocessing.run_preprocessing --process-parallel --translate
```

### Command Line Options

- `--process-parallel`: Process parallel corpus data
- `--process-monolingual`: Process monolingual data
- `--all`: Process all available data
- `--translate`: Translate Russian text to English using the neural model
- `--parallel-corpus {wikimedia,tatoeba,all}`: Which parallel corpus to process
- `--src-lang`: Source language code (default: "sah")
- `--tgt-lang`: Target language code (default: "en")
- `--skip-analysis`: Skip analysis steps
- `--max-lines`: Maximum number of lines to process (for testing)
- `--batch-size`: Batch size for translation (default: 8)

### Dependencies

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- sentencepiece
- tqdm
- pandas (for analysis)
- matplotlib (for analysis)
- seaborn (for analysis)

### Installation

```bash
pip install torch transformers sentencepiece tqdm pandas matplotlib seaborn
```

### Directory Structure

- `project/data/raw/opus/`: Raw parallel corpus data (Wikimedia, Tatoeba)
- `project/data/raw/oscar/`: Raw monolingual Yakut data
- `project/data/processed/parallel/`: Processed parallel corpus (Yakut-Russian-English)
- `project/data/processed/monolingual/`: Processed monolingual data
- `project/data/processed/analysis/`: Analysis results and statistics
- `project/data/muri_instructions/`: MURI-based instruction tuning data
- `project/src/preprocessing/analysis/`: Analysis tools and scripts
  - `parallel/`: Parallel corpus analysis
  - `mono/`: Monolingual corpus analysis
  - `english/`: English translation analysis

## Contents

- `text_cleaner.py`: Core text cleaning utilities
- `process_oscar.py`: Script for processing the OSCAR monolingual corpus
- `process_parallel.py`: Script for processing parallel corpora (Wikimedia, Tatoeba)

## Usage

### Processing OSCAR Corpus

```bash
# Extract and process OSCAR corpus
python -m src.preprocessing.process_oscar --input-dir project/data/raw/oscar --output-dir project/data/processed/monolingual

# Extract only without processing
python -m src.preprocessing.process_oscar --input-dir project/data/raw/oscar --extract-only

# Process only without extraction
python -m src.preprocessing.process_oscar --input-dir project/data/raw/oscar --output-dir project/data/processed/monolingual --clean-only
```

### Processing Parallel Corpora

```bash
# Process all parallel corpora (Wikimedia and Tatoeba)
python -m src.preprocessing.process_parallel --input-dir project/data/raw/opus --output-dir project/data/processed/parallel

# Process only Wikimedia corpus
python -m src.preprocessing.process_parallel --input-dir project/data/raw/opus --output-dir project/data/processed/parallel --corpus wikimedia

# Process only Tatoeba corpus
python -m src.preprocessing.process_parallel --input-dir project/data/raw/opus --output-dir project/data/processed/parallel --corpus tatoeba
```

### Running the Full Pipeline

For convenience, a runner script is provided to execute the entire preprocessing pipeline:

```bash
# Run the full preprocessing pipeline
python -m scripts.run_preprocessing

# Process only OSCAR corpus
python -m scripts.run_preprocessing --oscar-only

# Process only parallel corpora
python -m scripts.run_preprocessing --parallel-only

# Create only tokenizer samples (after processing)
python -m scripts.run_preprocessing --samples-only
```

## Customization

The cleaning process can be customized by modifying the parameters in the scripts. Key configurable options include:

- URL removal
- HTML tag removal
- Unicode normalization
- Minimum character/word requirements
- Length ratio filtering for parallel data

## Output

Processed data will be stored in the following locations:

- Monolingual data: `project/data/processed/monolingual/`
- Parallel data: `project/data/processed/parallel/`
- Tokenizer samples: `project/data/processed/tokenizer_samples/`

## Statistics

After processing, you can run the data statistics notebook to analyze the results:

```bash
jupyter notebook project/scripts/data_statistics.ipynb
```

This will provide insights into:
- Character distributions
- Sentence length statistics
- Filtering effectiveness
- Data quality metrics 