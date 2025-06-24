# Project Title

Tokenization Matters: Improving Low-Resource Language Modeling for Yakut with Custom Tokenizer

# Project Description

This repository contains the codebase and datasets for developing NLP tools and models for the low-resource language, Yakut (Sakha).

## Project Overview

Project goals:
- Process Yakut-Russian parallel corpora and monolingual text
- Develop custom tokenizer for Yakut
- Integrate the tokenizer with a pre-trained LLM
- Fine-tune the resultant language model for Yakut
- Evaluate model performance

## Repository Structure

```
project/
├── data/                     # Data storage
│   ├── raw/                      # Original corpus data from OPUS and OSCAR
│   │   ├── opus/                     # OPUS parallel corpus data
│   │   ├── oscar/                    # OSCAR monolingual corpus data
|   |   └── xquad/                    # XQuAD english version data
│   ├── processed/                # Cleaned and preprocessed data
│   │   ├── analysis/parallel/        # Statitic & visual analysis results for the data corpus
│   │   ├── masked/                   # Dataset used for MLM task (Phase4)
│   │   ├── monolingual/              # Text data used for training the tokenizer
|   |   ├── parallel/                 # Parallel (multiple) corpus data for model training (Phase1)
|   |   ├── synthetic/                # Evaluation dataset for the final model
|   |   └── xquad/                    # XQuAD data used for model training (Phase3)
│   ├── muri_instructions/        # MURI-based instruction data for model training (Phase2)
│   └── README.md                 # Data documentation and guidelines
├── results/                 # All the results of evaluations
|   ├── synthetic/                # Results for the Synthetic Dataset
|   |    ├── synthetic_llama_eval     # Results of Llama 3.2 on Synthetic Dataset
|   |    └── synthetic_our_model_eval # Results of our model on Synthetic Dataset
|   └── xquad/                    # Results for the XQuAD Dataset
|        ├── xquad_llama_eval         # Results of Llama 3.2 on XQuAD Dataset
|        └── xquad_our_model_eval     # Results of our model on XQuAD Dataset
├── src/                     # Source code
│   ├── evaluation/               # Evaluation scripts and metrics
|   |   ├── model_synthetic_eval      # Evaluation of Llama 3.2 & our model on the synthetic dataset
|   |   ├── synthetic_dataset_translation # Translation of Synthetic dataset from English to Yakut
|   |   └── xquads_translations       # Translation of XQuAD dataset from English to Yakut
│   ├── preprocessing/            # Data cleaning and preparation
|   |   ├── analysis/                 # Data analysis tools
|   |   ├── create_muri_instructuions # File to create instructions from MURI Dataset
│   │   ├── process_oscar.py          # OSCAR corpus processing
│   │   ├── process_parallel.py       # Parallel corpus processing
|   |   ├── run_preprocessing.py      # File to run the preprocessing tasks
│   │   ├── text_cleaner.py           # Text cleaning utilities for Yakut language
│   │   └── translate_using_model.py  # Neural translation model
│   ├── tokenizer/                # Custom tokenizer code implementation
|   |   └── tokenizer_training        # Code file for custom tokenizer training
│   ├── training/                 # Model implementation and fine-tuning files
|   |   ├── phase_1_training          # File for the initial fine-tuning phase
|   |   └── phase2_training&XQuAD_evaluation # File for the second phase training and evluation on XQuAD
│   └── utils/                    # Helper functions
│       ├── check_uniqueness.py       # Script to to check duplicate entries in synthetic JSON dataset
│       ├── csv_to_json.py            # Script to convert synthetic CSV file to JSON file
│       └── run_translation.sh        # Script for translation pipeline (from Yakut to English)
├── tokenizers               # Necessary files for our tokenizer
|   ├── trained tokenizer/        # Files of the trained tokenizer
|   └── yakut-llama-model         # Files of the trained model
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Data Sources

- [OPUS Sakha-Russian corpus](https://opus.nlpl.eu/results/sah&ru/corpus-result-table)
- [OSCAR corpus](https://oscar-project.github.io/documentation/versions/oscar-2019/#downloading-oscar)
- [Sakha-NLP Raw-Datasets](https://github.com/nlp-sakha/sakha-embeddings?tab=readme-ov-file)
- [MURI-IT Dataset](https://huggingface.co/datasets/akoksal/muri-it)

## Reproducing Results

This section provides high-level instructions to reproduce the results presented in this project. The complete pipeline involves data preprocessing, tokenizer training, model fine-tuning, and evaluation across multiple phases.

### Prerequisites

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - Download raw datasets (OPUS, OSCAR, XQuAD) into `data/raw/`
   

### Pipeline Overview

The reproduction process follows these main phases:

#### Phase 1: Data Preprocessing and Analysis
```bash
python src/preprocessing/run_preprocessing.py --all
```

#### Phase 2: Custom Tokenizer Development
```bash
jupyter notebook src/tokenizer/tokenizer_training.ipynb
```

#### Phase 3: Model Training
**Phase 1 Training (Initial Fine-tuning)**
```bash
jupyter notebook src/training/phase_1_training.ipynb
```

**Phase 2 Training (Instruction Following)**
```bash
jupyter notebook src/training/phase2_training&Xquad_evaluation.ipynb
```

#### Phase 4: Dataset Translation and Evaluation
```bash
python src/evaluation/synthetic_dataset_translation.py
python src/evaluation/xquads_translation.py

jupyter notebook src/evaluation/model_synthetic_eval.ipynb
```

### Expected Outputs

The pipeline generates the following key results:

1. **Data Analysis Reports** (`data/processed/analysis/`)
   - Statistical analysis of parallel and monolingual corpora

2. **Trained Models** (`tokenizers/`)
   - Custom Yakut tokenizer
   - Fine-tuned language model with integrated tokenizer

3. **Evaluation Results** (`results/`)
   - Performance metrics on synthetic dataset
   - XQuAD question-answering evaluation results
   - Comparative analysis with baseline models (LLaMA 3.2)

### Key Evaluation Metrics

- **Question Answering**: Exact Match (EM), F1 scores on XQuAD
- **Language Generation**: Fluency and coherence metrics
