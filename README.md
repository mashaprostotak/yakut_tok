# Project Title

SOMETHINGGGGG HEREEEEE

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
│   ├── raw/                    # Original corpus data from OPUS and OSCAR
│   │   ├── opus/                 # OPUS parallel corpus data
│   │   ├── oscar/                # OSCAR monolingual corpus data
|   |   └── xquad/                # XQuAD english version data
│   ├── processed/              # Cleaned and preprocessed data
│   │   ├── analysis/parallel/    # Statitic & visual analysis results for the data corpus
│   │   ├── masked/               # Dataset used for MLM task (Phase4)
│   │   ├── monolingual/          # Text data used for training the tokenizer
|   |   ├── parallel/             # Parallel (multiple) corpus data for model training (Phase1)
|   |   ├── synthetic/            # Evaluation dataset for the final model
|   |   └── xquad/                # XQuAD data used for model training (Phase3)
│   ├── muri_instructions/        # MURI-based instruction data for model training (Phase2)
│   └── README.md                 # Data documentation and guidelines
├── src/                      # Source code
│   ├── preprocessing/          # Data cleaning and preparation
│   │   ├── text_cleaner.py       # Text cleaning utilities for Yakut language
│   │   ├── process_oscar.py      # OSCAR corpus processing
│   │   ├── process_parallel.py   # Parallel corpus processing
│   │   ├── translate_using_model.py # Neural translation model
│   │   ├── analysis/             # Data analysis tools
│   │   └── README.md             # Preprocessing documentation
│   ├── tokenization/           # Custom tokenizer code implementation
│   ├── training/               # Model implementation and fine-tuning files
│   ├── evaluation/             # Evaluation scripts and metrics
│   └── utils/                  # Helper functions
│       ├── check_uniqueness.py   # Script to to check duplicate entries in synthetic JSON dataset
│       ├── csv_to_json.py        # Script to convert synthetic CSV file to JSON file
│       └── run_translation.sh    # Script for translation pipeline (from Yakut to English)
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Data Sources

- [OPUS Sakha-Russian corpus](https://opus.nlpl.eu/results/sah&ru/corpus-result-table)
- [OSCAR corpus](https://oscar-project.github.io/documentation/versions/oscar-2019/#downloading-oscar)
- [Sakha-NLP Raw-Datasets](https://github.com/nlp-sakha/sakha-embeddings?tab=readme-ov-file)


## Key Components

### Data Processing

Further instructions for tokenizer development, model training, and evaluation will be added as development progresses. 
