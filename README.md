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
│   ├── raw/                  # Original corpus data from OPUS and OSCAR
│   │   ├── opus/             # OPUS parallel corpus data
│   │   └── oscar/            # OSCAR monolingual corpus data
│   ├── processed/            # Cleaned and preprocessed data
│   │   ├── parallel/         # Processed parallel corpus data
│   │   ├── monolingual/      # Processed monolingual data
│   │   └── analysis/         # Data analysis results
│   ├── muri_instructions/    # MURI-based instruction data
│   └── README.md             # Data documentation and guidelines
├── src/                      # Source code
│   ├── preprocessing/        # Data cleaning and preparation
│   │   ├── text_cleaner.py   # Text cleaning utilities for Yakut language
│   │   ├── process_oscar.py  # OSCAR corpus processing
│   │   ├── process_parallel.py # Parallel corpus processing
│   │   ├── translate_using_model.py # Neural translation model
│   │   ├── analysis/         # Data analysis tools
│   │   └── README.md         # Preprocessing documentation
│   ├── tokenization/         # Custom tokenizer implementation
│   ├── models/               # Model implementation and fine-tuning
│   ├── evaluation/           # Evaluation scripts and metrics
│   └── utils/                # Helper functions
│       ├── download_data.py  # Script to download OPUS and OSCAR corpora
│       └── run_translation.sh # Translation pipeline script
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Data Sources

- [OPUS Sakha-Russian corpus](https://opus.nlpl.eu/results/sah&ru/corpus-result-table)
- [OSCAR corpus](https://oscar-project.github.io/documentation/versions/oscar-2019/#downloading-oscar)
- MOREEE HEREEEE


## Key Components

### Data Processing

Further instructions for tokenizer development, model training, and evaluation will be added as development progresses. 
