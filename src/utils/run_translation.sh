CORPUS=${1:-all}

echo "Running preprocessing pipeline for corpus: $CORPUS"
echo "Using model-based Russian to English translation"


mkdir -p data/processed/parallel
mkdir -p data/processed/analysis


python3 project/src/preprocessing/run_preprocessing.py \
  --process-parallel \
  --translate \
  --parallel-corpus $CORPUS \
  --src-lang sah \
  --tgt-lang en \
  --skip-analysis

echo "Translation completed successfully!"
echo "Translated files are in data/processed/parallel/" 