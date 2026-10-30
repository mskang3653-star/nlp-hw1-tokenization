# NLP Homework 1: Tokenization and Word Vectors

An implementation and evaluation of byte pair encoding (BPE) tokenizers and Word2Vec models for a natural language processing course assignment.

## Contents

- `bpe.py`: BPE implementation using the Python standard library.
- `test_bpe_public.py`: public tests for the BPE implementation.
- `hw1.ipynb`: experiments and analysis.
- `hw1_helpers.py`: corpus, tokenizer, and evaluation helpers.
- `prepare_corpus.py`: prepares deterministic corpus slices.
- `artifacts/bpe_*.json`: trained byte-level BPE tokenizers with several vocabulary sizes.
- `results/`: experiment tables and a plot.
- `data/checksums.txt`: SHA-256 checksums of the corpus slices used in the experiments.

The corpus text and Word2Vec model binaries are not included. The corpus preparation script defaults to the `Salesforce/wikitext` dataset, configuration `wikitext-103-raw-v1`.

## Setup and checks

```bash
python -m pip install -r requirements.txt
python test_bpe_public.py
```

To regenerate the corpus files before running the notebook:

```bash
python prepare_corpus.py
```

The notebook uses `data/corpus_train.txt`, `data/corpus_valid.txt`, and `data/corpus_small.txt`. Their expected checksums are in `data/checksums.txt`.

## Load a saved tokenizer

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("artifacts/bpe_8000.json")
encoded = tokenizer.encode("A short example sentence.")
print(encoded.tokens)
```

The saved tokenizer JSON files are produced with the `tokenizers` library. The custom implementation in `bpe.py` follows the assignment's separate BPE specification.
