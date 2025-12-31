# CLI Contract: train_model.py

**Feature**: 001-model-training
**Version**: 1.0.0
**Date**: 2025-12-30

## Overview

This document defines the command-line interface contract for the AI detection model training script.

---

## Command Signature

```bash
python train_model.py [OPTIONS]
```

---

## Options

### Feature Extraction Method (Mutually Exclusive)

**Default**: Word frequency baseline (no flag)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tfidf` | boolean | `false` | Use TF-IDF weighted features |
| `--embeddings` | boolean | `false` | Use GloVe mean word embeddings |
| `--embeddings-tfidf` | boolean | `false` | Use TF-IDF weighted GloVe embeddings |

**Constraints**:
- At most ONE feature extraction flag can be specified
- If multiple flags provided, script exits with error
- No flag means word frequency baseline (FR-005)

### Hyperparameters (Optional)

| Flag | Type | Default | Description | Validation |
|------|------|---------|-------------|------------|
| `--lr` | float | `0.001` | Learning rate | Must be > 0 |
| `--iterations` | integer | `1000` | Number of training iterations | Must be > 0 |
| `--batch-size` | integer | `256` | Batch size for gradient descent | Must be > 0 and ≤ 20000 |

---

## Usage Examples

### 1. Basic Word Frequency Training (Baseline)

```bash
python train_model.py
```

**Behavior**:
- Samples 10,000 human + 10,000 AI articles
- Builds word-label frequency dictionaries
- Trains with word frequency features (bias + human_count + ai_count)
- Uses defaults: lr=0.001, iterations=1000, batch_size=256
- Saves to `models/wordfreq_20250130_1430.npy` (example timestamp)

**Expected Output**:
```
Loading articles from data/human/...
Loaded 15432 human articles, sampling 10000
Loading articles from data/google_gemini-2.0-flash-lite-001/...
Loaded 12876 AI articles, sampling 10000
Building word frequency dictionaries...
Extracting features from training data...
Training input:
 [[1.   0.5  0.3]
  [1.   0.4  0.6]
  ...]
Training with lr=0.001, iterations=1000, batch_size=256
Iteration 100, Cost: 0.523412
Iteration 200, Cost: 0.412345
...
Iteration 1000, Cost: 0.234567
Training complete. Final cost: 0.234567
Model saved to models/wordfreq_20250130_1430.npy
Normalization parameters saved to models/wordfreq_mean_var_20250130_1430.npy
```

---

### 2. TF-IDF Enhanced Training

```bash
python train_model.py --tfidf
```

**Behavior**:
- Same as baseline, but uses TF-IDF weighted features
- Computes IDF scores from document frequencies
- Saves additional file: `models/tfidf_idf_20250130_1432.npy`

**Expected Output**:
```
Loading articles...
Building word frequency dictionaries and IDF scores...
Extracting TF-IDF weighted features...
Training with lr=0.001, iterations=1000, batch_size=256
...
Model saved to models/tfidf_20250130_1432.npy
Normalization parameters saved to models/tfidf_mean_var_20250130_1432.npy
IDF scores saved to models/tfidf_idf_20250130_1432.npy
```

---

### 3. GloVe Embeddings Training

```bash
python train_model.py --embeddings
```

**Behavior**:
- Validates GloVe availability before training (fail-fast)
- Extracts mean word embeddings (100-dimensional)
- May exclude articles with zero GloVe matches
- Reports exclusion statistics

**Expected Output**:
```
Loading articles...
Validating GloVe embeddings availability...
GloVe embeddings loaded successfully
Extracting embedding features from training data...
Excluded 42 articles with zero matches (0.21%)
Training on 19958 articles
Training with lr=0.001, iterations=1000, batch_size=256
...
Model saved to models/embeddings_20250130_1445.npy
Normalization parameters saved to models/embeddings_mean_var_20250130_1445.npy
```

---

### 4. TF-IDF Weighted Embeddings

```bash
python train_model.py --embeddings-tfidf
```

**Behavior**:
- Combines GloVe embeddings with TF-IDF weighting
- Computes IDF scores and loads GloVe
- May exclude articles with zero GloVe matches
- Saves both IDF scores and model

**Expected Output**:
```
Loading articles...
Validating GloVe embeddings availability...
Building IDF scores...
Extracting TF-IDF weighted embedding features...
Excluded 38 articles with zero matches (0.19%)
Training on 19962 articles
...
Model saved to models/embeddings_tfidf_20250130_1450.npy
Normalization parameters saved to models/embeddings_tfidf_mean_var_20250130_1450.npy
IDF scores saved to models/embeddings_tfidf_idf_20250130_1450.npy
```

---

### 5. Custom Hyperparameters

```bash
python train_model.py --tfidf --lr 0.01 --iterations 2000 --batch-size 512
```

**Behavior**:
- Uses TF-IDF features with custom hyperparameters
- Higher learning rate, more iterations, larger batch size

**Expected Output**:
```
Loading articles...
Building word frequency dictionaries and IDF scores...
Extracting TF-IDF weighted features...
Training with lr=0.01, iterations=2000, batch_size=512
Iteration 100, Cost: 0.421234
...
Iteration 2000, Cost: 0.156789
Training complete. Final cost: 0.156789
Model saved to models/tfidf_20250130_1455.npy
...
```

---

## Exit Codes

| Code | Meaning | Example Scenarios |
|------|---------|-------------------|
| `0` | Success | Training completed, model saved successfully |
| `1` | Validation Error | Missing data directories, insufficient articles, GloVe unavailable |
| `1` | Argument Error | Invalid hyperparameter values, mutually exclusive flags |
| `1` | Runtime Error | Disk full, permission denied writing to models/ |

---

## Error Messages

### 1. Missing Data Directories

```bash
python train_model.py
```

**Error Output**:
```
ERROR: Prerequisites failed:
  - Missing data/human/ directory
Exit code: 1
```

---

### 2. Insufficient Articles

```bash
python train_model.py
```

**Error Output**:
```
ERROR: Prerequisites failed:
  - Insufficient human articles: need 10000, found 8543
Exit code: 1
```

---

### 3. GloVe Unavailable

```bash
python train_model.py --embeddings
```

**Error Output**:
```
ERROR: Prerequisites failed:
  - GloVe embeddings not available: GloVe file not found at /path/to/glove.npy
  - Please download GloVe embeddings or use --tfidf or word frequency methods
Exit code: 1
```

---

### 4. Mutually Exclusive Flags

```bash
python train_model.py --tfidf --embeddings
```

**Error Output**:
```
usage: train_model.py [-h] [--tfidf | --embeddings | --embeddings-tfidf] [--lr LR] [--iterations ITERATIONS] [--batch-size BATCH_SIZE]
train_model.py: error: argument --embeddings: not allowed with argument --tfidf
Exit code: 2
```

---

### 5. Invalid Hyperparameter

```bash
python train_model.py --lr -0.5
```

**Error Output**:
```
ERROR: Learning rate must be > 0, got -0.5
Exit code: 1
```

---

## File Outputs

### Generated Files

All files created in `models/` directory with timestamped names:

| File Pattern | Description | Size | Format |
|--------------|-------------|------|--------|
| `{method}_{timestamp}.npy` | Model weights | ~1KB (3x1) or ~400B (100x1) | NumPy binary |
| `{method}_mean_var_{timestamp}.npy` | Normalization params | ~1KB | NumPy dict: `{mean: ndarray, var: ndarray}` |
| `{method}_idf_{timestamp}.npy` | IDF scores (TF-IDF methods only) | Varies (~100KB-1MB) | NumPy dict: `{word: float}` |

**Method Names**:
- `wordfreq` - Word frequency baseline
- `tfidf` - TF-IDF weighted features
- `embeddings` - GloVe mean embeddings
- `embeddings_tfidf` - TF-IDF weighted embeddings

**Timestamp Format**: `YYYYMMDD_HHMM` (e.g., `20250130_1430`)

---

## Help Output

```bash
python train_model.py --help
```

**Output**:
```
usage: train_model.py [-h] [--tfidf | --embeddings | --embeddings-tfidf]
                      [--lr LR] [--iterations ITERATIONS] [--batch-size BATCH_SIZE]

Train AI detection model with various feature extraction methods

optional arguments:
  -h, --help            show this help message and exit
  --tfidf               Use TF-IDF weighted features
  --embeddings          Use GloVe mean word embeddings
  --embeddings-tfidf    Use TF-IDF weighted GloVe embeddings
  --lr LR               Learning rate (default: 0.001)
  --iterations ITERATIONS
                        Training iterations (default: 1000)
  --batch-size BATCH_SIZE
                        Batch size (default: 256)
```

---

## Performance Expectations

Based on SC-001: Train baseline model in <10 minutes on 20,000 articles

| Method | Expected Training Time | Memory Usage |
|--------|------------------------|--------------|
| Word Frequency | 2-5 minutes | ~200MB |
| TF-IDF | 3-6 minutes | ~250MB |
| Embeddings | 5-8 minutes | ~500MB (includes GloVe) |
| Embeddings-TF-IDF | 6-10 minutes | ~500MB |

**Note**: Times are approximate and depend on hardware (CPU speed, RAM).

---

## Contract Validation

### Input Contract:
- ✅ Flags are mutually exclusive (enforced by argparse)
- ✅ Hyperparameters validated (> 0)
- ✅ Data directories validated before training
- ✅ GloVe availability checked for embedding methods

### Output Contract:
- ✅ Model files created with correct naming pattern
- ✅ Timestamps reflect actual training time
- ✅ All required files saved (weights + normalization params + IDF if applicable)
- ✅ Progress logged to stdout during training
- ✅ Final cost/loss printed on completion

### Error Contract:
- ✅ Clear error messages for all failure modes
- ✅ Non-zero exit codes on error
- ✅ No partial state (fail before training or after complete save)
- ✅ Help message available via --help

---

## Versioning

**Version**: 1.0.0

**Breaking Changes**: N/A (initial version)

**Future Compatibility Notes**:
- File naming pattern (`{method}_{timestamp}.npy`) should remain stable
- Adding new feature extraction methods would be backward compatible
- Changing hyperparameter defaults could affect reproducibility
