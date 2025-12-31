# Quickstart: AI Detection Model Training

**Feature**: 001-model-training
**Date**: 2025-12-30

## Overview

This guide helps you quickly train an AI detection model using the `train_model.py` script.

---

## Prerequisites

### Required

1. **Python 3.12+** installed
2. **Data directories** with article CSV files:
   - `data/human/` - At least 10,000 human-written articles
   - `data/google_gemini-2.0-flash-lite-001/` - At least 10,000 AI-generated articles
3. **Dependencies** installed (via `uv` or `pip`):
   ```bash
   uv sync  # or: pip install -e .
   ```

### Optional (for embedding methods)

4. **GloVe embeddings** downloaded and accessible via `nai.embeddings.glove` module

---

## Quick Start (3 steps)

### 1. Verify Data

```bash
# Check human articles
ls data/human/*.csv | wc -l

# Check AI articles
ls data/google_gemini-2.0-flash-lite-001/*.csv | wc -l
```

Both should show at least 1 CSV file with 10,000+ articles total.

---

### 2. Run Training

**Baseline (Word Frequency)**:
```bash
python train_model.py
```

**With TF-IDF**:
```bash
python train_model.py --tfidf
```

**With Embeddings** (requires GloVe):
```bash
python train_model.py --embeddings
```

**With TF-IDF Weighted Embeddings** (requires GloVe):
```bash
python train_model.py --embeddings-tfidf
```

---

### 3. Check Results

```bash
# List trained models
ls -lh models/

# Example output:
# wordfreq_20250130_1430.npy
# wordfreq_mean_var_20250130_1430.npy
# tfidf_20250130_1432.npy
# tfidf_mean_var_20250130_1432.npy
# tfidf_idf_20250130_1432.npy
```

---

## Training Methods Explained

### 1. Word Frequency (Baseline)

**Command**: `python train_model.py`

**What it does**:
- Counts how often each word appears with human vs AI labels
- Creates 3 features: bias term + human word count + AI word count
- Fast and simple baseline

**When to use**:
- First experiment to validate pipeline works
- Benchmark for comparing other methods
- Quick iterations during development

**Expected time**: 2-5 minutes

---

### 2. TF-IDF Weighted Features

**Command**: `python train_model.py --tfidf`

**What it does**:
- Same as word frequency, but weights words by importance (TF-IDF)
- Words that appear in few documents get higher weights
- Helps model focus on discriminative words

**When to use**:
- After baseline shows promise
- Want better performance than raw word counts
- Don't have GloVe embeddings available

**Expected time**: 3-6 minutes

---

### 3. GloVe Embeddings (Mean Pooling)

**Command**: `python train_model.py --embeddings`

**What it does**:
- Converts words to 100-dimensional semantic vectors (GloVe)
- Averages all word vectors in an article
- Captures semantic meaning, not just word presence

**When to use**:
- Want to capture semantic patterns in writing style
- Have GloVe embeddings available
- Willing to accept slightly longer training time

**Expected time**: 5-8 minutes

**Note**: Some articles may be excluded if they have no words in GloVe vocabulary (reported in output).

---

### 4. TF-IDF Weighted Embeddings (Best Performance)

**Command**: `python train_model.py --embeddings-tfidf`

**What it does**:
- Combines GloVe embeddings with TF-IDF importance weighting
- Weights each word's embedding vector by its IDF score
- Emphasizes semantically distinctive words

**When to use**:
- Want best possible performance
- Have GloVe embeddings and time for longer training
- After validating other methods work

**Expected time**: 6-10 minutes

---

## Custom Hyperparameters

### Learning Rate

**Flag**: `--lr <value>`

**Default**: 0.001

**When to change**:
- Training cost not decreasing → try smaller learning rate (0.0001)
- Training too slow → try larger learning rate (0.01)

**Example**:
```bash
python train_model.py --tfidf --lr 0.01
```

---

### Iterations

**Flag**: `--iterations <value>`

**Default**: 1000

**When to change**:
- Cost still decreasing at iteration 1000 → increase iterations (2000, 5000)
- Cost converged early → decrease iterations (500)

**Example**:
```bash
python train_model.py --iterations 2000
```

---

### Batch Size

**Flag**: `--batch-size <value>`

**Default**: 256

**When to change**:
- Noisy cost values → try smaller batch size (128, 64)
- Training too slow → try larger batch size (512, 1024)

**Example**:
```bash
python train_model.py --batch-size 512
```

---

### Combined Example

```bash
python train_model.py --tfidf --lr 0.01 --iterations 2000 --batch-size 512
```

---

## Common Issues

### Issue 1: "Missing data/human/ directory"

**Cause**: Data directories not set up

**Solution**:
```bash
# Ensure data directories exist
ls -d data/human data/google_gemini-2.0-flash-lite-001

# If missing, download/create them
```

---

### Issue 2: "Insufficient human articles: need 10000, found 8543"

**Cause**: Not enough articles in CSV files

**Solution**:
- Download more article CSV files to the data directories
- Or reduce sample size (requires code modification - not supported in MVP)

---

### Issue 3: "GloVe embeddings not available"

**Cause**: GloVe embeddings not downloaded

**Solution**:
```bash
# Option 1: Download GloVe (ask team for instructions)

# Option 2: Use non-embedding methods instead
python train_model.py --tfidf  # Works without GloVe
```

---

### Issue 4: Training cost not decreasing

**Cause**: Learning rate too high or too low

**Solution**:
```bash
# Try smaller learning rate
python train_model.py --lr 0.0001

# Or larger learning rate
python train_model.py --lr 0.01
```

**Monitor output**: Cost should gradually decrease over iterations.

---

### Issue 5: "Permission denied" writing to models/

**Cause**: models/ directory not writable

**Solution**:
```bash
# Create models directory with write permissions
mkdir -p models
chmod 755 models

# Or run from different directory
```

---

## Output Files Explained

### Model Weights File

**Pattern**: `{method}_{timestamp}.npy`

**Example**: `wordfreq_20250130_1430.npy`

**Contents**: Trained weight vector (3x1 or 100x1 depending on method)

**Usage**: Load with `numpy.load()` or `RegressionModel.load()`

---

### Normalization Parameters File

**Pattern**: `{method}_mean_var_{timestamp}.npy`

**Example**: `wordfreq_mean_var_20250130_1430.npy`

**Contents**: Dictionary with `{mean: ndarray, var: ndarray}`

**Usage**: Required for making predictions on new data (ensures same normalization as training)

---

### IDF Scores File (TF-IDF methods only)

**Pattern**: `{method}_idf_{timestamp}.npy`

**Example**: `tfidf_idf_20250130_1430.npy`

**Contents**: Dictionary mapping words to IDF scores `{word: float}`

**Usage**: Required for TF-IDF feature extraction at prediction time

---

## Next Steps

After training a model:

1. **Evaluate performance** (out of scope for this feature, but you can):
   - Load model with `RegressionModel.load()`
   - Make predictions on test set
   - Calculate accuracy, precision, recall

2. **Compare methods**:
   - Train all 4 methods
   - Compare final cost values (lower is better)
   - Test on holdout data to see which generalizes best

3. **Tune hyperparameters**:
   - Try different learning rates
   - Adjust iterations based on convergence
   - Experiment with batch sizes

4. **Deploy model** (future work):
   - Load trained model in prediction script
   - Classify new articles as AI or human-written

---

## Tips for Best Results

### 1. Start Simple

Train baseline first:
```bash
python train_model.py
```

Verify it works before trying complex methods.

---

### 2. Monitor Training

Watch for:
- **Decreasing cost**: Good - model is learning
- **Flat cost**: Try different learning rate
- **Increasing cost**: Learning rate too high

---

### 3. Reproducibility

The script uses a fixed random seed (42), so:
- Same command → same sampled articles → same results
- Enables fair comparison between methods

---

### 4. Experiment Tracking

Keep a log of experiments:
```bash
# Log training command and results
python train_model.py --tfidf --lr 0.01 | tee experiments/tfidf_lr001.log
```

---

### 5. Model Naming

Timestamps help track experiments:
- `wordfreq_20250130_1430.npy` - Trained at 2:30 PM on Jan 30, 2025
- `tfidf_20250130_1432.npy` - Trained 2 minutes later

No manual renaming needed!

---

## Performance Expectations

| Method | Training Time | Memory | Expected Final Cost |
|--------|---------------|--------|---------------------|
| Word Freq | 2-5 min | ~200MB | ~0.3-0.4 |
| TF-IDF | 3-6 min | ~250MB | ~0.25-0.35 |
| Embeddings | 5-8 min | ~500MB | ~0.2-0.3 |
| Embeddings-TF-IDF | 6-10 min | ~500MB | ~0.15-0.25 |

**Note**: Lower final cost = better fit to training data (but watch for overfitting on real test data!)

---

## Getting Help

**View all options**:
```bash
python train_model.py --help
```

**Check version**:
```bash
python --version  # Should be 3.12+
```

**Verify dependencies**:
```bash
uv pip list | grep -E 'numpy|pandas|nltk'
```

---

## Example Complete Workflow

```bash
# 1. Verify prerequisites
python --version  # Check Python 3.12+
ls data/human/*.csv  # Check human articles
ls data/google_gemini-2.0-flash-lite-001/*.csv  # Check AI articles

# 2. Train baseline
python train_model.py
# ... watch training progress ...
# Final cost: 0.345678

# 3. Try TF-IDF
python train_model.py --tfidf
# ... watch training progress ...
# Final cost: 0.278901

# 4. Check models
ls -lh models/
# wordfreq_20250130_1430.npy
# wordfreq_mean_var_20250130_1430.npy
# tfidf_20250130_1432.npy
# tfidf_mean_var_20250130_1432.npy
# tfidf_idf_20250130_1432.npy

# 5. Experiment with hyperparameters
python train_model.py --tfidf --lr 0.01 --iterations 2000

# Done! Models ready for prediction.
```

---

## Summary

- **Quickest path**: `python train_model.py` (word frequency baseline)
- **Best accuracy**: `python train_model.py --embeddings-tfidf` (requires GloVe)
- **Best balance**: `python train_model.py --tfidf` (no GloVe needed, better than baseline)
- **Customization**: Add `--lr`, `--iterations`, `--batch-size` flags as needed
- **Output**: Timestamped model files in `models/` directory

Happy training! 🚀
