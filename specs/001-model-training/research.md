# Research: Model Training Script

**Feature**: 001-model-training
**Date**: 2025-12-30
**Status**: Complete

## Overview

This document resolves technical clarifications and documents best practices for implementing the AI detection model training script.

---

## Research Items

### 1. Testing Framework Selection

**Decision**: pytest

**Rationale**:
- Industry standard for Python projects
- Already compatible with numpy, pandas ecosystem
- Simple test discovery and execution
- Good integration with CI/CD pipelines
- Rich plugin ecosystem (pytest-cov for coverage, pytest-mock for mocking)

**Alternatives Considered**:
- **unittest** (Python stdlib): More verbose, requires class-based tests, less modern
- **nose2**: Less actively maintained than pytest, smaller community
- **No testing framework**: Out of scope for MVP per spec.md, but pytest recommended for future

**Recommendation**: Add `pytest>=8.0.0` to pyproject.toml dependencies when testing is implemented (future work, not in current scope per FR requirements).

---

### 2. CSV Reading Best Practices

**Decision**: Use pandas.read_csv() with specific column selection and error handling

**Rationale**:
- pandas already in dependencies (2.3.3+)
- Efficient memory usage with `usecols=['content']` to load only needed column
- Built-in CSV dialect detection and error handling
- Easy random sampling with `df.sample(n=10000)`
- Natural integration with numpy arrays via `.values`

**Implementation Pattern**:
```python
import pandas as pd

def load_articles(folder_path, n_samples=10000):
    """Load and sample articles from CSV files."""
    # Get all CSV files in folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # Read and concatenate all CSVs
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, usecols=['content'])
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Sample n articles
    if len(combined) < n_samples:
        raise ValueError(f"Insufficient articles: need {n_samples}, found {len(combined)}")

    sampled = combined.sample(n=n_samples, random_state=42)
    return sampled['content'].values
```

**Alternatives Considered**:
- **csv module** (stdlib): More manual, requires iteration, no built-in sampling
- **numpy.genfromtxt()**: Less flexible for mixed data types, poor CSV dialect handling

---

### 3. Command-Line Argument Parsing

**Decision**: argparse (Python stdlib)

**Rationale**:
- No additional dependencies needed
- Sufficient for the required flags (--tfidf, --embeddings, --embeddings-tfidf, --lr, --iterations, --batch-size)
- Built-in help generation and type conversion
- Mutually exclusive groups for feature extraction methods

**Implementation Pattern**:
```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train AI detection model with various feature extraction methods'
    )

    # Feature extraction method (mutually exclusive)
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        '--tfidf',
        action='store_true',
        help='Use TF-IDF weighted features'
    )
    feature_group.add_argument(
        '--embeddings',
        action='store_true',
        help='Use GloVe mean word embeddings'
    )
    feature_group.add_argument(
        '--embeddings-tfidf',
        action='store_true',
        help='Use TF-IDF weighted GloVe embeddings'
    )

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--iterations', type=int, default=1000, help='Training iterations (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')

    return parser.parse_args()
```

**Alternatives Considered**:
- **click**: More features than needed, adds dependency
- **sys.argv manual parsing**: Error-prone, no type conversion, no help generation

---

### 4. Timestamp Format for Model Files

**Decision**: `YYYYMMDD_HHMM` format using datetime.now().strftime()

**Rationale**:
- Sortable by name (chronological order)
- Filesystem-safe (no colons or special characters)
- Human-readable
- Sufficient granularity (minute-level) for typical training workflows

**Implementation Pattern**:
```python
from datetime import datetime

def generate_model_filename(method):
    """Generate timestamped model filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    return f"models/{method}_{timestamp}.npy"

# Examples:
# wordfreq_20250130_1430.npy
# tfidf_20250130_1432.npy
# embeddings_20250130_1445.npy
```

**Alternatives Considered**:
- **ISO 8601 format** (2025-01-30T14:30:00): Contains colons, problematic on Windows
- **Unix timestamp**: Not human-readable
- **UUID**: No temporal information, not sortable

---

### 5. Random Sampling Strategy

**Decision**: Fixed random seed (random_state=42) for reproducibility

**Rationale**:
- Ensures reproducible results across runs for debugging
- Allows fair comparison between different feature extraction methods
- Standard practice in ML experiments (seed value 42 is conventional)
- Can be made configurable later if needed

**Implementation Pattern**:
```python
# In pandas sampling
sampled_human = df_human.sample(n=10000, random_state=42)
sampled_ai = df_ai.sample(n=10000, random_state=42)
```

**Alternatives Considered**:
- **Random seed from timestamp**: Non-reproducible
- **User-specified seed**: Added complexity, not required by spec
- **No seed (truly random)**: Makes debugging difficult

---

### 6. Error Handling Strategy

**Decision**: Fail-fast with descriptive error messages, check prerequisites upfront

**Rationale**:
- Aligns with FR-018 (validate GloVe availability before training)
- Prevents wasted computation time
- Clear user feedback for resolution
- Validates all inputs before starting expensive operations

**Implementation Pattern**:
```python
def validate_prerequisites(args):
    """Validate all prerequisites before training."""
    errors = []

    # Check data directories exist
    if not os.path.exists('data/human'):
        errors.append("Missing data/human/ directory")
    if not os.path.exists('data/google_gemini-2.0-flash-lite-001'):
        errors.append("Missing data/google_gemini-2.0-flash-lite-001/ directory")

    # Check sufficient articles
    human_count = count_articles('data/human')
    if human_count < 10000:
        errors.append(f"Insufficient human articles: need 10000, found {human_count}")

    ai_count = count_articles('data/google_gemini-2.0-flash-lite-001')
    if ai_count < 10000:
        errors.append(f"Insufficient AI articles: need 10000, found {ai_count}")

    # Check GloVe availability for embedding methods
    if args.embeddings or args.embeddings_tfidf:
        try:
            from nai.embeddings.glove import GloVeEmbeddings
            glove = GloVeEmbeddings()  # Will raise if not available
        except Exception as e:
            errors.append(f"GloVe embeddings not available: {e}")

    # Check/create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created models/ directory")

    if errors:
        print("ERROR: Prerequisites failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
```

**Alternatives Considered**:
- **Try-catch during training**: Wastes computation time
- **Silent fallbacks**: Confusing user experience, violates FR-018
- **Warnings only**: User might miss critical issues

---

### 7. Progress Reporting Pattern

**Decision**: Print progress every N iterations with iteration number and cost

**Rationale**:
- Satisfies FR-015 (display training progress with cost/loss values)
- Prevents log spam (not every iteration)
- Provides feedback that training is progressing
- Shows convergence behavior via cost values

**Implementation Pattern**:
```python
def train_with_progress(model, train_x, train_y, frequencies, idf_scores, lr, iterations, batch_size):
    """Train model with progress reporting."""
    print(f"Training with lr={lr}, iterations={iterations}, batch_size={batch_size}")
    print(f"Training on {len(train_x)} articles...")

    # Existing model.train() already handles this, but we can wrap it
    costs, weights = model.train(train_x, train_y, frequencies, idf_scores, lr, iterations, batch_size)

    # Print final cost
    print(f"Training complete. Final cost: {costs[-1]:.6f}")

    return costs, weights
```

**Note**: The existing `batch_gradient_descent` function in `nai.train.cost` already prints iteration progress, so minimal additional code needed.

**Alternatives Considered**:
- **Progress bar (tqdm)**: Already in dependencies, could enhance UX but not required
- **No progress**: Violates FR-015
- **Log file**: Added complexity, not required

---

### 8. Model Weight Shapes

**Decision**: Initialize RegressionModel with correct shape based on feature method

**Rationale**:
- Word frequency methods: 3 features (bias + human_count + ai_count) → shape (3, 1)
- Embedding methods: 100 features (GloVe dimension) → shape (100, 1)
- RegressionModel already supports shape parameter in __init__

**Implementation Pattern**:
```python
def create_model(args):
    """Create model with appropriate shape."""
    if args.embeddings or args.embeddings_tfidf:
        # GloVe is 100-dimensional
        return RegressionModel(shape=(100, 1))
    else:
        # Word frequency features: bias + positive + negative
        return RegressionModel(shape=(3, 1))
```

---

## Summary

All technical clarifications resolved:
- ✅ Testing: pytest (future work)
- ✅ CSV reading: pandas.read_csv() with sampling
- ✅ CLI parsing: argparse with mutually exclusive groups
- ✅ Timestamps: YYYYMMDD_HHMM format
- ✅ Sampling: Fixed random seed (42) for reproducibility
- ✅ Error handling: Fail-fast with upfront validation
- ✅ Progress reporting: Print iterations and cost values
- ✅ Model shapes: 3x1 for frequency, 100x1 for embeddings

No blocking issues identified. Implementation can proceed to Phase 1 (Design & Contracts).
