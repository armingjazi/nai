# Data Model: Model Training Script

**Feature**: 001-model-training
**Date**: 2025-12-30

## Overview

This document defines the data structures and their relationships for the AI detection model training system.

---

## Core Entities

### 1. Article

**Purpose**: Represents a single news article from the dataset

**Attributes**:
| Attribute | Type | Required | Description | Validation Rules |
|-----------|------|----------|-------------|------------------|
| id | string | Yes | Unique identifier from CSV | Non-empty |
| title | string | Yes | Article headline | Non-empty |
| publication | string | Yes | Source publication name | Non-empty |
| author | string | No | Article author | May be empty |
| date | string | Yes | Publication date | Date format |
| year | integer | Yes | Publication year | Positive integer |
| month | integer | Yes | Publication month | 1-12 |
| url | string | Yes | Source URL | Valid URL format |
| content | string | Yes | Full article text | Non-empty, used for training |

**Source**: CSV files in `data/human/` and `data/google_gemini-2.0-flash-lite-001/`

**Lifecycle States**: N/A (immutable source data)

---

### 2. TrainingDataset

**Purpose**: Combined, labeled, and sampled dataset ready for training

**Attributes**:
| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| texts | list[string] | Article content from 'content' column | Length: 20,000 |
| labels | ndarray[int] | Binary labels (0=human, 1=AI) | Shape: (20000,), values: {0, 1} |
| size | integer | Total number of samples | Always 20,000 |
| human_count | integer | Number of human articles | Always 10,000 |
| ai_count | integer | Number of AI articles | Always 10,000 |

**Relationships**:
- Created by sampling 10,000 Article instances from each class
- Consumed by FeatureExtractor to produce TrainingFeatures

**Invariants**:
- `len(texts) == len(labels) == 20000`
- `human_count + ai_count == size`
- Labels are balanced: `sum(labels == 0) == sum(labels == 1) == 10000`

---

### 3. WordFrequencies

**Purpose**: Dictionary mapping (word, label) pairs to occurrence counts

**Attributes**:
| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| frequencies | dict[(str, int), int] | (word, label) → count | {("the", 0): 45000, ("the", 1): 42000} |

**Relationships**:
- Built from TrainingDataset texts + labels
- Used by word frequency and TF-IDF feature extractors
- Consumed by model training process

**Creation**: `nai.train.frequency.build_freqs(texts, labels)`

---

### 4. IDFScores

**Purpose**: Inverse Document Frequency scores for words

**Attributes**:
| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| idf_scores | dict[str, float] | word → IDF score | {"ai": 2.4, "human": 2.1, "the": 0.5} |
| doc_freqs | dict[str, int] | word → document frequency | {"ai": 500, "human": 600, "the": 19000} |
| total_docs | integer | Total number of documents | 20000 |

**Relationships**:
- Computed from TrainingDataset texts
- Used by TF-IDF and embeddings-TF-IDF feature extractors
- Saved to disk for prediction time reuse

**Creation**: `nai.train.frequency.build_freqs_docs()` + `compute_idf()`

**Formula**: `idf(word) = log((total_docs + 1) / (doc_freq + 1)) + 1`

---

### 5. TrainingFeatures

**Purpose**: Numerical feature matrix ready for model training

**Attributes**:
| Attribute | Type | Description | Shape |
|-----------|------|-------------|-------|
| features | ndarray[float] | Feature matrix | (20000, 3) or (20000, 100) |
| mean | ndarray[float] | Feature means for normalization | (2,) or (100,) |
| variance | ndarray[float] | Feature variances for normalization | (2,) or (100,) |

**Variants by Feature Extraction Method**:

| Method | Shape | Columns | Source |
|--------|-------|---------|--------|
| Word Frequency | (N, 3) | [bias, human_word_count, ai_word_count] | `extract_features()` |
| TF-IDF | (N, 3) | [bias, human_tfidf_sum, ai_tfidf_sum] | `extract_features_idf()` |
| Embeddings | (N, 100) | [100 GloVe dimensions (mean pooling)] | `extract_features_glove_batch()` |
| Embeddings-TF-IDF | (N, 100) | [100 GloVe dimensions (IDF-weighted)] | `extract_features_glove_tfidf_batch()` |

**Relationships**:
- Created from TrainingDataset using one of four feature extraction methods
- Normalized using `nai.train.normalization.normalize()`
- Consumed by RegressionModel.train()

**Invariants**:
- All features normalized: `mean ≈ 0`, `variance ≈ 1` (except bias column for frequency methods)
- No NaN or Inf values
- For embeddings: some articles may be excluded if no GloVe matches (reported in logs)

---

### 6. ModelWeights

**Purpose**: Learned parameters of the logistic regression model

**Attributes**:
| Attribute | Type | Description | Shape |
|-----------|------|-------------|-------|
| weights | ndarray[float] | Trained weight vector | (3, 1) or (100, 1) |
| mean | ndarray[float] | Training feature means | (2,) or (100,) |
| variance | ndarray[float] | Training feature variances | (2,) or (100,) |
| method | string | Feature extraction method | {"wordfreq", "tfidf", "embeddings", "embeddings_tfidf"} |
| timestamp | string | Training timestamp | YYYYMMDD_HHMM format |

**Relationships**:
- Produced by `RegressionModel.train()`
- Persisted to disk as NumPy .npy files
- Loaded at prediction time via `RegressionModel.load()`

**File Structure**:
```
models/
├── {method}_{timestamp}.npy              # weights
├── {method}_mean_var_{timestamp}.npy     # {mean: ndarray, var: ndarray}
└── {method}_idf_{timestamp}.npy          # IDF scores (TF-IDF methods only)
```

---

### 7. TrainingProgress

**Purpose**: Runtime training metrics reported during training

**Attributes**:
| Attribute | Type | Description |
|-----------|------|-------------|
| iteration | integer | Current iteration number (1 to max_iterations) |
| cost | float | Current cost/loss value |
| costs_history | list[float] | Historical cost values from all iterations |

**Relationships**:
- Generated during `batch_gradient_descent()` execution
- Printed to stdout for user feedback (FR-015)
- Returned as list from model.train()

**Display Format**: Printed by existing `nai.train.cost.batch_gradient_descent()` function

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Loading & Sampling                                  │
│                                                              │
│  CSV Files (data/human/, data/ai/)                          │
│         ↓                                                    │
│  pandas.read_csv() + sample(10000)                          │
│         ↓                                                    │
│  TrainingDataset (texts: 20000, labels: [0]*10000 + [1]*10000)│
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Extraction (branch by CLI flag)                  │
│                                                              │
│  [--tfidf]                                                   │
│   TrainingDataset → build_freqs_docs() → WordFrequencies    │
│                  → compute_idf() → IDFScores                │
│                  → extract_features_idf() → TrainingFeatures (3 cols) │
│                                                              │
│  [--embeddings]                                              │
│   TrainingDataset → GloVeEmbeddings.load()                 │
│                  → extract_features_glove_batch()           │
│                  → TrainingFeatures (100 cols)              │
│                                                              │
│  [--embeddings-tfidf]                                        │
│   TrainingDataset → compute_idf() + GloVeEmbeddings        │
│                  → extract_features_glove_tfidf_batch()     │
│                  → TrainingFeatures (100 cols)              │
│                                                              │
│  [default: word freq]                                        │
│   TrainingDataset → build_freqs() → WordFrequencies         │
│                  → extract_features() → TrainingFeatures (3 cols) │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Normalization                                            │
│                                                              │
│  TrainingFeatures → normalize() → (normalized_features, mean, var) │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Model Training                                            │
│                                                              │
│  RegressionModel.train(features, labels, lr, iterations, batch_size) │
│         ↓                                                    │
│  batch_gradient_descent() → (costs, weights)                │
│         ↓                                                    │
│  TrainingProgress (printed to stdout)                       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Persistence                                               │
│                                                              │
│  ModelWeights → save to models/{method}_{timestamp}.npy    │
│  (mean, var) → save to models/{method}_mean_var_{timestamp}.npy │
│  IDFScores → save to models/{method}_idf_{timestamp}.npy (if applicable) │
└─────────────────────────────────────────────────────────────┘
```

---

## Validation Rules Summary

### Input Validation (before training):
1. ✅ Human articles directory exists and contains ≥10,000 articles
2. ✅ AI articles directory exists and contains ≥10,000 articles
3. ✅ All CSV files have 'content' column
4. ✅ GloVe embeddings available (if --embeddings or --embeddings-tfidf)
5. ✅ models/ directory exists or can be created

### Runtime Invariants:
1. ✅ TrainingDataset is balanced: 10,000 human + 10,000 AI
2. ✅ TrainingFeatures shape matches model weight shape
3. ✅ No NaN or Inf values in features after normalization
4. ✅ Learning rate > 0, iterations > 0, batch_size > 0

### Output Validation:
1. ✅ Model files successfully saved with correct timestamps
2. ✅ Model can be loaded without errors
3. ✅ Saved mean/var can be loaded and applied to new data

---

## Notes

- **Existing implementations**: All entity transformations use existing functions from `nai.train`, `nai.embeddings`, `nai.process` modules
- **Memory considerations**: Entire dataset (20,000 articles) must fit in memory; this is acceptable per SC-005
- **Reproducibility**: Random seed (42) ensures same sampling across runs
- **Error handling**: Fail-fast validation before any expensive operations (data loading, feature extraction, training)
