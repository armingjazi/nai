# Implementation Plan: Model Training Script

**Branch**: `001-model-training` | **Date**: 2025-12-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-model-training/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a CLI training script that trains AI vs Human authorship detection models using four different feature extraction methods: (1) word frequency baseline, (2) TF-IDF weighting, (3) GloVe embeddings, and (4) TF-IDF weighted embeddings. The script samples 10,000 articles from each class, trains a logistic regression model using existing nai.train and nai.embeddings components, and saves timestamped model artifacts to the models/ directory.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: numpy 2.4.0+, pandas 2.3.3+, nltk 3.9.2+, existing nai.train and nai.embeddings modules
**Storage**: CSV files (data/human/, data/google_gemini-2.0-flash-lite-001/), NumPy binary files (.npy) for model persistence
**Testing**: NEEDS CLARIFICATION (pytest recommended for Python projects)
**Target Platform**: Local development machine (macOS/Linux/Windows), command-line execution
**Project Type**: Single project (CLI script within existing nai package)
**Performance Goals**: Train baseline model in <10 minutes on 20,000 articles, in-memory processing
**Constraints**: Memory footprint must accommodate 20,000 articles + embeddings, must reuse existing nai.train.model.RegressionModel
**Scale/Scope**: Single training script, 4 feature extraction methods, ~20,000 articles training set, 100-dimensional embeddings

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ✓ PASS - No project constitution defined yet

The project constitution file (`.specify/memory/constitution.md`) contains only a template. No specific architectural principles, testing requirements, or quality gates are currently enforced.

**Recommendations for future**:
- Define testing strategy (unit tests for data loading, feature extraction, model training)
- Establish code organization principles (separation of concerns: data loading, feature extraction, training, persistence)
- Set performance benchmarks and monitoring requirements

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
nai/                          # Existing package
├── __init__.py
├── embeddings/               # Existing: GloVe embeddings (glove.py, embedding_feature.py)
├── train/                    # Existing: Training components
│   ├── model.py             # RegressionModel class (reused)
│   ├── feature.py           # extract_features, extract_features_idf
│   ├── frequency.py         # build_freqs, build_freqs_docs, compute_idf
│   ├── activation.py        # sigmoid
│   ├── cost.py              # batch_gradient_descent
│   └── normalization.py     # normalize, normalize_with_mean_var
├── process/                  # Existing: Text processing
│   └── text.py              # process_text_to_words
├── predict/                  # Existing: Prediction utilities
└── generate/                 # Existing: Generation utilities

train_model.py               # NEW: Main training script (this feature)

data/                         # Existing data directories
├── human/                    # Human-written articles (CSV files)
└── google_gemini-2.0-flash-lite-001/  # AI-generated articles (CSV files)

models/                       # NEW: Model artifacts directory
└── [timestamped .npy files]  # Created by training script

tests/                        # FUTURE: Test directory (not in scope)
└── [future test files]
```

**Structure Decision**: Single project structure. The new training script (`train_model.py`) will be added at the repository root and will orchestrate existing components from the `nai` package. No new modules needed - all feature extraction and model training logic already exists in `nai.train`, `nai.embeddings`, and `nai.process`.

## Complexity Tracking

No complexity violations - this feature reuses existing components and adds a single orchestration script.

---

## Planning Summary

### Phase 0: Research ✅ COMPLETE

**Deliverable**: `research.md`

**Resolved Items**:
- Testing framework selection: pytest (future work, out of MVP scope)
- CSV reading approach: pandas with usecols and sampling
- CLI argument parsing: argparse with mutually exclusive groups
- Timestamp format: YYYYMMDD_HHMM for filesafe, sortable names
- Random sampling: Fixed seed (42) for reproducibility
- Error handling: Fail-fast validation before training
- Progress reporting: Print iterations and cost values
- Model weight shapes: 3x1 for frequency, 100x1 for embeddings

**Key Decisions**:
- No new dependencies needed beyond existing project deps
- Reuse all existing nai.train and nai.embeddings components
- Single Python script at repository root (train_model.py)

---

### Phase 1: Design & Contracts ✅ COMPLETE

**Deliverables**:
- `data-model.md` - 7 core entities defined with validation rules
- `contracts/cli-contract.md` - CLI interface specification with examples
- `quickstart.md` - User guide for running the training script
- `CLAUDE.md` - Updated agent context with Python 3.12 + dependencies

**Data Model Entities**:
1. Article (CSV source data)
2. TrainingDataset (20,000 balanced samples)
3. WordFrequencies (word-label count dictionaries)
4. IDFScores (inverse document frequency values)
5. TrainingFeatures (normalized numerical matrices)
6. ModelWeights (learned parameters + normalization params)
7. TrainingProgress (runtime cost/loss tracking)

**CLI Contract**:
- 4 feature extraction methods (default + 3 flags)
- 3 optional hyperparameter flags (--lr, --iterations, --batch-size)
- Timestamped model file outputs
- Comprehensive error messages and exit codes
- Help documentation via --help

---

### Constitution Re-Check ✅ PASS

**Status**: No violations

The design maintains simplicity by:
- Reusing 100% of existing components (no new modules)
- Adding only one orchestration script
- No new dependencies
- Clear separation of concerns (data loading → feature extraction → training → persistence)

---

### Next Steps

**Immediate**: Run `/speckit.tasks` to generate implementation task list

**Implementation order** (recommended):
1. Create train_model.py skeleton with argparse
2. Implement data loading and sampling (pandas)
3. Implement prerequisite validation
4. Wire up existing feature extraction methods
5. Wire up existing model training
6. Implement timestamped file saving
7. Test all 4 feature extraction methods
8. Document edge cases and error handling

**Estimated Implementation Time**: 2-4 hours (mostly wiring existing components)

**Testing Strategy** (future work):
- Unit tests for data loading and sampling
- Integration tests for each feature extraction method
- End-to-end test: train → save → load → predict
