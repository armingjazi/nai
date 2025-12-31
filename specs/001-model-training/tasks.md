# Tasks: Model Training Script

**Feature**: 001-model-training
**Input**: Design documents from `/specs/001-model-training/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Not included (out of MVP scope per spec.md)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each feature extraction method.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- Repository root: `/Users/armin/Projects/nai/`
- Training script: `train_model.py`
- Existing modules: `nai/train/`, `nai/embeddings/`, `nai/process/`
- Model outputs: `models/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create basic script structure and prerequisite validation shared by all user stories

- [ ] T001 Create train_model.py skeleton at repository root with main() function and argparse setup
- [ ] T002 Implement prerequisite validation function (check data directories exist, check article counts >= 10000, create models/ directory if needed)
- [ ] T003 [P] Implement CSV article loading function using pandas (load from data/human/ and data/google_gemini-2.0-flash-lite-001/, sample 10000 from each, return texts and labels arrays)
- [ ] T004 [P] Implement argparse CLI with mutually exclusive feature flags (--tfidf, --embeddings, --embeddings-tfidf) and hyperparameter flags (--lr, --iterations, --batch-size with defaults)
- [ ] T005 Implement timestamp generator function for model filenames (format: YYYYMMDD_HHMM)

**Checkpoint**: Basic infrastructure ready - all user stories can use data loading, validation, and CLI parsing

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core training pipeline components shared by all feature extraction methods

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Import existing components from nai.train.model (RegressionModel), nai.train.normalization (normalize), and nai.process.text (process_text_to_words)
- [ ] T007 Implement model save function (save weights, mean_var, and optionally IDF scores with timestamped filenames to models/ directory)
- [ ] T008 Implement training progress display wrapper (print training start message, iteration/cost updates are already in batch_gradient_descent, print final cost)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Word Frequency Training (Priority: P1) 🎯 MVP

**Goal**: Train baseline AI detection model using simple word frequency features (no CLI flags)

**Independent Test**: Run `python train_model.py` without flags, verify models/wordfreq_YYYYMMDD_HHMM.npy and models/wordfreq_mean_var_YYYYMMDD_HHMM.npy are created

### Implementation for User Story 1

- [ ] T009 [US1] Implement word frequency training path in main(): load articles, build frequency dict using build_freqs() from nai.train.frequency
- [ ] T010 [US1] Extract word frequency features using extract_features() from nai.train.feature for all 20000 articles
- [ ] T011 [US1] Create RegressionModel with shape=(3,1), train using model.train() with word frequencies
- [ ] T012 [US1] Save word frequency model using save function with method="wordfreq"
- [ ] T013 [US1] Add error handling for missing/corrupted CSV files and insufficient article counts in word frequency path

**Checkpoint**: User Story 1 complete - baseline training works end-to-end, can demo `python train_model.py`

---

## Phase 4: User Story 2 - TF-IDF Enhanced Training (Priority: P2)

**Goal**: Enable TF-IDF weighted feature training via --tfidf flag

**Independent Test**: Run `python train_model.py --tfidf`, verify models/tfidf_*.npy, models/tfidf_mean_var_*.npy, and models/tfidf_idf_*.npy are created

### Implementation for User Story 2

- [ ] T014 [US2] Implement TF-IDF training path in main(): detect --tfidf flag, build frequency and document frequency dicts using build_freqs_docs() from nai.train.frequency
- [ ] T015 [US2] Compute IDF scores using compute_idf() from nai.train.frequency
- [ ] T016 [US2] Extract TF-IDF weighted features using extract_features_idf() from nai.train.feature for all 20000 articles
- [ ] T017 [US2] Create RegressionModel with shape=(3,1), train using model.train() with TF-IDF features
- [ ] T018 [US2] Save TF-IDF model including IDF scores file using save function with method="tfidf"
- [ ] T019 [US2] Verify IDF smoothing is applied (already in compute_idf) and handle edge cases

**Checkpoint**: User Stories 1 AND 2 both work independently - can demo both `python train_model.py` and `python train_model.py --tfidf`

---

## Phase 5: User Story 3 - Word Embeddings Training (Priority: P3)

**Goal**: Enable GloVe mean embedding training via --embeddings flag

**Independent Test**: Run `python train_model.py --embeddings` (with GloVe available), verify models/embeddings_*.npy and models/embeddings_mean_var_*.npy are created, check exclusion statistics printed

### Implementation for User Story 3

- [ ] T020 [US3] Add GloVe availability check to prerequisite validation: try to load GloVeEmbeddings from nai.embeddings.glove, fail immediately with clear error if unavailable when --embeddings flag used
- [ ] T021 [US3] Implement embeddings training path in main(): detect --embeddings flag, load GloVe embeddings
- [ ] T022 [US3] Extract embedding features using extract_features_glove_batch() from nai.embeddings.embedding_feature for all articles
- [ ] T023 [US3] Report exclusion statistics (number and percentage of articles with zero GloVe matches)
- [ ] T024 [US3] Create RegressionModel with shape=(100,1) for embeddings, train using model.train_with_embeddings()
- [ ] T025 [US3] Save embeddings model using save function with method="embeddings"
- [ ] T026 [US3] Handle edge case where all articles have zero embedding matches (fail with clear error)

**Checkpoint**: User Stories 1, 2, AND 3 all work independently - can demo all three training methods

---

## Phase 6: User Story 4 - TF-IDF Weighted Embeddings Training (Priority: P4)

**Goal**: Enable TF-IDF weighted GloVe embedding training via --embeddings-tfidf flag (most sophisticated method)

**Independent Test**: Run `python train_model.py --embeddings-tfidf` (with GloVe available), verify models/embeddings_tfidf_*.npy, models/embeddings_tfidf_mean_var_*.npy, and models/embeddings_tfidf_idf_*.npy are created

### Implementation for User Story 4

- [ ] T027 [US4] Implement embeddings-tfidf training path in main(): detect --embeddings-tfidf flag, load GloVe and compute IDF scores
- [ ] T028 [US4] Extract TF-IDF weighted embedding features using extract_features_glove_tfidf_batch() from nai.embeddings.embedding_feature
- [ ] T029 [US4] Report exclusion statistics for zero-match articles
- [ ] T030 [US4] Create RegressionModel with shape=(100,1), train using model.train_with_embeddings_tfidf()
- [ ] T031 [US4] Save embeddings-tfidf model including IDF scores using save function with method="embeddings_tfidf"
- [ ] T032 [US4] Verify default IDF value (1.0) is used for words in GloVe but not in training corpus (already in extract_features_glove_tfidf)

**Checkpoint**: All 4 user stories complete - all feature extraction methods functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements and validation across all training methods

- [ ] T033 [P] Add comprehensive error messages for all validation failures (missing directories, insufficient articles, invalid hyperparameters)
- [ ] T034 [P] Verify hyperparameter validation (--lr > 0, --iterations > 0, --batch-size > 0 and <= 20000)
- [ ] T035 [P] Test all 4 training methods end-to-end (run each command, verify outputs, check model file sizes)
- [ ] T036 [P] Validate timestamp format consistency across all saved model files
- [ ] T037 Manual validation using quickstart.md guide: run each training method from quickstart examples, verify outputs match expectations
- [ ] T038 Add docstring comments to main functions (prerequisite validation, data loading, save model)
- [ ] T039 Verify random seed (42) is consistently used for reproducible sampling in all training paths

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001-T005) completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase (T006-T008) completion
  - User stories can then proceed in parallel (if staffed) or sequentially by priority
  - Each story is independently testable
- **Polish (Phase 7)**: Depends on desired user stories being complete (recommend at least US1-US2)

### User Story Dependencies

- **User Story 1 (P1) - Word Frequency**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P2) - TF-IDF**: Can start after Foundational - No dependencies on other stories (uses same data loading as US1)
- **User Story 3 (P3) - Embeddings**: Can start after Foundational - Requires GloVe availability check (T020) but otherwise independent
- **User Story 4 (P4) - Embeddings-TF-IDF**: Can start after Foundational - Combines IDF (like US2) and embeddings (like US3) but is independently testable

### Within Each User Story

Tasks within a story are sequential (no [P] markers within user stories) because they all modify the same train_model.py file:
1. Implement training path detection (--flag handling)
2. Feature extraction specific to that method
3. Model creation with correct shape
4. Model training
5. Model saving with correct file naming
6. Error handling specific to that method

### Parallel Opportunities

**Setup Phase** (all can run in parallel):
- T003 (data loading) and T004 (CLI) are independent
- T005 (timestamp) is independent

**Foundational Phase** (limited parallelism):
- T006 (imports) must complete first
- T007 (save function) and T008 (progress display) can run in parallel after T006

**User Stories** (maximum parallelism):
- Once Foundational complete, ALL user stories (US1, US2, US3, US4) can be worked on in parallel by different developers
- Each story modifies different sections of train_model.py (if using feature branches)
- OR: implement sequentially in priority order (P1 → P2 → P3 → P4) on main branch

**Polish Phase** (all can run in parallel):
- T033, T034, T036, T038, T039 all modify different aspects
- T035 and T037 are validation tasks (can run after other polish tasks)

---

## Parallel Example: Foundational Phase

```bash
# After T006 imports complete:
Task: "Implement model save function (T007)"
Task: "Implement training progress display wrapper (T008)"
```

## Parallel Example: User Stories (Multi-Developer)

```bash
# After Foundational phase complete, launch all stories in parallel:
Developer A: "User Story 1 tasks (T009-T013)"
Developer B: "User Story 2 tasks (T014-T019)"
Developer C: "User Story 3 tasks (T020-T026)"
Developer D: "User Story 4 tasks (T027-T032)"
```

## Parallel Example: Polish Phase

```bash
# Launch all polish tasks together:
Task: "Add comprehensive error messages (T033)"
Task: "Verify hyperparameter validation (T034)"
Task: "Test all 4 methods end-to-end (T035)"
Task: "Validate timestamp format (T036)"
Task: "Add docstrings (T038)"
Task: "Verify random seed usage (T039)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T008) - **CRITICAL BLOCKER**
3. Complete Phase 3: User Story 1 (T009-T013)
4. **STOP and VALIDATE**: Run `python train_model.py`, verify model files created, verify can load model
5. Deploy/demo baseline training capability

**Estimated Time**: 1.5-2 hours for MVP (US1 only)

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready (~30 minutes)
2. Add User Story 1 → Test independently → **MVP deployed!** (~1 hour)
3. Add User Story 2 → Test independently → **TF-IDF available** (~30 minutes)
4. Add User Story 3 → Test independently → **Embeddings available** (~45 minutes)
5. Add User Story 4 → Test independently → **All methods available** (~30 minutes)
6. Polish phase → Production ready (~30 minutes)

**Total Estimated Time**: 4 hours for complete implementation

### Parallel Team Strategy

With 2 developers:

1. Both complete Setup + Foundational together (~30 minutes)
2. Once Foundational done:
   - Developer A: User Story 1 (MVP) + User Story 2
   - Developer B: User Story 3 + User Story 4
3. Merge and validate all stories work independently (~15 minutes)
4. Both run Polish phase tasks in parallel (~20 minutes)

**Total Estimated Time**: ~2.5 hours with 2 developers

---

## Notes

- [P] tasks = different files or independent validations, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story implements one feature extraction method and should be independently testable
- No test tasks included (tests out of scope per spec.md FR requirements)
- Commit after each user story phase to maintain clean git history
- Stop at any checkpoint to validate story independently
- All user stories modify train_model.py - use feature branches for parallel development or implement sequentially

---

## Task Summary

**Total Tasks**: 39

**By Phase**:
- Phase 1 (Setup): 5 tasks
- Phase 2 (Foundational): 3 tasks (BLOCKS all user stories)
- Phase 3 (US1 - Word Frequency): 5 tasks 🎯 MVP
- Phase 4 (US2 - TF-IDF): 6 tasks
- Phase 5 (US3 - Embeddings): 7 tasks
- Phase 6 (US4 - Embeddings-TF-IDF): 6 tasks
- Phase 7 (Polish): 7 tasks

**Parallel Opportunities**:
- Setup: 3 parallel tasks (T003, T004, T005)
- Foundational: 2 parallel tasks (T007, T008)
- User Stories: 4 parallel stories (US1-US4) if multi-developer team
- Polish: 6 parallel tasks (T033, T034, T036, T038, T039)

**Independent Test Criteria**:
- **US1**: Run `python train_model.py`, verify wordfreq model files created
- **US2**: Run `python train_model.py --tfidf`, verify tfidf model + IDF files created
- **US3**: Run `python train_model.py --embeddings`, verify embeddings model created and exclusions reported
- **US4**: Run `python train_model.py --embeddings-tfidf`, verify embeddings_tfidf model + IDF files created

**Suggested MVP Scope**: User Story 1 only (baseline word frequency training) - delivers end-to-end working training pipeline

**Format Validation**: ✅ All tasks follow checklist format with checkboxes, task IDs, optional [P]/[Story] labels, and file path descriptions
