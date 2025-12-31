# Feature Specification: Model Training Script

**Feature Branch**: `001-model-training`
**Created**: 2025-12-30
**Status**: Draft
**Input**: User description: "create a python script to train a model based on the data (all the news dataset), the components exist already, we just need to fill the gaps and have the script that trains the model and saves it into a file. I want to pass to the script these options: 1- no option simple linear regression with word freq feature extraction. 2- --tfidf for term frequency inverse doc freq 3- --embeddings simple mean word embeddings 4- --embeddings-tfidf embeddings weighted by tf idf"

## Clarifications

### Session 2025-12-30

- Q: What are the CSV column names and what is the classification task? → A: CSV columns are `id,title,publication,author,date,year,month,url,content`. The text field is `content`. This is AI vs Human authorship detection, not sentiment analysis. Labels don't exist yet - they need to be created by sampling from data/human/ folder (label=0 for human-written) and data/google_gemini-2.0-flash-lite-001/ folder (label=1 for AI-generated), then combining into a labeled training dataset.
- Q: How many articles to sample from each folder? → A: 10,000 from human folder and 10,000 from AI folder (balanced dataset of 20,000 total articles)
- Q: What file naming convention for saved models? → A: Timestamped descriptive names like models/wordfreq_20250130_1430.npy, models/tfidf_20250130_1430.npy
- Q: How should users specify hyperparameters? → A: Optional flags with defaults: --lr 0.001 --iterations 1000 --batch-size 256
- Q: What should happen when GloVe embeddings are missing for --embeddings or --embeddings-tfidf? → A: Fail immediately with clear error message indicating GloVe path/installation needed

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Word Frequency Training (Priority: P1)

A data scientist wants to quickly train a baseline AI detection model using simple word frequency features on the news articles dataset. They run the training script without any options and get a trained model saved to disk that can distinguish between human-written and AI-generated articles.

**Why this priority**: This is the simplest and most fundamental training approach. It provides the baseline model that validates the entire training pipeline works end-to-end before adding more complex features.

**Independent Test**: Can be fully tested by running the script without options on samples from human and AI article folders, verifying a model file is created and can be loaded for predictions.

**Acceptance Scenarios**:

1. **Given** human articles exist in data/human/ and AI articles exist in data/google_gemini-2.0-flash-lite-001/, **When** the user runs the training script without options, **Then** the script samples articles from both folders, labels them (0=human, 1=AI), trains a linear regression model using word frequency features, and saves the model to a file
2. **Given** training completes successfully, **When** the user checks the output directory, **Then** model weights and normalization parameters are saved to files
3. **Given** a saved model file exists, **When** the user loads the model, **Then** the model can make predictions on new articles to detect AI authorship

---

### User Story 2 - TF-IDF Enhanced Training (Priority: P2)

A data scientist wants to improve AI detection performance by using TF-IDF weighting instead of raw word frequencies. They run the training script with the --tfidf flag and get a model that weights important discriminative words more heavily to better distinguish AI from human writing patterns.

**Why this priority**: TF-IDF is a well-established improvement over raw frequencies that helps the model focus on more informative words. This is the next logical step after the baseline.

**Independent Test**: Can be fully tested by running the script with --tfidf flag and comparing the resulting model's performance against the baseline word frequency model.

**Acceptance Scenarios**:

1. **Given** human and AI article datasets exist, **When** the user runs the training script with --tfidf flag, **Then** the script samples and labels articles, computes IDF scores, and trains using TF-IDF weighted features
2. **Given** TF-IDF training completes, **When** the user checks the output, **Then** both model weights and IDF scores are saved for later prediction use
3. **Given** insufficient document frequency data, **When** training with --tfidf, **Then** the script applies smoothing to prevent division by zero

---

### User Story 3 - Word Embeddings Training (Priority: P3)

A data scientist wants to leverage semantic information by using word embeddings (GloVe) to detect AI-generated content. They run the training script with the --embeddings flag and get a model that uses mean word embeddings as features, capturing semantic patterns that distinguish AI from human writing styles.

**Why this priority**: Word embeddings provide richer semantic features but require external resources (GloVe) and more computation. This is an advanced option after frequency-based approaches are validated.

**Independent Test**: Can be fully tested by running the script with --embeddings flag with GloVe embeddings available and verifying the model uses 100-dimensional embedding features.

**Acceptance Scenarios**:

1. **Given** GloVe embeddings are available and human/AI article datasets exist, **When** the user runs the training script with --embeddings flag, **Then** the script samples and labels articles, extracts mean word embedding features, and trains the model
2. **Given** some articles have no matching words in GloVe, **When** training with embeddings, **Then** the script excludes those articles and reports the exclusion count
3. **Given** embedding training completes, **When** the user saves the model, **Then** embedding model weights and normalization parameters are saved separately from TF-IDF models

---

### User Story 4 - TF-IDF Weighted Embeddings Training (Priority: P4)

A data scientist wants to combine the benefits of semantic embeddings with TF-IDF importance weighting for optimal AI detection. They run the training script with the --embeddings-tfidf flag and get a model that weights word embeddings by their IDF scores to focus on distinctive semantic patterns.

**Why this priority**: This combines two advanced techniques and represents the most sophisticated feature extraction approach. It depends on both embeddings and IDF computation working correctly.

**Independent Test**: Can be fully tested by running the script with --embeddings-tfidf flag and verifying the model uses IDF-weighted embeddings and achieves improved performance over unweighted embeddings.

**Acceptance Scenarios**:

1. **Given** GloVe embeddings and human/AI article datasets exist, **When** the user runs the training script with --embeddings-tfidf flag, **Then** the script samples and labels articles, computes IDF scores, and trains using TF-IDF weighted embeddings
2. **Given** TF-IDF weighted embedding training completes, **When** the user saves the model, **Then** both model weights and IDF scores are saved for prediction
3. **Given** a word appears in GloVe but not in the training corpus, **When** computing TF-IDF weighted embeddings, **Then** the script uses a default IDF value

---

### Edge Cases

- What happens when one or both article folders (human/AI) are empty or missing?
- What happens when either folder contains fewer than 10,000 articles?
- How does the system handle CSV files with missing or corrupted 'content' column data?
- How does the system handle missing GloVe embeddings when --embeddings or --embeddings-tfidf flags are used? → Script fails immediately with clear error message indicating GloVe path/installation needed
- What happens if the models/ directory does not exist?
- How does the script behave if training is interrupted midway?
- What happens when all articles in a batch have zero embedding matches?
- How does the system handle extremely imbalanced datasets (e.g., significantly more human than AI articles or vice versa)?
- What happens when CSV files have different column structures or missing columns?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Script MUST accept a command-line option to specify which feature extraction method to use (none for word frequency, --tfidf, --embeddings, or --embeddings-tfidf)
- **FR-002**: Script MUST load article CSV files from data/human/ directory (human-written articles) and data/google_gemini-2.0-flash-lite-001/ directory (AI-generated articles)
- **FR-003**: Script MUST read the 'content' column from CSV files (columns: id, title, publication, author, date, year, month, url, content)
- **FR-004**: Script MUST sample 10,000 articles from the human folder and 10,000 articles from the AI folder, creating a balanced combined training dataset of 20,000 articles with labels (0=human, 1=AI)
- **FR-005**: When no option is provided, script MUST train using simple word frequency features (human word counts and AI word counts with bias term)
- **FR-006**: When --tfidf flag is provided, script MUST compute document frequencies, calculate IDF scores, and train using TF-IDF weighted features
- **FR-007**: When --embeddings flag is provided, script MUST load GloVe embeddings and train using mean word embedding features
- **FR-008**: When --embeddings-tfidf flag is provided, script MUST compute IDF scores, load GloVe embeddings, and train using TF-IDF weighted embedding features
- **FR-009**: Script MUST build word-label frequency dictionaries from the labeled training data
- **FR-010**: Script MUST normalize features using mean and variance before training
- **FR-011**: Script MUST train a linear regression model with sigmoid activation using gradient descent
- **FR-012**: Script MUST save trained model weights to a timestamped file in models/ directory with format {method}_{YYYYMMDD}_{HHMM}.npy (e.g., wordfreq_20250130_1430.npy, tfidf_20250130_1430.npy)
- **FR-013**: Script MUST save normalization parameters (mean and variance) to a timestamped file with format {method}_mean_var_{YYYYMMDD}_{HHMM}.npy
- **FR-014**: For TF-IDF based methods, script MUST save IDF scores to a timestamped file with format {method}_idf_{YYYYMMDD}_{HHMM}.npy
- **FR-015**: Script MUST display training progress including cost/loss values during training
- **FR-016**: Script MUST allow user to optionally specify hyperparameters via command-line flags (--lr, --iterations, --batch-size) with defaults (learning rate=0.001, iterations=1000, batch size=256)
- **FR-017**: Script MUST handle articles with no embedding matches by excluding them and reporting exclusion statistics
- **FR-018**: Script MUST validate GloVe embeddings availability when --embeddings or --embeddings-tfidf flags are used, failing immediately with a clear error message if unavailable
- **FR-019**: Script MUST use existing components from nai.train and nai.embeddings modules without reimplementing core functionality

### Key Entities *(include if feature involves data)*

- **News Articles**: Text data from CSV files with columns (id, title, publication, author, date, year, month, url, content), separated into human-written and AI-generated folders
- **Article Content**: The 'content' column contains the full article text used for feature extraction
- **Labels**: Binary classification labels assigned during data preparation (0=human-written from data/human/, 1=AI-generated from data/google_gemini-2.0-flash-lite-001/)
- **Word Frequencies**: Dictionary mapping (word, label) pairs to their occurrence counts across the training corpus
- **IDF Scores**: Dictionary mapping words to their inverse document frequency values, measuring word importance in distinguishing AI from human writing
- **GloVe Embeddings**: Pre-trained word vectors providing semantic representations of words in 100-dimensional space
- **Model Weights**: Learned parameters of the linear regression model, shape depends on feature extraction method (3x1 for frequency-based, 100x1 for embeddings)
- **Normalization Parameters**: Mean and variance computed from training features, required for consistent prediction
- **Training Features**: Numerical representations of articles extracted using one of four methods (word freq, TF-IDF, embeddings, or weighted embeddings)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: User can train a baseline model by running the script without options in under 10 minutes on the full news dataset
- **SC-002**: User can train any of the four model variants (word freq, TF-IDF, embeddings, embeddings-TF-IDF) by specifying the appropriate command-line flag
- **SC-003**: Trained models are saved to disk and can be successfully loaded for making predictions without errors
- **SC-004**: User can see training progress including iteration number and cost values during training
- **SC-005**: Script completes training on the full news dataset without running out of memory or crashing
- **SC-006**: User receives clear error messages when required data (CSV files or GloVe embeddings) are missing
- **SC-007**: When using embedding-based methods, user is informed about the number and percentage of articles excluded due to zero matches
- **SC-008**: Model files are saved with timestamped descriptive names that indicate the feature extraction method and training time (e.g., models/wordfreq_20250130_1430.npy)

## Assumptions

- Article CSV files are located in data/human/ (human-written) and data/google_gemini-2.0-flash-lite-001/ (AI-generated) directories with consistent column structure (id, title, publication, author, date, year, month, url, content)
- The 'content' column contains sufficient text for feature extraction and classification
- GloVe embeddings are already downloaded and accessible via the existing nai.embeddings.glove module
- The combined dataset (after sampling) fits in memory for building frequency dictionaries and IDF scores
- Default hyperparameters are: learning rate=0.001, iterations=1000, batch size=256, which are reasonable values for the 20,000 article dataset
- The AI detection task is binary classification (0=human, 1=AI)
- Users have basic command-line knowledge and can run Python scripts with optional flags
- The existing RegressionModel class and feature extraction functions in the codebase are complete and tested
- Both human and AI article folders contain at least 10,000 articles each for training

## Out of Scope

- Model evaluation and testing functionality (accuracy, precision, recall metrics)
- Cross-validation or train/test splitting
- Hyperparameter tuning or grid search
- Support for multi-class classification
- Model comparison or performance benchmarking
- Web interface or GUI for training
- Distributed or GPU-accelerated training
- Real-time training progress visualization
- Support for other embedding types beyond GloVe
- Custom model architectures beyond linear regression
