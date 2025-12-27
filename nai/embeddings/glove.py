from typing import Optional, Set
import numpy as np


class GloVeEmbeddings:
    """Manages pre-trained GloVe word vectors."""

    def __init__(self, file_path: str, vocab: Optional[Set[str]] = None) -> None:
        """Load GloVe embeddings with optional vocabulary filtering.

        Args:
            file_path: Path to GloVe .txt file (e.g., glove.6B.100d.txt)
            vocab: Optional set of words to load. If provided, only these words
                   are loaded into memory for efficient filtering.
        """
        if vocab is not None:
            vocab = set(word.lower() for word in vocab)

        self.embeddings = self._load_embeddings(file_path, vocab)
        self.dimension = 100
        self.vocabulary = set(self.embeddings.keys())

    def _load_embeddings(
        self, file_path: str, vocab: Optional[Set[str]]
    ) -> dict[str, np.ndarray]:
        """Load embeddings from GloVe .txt file.

        GloVe format: word dim1 dim2 ... dim100 (space-separated)
        Each line has 101 values: word + 100 floating-point dimensions.
        """
        embeddings = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                values = line.split()

                if len(values) != 101:
                    print(f"Warning: Line {line_num} has {len(values)} values, expected 101")
                    continue

                word = values[0].lower()

                if vocab is not None and word not in vocab:
                    continue

                try:
                    vector = np.array(values[1:], dtype=np.float32)
                    embeddings[word] = vector
                except ValueError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue

        print(f"Loaded {len(embeddings)} word embeddings")
        return embeddings

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for word (case-insensitive).

        Args:
            word: Word to look up

        Returns:
            100-dimensional numpy array if word exists, None otherwise
        """
        return self.embeddings.get(word.lower(), None)

    def has_word(self, word: str) -> bool:
        """Check if word exists in vocabulary (case-insensitive).

        Args:
            word: Word to check

        Returns:
            True if word is in vocabulary, False otherwise
        """
        return word.lower() in self.embeddings

    def vocabulary_size(self) -> int:
        """Get number of loaded words.

        Returns:
            Count of words in vocabulary
        """
        return len(self.embeddings)
