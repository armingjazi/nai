import numpy as np
from typing import Tuple, Optional, List
from nai.embeddings.glove import GloVeEmbeddings
from nai.process.text import process_text_to_words


def extract_features_glove(headline: str, glove: GloVeEmbeddings) -> Optional[np.ndarray]:
    """Extract mean embedding feature from headline.

    Tokenizes the headline, looks up embeddings for each word, and computes
    the mean of all available embeddings. Headlines with zero matches are
    excluded by returning None.

    Args:
        headline: News headline text
        glove: Loaded GloVe embeddings

    Returns:
        np.ndarray of shape (1, 100) - mean of word embeddings
        None if no words have embeddings (zero-match exclusion)
    """
    words = process_text_to_words(headline)

    embeddings = []
    for word in words:
        vec = glove.get_vector(word)
        if vec is not None:
            embeddings.append(vec)

    if len(embeddings) == 0:
        return None

    mean_vector = np.mean(embeddings, axis=0)
    return mean_vector[np.newaxis, :]


def extract_features_glove_batch(
    headlines: List[str],
    glove: GloVeEmbeddings
) -> Tuple[np.ndarray, List[int]]:
    """Extract features for multiple headlines with zero-match filtering.

    Args:
        headlines: List of headline strings
        glove: Loaded GloVe embeddings

    Returns:
        Tuple of (features, excluded_indices)
        - features: np.ndarray of shape (N, 100) where N <= len(headlines)
        - excluded_indices: List of indices of headlines that were excluded
    """
    features_list = []
    excluded_indices = []

    for i, headline in enumerate(headlines):
        feature = extract_features_glove(headline, glove)
        if feature is not None:
            features_list.append(feature[0])
        else:
            excluded_indices.append(i)

    if len(features_list) == 0:
        return np.zeros((0, 100), dtype=np.float32), excluded_indices

    features = np.array(features_list, dtype=np.float32)
    return features, excluded_indices


def extract_features_glove_tfidf(headline: str, glove: GloVeEmbeddings, idf_scores: dict) -> Optional[np.ndarray]:
    """Extract TF-IDF weighted embedding feature from headline.

    Instead of simple mean pooling, weight each word embedding by its IDF score.
    This gives more importance to discriminative words while preserving semantic information.

    Args:
        headline: News headline text
        glove: Loaded GloVe embeddings
        idf_scores: Dictionary mapping words to IDF scores

    Returns:
        np.ndarray of shape (1, 100) - TF-IDF weighted sum of word embeddings
        None if no words have embeddings (zero-match exclusion)
    """
    words = process_text_to_words(headline)

    weighted_embeddings = []
    total_weight = 0.0

    for word in words:
        vec = glove.get_vector(word)
        if vec is not None:
            idf = idf_scores.get(word, 1.0)
            weighted_embeddings.append(vec * idf)
            total_weight += idf

    if len(weighted_embeddings) == 0:
        return None

    weighted_sum = np.sum(weighted_embeddings, axis=0)

    if total_weight > 0:
        normalized_vector = weighted_sum / total_weight
    else:
        normalized_vector = weighted_sum

    return normalized_vector[np.newaxis, :]


def extract_features_glove_tfidf_batch(
    headlines: List[str],
    glove: GloVeEmbeddings,
    idf_scores: dict
) -> Tuple[np.ndarray, List[int]]:
    """Extract TF-IDF weighted features for multiple headlines.

    Args:
        headlines: List of headline strings
        glove: Loaded GloVe embeddings
        idf_scores: Dictionary mapping words to IDF scores

    Returns:
        Tuple of (features, excluded_indices)
        - features: np.ndarray of shape (N, 100) where N <= len(headlines)
        - excluded_indices: List of indices of headlines that were excluded
    """
    features_list = []
    excluded_indices = []

    for i, headline in enumerate(headlines):
        feature = extract_features_glove_tfidf(headline, glove, idf_scores)
        if feature is not None:
            features_list.append(feature[0])
        else:
            excluded_indices.append(i)

    if len(features_list) == 0:
        return np.zeros((0, 100), dtype=np.float32), excluded_indices

    features = np.array(features_list, dtype=np.float32)
    return features, excluded_indices


def extract_features_glove_max(headline: str, glove: GloVeEmbeddings) -> Optional[np.ndarray]:
    """Extract max pooling embedding feature from headline.

    Takes the maximum activation along each dimension rather than the mean.
    This preserves the strongest signal from any single word.

    Args:
        headline: News headline text
        glove: Loaded GloVe embeddings

    Returns:
        np.ndarray of shape (1, 100) - max of word embeddings across each dimension
        None if no words have embeddings (zero-match exclusion)
    """
    words = process_text_to_words(headline)

    embeddings = []
    for word in words:
        vec = glove.get_vector(word)
        if vec is not None:
            embeddings.append(vec)

    if len(embeddings) == 0:
        return None

    max_vector = np.max(embeddings, axis=0)
    return max_vector[np.newaxis, :]


def extract_features_glove_max_batch(
    headlines: List[str],
    glove: GloVeEmbeddings
) -> Tuple[np.ndarray, List[int]]:
    """Extract max pooling features for multiple headlines.

    Args:
        headlines: List of headline strings
        glove: Loaded GloVe embeddings

    Returns:
        Tuple of (features, excluded_indices)
        - features: np.ndarray of shape (N, 100) where N <= len(headlines)
        - excluded_indices: List of indices of headlines that were excluded
    """
    features_list = []
    excluded_indices = []

    for i, headline in enumerate(headlines):
        feature = extract_features_glove_max(headline, glove)
        if feature is not None:
            features_list.append(feature[0])
        else:
            excluded_indices.append(i)

    if len(features_list) == 0:
        return np.zeros((0, 100), dtype=np.float32), excluded_indices

    features = np.array(features_list, dtype=np.float32)
    return features, excluded_indices


def compute_coverage(
    headline: str,
    glove: GloVeEmbeddings
) -> Tuple[int, int, float]:
    """Compute vocabulary coverage for headline.

    Args:
        headline: Headline text
        glove: Loaded GloVe embeddings

    Returns:
        (matched_words, total_words, coverage_ratio)
    """
    words = process_text_to_words(headline)
    total_words = len(words)

    if total_words == 0:
        return (0, 0, 0.0)

    matched_words = sum(1 for word in words if glove.has_word(word))
    coverage = matched_words / total_words

    return (matched_words, total_words, coverage)
