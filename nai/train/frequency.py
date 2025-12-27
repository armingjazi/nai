from collections import defaultdict
import math
import numpy as np
from nai.process.text import process_text_to_words


def build_freqs(texts, ys, word_processor=process_text_to_words):
    """Build frequencies.
    Input:
        texts: a list of texts
        ys: an m x 1 array with the sentiment label of each text
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """

    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, text in zip(yslist, texts):
        for word in word_processor(text):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def build_freqs_docs(texts, labels, word_processor=process_text_to_words):
    """Build frequencies.
    Input:
        texts: a list of texts
        ys: an m x 1 array with the sentiment label of each text
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    freqs = defaultdict(int)
    doc_freqs = defaultdict(int)
    N = len(texts)

    for text, label in zip(texts, labels):
        words = word_processor(text)
        unique_words = set(words)  # **Set to count once per doc**
        for word in unique_words:
            doc_freqs[word] += 1
        for word in words:
            freqs[(word, label)] += 1

    return freqs, doc_freqs, N

def compute_idf(doc_freqs, total_docs):
    """
    Compute the IDF scores
    Inputs:
        doc_freqs: dict mapping word -> doc frequency
        total_docs: total number of docs
    Returns:
        idf_scores: dict mapping word -> idf score
    """
    idf_scores = {}
    for word, df in doc_freqs.items():
        idf_scores[word] = math.log((total_docs + 1) / (df + 1)) + 1  # +1 for smoothing
    return idf_scores