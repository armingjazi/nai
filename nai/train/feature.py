import numpy as np
from nai.process.text import process_text_to_words


def extract_features(headline, frequencies):
    """Extract features from news data.
    Input:
        headlline: a headlline sentence
        frequencies: a dictionary mapping each (word, sentiment) pair to its frequency
    Output:
        features: a list of features for the headline (a 1 x 3 array) (batch, bias, positive, negative)
    """
    words = process_text_to_words(headline)
    
    x = np.zeros(3) 
    
    # bias term is set to 1
    x[0] = 1 
    for word in words:
        
        x[1] += frequencies.get((word, 1), 0)
        
        x[2] += frequencies.get((word, 0), 0)
            
    
    # adding batch dimension
    x = x[None, :]  

    return x

def extract_features_idf(headline, freqs, idf_scores):
    """
    Extract TF-IDF weighted features
    TF: term frequency
    IDF: inverse document frequency
    TF-IDF = TF * IDF
    TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents (corpus).
    TF-IDF is often used in text mining and information retrieval to evaluate the significance of a word in a document.
    TF-IDF is a common feature representation in natural language processing (NLP) tasks, such as text classification and information retrieval.
    Inputs:
        headline: single news text
        freqs: word-label frequency dict
        idf_scores: word -> idf
    Output:
        feature vector (1 x 3)
    """
    words = process_text_to_words(headline)
    
    x = np.zeros(3)
    x[0] = 1  # bias term

    for word in words:
        idf = idf_scores.get(word, 1)  # default IDF=1 if unseen word
        x[1] += freqs.get((word, 1), 0) * idf  # positive class
        x[2] += freqs.get((word, 0), 0) * idf  # negative class

    return x[None, :]  # maintain batch dimension
