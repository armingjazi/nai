import numpy as np

from nai.train.activation import sigmoid
from nai.train.cost import batch_gradient_descent
from nai.train.feature import extract_features_idf
from nai.train.normalization import normalize, normalize_with_mean_var
from nai.embeddings.glove import GloVeEmbeddings
from nai.embeddings.embedding_feature import (
    extract_features_glove_batch,
    extract_features_glove_tfidf_batch,
    extract_features_glove_max_batch
)


class RegressionModel:
    def __init__(self, shape=(3, 1)):
        self.weights = np.zeros(shape)  # Initialize weights to zero
        self.mean = None
        self.var = None

    def predict(self, inputs):
        """
        Predict using the learned weights.
        :param inputs: Feature matrix (already TF-IDF processed)
        :return: Predicted probabilities
        """
        z = np.dot(inputs, self.weights)
        return sigmoid(z)
    
    def predict_sentiments(self, headlines, frequencies, idf_scores):
        """
        Predict using the learned weights.
        :param headlines: List of headlines
        :param frequencies: Frequencies of words
        :param idf_scores: IDF scores of words
        :return: Predicted probabilities
        """
        n = len(headlines)
        predict_input = np.zeros((n, 3))

        for i in range(n):
            predict_input[i, :] = extract_features_idf(headlines[i], frequencies, idf_scores)

        predict_input[:, 1:] = normalize_with_mean_var(predict_input[:, 1:], self.mean, self.var)

        return self.predict(predict_input)    

    def predict_sentiment(self, headline, frequencies, idf_scores):
        """
        Make a prediction using the model
        """
        features = extract_features_idf(headline, frequencies, idf_scores)
        return self.predict(features)
    
    def train(self, train_x, train_y, frequencies, idf_scores, learning_rate, iterations, batch_size=2048, optimizer=batch_gradient_descent):
        train_input = np.zeros((len(train_x), 3))

        print("Extracting features from training data...")
        
        for i in range(len(train_x)):
            train_input[i, :] = extract_features_idf(train_x[i], frequencies, idf_scores)

        train_input[:, 1:], self.mean, self.var = normalize(train_input[:, 1:])

        # labels
        train_target = train_y.reshape(-1, 1)

        print(f"Training input: \n {train_input[:5]} \n {train_input[-5:]} ")
        print(f"Training label: \n {train_target[:5]} \n {train_target[-5:]} ")

        costs, weights = optimizer(train_input, train_target, self.weights, learning_rate, iterations, batch_size)

        self.weights = weights

        return costs, weights
    
    def load(self, model_file, mean_var_file):
        """
        Load the model weights from a file
        """
        self.weights = np.load(model_file, allow_pickle=True)
        mean_var = np.load(mean_var_file, allow_pickle=True).item()
        self.mean = mean_var['mean']
        self.var = mean_var['var']
        if self.mean is None or self.var is None:
            raise ValueError("Mean and variance must be loaded from the file.")
        if self.weights is None:
            raise ValueError("Weights must be loaded from the file.")
        
        print(f"Model loaded from {model_file}")

    def save(self, model_file, mean_var_file):
        """
        Save the model weights to a file
        """
        np.save(model_file, self.weights)
        np.save(mean_var_file, {'mean': self.mean, 'var': self.var})

        print(f"Model saved to {model_file}")

    def train_with_embeddings(
        self,
        train_x,
        train_y,
        glove: GloVeEmbeddings,
        learning_rate: float,
        iterations: int,
        batch_size: int = 1024
    ):
        """Train model using embedding features.

        Args:
            train_x: List of headline strings
            train_y: Labels (numpy array)
            glove: Loaded GloVe embeddings
            learning_rate: Learning rate for gradient descent
            iterations: Number of training iterations
            batch_size: Batch size for gradient descent

        Returns:
            (costs, weights): Training costs and final weights
        """
        print("Extracting embedding features from training data...")

        train_input, excluded = extract_features_glove_batch(train_x, glove)

        if len(excluded) > 0:
            print(f"Excluded {len(excluded)} headlines with zero matches ({len(excluded)/len(train_x)*100:.2f}%)")
            train_y = np.delete(train_y, excluded)

        train_input, self.mean, self.var = normalize(train_input)

        train_target = train_y.reshape(-1, 1)

        print(f"Training input shape: {train_input.shape}")
        print(f"Training label shape: {train_target.shape}")

        costs, weights = batch_gradient_descent(
            train_input, train_target, self.weights,
            learning_rate, iterations, batch_size
        )

        self.weights = weights
        return costs, weights

    def predict_sentiments_with_embeddings(
        self,
        headlines,
        glove: GloVeEmbeddings
    ):
        """Predict sentiments using embedding features.

        Args:
            headlines: List of headline strings
            glove: Loaded GloVe embeddings

        Returns:
            Tuple of (predicted probabilities, excluded indices)
        """
        features, excluded = extract_features_glove_batch(headlines, glove)

        if len(excluded) > 0:
            print(f"Warning: Excluded {len(excluded)} headlines with zero matches during prediction")

        features = normalize_with_mean_var(features, self.mean, self.var)

        predictions = self.predict(features)
        return predictions, excluded

    def save_embeddings_model(self, model_file: str, mean_var_file: str):
        """Save embedding model (separate from TF-IDF).

        Args:
            model_file: Path to save model weights
            mean_var_file: Path to save mean/variance
        """
        np.save(model_file, self.weights)
        np.save(mean_var_file, {'mean': self.mean, 'var': self.var})
        print(f"Embedding model saved to {model_file}")

    def load_embeddings_model(self, model_file: str, mean_var_file: str):
        """Load embedding model.

        Args:
            model_file: Path to model weights file
            mean_var_file: Path to mean/variance file
        """
        self.weights = np.load(model_file, allow_pickle=True)
        mean_var = np.load(mean_var_file, allow_pickle=True).item()
        self.mean = mean_var['mean']
        self.var = mean_var['var']
        print(f"Embedding model loaded from {model_file}")

    def train_with_embeddings_tfidf(
        self,
        train_x,
        train_y,
        glove: GloVeEmbeddings,
        idf_scores: dict,
        learning_rate: float,
        iterations: int,
        batch_size: int = 1024
    ):
        """Train model using TF-IDF weighted embedding features.

        Args:
            train_x: List of headline strings
            train_y: Labels (numpy array)
            glove: Loaded GloVe embeddings
            idf_scores: Dictionary mapping words to IDF scores
            learning_rate: Learning rate for gradient descent
            iterations: Number of training iterations
            batch_size: Batch size for gradient descent

        Returns:
            (costs, weights): Training costs and final weights
        """
        print("Extracting TF-IDF weighted embedding features from training data...")

        train_input, excluded = extract_features_glove_tfidf_batch(train_x, glove, idf_scores)

        if len(excluded) > 0:
            print(f"Excluded {len(excluded)} headlines with zero matches ({len(excluded)/len(train_x)*100:.2f}%)")
            train_y = np.delete(train_y, excluded)

        train_input, self.mean, self.var = normalize(train_input)

        train_target = train_y.reshape(-1, 1)

        print(f"Training input shape: {train_input.shape}")
        print(f"Training label shape: {train_target.shape}")

        costs, weights = batch_gradient_descent(
            train_input, train_target, self.weights,
            learning_rate, iterations, batch_size
        )

        self.weights = weights
        return costs, weights

    def predict_sentiments_with_embeddings_tfidf(
        self,
        headlines,
        glove: GloVeEmbeddings,
        idf_scores: dict
    ):
        """Predict sentiments using TF-IDF weighted embedding features.

        Args:
            headlines: List of headline strings
            glove: Loaded GloVe embeddings
            idf_scores: Dictionary mapping words to IDF scores

        Returns:
            Tuple of (predicted probabilities, excluded indices)
        """
        features, excluded = extract_features_glove_tfidf_batch(headlines, glove, idf_scores)

        if len(excluded) > 0:
            print(f"Warning: Excluded {len(excluded)} headlines with zero matches during prediction")

        features = normalize_with_mean_var(features, self.mean, self.var)

        predictions = self.predict(features)
        return predictions, excluded

    def train_with_embeddings_max(
        self,
        train_x,
        train_y,
        glove: GloVeEmbeddings,
        learning_rate: float,
        iterations: int,
        batch_size: int = 1024
    ):
        """Train model using max pooling embedding features.

        Args:
            train_x: List of headline strings
            train_y: Labels (numpy array)
            glove: Loaded GloVe embeddings
            learning_rate: Learning rate for gradient descent
            iterations: Number of training iterations
            batch_size: Batch size for gradient descent

        Returns:
            (costs, weights): Training costs and final weights
        """
        print("Extracting max pooling embedding features from training data...")

        train_input, excluded = extract_features_glove_max_batch(train_x, glove)

        if len(excluded) > 0:
            print(f"Excluded {len(excluded)} headlines with zero matches ({len(excluded)/len(train_x)*100:.2f}%)")
            train_y = np.delete(train_y, excluded)

        train_input, self.mean, self.var = normalize(train_input)

        train_target = train_y.reshape(-1, 1)

        print(f"Training input shape: {train_input.shape}")
        print(f"Training label shape: {train_target.shape}")

        costs, weights = batch_gradient_descent(
            train_input, train_target, self.weights,
            learning_rate, iterations, batch_size
        )

        self.weights = weights
        return costs, weights

    def predict_sentiments_with_embeddings_max(
        self,
        headlines,
        glove: GloVeEmbeddings
    ):
        """Predict sentiments using max pooling embedding features.

        Args:
            headlines: List of headline strings
            glove: Loaded GloVe embeddings

        Returns:
            Tuple of (predicted probabilities, excluded indices)
        """
        features, excluded = extract_features_glove_max_batch(headlines, glove)

        if len(excluded) > 0:
            print(f"Warning: Excluded {len(excluded)} headlines with zero matches during prediction")

        features = normalize_with_mean_var(features, self.mean, self.var)

        predictions = self.predict(features)
        return predictions, excluded