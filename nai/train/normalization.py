import numpy as np

def normalize_with_mean_var(X, mean, var, epsilon=1e-5):
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return X_norm

def normalize(X, epsilon=1e-5):
    mean = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    X_norm = normalize_with_mean_var(X, mean, var, epsilon)
    return X_norm, mean, var
