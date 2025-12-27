import numpy as np


def sigmoid(inputs): 
    '''
    Input:
        inputs: is the input (can be a scalar or an array)
    Output:
        activations: the sigmoid of inputs
    '''
    return np.where(
        inputs >= 0,
        1 / (1 + np.exp(-inputs)),
        np.exp(inputs) / (1 + np.exp(inputs))
    )
