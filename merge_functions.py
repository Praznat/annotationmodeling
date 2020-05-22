import pandas as pd
import numpy as np

def numerical_mean(values, weights):
    return np.array([np.array(values[i]) * weights[i] for i in range(len(values))]).sum(axis=0) / np.sum(weights)

