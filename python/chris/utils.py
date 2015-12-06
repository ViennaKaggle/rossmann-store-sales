import time, datetime

import numpy as np
import skutils

@skutils.score(greater_is_better=False)
def rmspe(y, y_pred):
    """Root Mean Square Percentage Error
    Details about this function can be found on kaggle 
    https://www.kaggle.com/c/rossmann-store-sales/details/evaluation"""
    # Convert the values to numpy arrays
    y, y_pred = np.array(y), np.array(y_pred)
    # Create a weight vector, and initialize with zeros
    w = np.zeros(y.shape, dtype=float)
    # Create a binary maks containing indecies
    # of all non-zero values 
    idx = y != 0
    # Add weights for all non-zero values
    w[idx] = 1.0 / (y[idx] ** 2)
    # return the error value
    return np.sqrt(np.mean(w * (y[idx] - y_pred[idx]) ** 2))

def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
