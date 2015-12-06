import numpy as np
from sklearn.metrics import make_scorer


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w


def RMSPE(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


def RMSPE_for_log(y, yhat):
    y = np.expm1(y)
    yhat = np.expm1(yhat)
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe


# Create a Scorer
rmspe_scorer = make_scorer(RMSPE, greater_is_better=False)
rmspe_for_log_scorer = make_scorer(RMSPE_for_log, greater_is_better=False)

