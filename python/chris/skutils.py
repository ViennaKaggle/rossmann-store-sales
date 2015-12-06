from functools import wraps

from sklearn.metrics import make_scorer
from sklearn import cross_validation as cv

def score(*args, **kwargs):
    """Decorator, that transform a function to a scorer.
    A scorer has the arguments estimator, X, y_true, sample_weight=None
    """
    decorator_args = args
    decorator_kwargs = kwargs
    def score_decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            func_args = args
            func_kwargs = kwargs
            scorer = make_scorer(func, *decorator_args, **decorator_kwargs)
            return scorer(*func_args, **func_kwargs)
        return func_wrapper
    return score_decorator

def folds(y, n_folds=4, **kwargs):
    return cv.KFold(n=len(y), n_folds=n_folds, shuffle=True, random_state=42, **kwargs)

def cross_val(estimator, X, y, n_jobs=-1, **kwargs):
    # Extract values from pandas DF
    if 'values' in X:
        X = X.values
    if 'values' in y:
        y = y.values
    # Return Cross validation score
    return cv.cross_val_score(estimator, X, y, cv=folds(y), n_jobs=n_jobs, **kwargs)
