from sklearn.metrics import make_scorer

def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error
    Details about this function can be found on kaggle 
    https://www.kaggle.com/c/rossmann-store-sales/details/evaluation"""
    idx = y_true != 0
    return np.sqrt(np.mean(((y_true[idx] - y_pred[idx]) / y_true[idx]) ** 2))

# Create a Scorer
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
