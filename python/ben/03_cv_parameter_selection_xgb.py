
# coding: utf-8

# In[1]:

from time import time
import datetime
from operator import itemgetter
import csv

import utils
import data_utils

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation as cv
from sklearn.grid_search import RandomizedSearchCV

import xgboost as xgb

from scipy.stats import randint as sp_randint


# In[2]:

train, test = data_utils.load_transformed_data()
X_train, y_train = data_utils.get_raw_values(train)


# In[3]:

rfr_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'oob_score': False, 'bootstrap': False, 'min_samples_leaf': 6, 'min_samples_split': 40, 'max_features': 13, 'max_depth': 54}
rfr_params = {'n_estimators': 15, 'random_state': 42, 'n_jobs': -1}


# In[4]:

# Utility function to report best scores
def report(grid_scores, n_top=20):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(score.cv_validation_scores)
        print("Mean validation score: {0:.10f} (std: {1:.10f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# In[5]:

n_features = X_train.shape[1]
model = xgb.XGBRegressor()
# specify parameters and distributions to sample from
param_dist = {"objective": ["reg:linear"],
#              "booster" : ["gbtree"],
#              "eta": [0.1, 0.3, 0.5, 0.7],
              "max_depth": sp_randint(1, 2),
              "subsample": [0.1, 0.3, 0.5, 0.7, 0.9],
              "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
              "silent": [1],
              "seed": [42]
             }

# run randomized search
n_iter_search = 1
folds = cv.KFold(n=len(y_train), n_folds=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(model,
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   cv=folds,
                                   n_jobs=-1,
                                   scoring=utils.rmspe_scorer,
                                   iid=False,
                                   error_score=-99.99,
                                   verbose=1
                                  )
start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))


# In[6]:

report(random_search.grid_scores_, n_iter_search)

